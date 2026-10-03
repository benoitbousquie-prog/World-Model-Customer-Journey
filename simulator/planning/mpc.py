from __future__ import annotations

import numpy as np
import torch

from simulator.env.actions import Action

class WorldModelWrapper:
    """Wraps torch world model and environment reward parameters to support planning."""
    def __init__(self, model, purchase_value: float, discount_cost: float, churn_penalty: float, device: str = "cpu", beta: float = 0.0):
        self.model = model
        self.purchase_value = float(purchase_value)
        self.discount_cost = float(discount_cost)
        self.churn_penalty = float(churn_penalty)
        self.device = device
        self.beta = float(beta)

    def _to_tensors(self, s, a):
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device).view(1, 3)
        a_t = torch.tensor([int(a)], dtype=torch.long, device=self.device)
        return s_t, a_t

    @torch.no_grad()
    def predict(self, s, a):
        s_t, a_t = self._to_tensors(s, a)
        s2, p_buy, p_churn = self.model(s_t, a_t)

        # Optional bias injection beta (simple optimism on discount purchase prob)
        # If beta>0, increase p_buy for discount during imagination
        if int(a) == int(Action.DISCOUNT) and self.beta != 0.0:
            p_buy = torch.clamp(p_buy + self.beta, 0.0, 1.0)

        return (
            s2.squeeze(0).detach().cpu().numpy(),
            float(p_buy.squeeze(0).detach().cpu()),
            float(p_churn.squeeze(0).detach().cpu())
        )

    def expected_immediate_reward(self, s, a):
        _, p_buy, p_churn = self.predict(s, a)
        disc = self.discount_cost if int(a) == int(Action.DISCOUNT) else 0.0
        return self.purchase_value * p_buy - disc - self.churn_penalty * p_churn

    def sample_step(self, s, a, rng: np.random.Generator):
        s2, p_buy, p_churn = self.predict(s, a)
        buy = int(rng.random() < p_buy)
        churn = int(rng.random() < p_churn)
        disc = self.discount_cost if int(a) == int(Action.DISCOUNT) else 0.0
        r = self.purchase_value * buy - disc - self.churn_penalty * churn
        done = bool(churn)
        return s2, r, done

class MPCPlanner:
    def __init__(self, model_wrapper: WorldModelWrapper, horizon: int, gamma: float, K: int, particles: int, seed: int = 0):
        self.m = model_wrapper
        self.H = int(horizon)
        self.gamma = float(gamma)
        self.K = int(K)
        self.particles = int(particles)
        self.rng = np.random.default_rng(seed)

    def act(self, s):
        # random shooting over discrete actions
        actions = np.array([0,1,2,3], dtype=int)
        seqs = self.rng.integers(0, 4, size=(self.K, self.H), dtype=int)

        best_idx = 0
        best_val = -1e18

        for i in range(self.K):
            a_seq = seqs[i]
            # particle estimate
            v = 0.0
            for _p in range(self.particles):
                sp = np.array(s, dtype=np.float32)
                disc = 1.0
                done = False
                ret = 0.0
                for h in range(self.H):
                    if done:
                        break
                    a = Action(int(a_seq[h]))
                    sp, r, done = self.m.sample_step(sp, a, self.rng)
                    ret += disc * r
                    disc *= self.gamma
                v += ret
            v /= float(self.particles)

            if v > best_val:
                best_val = v
                best_idx = i

        return Action(int(seqs[best_idx, 0]))
