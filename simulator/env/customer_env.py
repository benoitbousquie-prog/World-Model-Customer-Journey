from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from simulator.env.actions import Action, ACTION_NAMES

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))

@dataclass
class EnvConfig:
    episode_length: int
    purchase_value: float
    churn_penalty: float
    discount_rate: float
    noise_std: float

    uplift: dict
    churn: dict
    dynamics: dict

class CustomerEnv:
    """Ground-truth environment (authoritative reality).

    State s=(E,F,P) in [0,1]^3.
    Multiple purchases allowed (<=1 per step).
    Churn ends the episode.
    """

    def __init__(self, cfg: EnvConfig, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.E = 0.0
        self.F = 0.0
        self.P = 0.0
        self.done = False

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        # Initial states: mildly engaged, low fatigue, modest propensity
        self.E = float(self.rng.beta(2, 2))  # centered around 0.5
        self.F = float(self.rng.beta(2, 6))  # skew low
        self.P = float(self.rng.beta(2, 3))  # modest
        self.done = False
        return self._obs()

    def _obs(self):
        return np.array([self.E, self.F, self.P], dtype=np.float32)

    def step(self, action: Action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        a_name = ACTION_NAMES[Action(int(action))]

        # ---- purchase probability (depends on P, uplift, and fatigue suppression)
        uplift = float(self.cfg.uplift[a_name])
        p_buy = np.clip(self.P + uplift - float(self.cfg.dynamics["purchase_fatigue_suppress"]) * self.F, 0.0, 1.0)
        purchase = int(self.rng.random() < p_buy)

        # ---- churn probability (fatigue-driven)
        tau = float(self.cfg.churn["tau"])
        k = float(self.cfg.churn["k"])
        p_churn = sigmoid(k * (self.F - tau))
        churn = int(self.rng.random() < p_churn)

        # ---- reward (revenue-focused)
        discount_cost = (self.cfg.purchase_value * self.cfg.discount_rate) if a_name == "discount" else 0.0
        reward = (self.cfg.purchase_value * purchase) - discount_cost - (self.cfg.churn_penalty * churn)

        # ---- dynamics update with small noise
        ns = float(self.cfg.noise_std)
        epsE, epsF, epsP = self.rng.normal(0.0, ns, size=3)

        # engagement update
        dE = self.cfg.dynamics["E"]
        E_next = self.E                     + float(dE["email_boost"]) * (1 if a_name == "email" else 0)                     + float(dE["reco_boost"]) * (1 if a_name == "reco" else 0)                     - float(dE["fatigue_drag"]) * self.F                     + float(epsE)
        E_next = _clip01(E_next)

        # fatigue update
        dF = self.cfg.dynamics["F"]
        contact = 1 if a_name != "none" else 0
        F_next = float(dF["decay"]) * self.F                     + float(dF["contact_add"]) * contact                     + float(dF["discount_add"]) * (1 if a_name == "discount" else 0)                     + float(epsF)
        F_next = _clip01(F_next)

        # propensity update
        dP = self.cfg.dynamics["P"]
        P_next = float(dP["decay"]) * self.P                     + float(dP["engagement_gain"]) * E_next                     - float(dP["fatigue_drag"]) * F_next                     + float(dP["purchase_bump"]) * purchase                     + float(epsP)
        P_next = _clip01(P_next)

        # apply terminal condition
        self.E, self.F, self.P = E_next, F_next, P_next
        self.t += 1
        if churn == 1 or self.t >= self.cfg.episode_length:
            self.done = True

        info = {
            "t": self.t,
            "p_buy": float(p_buy),
            "p_churn": float(p_churn),
            "purchase": int(purchase),
            "churn": int(churn),
            "action_name": a_name,
            "discount_cost": float(discount_cost),
        }
        return self._obs(), float(reward), self.done, info
