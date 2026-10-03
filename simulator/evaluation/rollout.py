from __future__ import annotations

import numpy as np
from tqdm import tqdm

from simulator.env.actions import Action


def eval_policy(env, policy, n_episodes: int, seed: int = 0):
    """
    Evaluate a policy in the *ground-truth* environment.
    Returns undiscounted episodic return and diagnostics.
    """
    ep_returns = []
    ep_purchases = []
    ep_churns = []
    ep_discounts = []

    for ep in tqdm(range(n_episodes), desc="Evaluating (true env)", leave=False):
        s = env.reset(seed=seed + ep + 100000)
        done = False
        G = 0.0
        purchases = 0
        churned = 0
        discounts = 0

        while not done:
            a = policy.act(s)
            if int(a) == int(Action.DISCOUNT):
                discounts += 1
            s, r, done, info = env.step(a)
            G += r
            purchases += int(info["purchase"])
            churned = max(churned, int(info["churn"]))

        ep_returns.append(G)
        ep_purchases.append(purchases)
        ep_churns.append(churned)
        ep_discounts.append(discounts)

    return {
        "return_mean": float(np.mean(ep_returns)),
        "return_std": float(np.std(ep_returns)),
        "purchase_rate": float(np.mean(ep_purchases) / env.cfg.episode_length),
        "churn_rate": float(np.mean(ep_churns)),
        "discounts_per_ep": float(np.mean(ep_discounts)),
    }


def eval_policy_in_model(model_wrapper, policy, n_episodes: int, episode_length: int, seed: int = 0):
    """
    Evaluate the same policy *inside the world model* (imagination).
    This supports the PDF requirement: gap widens with horizon H and injected bias beta.
    """
    rng = np.random.default_rng(seed)
    ep_returns = []
    ep_discounts = []
    ep_done = []

    for ep in tqdm(range(n_episodes), desc="Evaluating (world model)", leave=False):
        # Same initial-state distribution as env.reset()
        E = float(rng.beta(2, 2))
        F = float(rng.beta(2, 6))
        P = float(rng.beta(2, 3))
        s = np.array([E, F, P], dtype=np.float32)

        done = False
        G = 0.0
        discounts = 0

        for t in range(int(episode_length)):
            if done:
                break
            a = policy.act(s)
            if int(a) == int(Action.DISCOUNT):
                discounts += 1
            s, r, done = model_wrapper.sample_step(s, a, rng)
            G += r

        ep_returns.append(G)
        ep_discounts.append(discounts)
        ep_done.append(int(done))

    return {
        "sim_return_mean": float(np.mean(ep_returns)),
        "sim_return_std": float(np.std(ep_returns)),
        "sim_discounts_per_ep": float(np.mean(ep_discounts)),
        "sim_done_rate": float(np.mean(ep_done)),
    }
