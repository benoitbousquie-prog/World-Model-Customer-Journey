from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from simulator.config import load_config
from simulator.env.customer_env import CustomerEnv, EnvConfig
from simulator.models.world_model_torch import WorldModel, WorldModelConfig
from simulator.planning.mpc import WorldModelWrapper, MPCPlanner
from simulator.planning.baselines import RandomPolicy, MyopicPolicy
from simulator.evaluation.rollout import eval_policy, eval_policy_in_model


def pick_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def make_env(cfg: dict, seed: int):
    ecfg = EnvConfig(
        episode_length=int(cfg["env"]["episode_length"]),
        purchase_value=float(cfg["env"]["purchase_value"]),
        churn_penalty=float(cfg["env"]["churn_penalty"]),
        discount_rate=float(cfg["env"]["discount_rate"]),
        noise_std=float(cfg["env"]["noise_std"]),
        uplift=dict(cfg["env"]["uplift"]),
        churn=dict(cfg["env"]["churn"]),
        dynamics=dict(cfg["env"]["dynamics"]),
    )
    return CustomerEnv(ecfg, seed=seed)


def load_world_model(cfg: dict, regime: str, device: str):
    ckpt_path = Path("results/checkpoints") / f"world_model_{regime}.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint {ckpt_path}. Train first.")
    wm_cfg = WorldModelConfig(hidden_sizes=list(cfg["model"]["hidden_sizes"]), device=device)
    model = WorldModel(wm_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


KEY_COLS = ["regime", "policy", "H", "gamma", "beta", "seed"]


def already_done(df: pd.DataFrame, row_key: dict) -> bool:
    if df is None or len(df) == 0:
        return False
    mask = np.ones(len(df), dtype=bool)
    for k in KEY_COLS:
        mask &= (df[k] == row_key[k])
    return bool(mask.any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Override config (e.g. configs/confounded_accurate.yaml)")
    ap.add_argument("--base_config", default="configs/default.yaml")
    args = ap.parse_args()

    base = load_config(args.base_config)
    override = load_config(args.config)
    if not isinstance(override, dict):
        raise SystemExit(f"Override config must be dict; got {type(override)}")
    cfg = deep_update(base, override)

    device = pick_device(cfg.get("model", {}).get("device", "auto"))

    purchase_value = float(cfg["env"]["purchase_value"])
    discount_cost = purchase_value * float(cfg["env"]["discount_rate"])
    churn_penalty = float(cfg["env"]["churn_penalty"])
    episode_length = int(cfg["env"]["episode_length"])

    seeds = list(cfg["sweeps"]["seeds"])
    horizons = list(cfg["sweeps"]["horizons"])
    gammas = list(cfg["sweeps"]["gammas"])
    betas = list(cfg["sweeps"]["betas"])
    eval_episodes = int(cfg["sweeps"]["eval_episodes"])

    resume = bool(cfg["sweeps"].get("resume", True))
    out_path = Path(cfg["sweeps"]["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if resume and out_path.exists():
        existing = pd.read_parquet(out_path)
        print(f"Resuming: {out_path} ({len(existing)} rows)")
    else:
        existing = pd.DataFrame(columns=KEY_COLS + [
            "return_mean", "return_std", "purchase_rate", "churn_rate", "discounts_per_ep",
            "sim_return_mean", "sim_return_std", "sim_discounts_per_ep", "sim_done_rate",
            "reality_gap_mean"
        ])

    # CONF0UNDED ONLY
    regime = "confounded"
    model = load_world_model(cfg, regime=regime, device=device)

    for beta in betas:
        wm_wrap = WorldModelWrapper(
            model=model,
            purchase_value=purchase_value,
            discount_cost=discount_cost,
            churn_penalty=churn_penalty,
            device=device,
            beta=float(beta),
        )

        for gamma in gammas:
            for seed in seeds:
                env = make_env(cfg, seed=seed)

                baselines = [
                    ("random", RandomPolicy(seed=seed)),
                    ("myopic", MyopicPolicy(wm_wrap, gamma=float(gamma))),
                ]

                for pol_name, pol in baselines:
                    row_key = {
                        "regime": regime, "policy": pol_name, "H": 1,
                        "gamma": float(gamma), "beta": float(beta), "seed": int(seed)
                    }
                    if already_done(existing, row_key):
                        continue

                    true_m = eval_policy(env, pol, n_episodes=eval_episodes, seed=seed)
                    sim_m = eval_policy_in_model(
                        wm_wrap, pol,
                        n_episodes=eval_episodes,
                        episode_length=episode_length,
                        seed=seed + 555,
                    )
                    row = {**row_key, **true_m, **sim_m}
                    row["reality_gap_mean"] = float(row["sim_return_mean"] - row["return_mean"])

                    existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
                    existing.to_parquet(out_path, index=False)
                    print("Wrote:", row_key)

                for H in horizons:
                    row_key = {
                        "regime": regime, "policy": "mpc", "H": int(H),
                        "gamma": float(gamma), "beta": float(beta), "seed": int(seed)
                    }
                    if already_done(existing, row_key):
                        continue

                    mpc = MPCPlanner(
                        model_wrapper=wm_wrap,
                        horizon=int(H),
                        gamma=float(gamma),
                        K=int(cfg["planner"]["K"]),
                        particles=int(cfg["planner"]["particles"]),
                        seed=seed + 999,
                    )

                    true_m = eval_policy(env, mpc, n_episodes=eval_episodes, seed=seed + 1)
                    sim_m = eval_policy_in_model(
                        wm_wrap, mpc,
                        n_episodes=eval_episodes,
                        episode_length=episode_length,
                        seed=seed + 777,
                    )
                    row = {**row_key, **true_m, **sim_m}
                    row["reality_gap_mean"] = float(row["sim_return_mean"] - row["return_mean"])

                    existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
                    existing.to_parquet(out_path, index=False)
                    print("Wrote:", row_key)

    print(f"Done. Confounded grid written to {out_path} with {len(existing)} rows.")


if __name__ == "__main__":
    main()
