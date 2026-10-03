from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from simulator.config import load_config
from simulator.env.customer_env import CustomerEnv, EnvConfig
from simulator.env.policies import RandomizedLoggingPolicy, ConfoundedLoggingPolicy
from simulator.env.actions import Action
from simulator.data.schema import COLUMNS

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--regime", required=True, choices=["randomized", "confounded"])
    ap.add_argument("--out", default=None, help="Optional output path (parquet).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_episodes = int(cfg["data"]["n_episodes"])
    seed = int(cfg["data"]["seed"])

    env = make_env(cfg, seed=seed)

    if args.regime == "randomized":
        pol = RandomizedLoggingPolicy(cfg["logging_policies"]["randomized"]["action_probs"], seed=seed + 1)
    else:
        c = cfg["logging_policies"]["confounded"]
        pol = ConfoundedLoggingPolicy(
            discount_when_low_P=float(c["discount_when_low_P"]),
            email_when_high_E=float(c["email_when_high_E"]),
            none_when_high_F=float(c["none_when_high_F"]),
            seed=seed + 2,
        )

    rows = []
    for ep in tqdm(range(n_episodes), desc=f"Generating {args.regime} logs"):
        s = env.reset(seed=seed + ep)
        done = False
        t = 0
        while not done:
            a = pol.act(s)
            s2, r, done, info = env.step(a)
            rows.append([
                ep, t,
                float(s[0]), float(s[1]), float(s[2]),
                int(a),
                float(r),
                int(info["purchase"]), int(info["churn"]),
                float(info["p_buy"]), float(info["p_churn"]),
                float(s2[0]), float(s2[1]), float(s2[2]),
                int(done),
            ])
            s = s2
            t += 1

    df = pd.DataFrame(rows, columns=COLUMNS)

    out_dir = Path("results/datasets")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (out_dir / f"dataset_{args.regime}.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")

    # Quick summary
    print("\nSummary:")
    print(df[["reward","purchase","churn"]].mean())
    print("Action frequencies:", df["action"].value_counts(normalize=True).sort_index().to_dict())

if __name__ == "__main__":
    main()
