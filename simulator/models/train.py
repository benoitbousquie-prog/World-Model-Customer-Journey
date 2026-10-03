from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from simulator.config import load_config
from simulator.models.world_model_torch import WorldModel, WorldModelConfig

class Transitions(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.s = df[["E","F","P"]].to_numpy(np.float32)
        self.a = df["action"].to_numpy(np.int64)
        self.s2 = df[["E_next","F_next","P_next"]].to_numpy(np.float32)
        self.buy = df["purchase"].to_numpy(np.float32)
        self.churn = df["churn"].to_numpy(np.float32)

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.s[i]),
            torch.tensor(self.a[i], dtype=torch.long),
            torch.from_numpy(self.s2[i]),
            torch.tensor(self.buy[i], dtype=torch.float32),
            torch.tensor(self.churn[i], dtype=torch.float32),
        )

def pick_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--regime", required=True, choices=["randomized","confounded"])
    args = ap.parse_args()

    cfg = load_config(args.config)

    data_path = Path("results/datasets") / f"dataset_{args.regime}.parquet"
    if not data_path.exists():
        raise SystemExit(f"Missing dataset: {data_path}. Run generate_logs first.")

    df = pd.read_parquet(data_path)

    # Train/val split by episode id
    eps = df["episode_id"].unique()
    rng = np.random.default_rng(0)
    rng.shuffle(eps)
    cut = int(0.9 * len(eps))
    train_eps = set(eps[:cut])
    train_df = df[df["episode_id"].isin(train_eps)].copy()
    val_df = df[~df["episode_id"].isin(train_eps)].copy()

    train_ds = Transitions(train_df)
    val_ds = Transitions(val_df)

    device = pick_device(cfg["model"]["device"])
    wm_cfg = WorldModelConfig(hidden_sizes=list(cfg["model"]["hidden_sizes"]), device=device)
    model = WorldModel(wm_cfg).to(device)

    lr = float(cfg["model"]["lr"])
    wd = float(cfg["model"]["weight_decay"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    bce = nn.BCELoss()
    mse = nn.MSELoss()

    batch_size = int(cfg["model"]["batch_size"])
    epochs = int(cfg["model"]["epochs"])
    grad_clip = float(cfg["model"]["grad_clip"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    ckpt_dir = Path("results/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"world_model_{args.regime}.pt"

    for ep in range(1, epochs + 1):
        model.train()
        tl = []
        for s, a, s2, buy, churn in tqdm(train_loader, desc=f"Train {args.regime} ep{ep}/{epochs}", leave=False):
            s, a, s2 = s.to(device), a.to(device), s2.to(device)
            buy, churn = buy.to(device), churn.to(device)

            pred_s2, p_buy, p_churn = model(s, a)
            loss = mse(pred_s2, s2) + bce(p_buy, buy) + bce(p_churn, churn)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tl.append(float(loss.detach().cpu()))

        model.eval()
        vl = []
        with torch.no_grad():
            for s, a, s2, buy, churn in val_loader:
                s, a, s2 = s.to(device), a.to(device), s2.to(device)
                buy, churn = buy.to(device), churn.to(device)
                pred_s2, p_buy, p_churn = model(s, a)
                loss = mse(pred_s2, s2) + bce(p_buy, buy) + bce(p_churn, churn)
                vl.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(tl))
        val_loss = float(np.mean(vl))
        print(f"[{args.regime}] epoch {ep}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "config": cfg, "regime": args.regime}, ckpt_path)

    print(f"Saved best checkpoint to {ckpt_path} (val_loss={best_val:.4f})")

if __name__ == "__main__":
    main()
