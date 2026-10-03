#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOMIZED_GRID = Path("results/grids/grid_pdf_strategy1.parquet")
CONFOUNDED_GRID = Path("results/grids/grid_confounded_accurate.parquet")


def save_both(fig, stem: str, dpi: int = 200):
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    print("Wrote:", pdf_path)
    print("Wrote:", png_path)


def load_data():
    if not RANDOMIZED_GRID.exists():
        raise FileNotFoundError(f"Missing {RANDOMIZED_GRID}")
    if not CONFOUNDED_GRID.exists():
        raise FileNotFoundError(f"Missing {CONFOUNDED_GRID}")

    r = pd.read_parquet(RANDOMIZED_GRID)
    c = pd.read_parquet(CONFOUNDED_GRID)

    # Keep only MPC rows for gap heatmaps
    r_mpc = r[(r["policy"] == "mpc") & (r["regime"] == "randomized")].copy()
    c_mpc = c[(c["policy"] == "mpc") & (c["regime"] == "confounded")].copy()

    return r, c, r_mpc, c_mpc


def heatmap_gap(df_mpc: pd.DataFrame, gamma: float, title: str):
    d = df_mpc[df_mpc["gamma"] == gamma].copy()
    if d.empty:
        raise ValueError(f"No rows for gamma={gamma} in dataframe")

    # mean gap over seeds
    agg = d.groupby(["beta", "H"], as_index=False)["reality_gap_mean"].mean()
    piv = agg.pivot(index="beta", columns="H", values="reality_gap_mean").sort_index()

    fig, ax = plt.subplots(figsize=(9, 3.8))
    sns.heatmap(
        piv,
        ax=ax,
        cmap="RdBu_r",
        center=0.0,
        cbar_kws={"label": "Reality gap (sim − true)"},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(title)
    ax.set_xlabel("Planning horizon H")
    ax.set_ylabel("Injected bias β")
    return fig


def line_return_vs_H(df: pd.DataFrame, regime: str, gamma: float, beta: float):
    d = df[(df["regime"] == regime) & (df["gamma"] == gamma) & (df["beta"] == beta)].copy()
    if d.empty:
        raise ValueError(f"No rows for regime={regime}, gamma={gamma}, beta={beta}")

    # average over seeds
    agg = d.groupby(["policy", "H"], as_index=False)["return_mean"].mean()

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for pol in ["myopic", "mpc"]:
        dd = agg[agg["policy"] == pol].sort_values("H")
        if dd.empty:
            continue
        ax.plot(dd["H"], dd["return_mean"], marker="o", label=pol)

    ax.set_title(f"True return vs horizon (regime={regime}, γ={gamma}, β={beta})")
    ax.set_xlabel("Horizon H")
    ax.set_ylabel("True return (ground truth)")
    ax.legend()
    return fig


def main():
    r, c, r_mpc, c_mpc = load_data()

    # ---- Gap heatmaps ----
    # Choose representative gammas (adjust if you want)
    for gamma in [0.95, 0.99]:
        if not r_mpc[r_mpc["gamma"] == gamma].empty:
            fig = heatmap_gap(
                r_mpc, gamma=gamma,
                title=f"Randomized regime: reality gap heatmap (γ={gamma})"
            )
            save_both(fig, f"fig_gap_heatmap_randomized_gamma_{gamma}")
            plt.close(fig)

        if not c_mpc[c_mpc["gamma"] == gamma].empty:
            fig = heatmap_gap(
                c_mpc, gamma=gamma,
                title=f"Confounded regime: reality gap heatmap (γ={gamma})"
            )
            save_both(fig, f"fig_gap_heatmap_confounded_gamma_{gamma}")
            plt.close(fig)

    # For the paper placeholders in main.tex we also write default names (γ=0.95 if available)
    default_gamma = 0.95
    if not r_mpc[r_mpc["gamma"] == default_gamma].empty:
        fig = heatmap_gap(r_mpc, gamma=default_gamma, title=f"Randomized regime: reality gap heatmap (γ={default_gamma})")
        save_both(fig, "fig_gap_heatmap_randomized")
        plt.close(fig)

    if not c_mpc[c_mpc["gamma"] == default_gamma].empty:
        fig = heatmap_gap(c_mpc, gamma=default_gamma, title=f"Confounded regime: reality gap heatmap (γ={default_gamma})")
        save_both(fig, "fig_gap_heatmap_confounded")
        plt.close(fig)

    # ---- True return vs H lines (MPC vs myopic) ----
    # Choose beta=0 as default for clean comparison; adjust as needed
    for regime, df in [("randomized", r), ("confounded", c)]:
        for gamma in [0.95, 0.99]:
            try:
                fig = line_return_vs_H(df, regime=regime, gamma=gamma, beta=0.0)
                save_both(fig, f"fig_return_vs_H_{regime}_gamma_{gamma}_beta_0")
                plt.close(fig)
            except ValueError:
                pass

    print("\nDone. Figures written to ./figures/")

if __name__ == "__main__":
    main()
