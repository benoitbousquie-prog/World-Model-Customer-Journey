from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="World Model Journeys Explorer", layout="wide")

grid_path = Path("results/grids/grid_pdf_strategy1.parquet")

st.title("World Model Journeys — Explorer (precomputed results)")

if not grid_path.exists():
    st.error(f"Missing {grid_path}. Run sweeps first.")
    st.stop()

df = pd.read_parquet(grid_path)

col1, col2, col3, col4 = st.columns(4)
with col1:
    regime = st.selectbox("Training regime", sorted(df["regime"].unique()))
with col2:
    gamma = st.selectbox("Gamma (γ)", sorted(df["gamma"].unique()))
with col3:
    # allow selecting one beta for the line charts
    beta = st.selectbox("Bias β (for line charts)", sorted(df["beta"].unique()))
with col4:
    show_policies = st.multiselect("Policies", sorted(df["policy"].unique()), default=sorted(df["policy"].unique()))

d = df[
    (df["regime"] == regime) &
    (df["gamma"] == gamma) &
    (df["beta"] == beta) &
    (df["policy"].isin(show_policies))
].copy()

st.subheader("Revenue vs Horizon")
fig = px.line(
    d.sort_values("H"),
    x="H", y="return_mean", color="policy",
    error_y="return_std",
    markers=True,
    title="True return in ground-truth env (mean ± std)"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Churn rate vs Horizon")
fig2 = px.line(d.sort_values("H"), x="H", y="churn_rate", color="policy", markers=True)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Discounts per episode vs Horizon")
fig3 = px.line(d.sort_values("H"), x="H", y="discounts_per_ep", color="policy", markers=True)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Reality gap heatmap (MPC only): simulated − true")
d_mpc = df[(df["regime"] == regime) & (df["gamma"] == gamma) & (df["policy"] == "mpc")].copy()
if len(d_mpc) == 0:
    st.info("No MPC rows for this regime/gamma yet (sweeps may still be running).")
else:
    heat = d_mpc.pivot_table(index="beta", columns="H", values="reality_gap_mean", aggfunc="mean")
    fig4 = px.imshow(
        heat.sort_index(),
        aspect="auto",
        title="Reality gap (mean over seeds) — positive means the world model is optimistic",
        labels=dict(x="Horizon H", y="Bias β", color="sim - true")
    )
    st.plotly_chart(fig4, use_container_width=True)

st.subheader("Raw table (filtered)")
st.dataframe(d.sort_values(["policy", "H"]), use_container_width=True)
