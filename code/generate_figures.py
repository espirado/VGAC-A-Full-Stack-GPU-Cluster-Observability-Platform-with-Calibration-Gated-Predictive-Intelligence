#!/usr/bin/env python3
"""
Paper 4 Figure Generation

Generates publication-quality figures from experiment results:
  1. 5×5 AUROC heatmap
  2. 5×5 ECE heatmap
  3. JS divergence vs ECE degradation scatter + regression
  4. Few-shot recalibration trajectory
  5. SHAP feature importance
  6. JS divergence bar chart
  7. Brier decomposition bar chart
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

ENV_ORDER = ["EKS-P3", "EKS-Nov", "Slurm-HPC", "Alibaba-2020", "Alibaba-2023"]
ENV_SHORT = ["EKS-P3", "EKS-Nov", "Slurm", "Ali-20", "Ali-23"]


def load_transfer_matrix(path: Path | None = None) -> pd.DataFrame:
    path = path or RESULTS_DIR / "transfer_matrix.csv"
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Figure 1: Transfer Matrix Heatmaps (5×5 AUROC + ECE)
# ---------------------------------------------------------------------------

def plot_transfer_heatmaps(df: pd.DataFrame, model: str = "GB"):
    """Side-by-side AUROC and ECE heatmaps."""
    sub = df[df["model"] == model].copy()

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    for ax, metric, title, cmap, fmt, vmin, vmax in [
        (axes[0], "auroc", "(a) AUROC (discrimination)", "YlGnBu", ".3f", 0.4, 1.0),
        (axes[1], "ece",   "(b) ECE (calibration)",      "YlOrRd", ".3f", 0.0, 0.8),
    ]:
        # Build matrix
        envs_present = [e for e in ENV_ORDER if e in sub["source"].unique()]
        n = len(envs_present)
        matrix = np.full((n, n), np.nan)

        for i, src in enumerate(envs_present):
            for j, tgt in enumerate(envs_present):
                row = sub[(sub["source"] == src) & (sub["target"] == tgt)]
                if len(row) > 0:
                    matrix[i, j] = row[metric].iloc[0]

        short_labels = [ENV_SHORT[ENV_ORDER.index(e)] for e in envs_present]

        sns.heatmap(
            matrix, ax=ax, annot=True, fmt=fmt, cmap=cmap,
            xticklabels=short_labels, yticklabels=short_labels,
            vmin=vmin, vmax=vmax, linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")

    plt.tight_layout()
    out = FIGURES_DIR / "transfer_matrix_5x5.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: JS Divergence vs ECE Degradation
# ---------------------------------------------------------------------------

def plot_js_vs_ece(path: Path | None = None):
    """Scatter plot: JS divergence predicts ECE degradation."""
    path = path or RESULTS_DIR / "js_ece_merged.csv"
    if not path.exists():
        print(f"  SKIP js_vs_ece: {path} not found")
        return

    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(4, 3))

    # Color by scheduler match
    colors = df["scheduler_mismatch"].map({0: "#2196F3", 1: "#F44336"}).values
    labels_mapped = df["scheduler_mismatch"].map({0: "Same family", 1: "Cross-family"}).values

    for label, color in [("Same family", "#2196F3"), ("Cross-family", "#F44336")]:
        mask = labels_mapped == label
        ax.scatter(df.loc[mask, "js_divergence"], df.loc[mask, "ece_ratio"],
                   c=color, label=label, s=40, alpha=0.8, edgecolors="white", linewidths=0.5)

    # Regression line
    from numpy.polynomial.polynomial import polyfit
    x = df["js_divergence"].values
    y = df["ece_ratio"].values
    b, m = polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, b + m * x_line, "--", color="gray", linewidth=1.5, alpha=0.7)

    # R² annotation
    reg_path = RESULTS_DIR / "js_regression.json"
    if reg_path.exists():
        with open(reg_path) as f:
            reg = json.load(f)
        ax.text(0.05, 0.95, f"$R^2 = {reg['r2']:.2f}$",
                transform=ax.transAxes, fontsize=9, va="top")

    ax.set_xlabel("JS divergence (pending_ratio)")
    ax.set_ylabel("ECE degradation ratio")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title("JS Divergence Predicts ECE Degradation")

    plt.tight_layout()
    out = FIGURES_DIR / "js_vs_ece.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Recalibration Trajectory
# ---------------------------------------------------------------------------

def plot_recalibration_trajectory(path: Path | None = None):
    """ECE recovery as function of recalibration sample size."""
    path = path or RESULTS_DIR / "fewshot_recalibration.csv"
    if not path.exists():
        print(f"  SKIP recalibration_trajectory: {path} not found")
        return

    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(4.5, 3))

    # Aggregate by (source→target, n_samples)
    pairs = df.groupby(["source", "target"])
    for (src, tgt), grp in pairs:
        grp = grp.sort_values("n_samples")
        label = f"{src[:6]}→{tgt[:6]}"
        ax.plot(grp["n_samples"], grp["recal_ece"], "o-", markersize=4, label=label, alpha=0.7)

    # Reference: add raw ECE as horizontal line at n=0
    ax.axhline(y=0.05, color="green", linestyle="--", linewidth=1, alpha=0.7, label="SLO threshold (0.05)")

    ax.set_xlabel("Target samples for recalibration")
    ax.set_ylabel("ECE after recalibration")
    ax.set_title("Few-Shot Recalibration Recovery")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = FIGURES_DIR / "recalibration_trajectory.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: SHAP Feature Importance
# ---------------------------------------------------------------------------

def plot_shap_importance(path: Path | None = None):
    """Bar chart of SHAP values per environment."""
    path = path or RESULTS_DIR / "shap_importance.csv"
    if not path.exists():
        print(f"  SKIP shap_importance: {path} not found")
        return

    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(5, 3))

    envs = df["env"].unique()
    features = df["feature"].unique()
    n_envs = len(envs)
    n_feats = len(features)
    width = 0.8 / n_envs

    colors = sns.color_palette("Set2", n_envs)
    x = np.arange(n_feats)

    for i, env in enumerate(envs):
        env_df = df[df["env"] == env]
        vals = [env_df[env_df["feature"] == f]["mean_abs_shap"].values[0] if f in env_df["feature"].values else 0 for f in features]
        ax.bar(x + i * width, vals, width, label=env, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width * (n_envs - 1) / 2)
    ax.set_xticklabels(features, rotation=20, ha="right")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title("Feature Importance Across Environments")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out = FIGURES_DIR / "shap_importance.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: JS Divergence Bar Chart
# ---------------------------------------------------------------------------

def plot_js_divergence_bars(path: Path | None = None):
    """Horizontal bar chart of JS divergence between environment pairs."""
    path = path or RESULTS_DIR / "js_divergence_pairs.csv"
    if not path.exists():
        print(f"  SKIP js_divergence_bars: {path} not found")
        return

    df = pd.read_csv(path)
    df["pair"] = df["source"].str[:6] + " ↔ " + df["target"].str[:6]
    df = df.sort_values("js_divergence", ascending=True)

    # Deduplicate symmetric pairs
    seen = set()
    unique_rows = []
    for _, row in df.iterrows():
        key = tuple(sorted([row["source"], row["target"]]))
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    df = pd.DataFrame(unique_rows)
    df = df.sort_values("js_divergence", ascending=True)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    colors = ["#F44336" if row["scheduler_mismatch"] else "#2196F3" for _, row in df.iterrows()]
    ax.barh(df["pair"], df["js_divergence"], color=colors, alpha=0.85)
    ax.set_xlabel("Jensen-Shannon Divergence (pending_ratio)")
    ax.set_title("Distribution Shift Between Environments")

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#2196F3", label="Same scheduler family"),
        Patch(color="#F44336", label="Cross-scheduler"),
    ], fontsize=7, loc="lower right")

    plt.tight_layout()
    out = FIGURES_DIR / "js_divergence.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Brier Decomposition
# ---------------------------------------------------------------------------

def plot_brier_decomposition(path: Path | None = None):
    """Stacked bar chart of Brier decomposition."""
    path = path or RESULTS_DIR / "brier_decomposition.csv"
    if not path.exists():
        print(f"  SKIP brier_decomposition: {path} not found")
        return

    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(5, 3))

    x = np.arange(len(df))
    width = 0.6

    ax.bar(x, df["reliability"], width, label="Reliability ↓", color="#F44336", alpha=0.85)
    ax.bar(x, df["resolution"], width, bottom=df["reliability"], label="Resolution ↑", color="#4CAF50", alpha=0.85)
    ax.bar(x, df["uncertainty"], width, bottom=df["reliability"] + df["resolution"], label="Uncertainty", color="#9E9E9E", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df["env"], rotation=20, ha="right")
    ax.set_ylabel("Brier Score Component")
    ax.set_title("Brier Decomposition (GB, In-Domain)")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out = FIGURES_DIR / "brier_decomposition.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Paper 4 figures")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--figures-dir", type=str, default=str(FIGURES_DIR))
    args = parser.parse_args()

    global RESULTS_DIR, FIGURES_DIR
    RESULTS_DIR = Path(args.results_dir)
    FIGURES_DIR = Path(args.figures_dir)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Results from: {RESULTS_DIR}")
    print(f"Figures to:   {FIGURES_DIR}")

    # Load transfer matrix
    tm_path = RESULTS_DIR / "transfer_matrix.csv"
    if tm_path.exists():
        tm = load_transfer_matrix(tm_path)
        plot_transfer_heatmaps(tm)
    else:
        print(f"  SKIP transfer heatmaps: {tm_path} not found")

    plot_js_vs_ece()
    plot_recalibration_trajectory()
    plot_shap_importance()
    plot_js_divergence_bars()
    plot_brier_decomposition()

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
