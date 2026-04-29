from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Force non-interactive backend for headless execution
matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def plot_cross_composites(scores: dict, outdir: Path) -> None:
    """Plots a summary bar chart of the cross-modal composites and sub-scores."""
    cross_scores = scores.get("cross_scores", {})

    # Extract plottable scores
    labels = []
    gt_vals = []
    pr_vals = []

    if "cca_gt" in cross_scores:
        labels.append("CCA Mean")
        gt_vals.append(cross_scores["cca_gt"])
        pr_vals.append(cross_scores["cca_pred"])

    if "r2_neural_to_behavior_gt" in cross_scores:
        labels.append("R2 (N->B)")
        gt_vals.append(cross_scores["r2_neural_to_behavior_gt"])
        pr_vals.append(cross_scores["r2_neural_to_behavior_pred"])

    if "r2_behavior_to_neural_gt" in cross_scores:
        labels.append("R2 (B->N)")
        gt_vals.append(cross_scores["r2_behavior_to_neural_gt"])
        pr_vals.append(cross_scores["r2_behavior_to_neural_pred"])

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x - width / 2,
        gt_vals,
        width,
        label="Ground Truth",
        color="#1f77b4",
        edgecolor="k",
    )
    ax.bar(
        x + width / 2, pr_vals, width, label="Predicted", color="#ff7f0e", edgecolor="k"
    )

    ax.set_ylabel("Score")
    ax.set_title(
        f"Cross-Modal Similarities | Composite: {scores.get('cross_composite', np.nan):.4f}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(outdir / "cross_composites.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lead_lag(scores: dict, outdir: Path) -> None:
    """Plots the lead/lag time offsets between neural PC1 and behavior speed."""
    cross_scores = scores.get("cross_scores", {})
    lag_gt = cross_scores.get("lead_lag_gt", np.nan)
    lag_pr = cross_scores.get("lead_lag_pred", np.nan)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Ground Truth", "Predicted"],
        [lag_gt, lag_pr],
        color=["#1f77b4", "#ff7f0e"],
        edgecolor="k",
    )
    ax.set_ylabel("Lead/Lag (Frames)")
    ax.set_title(
        f"Neural PC1 vs Behavior Speed Lead/Lag | Score: {cross_scores.get('lead_lag_score', np.nan):.4f}"
    )
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(outdir / "lead_lag_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_full_cross_report(scores: dict, outdir: Path) -> None:
    """Executes the full suite of cross-modal visualizations."""
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating Cross-Modal Composites...")
    plot_cross_composites(scores, outdir)

    logger.info("Generating Lead/Lag Comparisons...")
    plot_lead_lag(scores, outdir)

    logger.info(f"Successfully exported all cross-modal plots to {outdir}")
