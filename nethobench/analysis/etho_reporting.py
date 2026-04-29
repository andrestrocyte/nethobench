from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------


def plot_loaded_data(gt_df: pd.DataFrame, inf_df: pd.DataFrame, outdir: Path) -> None:
    """
    Plots basic dataset statistics: total rows and unique sequence counts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Rows
    axes[0].bar(
        ["GT", "INF"],
        [len(gt_df), len(inf_df)],
        color=["tab:blue", "tab:orange"],
        alpha=0.8,
    )
    axes[0].set_title("Loaded Rows per Domain")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.3)

    # Sequences
    gt_seqs = gt_df["sequenceId"].nunique() if "sequenceId" in gt_df else 0
    inf_seqs = inf_df["sequenceId"].nunique() if "sequenceId" in inf_df else 0
    axes[1].bar(
        ["GT", "INF"], [gt_seqs, inf_seqs], color=["tab:blue", "tab:orange"], alpha=0.8
    )
    axes[1].set_title("Unique Sequences Loaded")
    axes[1].set_ylabel("Count")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "loaded_data_stats.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_error_distributions(errors_df: pd.DataFrame, outdir: Path) -> None:
    """
    Plots the RMSE for each tracked body part.
    """
    if errors_df.empty:
        logger.warning("No error data available to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    # Sort for better visual hierarchy
    plot_df = errors_df.sort_values("rmse_pos", ascending=False)

    ax.bar(
        plot_df["body_part"],
        plot_df["rmse_pos"],
        color="tab:blue",
        alpha=0.8,
        edgecolor="k",
    )
    ax.set_xticks(range(len(plot_df["body_part"])))
    ax.set_xticklabels(plot_df["body_part"], rotation=45, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Per-Body-Part Position Error (RMSE)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "body_part_rmse.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_kinematics(paired_df: pd.DataFrame, outdir: Path) -> None:
    """
    Computes and plots the speed distribution of the CENTER track for GT vs Inference.
    """
    if paired_df.empty or "CENTER_X_gt" not in paired_df.columns:
        logger.warning("Missing CENTER tracking data for kinematics plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    for label, color, name in [
        ("gt", "tab:blue", "Ground Truth"),
        ("inf", "tab:orange", "Inference"),
    ]:
        speeds = []
        for seq, seq_df in paired_df.sort_values("itemPosition").groupby("sequenceId"):
            coords = seq_df[[f"CENTER_X_{label}", f"CENTER_Y_{label}"]].to_numpy()
            if len(coords) > 1:
                vel = np.diff(coords, axis=0)
                speeds.append(np.linalg.norm(vel, axis=1))

        if speeds:
            all_speeds = np.concatenate(speeds)
            all_speeds = all_speeds[np.isfinite(all_speeds)]
            # Filter extreme outliers for visualization
            if len(all_speeds) > 0:
                p99 = np.percentile(all_speeds, 99)
                ax.hist(
                    all_speeds[all_speeds < p99],
                    bins=50,
                    alpha=0.5,
                    label=name,
                    color=color,
                    density=True,
                )

    ax.set_xlabel("Speed (pixels/frame)")
    ax.set_ylabel("Density")
    ax.set_title("Speed Distribution (CENTER Track)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "kinematics_speed_dist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_geometry(distances_df: pd.DataFrame, outdir: Path) -> None:
    """
    Plots the mean inter-limb distances comparing GT and Inference.
    """
    if distances_df.empty:
        logger.warning("No inter-limb distance data to plot.")
        return

    try:
        pivot = distances_df.pivot(index="pair", columns="label", values="mean")
    except Exception as e:
        logger.warning(f"Could not pivot distance dataframe for plotting: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    colors = {"gt": "tab:blue", "inf": "tab:orange"}
    plot_colors = [colors.get(c, "gray") for c in pivot.columns]

    pivot.plot(kind="bar", ax=ax, color=plot_colors, alpha=0.8, edgecolor="k")

    ax.set_ylabel("Mean Distance")
    ax.set_xlabel("Body Part Pair")
    ax.set_title("Inter-limb Distances (GT vs INF)")
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Source")

    fig.tight_layout()
    fig.savefig(
        outdir / "geometry_interlimb_distances.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def plot_embeddings(
    chunk_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]], outdir: Path
) -> None:
    """
    Plots the 2D chunk embedding (UMAP or PCA) colored by domain label.
    """
    if chunk_data is None:
        logger.warning("No chunk embedding data provided to plot.")
        return

    emb, labels, seq_ids = chunk_data
    if len(emb) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"gt": "tab:blue", "inf": "tab:orange", "gt_shuffled": "tab:gray"}
    names = {
        "gt": "Ground Truth",
        "inf": "Inference",
        "gt_shuffled": "GT (Shuffled Baseline)",
    }

    for lbl in np.unique(labels):
        mask = labels == lbl
        c = colors.get(lbl, "black")
        n = names.get(lbl, lbl)
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.5, label=n, color=c)

    ax.set_title("Movement Chunk Embeddings (5-Frame Windows)")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(outdir / "manifold_chunk_embeddings.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------
# Master Reporting Entrypoint
# ---------------------------------------------------------


def generate_full_etho_report(
    gt_df: pd.DataFrame,
    inf_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    distances_df: pd.DataFrame,
    chunk_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    outdir: Path,
) -> None:
    """
    Executes the full suite of behavioral visualizations and saves them to the specified directory.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating loaded data statistics...")
    plot_loaded_data(gt_df, inf_df, outdir)

    logger.info("Generating error distribution plots...")
    plot_error_distributions(errors_df, outdir)

    logger.info("Generating kinematics distributions...")
    plot_kinematics(paired_df, outdir)

    logger.info("Generating geometry comparisons...")
    plot_geometry(distances_df, outdir)

    logger.info("Generating 2D movement manifold embeddings...")
    plot_embeddings(chunk_data, outdir)

    logger.info(f"Successfully exported all behavioral plots to {outdir}")
