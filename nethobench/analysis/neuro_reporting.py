from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------


def plot_sanity_checks(
    gt_arr: np.ndarray, pred_arr: np.ndarray, region_names: list[str], outdir: Path
) -> None:
    """
    Plots the raw time series traces for a subset of sequences to verify
    scale and alignment before metrics are interpreted.
    """
    n_seq, n_time, n_reg = gt_arr.shape
    num_trials = min(4, n_seq)

    # Pick evenly spaced sequences to visualize
    seq_trials = np.linspace(0, n_seq - 1, num_trials, dtype=int)

    # --- Plot Ground Truth ---
    fig_gt, axes_gt = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes_gt = np.atleast_1d(axes_gt).flatten()

    for ax, seq in zip(axes_gt, seq_trials):
        for r in range(n_reg):
            ax.plot(gt_arr[seq, :, r], label=region_names[r], alpha=0.8)
        ax.set_title(f"Ground Truth - Sequence {seq}")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    axes_gt[-1].set_xlabel("Time step")
    axes_gt[-2].set_xlabel("Time step")

    handles_gt, labels_gt = axes_gt[0].get_legend_handles_labels()
    fig_gt.legend(
        handles_gt,
        labels_gt,
        loc="upper center",
        ncol=min(len(labels_gt), 8),
        bbox_to_anchor=(0.5, 1.03),
    )
    fig_gt.tight_layout(rect=(0, 0, 1, 0.95))

    fig_gt.savefig(outdir / "sanity_traces_gt.png", dpi=200, bbox_inches="tight")
    plt.close(fig_gt)

    # --- Plot Predictions ---
    fig_pr, axes_pr = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes_pr = np.atleast_1d(axes_pr).flatten()

    for ax, seq in zip(axes_pr, seq_trials):
        for r in range(n_reg):
            ax.plot(pred_arr[seq, :, r], label=region_names[r], alpha=0.8)
        ax.set_title(f"Prediction - Sequence {seq}")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    axes_pr[-1].set_xlabel("Time step")
    axes_pr[-2].set_xlabel("Time step")

    fig_pr.legend(
        handles_gt,
        labels_gt,
        loc="upper center",
        ncol=min(len(labels_gt), 8),
        bbox_to_anchor=(0.5, 1.03),
    )
    fig_pr.tight_layout(rect=(0, 0, 1, 0.95))

    fig_pr.savefig(outdir / "sanity_traces_pred.png", dpi=200, bbox_inches="tight")
    plt.close(fig_pr)


def plot_distributions(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    region_names: list[str],
    outdir: Path,
    bins: int = 60,
) -> None:
    """
    Plots the marginal distributions (histograms) of GT vs Predictions for every region.
    """
    n_seq, n_time, n_reg = gt_arr.shape
    ncols = 4
    nrows = int(np.ceil(n_reg / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for r, name in enumerate(region_names):
        gt_vals = gt_arr[:, :, r].ravel()
        pr_vals = pred_arr[:, :, r].ravel()

        gt_vals = gt_vals[np.isfinite(gt_vals)]
        pr_vals = pr_vals[np.isfinite(pr_vals)]

        if gt_vals.size == 0 or pr_vals.size == 0:
            axes[r].axis("off")
            continue

        # Trim top/bottom 0.1% to prevent extreme outliers from crushing the histogram
        lo = min(np.quantile(gt_vals, 0.001), np.quantile(pr_vals, 0.001))
        hi = max(np.quantile(gt_vals, 0.999), np.quantile(pr_vals, 0.999))

        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.min(gt_vals)), float(np.max(gt_vals))
            if lo >= hi:
                hi = lo + 1e-6

        edges = np.linspace(lo, hi, bins)
        ax = axes[r]
        ax.hist(
            gt_vals, bins=edges, alpha=0.5, label="GT", density=True, color="#1f77b4"
        )
        ax.hist(
            pr_vals, bins=edges, alpha=0.5, label="Pred", density=True, color="#ff7f0e"
        )
        ax.set_title(name, fontsize=10)
        ax.tick_params(axis="both", labelsize=8)

    # Hide unused subplots
    for ax in axes[n_reg:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.95, 0.98))
    fig.suptitle("GT vs Pred Distributions per Region", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    fig.savefig(outdir / "distributions_per_region.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_umap_embeddings(
    gt_arr: np.ndarray, pred_arr: np.ndarray, outdir: Path
) -> None:
    """
    Computes and plots a 2D projection of the dataset. Uses UMAP if available,
    falling back to PCA if umap-learn is not installed in the environment.
    """
    try:
        from umap import UMAP

        reducer = UMAP(n_components=2, random_state=42)
        method_name = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=42)
        method_name = "PCA (Fallback)"
        logger.info(
            "umap-learn not installed; falling back to PCA for embedding visualization."
        )

    # Flatten temporal and sequence dimensions
    gt_flat = gt_arr.reshape(-1, gt_arr.shape[-1])
    pr_flat = pred_arr.reshape(-1, pred_arr.shape[-1])

    # Filter out NaNs
    gt_flat = gt_flat[np.isfinite(gt_flat).all(axis=1)]
    pr_flat = pr_flat[np.isfinite(pr_flat).all(axis=1)]

    if gt_flat.size == 0 or pr_flat.size == 0:
        logger.warning("Not enough valid data points to generate UMAP embeddings.")
        return

    # Subsample heavily for visual clarity and performance
    max_points = 3000
    rng = np.random.default_rng(42)

    if len(gt_flat) > max_points:
        gt_flat = gt_flat[rng.choice(len(gt_flat), max_points, replace=False)]
    if len(pr_flat) > max_points:
        pr_flat = pr_flat[rng.choice(len(pr_flat), max_points, replace=False)]

    # Fit on GT, transform both
    emb_gt = reducer.fit_transform(gt_flat)
    emb_pr = reducer.transform(pr_flat)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        emb_gt[:, 0],
        emb_gt[:, 1],
        s=8,
        alpha=0.4,
        label="Ground Truth",
        color="#1f77b4",
    )
    ax.scatter(
        emb_pr[:, 0], emb_pr[:, 1], s=8, alpha=0.4, label="Predicted", color="#ff7f0e"
    )

    ax.set_title(f"Latent Neighborhood Embedding ({method_name} fitted on GT)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(outdir / "latent_embedding_2d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_final_composites(scores: dict[str, float], outdir: Path) -> None:
    """
    Generates and saves the final composite score bar charts (Metric level and Family level).
    """
    # 1. Parse dictionary into Metric vs Family dataframes
    metrics_data = []
    families_data = []

    # We maintain standard family groupings so colors are assigned correctly
    family_mapping = {
        "KL_or_JSD_score01": "distribution",
        "Mean_score01": "distribution",
        "QNT_score01": "distribution",
        "MOM_score01": "distribution",
        "Error_score01": "fidelity",
        "MI_score01": "fidelity",
        "AUTO_score01": "temporal_spectral",
        "BP_score01": "temporal_spectral",
        "TRJDIST_score01": "temporal_spectral",
        "PSDShape_score01": "temporal_spectral",
        "FC_score01": "relational",
        "CC_score01": "relational",
        "GRAPH_score01": "relational",
        "PartialCorr_score01": "relational",
        "CrossRegionMI_score01": "relational",
        "PrecisionMatrixSpectrum_score01": "relational",
        "LaggedCovariance_score01": "relational",
        "ImpulseResponse_score01": "relational",
        "PCA_score01": "geometry",
        "MANI_score01": "geometry",
        "CCA_score01": "geometry",
        "Dimensionality_score01": "geometry",
        "SubspaceAngle_score01": "geometry",
        "EigenspectrumShape_score01": "geometry",
        "LatentStateOccupancyK11_score01": "state_dynamics",
        "LatentStateOccupancyK12_score01": "state_dynamics",
        "LatentStateTransitionLag1K11_score01": "state_dynamics",
        "LatentStateTransitionLag2K11_score01": "state_dynamics",
        "LatentStateTransitionLag3K11_score01": "state_dynamics",
    }

    family_colors = {
        "distribution": "#4C78A8",
        "fidelity": "#F58518",
        "temporal_spectral": "#54A24B",
        "relational": "#E45756",
        "geometry": "#72B7B2",
        "state_dynamics": "#9D755D",
    }

    for k, v in scores.items():
        if not np.isfinite(v):
            continue

        if (
            k.endswith("_score01")
            and not k.startswith("family_")
            and k
            not in [
                "FINAL_COMPOSITE_SCORE",
                "FINAL_NEURO_COMPOSITE_SCORE",
                "composite_score",
            ]
        ):
            metrics_data.append(
                {"metric": k, "value": v, "family": family_mapping.get(k, "unknown")}
            )
        elif k.startswith("family_"):
            families_data.append({"family": k.replace("family_", ""), "value": v})

    plot_df = pd.DataFrame(metrics_data)
    family_plot_df = pd.DataFrame(families_data)
    final_score = scores.get("FINAL_COMPOSITE_SCORE", np.nan)

    # 2. Render Plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))

    # Left: Metric-level plot
    if not plot_df.empty:
        plot_colors = [family_colors.get(f, "#cccccc") for f in plot_df["family"]]
        axes[0].bar(
            plot_df["metric"],
            plot_df["value"],
            color=plot_colors,
            edgecolor="k",
            alpha=0.9,
        )
        axes[0].set_title("Metric-level neuro score terms")
        axes[0].set_ylabel("score (0-1)")
        axes[0].tick_params(axis="x", rotation=90)
        axes[0].grid(axis="y", alpha=0.25)

    # Right: Family-level plot
    if not family_plot_df.empty:
        family_plot_colors = [
            family_colors.get(f, "#cccccc") for f in family_plot_df["family"]
        ]
        axes[1].bar(
            family_plot_df["family"],
            family_plot_df["value"],
            color=family_plot_colors,
            edgecolor="k",
            alpha=0.9,
        )
        axes[1].set_title(f"Family scores | FINAL COMPOSITE = {final_score:.4f}")
        axes[1].set_ylabel("score (0-1)")
        axes[1].tick_params(axis="x", rotation=22)
        axes[1].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(outdir / "final_composites.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------
# Master Reporting Entrypoint
# ---------------------------------------------------------


def generate_full_neuro_report(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    region_names: list[str],
    scores: dict[str, float],
    outdir: Path,
) -> None:
    """
    Executes the full suite of neuro visualizations and saves them to the specified directory.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating Sanity Check Traces...")
    plot_sanity_checks(gt_arr, pred_arr, region_names, outdir)

    logger.info("Generating Marginal Distribution Histograms...")
    plot_distributions(gt_arr, pred_arr, region_names, outdir)

    logger.info("Generating 2D Latent Embeddings...")
    plot_umap_embeddings(gt_arr, pred_arr, outdir)

    logger.info("Generating Final Composite Charts...")
    plot_final_composites(scores, outdir)

    logger.info(f"Successfully exported all plots to {outdir}")
