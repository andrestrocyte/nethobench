from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from nethobench.utils.helpers import clip_fn, geometric_mean_scores
from nethobench.utils.calculation import merge_aligned
from nethobench.utils.validation import validate_multimodal_data
from nethobench.neuro.metrics.composites import load_and_run_neuro_full_analysis
from nethobench.etho.metrics import (
    position_kl_score,
    quadrant_score,
    stationary_score,
    velocity_distribution_score,
    acceleration_distribution_score,
    direction_score,
    syllable_score,
    trajectory_shape_score,
)
from nethobench.cross.metrics import (
    load_config,
    arrays_from_aligned,
    behavior_feature_matrix,
    behavior_feature_sequences,
    neural_feature_matrix,
    neural_feature_sequences,
    cca_mean,
    predictive_r2,
    lead_lag_peak,
    speed_from_behavior,
)


def compute_cross_scores(
    predictions_csv: Path, ground_truth_csv: Path, config: Union[Path, dict, None]
) -> Dict[str, object]:
    # Read GT once for config inference if needed
    sample_df = pd.read_csv(ground_truth_csv)
    cfg = load_config(config, sample_df=sample_df)
    aligned = merge_aligned(ground_truth_csv, predictions_csv, cfg)
    validate_multimodal_data(aligned, cfg)
    neuro_cols = cfg.get("neuro_cols")
    if not neuro_cols:
        raise ValueError("Config must include neuro_cols for cross-scores.")

    # --- Neuro axis ---
    gt_neuro = arrays_from_aligned(aligned, neuro_cols, "gt", cfg)
    pr_neuro = arrays_from_aligned(aligned, neuro_cols, "inf", cfg)
    neuro_scores = load_and_run_neuro_full_analysis(
        gt_neuro, pr_neuro
    )

    # --- Behavior axis (reuse ethobench metrics on merged df) ---
    beh_scores = {}
    beh_scores["position_kl_score"] = position_kl_score(aligned)[0]
    beh_scores["quadrant_score"] = quadrant_score(aligned)[0]
    beh_scores["stationary_score"] = stationary_score(aligned)[0]
    beh_scores["velocity_score"] = velocity_distribution_score(aligned)[0]
    beh_scores["acceleration_score"] = acceleration_distribution_score(aligned)[0]
    beh_scores["direction_score"] = direction_score(
        aligned,
        nose_label=cfg.get("body_axis", ["NOSE", "TAIL_BASE"])[0],
        tail_label=cfg.get("body_axis", ["NOSE", "TAIL_BASE"])[-1],
    )[0]
    beh_scores["syllable_score"] = syllable_score(aligned)[0]
    beh_scores["trajectory_shape_score"] = trajectory_shape_score(aligned)[0]
    beh_scores["composite_score"] = geometric_mean_scores(list(beh_scores.values()))

    # --- Cross-modal axis ---
    neuro_gt_flat = neural_feature_matrix(aligned, neuro_cols, "gt", cfg)
    neuro_pr_flat = neural_feature_matrix(aligned, neuro_cols, "inf", cfg)
    beh_gt_flat = behavior_feature_matrix(aligned, cfg, "gt")
    beh_pr_flat = behavior_feature_matrix(aligned, cfg, "inf")
    neuro_gt_seq = neural_feature_sequences(aligned, neuro_cols, "gt", cfg)
    neuro_pr_seq = neural_feature_sequences(aligned, neuro_cols, "inf", cfg)
    beh_gt_seq = behavior_feature_sequences(aligned, cfg, "gt")
    beh_pr_seq = behavior_feature_sequences(aligned, cfg, "inf")

    cca_gt = cca_mean(neuro_gt_flat, beh_gt_flat)
    cca_pr = cca_mean(neuro_pr_flat, beh_pr_flat)
    cca_alignment_score = (
        float(1.0 - min(1.0, abs(cca_gt - cca_pr)))
        if np.isfinite(cca_gt) and np.isfinite(cca_pr)
        else np.nan
    )

    r2_n2b_gt = predictive_r2(neuro_gt_seq, beh_gt_seq)
    r2_n2b_pr = predictive_r2(neuro_pr_seq, beh_pr_seq)
    n2b_similarity = (
        float(1.0 / (1.0 + abs(r2_n2b_gt - r2_n2b_pr)))
        if np.isfinite(r2_n2b_gt) and np.isfinite(r2_n2b_pr)
        else np.nan
    )

    r2_b2n_gt = predictive_r2(beh_gt_seq, neuro_gt_seq)
    r2_b2n_pr = predictive_r2(beh_pr_seq, neuro_pr_seq)
    b2n_similarity = (
        float(1.0 / (1.0 + abs(r2_b2n_gt - r2_b2n_pr)))
        if np.isfinite(r2_b2n_gt) and np.isfinite(r2_b2n_pr)
        else np.nan
    )

    speed_gt, _ = speed_from_behavior(aligned, cfg, "gt")
    speed_pr, _ = speed_from_behavior(aligned, cfg, "inf")
    lag_gt = lead_lag_peak(neuro_gt_seq, speed_gt)
    lag_pr = lead_lag_peak(neuro_pr_seq, speed_pr)
    lead_lag_score = (
        1.0 - min(1.0, abs(lag_gt - lag_pr) / 30.0)
        if (lag_gt == lag_gt) and (lag_pr == lag_pr)
        else np.nan
    )

    cross_scores = {
        "cca_gt": cca_gt,
        "cca_pred": cca_pr,
        "cca_alignment_score": cca_alignment_score,
        "r2_neural_to_behavior_gt": r2_n2b_gt,
        "r2_neural_to_behavior_pred": r2_n2b_pr,
        "neural_to_behavior_similarity": n2b_similarity,
        "r2_behavior_to_neural_gt": r2_b2n_gt,
        "r2_behavior_to_neural_pred": r2_b2n_pr,
        "behavior_to_neural_similarity": b2n_similarity,
        "lead_lag_gt": lag_gt,
        "lead_lag_pred": lag_pr,
        "lead_lag_score": lead_lag_score,
    }

    cross_vals = [
        v
        for k, v in cross_scores.items()
        if k.endswith("score") or k.endswith("similarity")
    ]
    cross_vals = [v for v in cross_vals if np.isfinite(v)]
    cross_composite = (
        float(np.prod(cross_vals) ** (1 / len(cross_vals))) if cross_vals else np.nan
    )
    cross_scores["cross_composite"] = cross_composite

    neuro_composite = neuro_scores.get("composite_score", np.nan)
    etho_composite = beh_scores.get("composite_score", np.nan)
    comps = [
        v for v in [neuro_composite, etho_composite, cross_composite] if np.isfinite(v)
    ]
    final_comp = float(np.mean(comps)) if comps else np.nan

    return {
        "neuro_scores": neuro_scores,
        "behavior_scores": beh_scores,
        "cross_scores": cross_scores,
        "neuro_composite": neuro_composite,
        "etho_composite": etho_composite,
        "cross_composite": cross_composite,
        "composite": final_comp,
    }


def run_cross_full_analysis(
    predictions_csv: Path,
    ground_truth_csv: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> Path:
    """
    Execute the full cross-modal analysis natively, saving JSON metrics and rendering plots.
    """
    try:
        from nethobench.cross.reporting import generate_full_cross_report
    except ImportError:

        def generate_full_cross_report(*args, **kwargs):
            pass  # Failsafe if reporting module is missing

    # Set up output directory
    outdir_root = Path(output_root) if output_root else Path.cwd() / "outputs"
    outdir = outdir_root / f"cross-analysis-{Path(predictions_csv).stem}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Compute all cross-modal scores
    scores = compute_cross_scores(predictions_csv, ground_truth_csv, config_path)

    # Export to JSON
    with (outdir / "scores.json").open("w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    # Generate Matplotlib Figures
    generate_full_cross_report(scores, outdir)

    return outdir

