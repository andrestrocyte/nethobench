from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from nethobench.utils.helpers import (
    load_gt_and_preds,
    _timestamped_outdir,
    _clip01,
    _geometric_mean_scores,
)
from nethobench.utils.calculation import _merge_aligned
from nethobench.etho.metrics import (
    stationary_score,
    velocity_distribution_score,
    acceleration_distribution_score,
    direction_score,
    quadrant_score,
    position_kl_score,
    syllable_score,
    trajectory_shape_score,
    body_part_errors,
    inter_limb_distances,
    dtw_trajectory_similarity,
    manifold_alignment_metrics,
    get_chunked_embeddings,
)


def compute_etho_scores(
    gt_dir: Optional[Path] = None,
    inf_dir: Optional[Path] = None,
    *,
    paired_df: Optional[pd.DataFrame] = None,
) -> Tuple[
    Dict[str, float], Dict[str, List[float]], Dict[str, float], Dict[str, float]
]:

    # Avoid redundant loading if already passed from run_etho_full_analysis
    if paired_df is None:
        if gt_dir is None or inf_dir is None:
            raise ValueError(
                "Must provide either paired_df, or both gt_dir and inf_dir."
            )
        gt_df, inf_df = load_gt_and_preds(gt_dir, inf_dir)
        paired_df = pd.merge(
            gt_df.sort_values(["sequenceId", "itemPosition"]),
            inf_df.sort_values(["sequenceId", "itemPosition"]),
            on=["sequenceId", "itemPosition"],
            suffixes=("_gt", "_inf"),
            how="inner",
        )

    pos_res = position_kl_score(paired_df)
    stat_res = stationary_score(paired_df)
    vel_res = velocity_distribution_score(paired_df)
    acc_res = acceleration_distribution_score(paired_df)
    dir_res = direction_score(paired_df)
    quad_res = quadrant_score(paired_df)
    syll_res = syllable_score(paired_df)
    traj_res = trajectory_shape_score(paired_df)

    # Calculate memory-optimized DTW similarity
    dtw_score, dtw_seq_scores = dtw_trajectory_similarity(paired_df)

    # Calculate memory-optimized Manifold similarities
    manifold_res = manifold_alignment_metrics(paired_df)

    scores = {
        "position_kl_score": float(pos_res[0]) if np.isfinite(pos_res[0]) else np.nan,
        "stationary_score": float(stat_res[0]) if np.isfinite(stat_res[0]) else np.nan,
        "velocity_score": float(vel_res[0]) if np.isfinite(vel_res[0]) else np.nan,
        "acceleration_score": float(acc_res[0]) if np.isfinite(acc_res[0]) else np.nan,
        "direction_score": float(dir_res[0]) if np.isfinite(dir_res[0]) else np.nan,
        "quadrant_score": float(quad_res[0]) if np.isfinite(quad_res[0]) else np.nan,
        "syllable_score": float(syll_res[0]) if np.isfinite(syll_res[0]) else np.nan,
        "trajectory_shape_score": (
            float(traj_res[0]) if np.isfinite(traj_res[0]) else np.nan
        ),
        "dtw_similarity_score": dtw_score,
        "procrustes_similarity": manifold_res["procrustes_sim"],
        "mmd_similarity": manifold_res["mmd_sim"],
    }

    # Only core structural metrics go into composite logic
    core_scores = {
        k: v
        for k, v in scores.items()
        if k not in ["procrustes_similarity", "mmd_similarity", "dtw_similarity_score"]
    }
    scores["composite_score"] = _geometric_mean_scores(list(core_scores.values()))

    all_seq_dicts = {
        "position_kl_score": pos_res[1],
        "stationary_score": stat_res[1],
        "velocity_score": vel_res[1],
        "acceleration_score": acc_res[1],
        "direction_score": dir_res[1],
        "quadrant_score": quad_res[1],
        "syllable_score": syll_res[1],
        "trajectory_shape_score": traj_res[1],
        "dtw_similarity_score": dtw_seq_scores,
    }

    # Generate weights dictionary (for DTW, default to 1 since it's sequence-level)
    all_weight_dicts = {
        "position_kl_score": pos_res[2],
        "stationary_score": stat_res[2],
        "velocity_score": vel_res[2],
        "acceleration_score": acc_res[2],
        "direction_score": dir_res[2],
        "quadrant_score": quad_res[2],
        "syllable_score": syll_res[2],
        "trajectory_shape_score": traj_res[2],
        "dtw_similarity_score": {seq: 1 for seq in dtw_seq_scores},
    }

    all_keys = set()
    for d in all_seq_dicts.values():
        all_keys.update(d.keys())
    sorted_seq_ids = sorted(list(all_keys))

    sequence_level_scores = {
        "sequence_length": [
            int(v)
            for v in paired_df.groupby("sequenceId")
            .size()
            .reindex(sorted_seq_ids, fill_value=0)
        ]
    }
    sequence_means = {}
    sequence_stds = {}

    for metric_name in all_seq_dicts.keys():
        d_scores = all_seq_dicts[metric_name]
        d_weights = all_weight_dicts[metric_name]
        sequence_level_scores[metric_name] = [
            float(d_scores[k]) if k in d_scores and np.isfinite(d_scores[k]) else np.nan
            for k in sorted_seq_ids
        ]
        vals = []
        wts = []
        for k in d_scores.keys():
            if k in d_weights and np.isfinite(d_scores[k]):
                vals.append(d_scores[k])
                wts.append(d_weights[k])
        if vals and sum(wts) > 0:
            weighted_mean = np.average(vals, weights=wts)
            sequence_means[metric_name] = float(weighted_mean)
            variance = np.average((np.array(vals) - weighted_mean) ** 2, weights=wts)
            sequence_stds[metric_name] = float(np.sqrt(variance))
        else:
            sequence_means[metric_name] = np.nan
            sequence_stds[metric_name] = np.nan

    return scores, sequence_level_scores, sequence_means, sequence_stds


def run_etho_full_analysis(
    gt_dir: Path, inf_dir: Path, *, output_root: Path | None = None
) -> Path:
    """
    Executes the full behavioral analysis pipeline headlessly, saves metrics,
    and generates visualization figures directly into the specified output directory.
    """
    import json

    try:
        from nethobench.etho.reporting import generate_full_etho_report
    except ImportError:

        def generate_full_etho_report(*args, **kwargs):
            pass  # Failsafe if reporting module has not been fully implemented yet

    outdir = _timestamped_outdir(output_root, prefix="etho-analysis")

    # 1. Load Data
    gt_df, inf_df = load_gt_and_preds(gt_dir, inf_dir)
    paired_df = pd.merge(
        gt_df.sort_values(["sequenceId", "itemPosition"]),
        inf_df.sort_values(["sequenceId", "itemPosition"]),
        on=["sequenceId", "itemPosition"],
        suffixes=("_gt", "_inf"),
        how="inner",
    )

    # 2. Extract Additional Features for the Report
    coord_pairs = []
    for col in gt_df.columns:
        if col.endswith("_X"):
            base = col[:-2]
            if f"{base}_Y" in gt_df.columns:
                coord_pairs.append((base, col, f"{base}_Y"))

    errors_df = body_part_errors(paired_df, coord_pairs)
    pairs_to_check = [
        ("NOSE", "TAIL_BASE"),
        ("LEFT_EAR", "RIGHT_EAR"),
        ("NOSE", "CENTER"),
    ]
    distances_df = inter_limb_distances(paired_df, pairs_to_check)
    chunk_data = get_chunked_embeddings(paired_df, chunk_size=5)

    # 3. Compute Structural Scores (Pass paired_df to avoid double-loading!)
    scores, seq_scores, seq_means, seq_stds = compute_etho_scores(paired_df=paired_df)

    # 4. Save JSON
    payload = {
        "global_scores": scores,
        "per_sequence": seq_scores,
        "per_sequence_mean": seq_means,
        "per_sequence_std": seq_stds,
    }
    with (outdir / "scores.json").open("w") as f:
        json.dump(payload, f, indent=2)

    # 5. Generate Matplotlib Figures
    generate_full_etho_report(
        gt_df, inf_df, paired_df, errors_df, distances_df, chunk_data, outdir
    )

    return outdir
