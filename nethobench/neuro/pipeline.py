from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Dict, Optional

import numpy as np
import pandas as pd
from nethobench.utils.helpers import load_and_align, timestamped_outdir
from nethobench.neuro.metrics.composites import calculate_neuro_composites
from nethobench.neuro.reporting import generate_full_neuro_report
from nethobench.utils.evaluation_constants import config as _bench_config


def compute_neuro_scores(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    per_sequence_stats: bool = False,
    neuro_cols: Optional[list[str]] = None,
    config: Optional[dict] = None,
) -> Dict[str, float]:
    """
    Compute neuro composite scores from prediction and ground-truth CSVs.

    Loads the CSV files, aligns them into 3D tensors of shape
    ``[n_sequences, n_timesteps, n_regions]``, and calculates the
    composite neuro scores.

    Args:
        predictions_csv: Path to the predictions CSV file.
        ground_truth_csv: Path to the ground-truth CSV file.
        per_sequence_stats: If True, raises a ``ValueError`` because
            per-sequence statistics are not supported for notebook-based
            neuro scores. Defaults to False.
        neuro_cols: Optional list of column names to use as neural regions.
            If None, all matching columns are inferred automatically.
        config: Optional configuration dictionary with keys such as
            ``sequence_key``, ``time_key``, and ``neuro_cols``.

    Returns:
        Dictionary mapping composite score names to float values.
    """
    if per_sequence_stats:
        raise ValueError(
            "per_sequence_stats is not supported for notebook-based neuro scores."
        )

    cfg = config or {}
    _bench_config.update_from_dict(cfg)
    seq_key = cfg.get("sequence_key", "sequenceId")
    time_key = cfg.get("time_key", "itemPosition")
    cols = neuro_cols if neuro_cols is not None else cfg.get("neuro_cols")

    # 1. Use the existing helper to load CSVs and reshape them into 3D tensors
    # shape: [n_sequences, n_timesteps, n_regions]
    gt_arr, pred_arr, overlap = load_and_align(
        Path(predictions_csv),
        Path(ground_truth_csv),
        neuro_cols=cols,
        seq_key=seq_key,
        time_key=time_key,
    )

    return calculate_neuro_composites(gt_arr, pred_arr)


def run_neuro_full_analysis(
    predictions_csv: Path,
    ground_truth_csv: Path,
    output_root: Optional[Path] = None,
    config: Optional[dict] = None,
) -> Dict[str, object]:
    """
    Execute the active neuro notebook headlessly, save figures, and export notebook-derived scores.
    """
    preds_path = Path(predictions_csv)
    outdir = timestamped_outdir(output_root, prefix=preds_path.stem)

    cfg = config or {}
    _bench_config.update_from_dict(cfg)
    seq_key = cfg.get("sequence_key", "sequenceId")
    time_key = cfg.get("time_key", "itemPosition")
    cols = cfg.get("neuro_cols")

    gt_arr, pred_arr, region_names = load_and_align(
        preds_path,
        Path(ground_truth_csv),
        neuro_cols=cols,
        seq_key=seq_key,
        time_key=time_key,
    )

    scores = calculate_neuro_composites(gt_arr, pred_arr)

    generate_full_neuro_report(gt_arr, pred_arr, region_names, scores, outdir)

    scores_path = outdir / "scores.json"
    scores_path.write_text(json.dumps(scores, indent=2))
    return scores