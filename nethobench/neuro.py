from __future__ import annotations

import json
from pathlib import Path
import runpy
import tempfile
from typing import Dict, Optional

import numpy as np
import pandas as pd
from nethobench.analysis.neuro_scoring import calculate_neuro_composites
from nethobench.helpers import _load_and_align, _timestamped_outdir
from nethobench.analysis.neuro_reporting import generate_full_neuro_report


def _compute_scores_from_arrays(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    region_names: Optional[list[str]] = None,
    ddconfig_path: Optional[Path] = None,
) -> Dict[str, float]:
    if gt.ndim != 3 or pred.ndim != 3:
        raise ValueError("Expected gt/pred arrays with shape [n_seq, n_time, n_reg].")
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} vs {pred.shape}")

    n_seq, n_time, n_reg = gt.shape
    region_names = region_names or [f"R{i}" for i in range(n_reg)]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        gt_path = tmpdir_path / "gt.csv"
        pred_path = tmpdir_path / "pred.csv"

        seq_ids = np.repeat(np.arange(n_seq), n_time)
        item_pos = np.tile(np.arange(n_time), n_seq)

        gt_df = pd.DataFrame(gt.reshape(-1, n_reg), columns=region_names)
        gt_df.insert(0, "itemPosition", item_pos)
        gt_df.insert(0, "sequenceId", seq_ids)
        gt_df.to_csv(gt_path, index=False)

        pred_df = pd.DataFrame(pred.reshape(-1, n_reg), columns=region_names)
        pred_df.index = seq_ids
        pred_df.to_csv(pred_path)

        return run_neuro_full_analysis(pred_path, gt_path)


def compute_neuro_scores(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    per_sequence_stats: bool = False,
    neuro_cols: Optional[list[str]] = None,
) -> Dict[str, float]:

    if per_sequence_stats:
        raise ValueError(
            "per_sequence_stats is not supported for notebook-based neuro scores."
        )

    # 1. Use the existing helper to load CSVs and reshape them into 3D tensors
    # shape: [n_sequences, n_timesteps, n_regions]
    gt_arr, pred_arr, overlap = _load_and_align(
        Path(predictions_csv),
        Path(ground_truth_csv),
        neuro_cols=neuro_cols,
    )

    return calculate_neuro_composites(gt_arr, pred_arr)


def run_neuro_full_analysis(
    predictions_csv: Path,
    ground_truth_csv: Path,
    output_root: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Execute the active neuro notebook headlessly, save figures, and export notebook-derived scores.
    """
    preds_path = Path(predictions_csv)
    outdir = _timestamped_outdir(output_root, prefix=preds_path.stem)
    gt_arr, pred_arr, region_names = _load_and_align(preds_path, Path(ground_truth_csv))

    scores = calculate_neuro_composites(gt_arr, pred_arr)

    generate_full_neuro_report(gt_arr, pred_arr, region_names, scores, outdir)

    scores_path = outdir / "scores.json"
    scores_path.write_text(json.dumps(scores, indent=2))
    return scores
