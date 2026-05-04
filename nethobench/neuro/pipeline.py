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
    gt_arr, pred_arr, overlap = load_and_align(
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
    outdir = timestamped_outdir(output_root, prefix=preds_path.stem)
    gt_arr, pred_arr, region_names = load_and_align(preds_path, Path(ground_truth_csv))

    scores = calculate_neuro_composites(gt_arr, pred_arr)

    generate_full_neuro_report(gt_arr, pred_arr, region_names, scores, outdir)

    scores_path = outdir / "scores.json"
    scores_path.write_text(json.dumps(scores, indent=2))
    return scores