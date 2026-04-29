from __future__ import annotations

import json
from pathlib import Path
import runpy
import tempfile
from typing import Dict, Optional

import numpy as np
import pandas as pd
from nethobench.helpers import _load_and_align, _timestamped_outdir
from nethobench.analysis.neuro_scoring import calculate_neuro_composites

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

