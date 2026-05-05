from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from nethobench.neuro.metrics.definitions import compute_fidelity_composite
from nethobench.neuro.metrics.composites import compute_composite_scores
from nethobench.utils.evaluation_constants import config as _bench_config


def compute_fidelity_scores(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    per_sequence_stats: bool = False,
    neuro_cols: Optional[list[str]] = None,
    ddconfig_path: Optional[Path] = None,
    config: Optional[dict] = None,
) -> Dict[str, float]:
    """
    Compute fidelity scores from prediction and ground-truth CSVs.

    Calculates the Error and Mutual Information (MI) scores using the legacy
    composite pipeline, then derives the ``family_fidelity`` and
    ``FIDELITY_SCORE`` values from them.

    Args:
        predictions_csv: Path to the predictions CSV file.
        ground_truth_csv: Path to the ground-truth CSV file.
        per_sequence_stats: Whether to compute per-sequence statistics.
            Passed through to the legacy composite scorer. Defaults to False.
        neuro_cols: Optional list of column names to treat as neural regions.
            If None, columns are inferred automatically.
        ddconfig_path: Optional path to a configuration file for the legacy
            scorer.
        config: Optional configuration dictionary with keys such as
            ``sequence_key``, ``time_key``, and ``neuro_cols``.

    Returns:
        Dictionary containing ``Error_score``, ``MI_score``,
        ``family_fidelity``, and ``FIDELITY_SCORE``.
    """
    cfg = config or {}
    _bench_config.update_from_dict(cfg)
    seq_key = cfg.get("sequence_key", "sequenceId")
    time_key = cfg.get("time_key", "itemPosition")
    cols = neuro_cols if neuro_cols is not None else cfg.get("neuro_cols")

    legacy_scores = compute_composite_scores(
        predictions_csv,
        ground_truth_csv,
        per_sequence_stats=per_sequence_stats,
        neuro_cols=cols,
        ddconfig_path=ddconfig_path,
        seq_key=seq_key,
        time_key=time_key,
    )
    out = {
        "Error_score": float(legacy_scores.get("Error_score", np.nan)),
        "MI_score": float(legacy_scores.get("MI_score", np.nan)),
    }
    fidelity = compute_fidelity_composite(out)
    out["family_fidelity"] = fidelity
    out["FIDELITY_SCORE"] = fidelity
    return out
