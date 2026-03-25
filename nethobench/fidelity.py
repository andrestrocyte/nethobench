from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .analysis.score_definitions import compute_fidelity_composite
from .legacy.legacy_metrics import compute_legacy_neuro_scores


def compute_fidelity_scores(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    per_sequence_stats: bool = False,
    neuro_cols: Optional[list[str]] = None,
    ddconfig_path: Optional[Path] = None,
) -> Dict[str, float]:
    legacy_scores = compute_legacy_neuro_scores(
        predictions_csv,
        ground_truth_csv,
        per_sequence_stats=per_sequence_stats,
        neuro_cols=neuro_cols,
        ddconfig_path=ddconfig_path,
    )
    out = {
        "Error_score01": float(legacy_scores.get("Error_score01", np.nan)),
        "MI_score01": float(legacy_scores.get("MI_score01", np.nan)),
    }
    fidelity = compute_fidelity_composite(out)
    out["family_fidelity"] = fidelity
    out["FIDELITY_SCORE"] = fidelity
    return out
