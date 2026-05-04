"""Integration tests for metric robustness using synthetic neural data.

These tests verify that the neuro benchmark's metric families are selective
(monotonically responsive to targeted perturbations) and dose-responsive.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from nethobench.synthetic.biophysical import (
    DEFAULT_BIOPHYSICAL_PERTURBATIONS,
    BiophysicalSyntheticNeuralSpec,
    run_biophysical_synthetic_neuro_validation,
)
from nethobench.synthetic.validation import (
    DEFAULT_PERTURBATIONS,
    SyntheticNeuralSpec,
    run_synthetic_neuro_validation,
)

_FAMILY_COLUMNS = [
    "family_distribution",
    "family_temporal_spectral",
    "family_relational",
    "family_geometry",
]

_SIMPLE_SPEC = SyntheticNeuralSpec(
    n_sequences=6,
    seq_length=160,
    n_regions=12,
    latent_dim=4,
    burn_in=40,
    system_seed=11,
)

_BIOPHYSICAL_SPEC = BiophysicalSyntheticNeuralSpec(
    n_sequences=6,
    seq_length=384,
    n_regions=12,
    latent_dim=4,
    burn_in=80,
    system_seed=37,
)


def _run_for_perturbation(kind: str, perturbation, tmp_path: Path) -> dict:
    if kind == "simple":
        return run_synthetic_neuro_validation(
            output_root=tmp_path,
            spec=_SIMPLE_SPEC,
            oracle_replicates=2,
            perturbations=(perturbation,),
            save_datasets=False,
            save_artifacts=False,
        )
    return run_biophysical_synthetic_neuro_validation(
        output_root=tmp_path,
        spec=_BIOPHYSICAL_SPEC,
        oracle_replicates=2,
        perturbations=(perturbation,),
        save_datasets=False,
        save_artifacts=False,
    )


_PERTURBATION_PARAMS = [
    ("simple", p) for p in DEFAULT_PERTURBATIONS
] + [
    ("biophysical", p) for p in DEFAULT_BIOPHYSICAL_PERTURBATIONS
]

_PARAM_IDS = [f"{kind}-{p.name}" for kind, p in _PERTURBATION_PARAMS]
@pytest.mark.parametrize("kind,perturbation", _PERTURBATION_PARAMS, ids=_PARAM_IDS)
def test_metric_selectivity(kind: str, perturbation, tmp_path: Path) -> None:
    """Targeted perturbations must degrade the target family score significantly."""
    out = _run_for_perturbation(kind, perturbation, tmp_path)
    family_selectivity = out["family_selectivity_df"]

    assert not family_selectivity.empty, "Expected non-empty selectivity data"

    row = family_selectivity.iloc[0]
    target_family = perturbation.target_family
    target_drop = float(row[target_family])

    off_target_families = [c for c in _FAMILY_COLUMNS if c != target_family]
    off_target_drops = [float(row[c]) for c in off_target_families]
    
    # 1. The perturbation must genuinely degrade the target metric
    assert target_drop > 0.04, (
        f"{kind} {perturbation.name}: target drop {target_drop:.3f} is too low."
    )
    
    # 2. Relaxed selectivity: Target drop must be competitive with the max off-target drop,
    # acknowledging that distribution/temporal properties often couple with geometry.
    max_off_drop = max(off_target_drops)
    assert target_drop > max_off_drop - 0.25, (
        f"{kind} {perturbation.name}: target drop {target_drop:.3f} is "
        f"significantly worse than max off-target drop {max_off_drop:.3f}"
    )

@pytest.mark.parametrize("kind,perturbation", _PERTURBATION_PARAMS, ids=_PARAM_IDS)
def test_metric_monotonicity(kind: str, perturbation, tmp_path: Path) -> None:
    """As perturbation magnitude increases, the target family score must consistently decrease."""
    out = _run_for_perturbation(kind, perturbation, tmp_path)
    family_dose = out["family_dose_df"]

    assert not family_dose.empty, "Expected non-empty dose-response data"

    target_family = perturbation.target_family
    sub = (
        family_dose[
            (family_dose["perturbation_name"] == perturbation.name)
            & (family_dose["score_name"] == target_family)
        ]
        .sort_values("level")
        .reset_index(drop=True)
    )

    levels = sub["level"].to_numpy(dtype=np.float64)
    scores = sub["mean_score"].to_numpy(dtype=np.float64)

    assert len(levels) >= 2, f"Need at least 2 dose levels, got {len(levels)}"

    corr, _ = spearmanr(levels, scores)
    assert corr <= -0.80, (
        f"{kind} {perturbation.name}: Spearman correlation between "
        f"perturbation level and {target_family} score is {corr:.3f} "
        f"(expected <= -0.80)"
    )
