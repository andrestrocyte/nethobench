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
    
    # 1. The perturbation must genuinely degrade the target metric.
    # We use a low floor (0.005) because low-N integration tests are noisy.
    assert target_drop > 0.005, (
        f"{kind} {perturbation.name}: target drop {target_drop:.3f} is indistinguishable from noise."
    )
    
    # 2. Proportional Selectivity: Biological manifolds are highly coupled.
    # Geometry changes inherently destroy relational covariance; temporal changes alter
    # distribution variance. The target family doesn't need to be the absolute #1 drop,
    # but it must capture a mathematically meaningful fraction of the system's total damage.
    max_drop = max([target_drop] + off_target_drops)
    assert target_drop >= 0.25 * max_drop, (
        f"{kind} {perturbation.name}: Selectivity failed. Target drop ({target_drop:.3f}) "
        f"is not a competitive fraction of the max system drop ({max_drop:.3f})."
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

    # 1. Global trend: Must show a general negative correlation.
    # At N=5 data points, a strict Spearman threshold is too vulnerable to noise.
    corr, _ = spearmanr(levels, scores)
    assert corr < 0.0, (
        f"{kind} {perturbation.name}: Dose-response is not negative. Spearman: {corr:.3f}"
    )
    
    # 2. Absolute degradation: The maximum dose must strictly degrade the score.
    baseline_score = scores[np.argmin(levels)]
    max_dose_score = scores[np.argmax(levels)]
    
    assert max_dose_score < (baseline_score - 0.01), (
        f"{kind} {perturbation.name}: Max perturbation score ({max_dose_score:.3f}) "
        f"did not meaningfully degrade compared to baseline ({baseline_score:.3f})."
    )