from __future__ import annotations

import numpy as np
import pytest

from nethobench.probabilistic import (
    energy_score,
    ensemble_calibration_diagnostics,
    monte_carlo_structural_scores,
    reconstruct_sample_major_ensemble,
    temporal_variogram_score,
)


def _ensemble(seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    target = rng.normal(size=(7, 48, 3))
    samples = target[None, ...] + rng.normal(scale=0.25, size=(9, *target.shape))
    return target, samples


def test_reconstructs_sample_major_pool_without_duplicating_target() -> None:
    target, samples = _ensemble()
    repeated = np.repeat(target[None, ...], samples.shape[0], axis=0)
    repeated = repeated.reshape(-1, target.shape[1], target.shape[2])
    pooled = samples.reshape(-1, target.shape[1], target.shape[2])
    recovered_target, recovered_samples = reconstruct_sample_major_ensemble(
        repeated,
        pooled,
        n_draws=samples.shape[0],
    )
    assert recovered_target.shape == target.shape
    assert recovered_samples.shape == samples.shape
    assert np.array_equal(recovered_target, target)
    assert np.array_equal(recovered_samples, samples)


def test_reconstruction_rejects_inconsistent_target_blocks() -> None:
    target, samples = _ensemble()
    repeated = np.repeat(target[None, ...], samples.shape[0], axis=0)
    repeated[3, 0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="inconsistent"):
        reconstruct_sample_major_ensemble(
            repeated.reshape(-1, target.shape[1], target.shape[2]),
            samples.reshape(-1, target.shape[1], target.shape[2]),
            n_draws=samples.shape[0],
        )


def test_monte_carlo_protocol_scores_one_draw_per_context() -> None:
    target, samples = _ensemble()
    seen_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def scorer(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        seen_shapes.append((gt.shape, pred.shape))
        return {"mean_prediction": float(np.mean(pred))}

    result = monte_carlo_structural_scores(target, samples, scorer)
    assert len(seen_shapes) == samples.shape[0]
    assert set(seen_shapes) == {(target.shape, target.shape)}
    assert result["n_contexts"] == target.shape[0]
    assert result["n_draws"] == samples.shape[0]
    assert len(result["replicates"]) == samples.shape[0]


def test_proper_scores_are_zero_for_perfect_deterministic_ensemble() -> None:
    target, _ = _ensemble()
    perfect = np.repeat(target[None, ...], 4, axis=0)
    assert np.isclose(energy_score(target, perfect), 0.0, atol=1e-12)
    assert np.isclose(
        temporal_variogram_score(target, perfect),
        0.0,
        atol=1e-12,
    )


def test_proper_scores_penalize_offset_and_temporal_shuffle() -> None:
    target, _ = _ensemble()
    offset = np.repeat((target + 1.5)[None, ...], 5, axis=0)
    shuffled = np.repeat(target[:, ::-1, :][None, ...], 5, axis=0)
    assert energy_score(target, offset) > 0.5
    assert temporal_variogram_score(target, shuffled) > 0.01


def test_calibration_diagnostics_have_expected_shape_and_bounds() -> None:
    target, samples = _ensemble()
    diagnostics = ensemble_calibration_diagnostics(target, samples)
    assert len(diagnostics["rank_histogram_counts"]) == samples.shape[0] + 1
    assert sum(diagnostics["rank_histogram_counts"]) == target.size
    for interval in diagnostics["central_intervals"].values():
        assert 0.0 <= interval["coverage"] <= 1.0
        assert interval["mean_width"] >= 0.0
