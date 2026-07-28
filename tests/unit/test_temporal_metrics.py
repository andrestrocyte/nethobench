from __future__ import annotations

import numpy as np

from nethobench.neuro.metrics.definitions import NEURO_FAMILY_METRICS
from nethobench.neuro.metrics.temporal import compute_temporal_metrics
from nethobench.neuro.metrics.temporal import frame_lags_from_seconds


def _ar_bundle(seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_sequences, n_time, n_regions = 8, 320, 4
    data = np.zeros((n_sequences, n_time, n_regions), dtype=np.float64)
    coefficients = np.asarray([0.92, 0.75, 0.55, 0.35])
    for sequence in range(n_sequences):
        noise = rng.normal(size=(n_time, n_regions))
        for time in range(1, n_time):
            data[sequence, time] = (
                coefficients * data[sequence, time - 1] + noise[time]
            )
    return data


def _directed_var(seed: int = 23) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = np.zeros((10, 400, 3), dtype=np.float64)
    transition = np.asarray(
        [
            [0.72, 0.00, 0.00],
            [0.58, 0.35, 0.00],
            [0.00, 0.52, 0.25],
        ]
    )
    for sequence in range(data.shape[0]):
        innovations = rng.normal(scale=0.6, size=(data.shape[1], 3))
        for time in range(1, data.shape[1]):
            data[sequence, time] = transition @ data[sequence, time - 1]
            data[sequence, time] += innovations[time]
    return data


def test_temporal_family_has_fixed_equal_components() -> None:
    metrics = NEURO_FAMILY_METRICS["temporal_spectral"]
    assert tuple(metrics) == ("TRJDIST_score", "ACF_score", "PSD_score")
    assert np.isclose(sum(metrics.values()), 1.0)
    assert len(set(metrics.values())) == 1


def test_physical_lags_are_rescaled_for_frame_rate() -> None:
    lag_seconds = (1 / 30, 2 / 30, 4 / 30, 8 / 30)
    assert frame_lags_from_seconds(lag_seconds, 30.0) == (1, 2, 4, 8)
    assert frame_lags_from_seconds(lag_seconds, 50.0) == (2, 3, 7, 13)


def test_identity_scores_one() -> None:
    ground_truth = _ar_bundle()
    scores = compute_temporal_metrics(ground_truth, ground_truth)
    for name in (
        "ACF_score",
        "PSD_score",
        "PSDShape_score",
        "PSDPower_score",
        "DirectionalDynamics_score",
    ):
        assert np.isclose(scores[name], 1.0, atol=1e-10), (name, scores[name])


def test_time_shuffle_is_detected() -> None:
    ground_truth = _ar_bundle()
    rng = np.random.default_rng(31)
    shuffled = ground_truth.copy()
    for sequence in range(shuffled.shape[0]):
        shuffled[sequence] = shuffled[sequence, rng.permutation(shuffled.shape[1])]
    scores = compute_temporal_metrics(ground_truth, shuffled)
    assert scores["ACF_score"] < 0.80
    assert scores["PSD_score"] < 0.80


def test_short_timescale_jitter_is_detected_by_increment_acf() -> None:
    ground_truth = _ar_bundle()
    rng = np.random.default_rng(61)
    jittered = ground_truth.copy()
    impulses = rng.random(jittered.shape) < 0.08
    jittered[impulses] += rng.normal(scale=2.0, size=int(impulses.sum()))
    scores = compute_temporal_metrics(ground_truth, jittered)
    assert scores["ACF_score"] < 0.90


def test_variance_contraction_cannot_hide_in_psd_shape() -> None:
    ground_truth = _ar_bundle()
    contracted = 0.1 * ground_truth
    scores = compute_temporal_metrics(ground_truth, contracted)
    assert scores["PSDShape_score"] > 0.99
    assert 0.005 < scores["PSDPower_score"] < 0.02
    assert scores["PSD_score"] < 0.12


def test_directional_diagnostic_detects_time_reversal() -> None:
    ground_truth = _directed_var()
    reversed_time = ground_truth[:, ::-1, :]
    scores = compute_temporal_metrics(ground_truth, reversed_time)
    assert scores["DirectionalDynamics_score"] < 0.85


def test_directional_diagnostic_uses_shared_region_mask() -> None:
    ground_truth = _ar_bundle()
    prediction = ground_truth.copy()
    ground_truth[:, :, 0] = 0.0
    prediction[:, :, 1] = 0.0
    scores = compute_temporal_metrics(ground_truth, prediction)
    assert np.isclose(scores["DirectionalDynamics_score"], 1.0, atol=1e-10)


def test_scores_are_bounded_with_sparse_missing_values() -> None:
    ground_truth = _ar_bundle()
    prediction = ground_truth + 0.15
    prediction[:, 20, :] = np.nan
    scores = compute_temporal_metrics(ground_truth, prediction)
    for value in scores.values():
        assert 0.0 <= value <= 1.0
