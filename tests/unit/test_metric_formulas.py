from __future__ import annotations

import numpy as np
from scipy import stats

from nethobench.neuro.metrics.refined import (
    correlation_graph_score,
    perfected_graph_score_legacy,
    perfected_moment_score_legacy,
    pooled_moment_score,
)
from nethobench.utils.calculation import EPS, correlation_score


def test_correlation_mapping_endpoints() -> None:
    values = np.arange(8, dtype=np.float64)
    assert np.isclose(correlation_score(values, values), 1.0)
    assert np.isclose(correlation_score(values, values[::-1]), 0.0)


def test_pooled_moment_score_matches_documented_distance() -> None:
    rng = np.random.default_rng(14)
    ground_truth = rng.gamma(shape=2.0, scale=1.2, size=(3, 120, 1))
    prediction = 1.7 * ground_truth + 0.3
    actual = pooled_moment_score(ground_truth, prediction)["score"]

    gt = ground_truth.reshape(-1)
    pred = prediction.reshape(-1)
    distance = (
        abs(np.log((np.var(pred) + EPS) / (np.var(gt) + EPS)))
        + 0.50 * abs(stats.skew(pred, bias=False) - stats.skew(gt, bias=False))
        + 0.25
        * abs(
            stats.kurtosis(pred, fisher=True, bias=False)
            - stats.kurtosis(gt, fisher=True, bias=False)
        )
    )
    expected = 1.0 / (1.0 + distance)
    assert np.isclose(actual, expected)
    assert np.isclose(
        perfected_moment_score_legacy(ground_truth, prediction)["score"],
        actual,
    )


def test_correlation_graph_identity_and_alias() -> None:
    rng = np.random.default_rng(19)
    latent = rng.normal(size=(4, 160, 3))
    mixing = np.asarray(
        [
            [1.0, 0.5, 0.2],
            [0.2, 0.9, 0.4],
            [0.4, 0.1, 1.1],
            [0.8, 0.3, 0.2],
            [0.1, 0.7, 0.6],
        ]
    )
    activity = latent @ mixing.T
    score = correlation_graph_score(activity, activity)
    alias = perfected_graph_score_legacy(activity, activity)
    assert np.isclose(score["score"], 1.0)
    assert score == alias
