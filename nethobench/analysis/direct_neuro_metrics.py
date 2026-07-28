"""Compatibility wrappers for the direct neural-metric implementations.

Historical analysis scripts imported ``*_score01`` helpers from this module,
while the maintained implementations now live in
``nethobench.neuro.metrics.direct`` and expose unscaled names.  Keep the
analysis API stable by delegating to those implementations and renaming only
the primary score key; no score transformation is applied.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from nethobench.neuro.metrics.direct import (
    compute_graph_score,
    compute_manifold_score,
    compute_moment_score,
    compute_trajectory_score,
)


def _as_score01(
    result: dict[str, object],
    *,
    source_key: str,
    target_key: str,
) -> dict[str, object]:
    """Return a shallow result copy with a canonical ``score01`` key."""
    scores = dict(result.get("scores", {}))
    value = scores.pop(source_key, np.nan)
    scores[target_key] = float(value) if np.isfinite(value) else np.nan
    return {**result, "scores": scores}


def _delegate(
    fn: Callable[[np.ndarray, np.ndarray], dict[str, object]],
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    *,
    source_key: str,
    target_key: str,
) -> dict[str, object]:
    gt = np.asarray(gt_arr, dtype=np.float64)
    pred = np.asarray(pred_arr, dtype=np.float64)
    if gt.shape != pred.shape or gt.ndim != 3:
        raise ValueError(
            "Expected matching [sequence, time, region] arrays, "
            f"got {gt.shape} and {pred.shape}."
        )
    return _as_score01(
        fn(gt, pred),
        source_key=source_key,
        target_key=target_key,
    )


def compute_moment_score01(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
) -> dict[str, object]:
    return _delegate(
        compute_moment_score,
        gt_arr,
        pred_arr,
        source_key="MOM_score",
        target_key="MOM_score01",
    )


def compute_graph_score01(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
) -> dict[str, object]:
    return _delegate(
        compute_graph_score,
        gt_arr,
        pred_arr,
        source_key="GRAPH_score",
        target_key="GRAPH_score01",
    )


def compute_manifold_score01(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
) -> dict[str, object]:
    return _delegate(
        compute_manifold_score,
        gt_arr,
        pred_arr,
        source_key="MANI_score",
        target_key="MANI_score01",
    )


def compute_trajectory_score01(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
) -> dict[str, object]:
    return _delegate(
        compute_trajectory_score,
        gt_arr,
        pred_arr,
        source_key="TRJDIST_score",
        target_key="TRJDIST_score01",
    )
