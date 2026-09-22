"""Explicit alternatives to the pinned scoring definitions.

Adapted from the V2 implementation-sensitivity analysis. Existing metrics and
composites remain unchanged. These diagnostics must be reported separately;
they are not a new default benchmark or validation of biological correctness.
"""
from __future__ import annotations

import numpy as np

from nethobench.neuro.metrics import additional as a

PROTOCOL = "sequence-directed-conditional-v1"


def _tensor(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 3 or any(size == 0 for size in value.shape):
        raise ValueError("Expected a nonempty (sequence, time, channel) tensor")
    return value


def _paired(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt, pred = _tensor(gt), _tensor(pred)
    if gt.shape != pred.shape:
        raise ValueError("Reference and prediction must have identical aligned shapes")
    return gt, pred


def _standardized_tensor(value: np.ndarray) -> np.ndarray:
    rows = a._pooled_rows(value)
    if not len(rows):
        return np.full(value.shape, np.nan)
    center, scale = a._fit_reference_scaler(rows)
    return (value - center) / scale


def sequence_pairs(value: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Pair finite endpoints within sequences, never across a sequence boundary.

    Missing endpoints are removed after pairing, so missing rows never compress
    the time axis. ``lag`` is measured in samples, not seconds.
    """
    value = _tensor(value)
    if not isinstance(lag, (int, np.integer)) or isinstance(lag, bool) or lag < 1:
        raise ValueError("lag must be a positive integer number of samples")
    if lag >= value.shape[1]:
        empty = np.empty((0, value.shape[2]))
        return empty, empty.copy()
    x = value[:, :-lag].reshape(-1, value.shape[2])
    y = value[:, lag:].reshape(-1, value.shape[2])
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    return x[valid], y[valid]


def _lag_cov(value: np.ndarray, lag: int) -> np.ndarray | None:
    x, y = sequence_pairs(value, lag)
    if len(x) < 4 or x.shape[1] < 2:
        return None
    return (x - x.mean(0)).T @ (y - y.mean(0)) / (len(x) - 1)


def _var_coeff(value: np.ndarray) -> np.ndarray | None:
    x, y = sequence_pairs(value, 1)
    if len(x) < 5 or x.shape[1] < 2:
        return None
    return np.linalg.solve(x.T @ x + 0.01 * np.eye(x.shape[1]), x.T @ y).T


def _directed_similarity(gt: np.ndarray | None, pred: np.ndarray | None) -> float:
    if gt is None or pred is None:
        return float("nan")
    mask = ~np.eye(len(gt), dtype=bool)
    gv, pv = gt[mask], pred[mask]
    return float(a.weighted_mean_available(
        {0: a.correlation_score(gv, pv), 1: a.rmse_similarity(gv, pv)},
        weights={0: a.WEIGHT_MATRIX_CORR, 1: a.WEIGHT_MATRIX_RMSE},
    ))


def compute_relational_sensitivity(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Sequence-aware lag/VAR agreement using all directed off-diagonal edges.

    Each tensor is median/IQR standardized separately, as in the V2 diagnostic.
    Lag covariance averages available lags 1, 2 and 4; ridge VAR(1) uses 0.01.
    These are statistical proxies, not causal or mechanistic estimates.
    """
    gt, pred = _paired(gt, pred)
    g, p = _standardized_tensor(gt), _standardized_tensor(pred)
    values = [_directed_similarity(_lag_cov(g, lag), _lag_cov(p, lag))
              for lag in (1, 2, 4)]
    available = [value for value in values if np.isfinite(value)]
    return {
        "protocol": PROTOCOL,
        "lag_units": "samples",
        "lagged_covariance_directed": float(np.mean(available)) if available else float("nan"),
        "var1_directed": _directed_similarity(_var_coeff(g), _var_coeff(p)),
        "reference_pair_counts": [len(sequence_pairs(gt, lag)[0]) for lag in (1, 2, 4)],
        "prediction_pair_counts": [len(sequence_pairs(pred, lag)[0]) for lag in (1, 2, 4)],
    }


def conditional_transition_agreement(
    reference_sequences: list[np.ndarray], prediction_sequences: list[np.ndarray],
    n_states: int, lag: int = 1,
) -> dict:
    """Uniform average of row-conditional agreement on reference-supported origins.

    Sequences may differ in count and length. Count transitions independently in
    each population. An origin absent from predictions scores zero if present in
    the reference. No reference-supported transitions yields NaN, not agreement.
    """
    for name, value in (("n_states", n_states), ("lag", lag)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    def counts(sequences: list[np.ndarray]) -> np.ndarray:
        result = np.zeros((n_states, n_states))
        for sequence in sequences:
            sequence = np.asarray(sequence)
            if sequence.ndim != 1 or (sequence.size and (
                not np.issubdtype(sequence.dtype, np.integer)
                or np.any(sequence < 0) or np.any(sequence >= n_states)
            )):
                raise ValueError("State sequences must contain integer labels in [0, n_states)")
            if len(sequence) > lag:
                np.add.at(result, (sequence[:-lag], sequence[lag:]), 1)
        return result
    gm, pm = counts(reference_sequences), counts(prediction_sequences)
    support = np.flatnonzero(gm.sum(axis=1) > 0)
    scores = [a._hist_similarity(gm[i], pm[i]) if pm[i].sum() > 0 else 0.0
              for i in support]
    return {
        "score": float(np.mean(scores)) if scores else float("nan"),
        "reference_origin_states": int(len(support)),
        "missing_prediction_origin_states": int(np.sum(pm[support].sum(axis=1) == 0)),
    }


def compute_conditional_state_sensitivity(gt: np.ndarray, pred: np.ndarray) -> dict:
    """V2 reference-fitted PCA/K=11 states with conditional transition scoring.

    Requires finite aligned tensors: rejecting missing rows avoids accidentally
    creating transitions across gaps. Fewer than 12 reference rows is undefined.
    Occupancy is intentionally excluded; these are not biological state labels.
    """
    gt, pred = _paired(gt, pred)
    if not np.isfinite(gt).all() or not np.isfinite(pred).all():
        raise ValueError("Conditional-state sensitivity requires finite tensors")
    result = {"protocol": PROTOCOL, "lag_units": "samples", "n_states": 11,
              "origin_weighting": "uniform_reference_supported", "transitions": {}}
    empty = {"score": float("nan"), "reference_origin_states": 0,
             "missing_prediction_origin_states": 0}
    if gt.shape[0] * gt.shape[1] < 12:
        result["transitions"] = {str(lag): dict(empty) for lag in (1, 2, 3)}
        return result
    ref = a._prepare_latent_state_reference(gt, pred, a._pooled_rows(gt), a._pooled_rows(pred))
    centers = None if ref is None else a._fit_kmeans_centers(
        ref["gt_proj_fit"], 11, max_fit_points=10000, random_state=11)
    if centers is None:
        result["transitions"] = {str(lag): dict(empty) for lag in (1, 2, 3)}
        return result
    gs = [a._assign_to_centers(seq, centers) for seq in ref["gt_seq_proj"]]
    ps = [a._assign_to_centers(seq, centers) for seq in ref["pred_seq_proj"]]
    result["transitions"] = {str(lag): conditional_transition_agreement(gs, ps, 11, lag)
                             for lag in (1, 2, 3)}
    return result
