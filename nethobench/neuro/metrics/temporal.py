"""Order-sensitive temporal and spectral metrics.

All metrics operate on arrays shaped ``[sequence, time, region]``. Temporal
structure is estimated separately for every sequence-region trace so sequence
boundaries are never joined into an artificial time series.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import welch

from nethobench.utils.evaluation_constants import config


EPS = 1e-12
DEFAULT_ACF_LAGS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 48)
DEFAULT_DIRECTIONAL_LAGS: tuple[int, ...] = (1, 2, 4, 8)
DEFAULT_ACF_LAG_SECONDS: tuple[float, ...] = tuple(
    lag / 30.0 for lag in DEFAULT_ACF_LAGS
)
DEFAULT_DIRECTIONAL_LAG_SECONDS: tuple[float, ...] = tuple(
    lag / 30.0 for lag in DEFAULT_DIRECTIONAL_LAGS
)


def _validate_pair(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gt = np.asarray(gt_arr, dtype=np.float64)
    pred = np.asarray(pred_arr, dtype=np.float64)
    if gt.ndim != 3 or pred.ndim != 3:
        raise ValueError(
            "Expected gt_arr and pred_arr with shape [sequence, time, region]."
        )
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} vs {pred.shape}.")
    return gt, pred


def _fill_missing_trace(trace: np.ndarray) -> np.ndarray | None:
    """Linearly interpolate sparse missing values while preserving the time grid."""
    x = np.asarray(trace, dtype=np.float64)
    valid = np.isfinite(x)
    if valid.sum() < max(8, int(np.ceil(0.8 * x.size))):
        return None
    if valid.all():
        return x
    positions = np.arange(x.size, dtype=np.float64)
    return np.interp(positions, positions[valid], x[valid])


def autocorrelation_at_lags(
    trace: np.ndarray,
    lags: Sequence[int] = DEFAULT_ACF_LAGS,
) -> np.ndarray:
    """Return Pearson autocorrelation at fixed positive frame lags."""
    x = _fill_missing_trace(trace)
    out = np.full(len(lags), np.nan, dtype=np.float64)
    if x is None:
        return out
    for index, lag_value in enumerate(lags):
        lag = int(lag_value)
        if lag < 1 or x.size - lag < 3:
            continue
        left = x[:-lag]
        right = x[lag:]
        left = left - np.mean(left)
        right = right - np.mean(right)
        denominator = float(
            np.sqrt(np.sum(left * left) * np.sum(right * right))
        )
        if denominator > EPS:
            out[index] = float(np.sum(left * right) / denominator)
    return np.clip(out, -1.0, 1.0)


def frame_lags_from_seconds(
    lag_seconds: Sequence[float],
    sampling_frequency_hz: float,
) -> tuple[int, ...]:
    """Convert physical lags to unique positive frame offsets."""
    fs = float(sampling_frequency_hz)
    if fs <= 0:
        raise ValueError("sampling_frequency_hz must be positive.")
    frame_lags: list[int] = []
    for seconds in lag_seconds:
        if float(seconds) <= 0:
            raise ValueError("All lag durations must be positive.")
        lag = max(1, int(round(float(seconds) * fs)))
        if lag not in frame_lags:
            frame_lags.append(lag)
    return tuple(frame_lags)


def _acf_trace_similarity(
    gt_trace: np.ndarray,
    pred_trace: np.ndarray,
    lags: Sequence[int],
) -> float:
    def _curve_similarity(gt_acf: np.ndarray, pred_acf: np.ndarray) -> float:
        valid = np.isfinite(gt_acf) & np.isfinite(pred_acf)
        if not valid.any():
            return np.nan
        gt_valid = gt_acf[valid]
        pred_valid = pred_acf[valid]
        discrepancy = float(np.linalg.norm(gt_valid - pred_valid))
        scale = float(np.linalg.norm(gt_valid) + np.linalg.norm(pred_valid))
        if scale <= EPS:
            return 1.0 if discrepancy <= EPS else 0.0
        return float(np.clip(1.0 - discrepancy / scale, 0.0, 1.0))

    raw_similarity = _curve_similarity(
        autocorrelation_at_lags(gt_trace, lags),
        autocorrelation_at_lags(pred_trace, lags),
    )
    increment_similarity = _curve_similarity(
        autocorrelation_at_lags(np.diff(gt_trace), lags),
        autocorrelation_at_lags(np.diff(pred_trace), lags),
    )
    valid_scores = np.asarray(
        [raw_similarity, increment_similarity],
        dtype=np.float64,
    )
    valid_scores = valid_scores[np.isfinite(valid_scores)]
    return float(np.mean(valid_scores)) if valid_scores.size else np.nan


def _spectral_components(
    trace: np.ndarray,
    *,
    sampling_frequency_hz: float,
    nperseg: int,
) -> tuple[np.ndarray, float] | None:
    x = _fill_missing_trace(trace)
    if x is None or x.size < 8:
        return None
    frequencies, density = welch(
        x,
        fs=float(sampling_frequency_hz),
        nperseg=min(int(nperseg), x.size),
        detrend="constant",
        scaling="density",
    )
    keep = (frequencies > 0) & np.isfinite(density) & (density >= 0)
    density = np.asarray(density[keep], dtype=np.float64)
    if density.size == 0:
        return None
    total = float(np.sum(density))
    if total <= EPS:
        return np.zeros_like(density), 0.0
    return density / total, total


def _psd_trace_similarity(
    gt_trace: np.ndarray,
    pred_trace: np.ndarray,
    *,
    sampling_frequency_hz: float,
    nperseg: int,
) -> tuple[float, float, float]:
    gt_components = _spectral_components(
        gt_trace,
        sampling_frequency_hz=sampling_frequency_hz,
        nperseg=nperseg,
    )
    pred_components = _spectral_components(
        pred_trace,
        sampling_frequency_hz=sampling_frequency_hz,
        nperseg=nperseg,
    )
    if gt_components is None or pred_components is None:
        return np.nan, np.nan, np.nan
    gt_shape, gt_power = gt_components
    pred_shape, pred_power = pred_components
    size = min(gt_shape.size, pred_shape.size)
    if size == 0:
        return np.nan, np.nan, np.nan
    if gt_power <= EPS and pred_power <= EPS:
        return 1.0, 1.0, 1.0
    if gt_power <= EPS or pred_power <= EPS:
        return 0.0, 0.0, 0.0
    gt_shape = gt_shape[:size]
    pred_shape = pred_shape[:size]
    gt_shape = gt_shape / max(float(np.sum(gt_shape)), EPS)
    pred_shape = pred_shape / max(float(np.sum(pred_shape)), EPS)
    shape_score = float(np.sum(np.sqrt(gt_shape * pred_shape)))
    shape_score = float(np.clip(shape_score, 0.0, 1.0))
    power_score = float(min(gt_power, pred_power) / max(gt_power, pred_power))
    power_score = float(np.clip(power_score, 0.0, 1.0))
    combined = float(np.sqrt(shape_score * power_score))
    return combined, shape_score, power_score


def _aggregate_unit_scores(values: np.ndarray) -> float:
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return np.nan
    return float(0.5 * np.mean(valid) + 0.5 * np.quantile(valid, 0.10))


def _standardize_sequence_pair(
    gt_sequence: np.ndarray,
    pred_sequence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Standardize a sequence pair using one shared usable-region mask.

    A region that is constant in only one member of the pair cannot support a
    directional covariance comparison.  Selecting usable columns separately
    could silently align different physical regions when the two members have
    the same number of constant columns, so the mask is deliberately shared.
    """
    gt = np.asarray(gt_sequence, dtype=np.float64)
    pred = np.asarray(pred_sequence, dtype=np.float64)
    if (
        gt.ndim != 2
        or pred.ndim != 2
        or gt.shape != pred.shape
        or gt.shape[0] < 5
    ):
        return None

    filled: list[np.ndarray] = []
    for sequence in (gt, pred):
        columns = [
            _fill_missing_trace(sequence[:, index])
            for index in range(sequence.shape[1])
        ]
        if any(column is None for column in columns):
            return None
        filled.append(np.column_stack(columns))

    gt_filled, pred_filled = filled
    gt_centered = gt_filled - np.mean(gt_filled, axis=0, keepdims=True)
    pred_centered = pred_filled - np.mean(pred_filled, axis=0, keepdims=True)
    gt_scale = np.std(gt_centered, axis=0, ddof=1)
    pred_scale = np.std(pred_centered, axis=0, ddof=1)
    usable = (gt_scale > EPS) & (pred_scale > EPS)
    if usable.sum() < 2:
        return None
    return (
        gt_centered[:, usable] / gt_scale[usable],
        pred_centered[:, usable] / pred_scale[usable],
    )


def _directional_sequence_similarity(
    gt_sequence: np.ndarray,
    pred_sequence: np.ndarray,
    lags: Sequence[int],
) -> float:
    standardized = _standardize_sequence_pair(gt_sequence, pred_sequence)
    if standardized is None:
        return np.nan
    gt, pred = standardized
    gt_matrices: list[np.ndarray] = []
    pred_matrices: list[np.ndarray] = []
    for lag_value in lags:
        lag = int(lag_value)
        if lag < 1 or gt.shape[0] - lag < 3:
            continue
        gt_matrices.append((gt[:-lag].T @ gt[lag:]) / (gt.shape[0] - lag - 1))
        pred_matrices.append(
            (pred[:-lag].T @ pred[lag:]) / (pred.shape[0] - lag - 1)
        )
    if not gt_matrices:
        return np.nan
    gt_stack = np.stack(gt_matrices)
    pred_stack = np.stack(pred_matrices)
    difference = float(np.linalg.norm(gt_stack - pred_stack))
    scale = float(np.linalg.norm(gt_stack) + np.linalg.norm(pred_stack))
    if scale <= EPS:
        return 1.0 if difference <= EPS else 0.0
    return float(np.clip(1.0 - difference / scale, 0.0, 1.0))


def compute_temporal_metrics(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    *,
    acf_lags: Sequence[int] | None = None,
    directional_lags: Sequence[int] | None = None,
    sampling_frequency_hz: float | None = None,
    welch_nperseg: int | None = None,
) -> dict[str, float]:
    """Compute bounded temporal scores and a directionality diagnostic.

    ``ACF_score`` and ``PSD_score`` are headline temporal-family components.
    ``PSDShape_score`` and ``PSDPower_score`` expose the two PSD terms so
    variance contraction cannot be hidden by normalized spectra.
    ``DirectionalDynamics_score`` is reported as a diagnostic sidecar and is
    not included in the fixed temporal-family composite.
    """
    gt, pred = _validate_pair(gt_arr, pred_arr)
    fs = (
        float(config.WELCH_SAMPLING_FREQUENCY)
        if sampling_frequency_hz is None
        else float(sampling_frequency_hz)
    )
    nperseg = (
        int(config.WELCH_NPERSEG)
        if welch_nperseg is None
        else int(welch_nperseg)
    )
    if fs <= 0 or nperseg < 8:
        raise ValueError("sampling_frequency_hz must be positive and nperseg >= 8.")
    if acf_lags is None:
        acf_lags = frame_lags_from_seconds(DEFAULT_ACF_LAG_SECONDS, fs)
    if directional_lags is None:
        directional_lags = frame_lags_from_seconds(
            DEFAULT_DIRECTIONAL_LAG_SECONDS,
            fs,
        )

    n_sequences, _, n_regions = gt.shape
    acf_scores = np.full((n_sequences, n_regions), np.nan, dtype=np.float64)
    psd_scores = np.full_like(acf_scores, np.nan)
    psd_shape_scores = np.full_like(acf_scores, np.nan)
    psd_power_scores = np.full_like(acf_scores, np.nan)

    for sequence_index in range(n_sequences):
        for region_index in range(n_regions):
            gt_trace = gt[sequence_index, :, region_index]
            pred_trace = pred[sequence_index, :, region_index]
            acf_scores[sequence_index, region_index] = _acf_trace_similarity(
                gt_trace,
                pred_trace,
                acf_lags,
            )
            psd, shape, power = _psd_trace_similarity(
                gt_trace,
                pred_trace,
                sampling_frequency_hz=fs,
                nperseg=nperseg,
            )
            psd_scores[sequence_index, region_index] = psd
            psd_shape_scores[sequence_index, region_index] = shape
            psd_power_scores[sequence_index, region_index] = power

    directional_scores = np.asarray(
        [
            _directional_sequence_similarity(gt[index], pred[index], directional_lags)
            for index in range(n_sequences)
        ],
        dtype=np.float64,
    )
    return {
        "ACF_score": _aggregate_unit_scores(acf_scores),
        "PSD_score": _aggregate_unit_scores(psd_scores),
        "PSDShape_score": _aggregate_unit_scores(psd_shape_scores),
        "PSDPower_score": _aggregate_unit_scores(psd_power_scores),
        "DirectionalDynamics_score": _aggregate_unit_scores(directional_scores),
    }


__all__ = [
    "DEFAULT_ACF_LAGS",
    "DEFAULT_ACF_LAG_SECONDS",
    "DEFAULT_DIRECTIONAL_LAGS",
    "DEFAULT_DIRECTIONAL_LAG_SECONDS",
    "autocorrelation_at_lags",
    "compute_temporal_metrics",
    "frame_lags_from_seconds",
]
