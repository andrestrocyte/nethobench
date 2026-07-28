"""Evaluation utilities for stochastic neural predictions.

The structural Monte Carlo protocol in this module never concatenates multiple
draws against duplicated ground truth. An ensemble has shape
``[draw, context, time, region]`` and each structural replicate scores one draw
against the original ``[context, time, region]`` target tensor.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


EPS = 1e-12
DEFAULT_VARIOGRAM_LAGS: tuple[int, ...] = (1, 2, 4, 8, 16)


def validate_ensemble(
    ground_truth: np.ndarray,
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return float64 target/ensemble tensors."""
    target = np.asarray(ground_truth, dtype=np.float64)
    ensemble = np.asarray(samples, dtype=np.float64)
    if target.ndim != 3:
        raise ValueError(
            "ground_truth must have shape [context, time, region], "
            f"got {target.shape}."
        )
    if ensemble.ndim != 4:
        raise ValueError(
            "samples must have shape [draw, context, time, region], "
            f"got {ensemble.shape}."
        )
    if ensemble.shape[1:] != target.shape:
        raise ValueError(
            f"Ensemble/target shape mismatch: {ensemble.shape[1:]} vs {target.shape}."
        )
    if ensemble.shape[0] < 1:
        raise ValueError("At least one stochastic draw is required.")
    if not np.isfinite(target).all() or not np.isfinite(ensemble).all():
        raise ValueError("Proper-score inputs must contain only finite values.")
    return target, ensemble


def reconstruct_sample_major_ensemble(
    repeated_ground_truth: np.ndarray,
    pooled_predictions: np.ndarray,
    *,
    n_draws: int,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Undo legacy sample-major pooling and verify target duplication exactly.

    Legacy files were written by reshaping ``[draw, context, time, region]`` to
    ``[draw * context, time, region]`` and repeating the target along the draw
    axis. This function reconstructs the ensemble, checks that every repeated
    target block agrees, and returns only one target copy.
    """
    repeated = np.asarray(repeated_ground_truth, dtype=np.float64)
    pooled = np.asarray(pooled_predictions, dtype=np.float64)
    if repeated.ndim != 3 or pooled.ndim != 3:
        raise ValueError("Pooled arrays must have shape [draw * context, time, region].")
    if repeated.shape != pooled.shape:
        raise ValueError(f"Shape mismatch: {repeated.shape} vs {pooled.shape}.")
    if int(n_draws) < 1 or repeated.shape[0] % int(n_draws) != 0:
        raise ValueError(
            f"Leading dimension {repeated.shape[0]} is not divisible by "
            f"n_draws={n_draws}."
        )
    n_contexts = repeated.shape[0] // int(n_draws)
    target_blocks = repeated.reshape(
        int(n_draws), n_contexts, repeated.shape[1], repeated.shape[2]
    )
    ensemble = pooled.reshape(
        int(n_draws), n_contexts, pooled.shape[1], pooled.shape[2]
    )
    target = target_blocks[0]
    if not np.allclose(
        target_blocks,
        target[None, ...],
        rtol=float(rtol),
        atol=float(atol),
        equal_nan=True,
    ):
        max_difference = float(
            np.nanmax(np.abs(target_blocks - target[None, ...]))
        )
        raise ValueError(
            "Repeated ground-truth blocks are inconsistent; sample-major "
            f"reconstruction is unsafe (max absolute difference {max_difference:.3g})."
        )
    return validate_ensemble(target, ensemble)


def monte_carlo_structural_scores(
    ground_truth: np.ndarray,
    samples: np.ndarray,
    scorer: Callable[[np.ndarray, np.ndarray], Mapping[str, float]],
) -> dict[str, Any]:
    """Score one stochastic draw per context in each Monte Carlo replicate."""
    target, ensemble = validate_ensemble(ground_truth, samples)
    replicates: list[dict[str, float | int]] = []
    for draw_index in range(ensemble.shape[0]):
        raw = scorer(target, ensemble[draw_index])
        row: dict[str, float | int] = {"draw": int(draw_index)}
        row.update({str(name): float(value) for name, value in raw.items()})
        replicates.append(row)

    metric_names = sorted(
        {
            name
            for row in replicates
            for name in row
            if name != "draw"
        }
    )
    summary: dict[str, dict[str, float | int]] = {}
    for metric_name in metric_names:
        values = np.asarray(
            [float(row.get(metric_name, np.nan)) for row in replicates],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            summary[metric_name] = {
                "n": 0,
                "mean": np.nan,
                "sd": np.nan,
                "sem": np.nan,
                "q025": np.nan,
                "q975": np.nan,
            }
            continue
        sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summary[metric_name] = {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "sd": sd,
            "sem": float(sd / np.sqrt(values.size)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
        }
    return {
        "protocol": "one_draw_per_context_per_replicate",
        "n_draws": int(ensemble.shape[0]),
        "n_contexts": int(target.shape[0]),
        "replicates": replicates,
        "summary": summary,
    }


def _robust_region_scale(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flattened = target.reshape(-1, target.shape[-1])
    center = np.median(flattened, axis=0)
    mad = np.median(np.abs(flattened - center), axis=0)
    scale = 1.4826 * mad
    fallback = np.std(flattened, axis=0, ddof=1)
    scale = np.where(scale > EPS, scale, fallback)
    scale = np.where(scale > EPS, scale, 1.0)
    return center, scale


def _standardized_ensemble(
    target: np.ndarray,
    ensemble: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center, scale = _robust_region_scale(target)
    return (target - center) / scale, (ensemble - center) / scale


def energy_score(
    ground_truth: np.ndarray,
    samples: np.ndarray,
    *,
    return_per_context: bool = False,
) -> float | tuple[float, np.ndarray]:
    """Compute the multivariate ensemble energy score (lower is better).

    Time-region targets are robustly standardized per region and Euclidean
    distances are divided by ``sqrt(time * region)``. The standard ensemble
    estimator includes all draw pairs, including the zero diagonal.
    """
    target, ensemble = validate_ensemble(ground_truth, samples)
    target, ensemble = _standardized_ensemble(target, ensemble)
    n_draws, n_contexts, n_time, n_regions = ensemble.shape
    dimension_scale = float(np.sqrt(n_time * n_regions))
    per_context = np.empty(n_contexts, dtype=np.float64)
    for context_index in range(n_contexts):
        observation = target[context_index].reshape(-1)
        forecasts = ensemble[:, context_index].reshape(n_draws, -1)
        observation_term = float(
            np.mean(np.linalg.norm(forecasts - observation[None, :], axis=1))
        )
        pairwise = forecasts[:, None, :] - forecasts[None, :, :]
        ensemble_term = float(
            np.mean(np.linalg.norm(pairwise, axis=2))
        )
        per_context[context_index] = (
            observation_term - 0.5 * ensemble_term
        ) / dimension_scale
    score = float(np.mean(per_context))
    return (score, per_context) if return_per_context else score


def temporal_variogram_score(
    ground_truth: np.ndarray,
    samples: np.ndarray,
    *,
    lags: Sequence[int] = DEFAULT_VARIOGRAM_LAGS,
    p: float = 0.5,
    return_per_context: bool = False,
) -> float | tuple[float, np.ndarray]:
    """Compute a lag-restricted temporal variogram score (lower is better).

    For each fixed lag, the observed ``|y(t+l)-y(t)|**p`` is compared with its
    ensemble expectation. Lag weights are proportional to ``1 / lag`` and are
    normalized to sum to one.
    """
    if not (0 < float(p) <= 2):
        raise ValueError("p must lie in (0, 2].")
    target, ensemble = validate_ensemble(ground_truth, samples)
    target, ensemble = _standardized_ensemble(target, ensemble)
    usable_lags = tuple(
        int(lag) for lag in lags if int(lag) >= 1 and int(lag) < target.shape[1]
    )
    if not usable_lags:
        raise ValueError("No variogram lag is shorter than the target sequence.")
    weights = 1.0 / np.asarray(usable_lags, dtype=np.float64)
    weights = weights / np.sum(weights)
    per_context = np.zeros(target.shape[0], dtype=np.float64)
    for weight, lag in zip(weights, usable_lags):
        observed = np.abs(target[:, lag:, :] - target[:, :-lag, :]) ** float(p)
        forecast = np.abs(
            ensemble[:, :, lag:, :] - ensemble[:, :, :-lag, :]
        ) ** float(p)
        expected = np.mean(forecast, axis=0)
        discrepancy = np.mean((observed - expected) ** 2, axis=(1, 2))
        per_context += float(weight) * discrepancy
    score = float(np.mean(per_context))
    return (score, per_context) if return_per_context else score


def ensemble_calibration_diagnostics(
    ground_truth: np.ndarray,
    samples: np.ndarray,
    *,
    central_levels: Sequence[float] = (0.5, 0.8, 0.9),
    rank_seed: int = 0,
) -> dict[str, Any]:
    """Return interval coverage, sharpness, ranks, and dispersion diagnostics."""
    target, ensemble = validate_ensemble(ground_truth, samples)
    standardized_target, standardized_ensemble = _standardized_ensemble(
        target, ensemble
    )
    intervals: dict[str, dict[str, float]] = {}
    for level_value in central_levels:
        level = float(level_value)
        if not 0 < level < 1:
            raise ValueError("Every central interval level must lie in (0, 1).")
        tail = 0.5 * (1.0 - level)
        lower = np.quantile(standardized_ensemble, tail, axis=0)
        upper = np.quantile(standardized_ensemble, 1.0 - tail, axis=0)
        covered = (standardized_target >= lower) & (standardized_target <= upper)
        intervals[f"{level:.3f}"] = {
            "nominal": level,
            "coverage": float(np.mean(covered)),
            "coverage_error": float(np.mean(covered) - level),
            "mean_width": float(np.mean(upper - lower)),
        }

    lower_count = np.sum(
        standardized_ensemble < standardized_target[None, ...],
        axis=0,
    )
    tie_count = np.sum(
        standardized_ensemble == standardized_target[None, ...],
        axis=0,
    )
    rng = np.random.default_rng(int(rank_seed))
    tie_offset = np.floor(rng.random(tie_count.shape) * (tie_count + 1)).astype(int)
    ranks = lower_count + tie_offset
    rank_counts = np.bincount(
        ranks.reshape(-1),
        minlength=ensemble.shape[0] + 1,
    )
    rank_frequencies = rank_counts / max(int(np.sum(rank_counts)), 1)

    ensemble_variance = float(
        np.mean(np.var(standardized_ensemble, axis=0, ddof=1))
    ) if ensemble.shape[0] > 1 else 0.0
    target_variance = float(np.var(standardized_target, ddof=1))
    variance_ratio = (
        ensemble_variance / target_variance
        if target_variance > EPS
        else np.nan
    )
    return {
        "n_draws": int(ensemble.shape[0]),
        "n_contexts": int(target.shape[0]),
        "central_intervals": intervals,
        "rank_histogram_counts": rank_counts.astype(int).tolist(),
        "rank_histogram_frequencies": rank_frequencies.tolist(),
        "mean_ensemble_variance": ensemble_variance,
        "pooled_target_variance": target_variance,
        "ensemble_to_target_variance_ratio": float(variance_ratio),
        "variance_ratio_note": (
            "Dispersion diagnostic only; pooled target variance includes "
            "between-context variation and is not itself a proper score."
        ),
    }


__all__ = [
    "DEFAULT_VARIOGRAM_LAGS",
    "energy_score",
    "ensemble_calibration_diagnostics",
    "monte_carlo_structural_scores",
    "reconstruct_sample_major_ensemble",
    "temporal_variogram_score",
    "validate_ensemble",
]
