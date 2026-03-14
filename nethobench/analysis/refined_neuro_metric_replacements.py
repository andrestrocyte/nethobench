from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .sensitive_metric_candidates import (
    autocorr_weighted_rmse_power_v6,
    bandpower_band_fraction_v2,
    corruption_region_permute_blend,
    corruption_region_shift_desync,
    corruption_time_shuffle_blend,
    crosscorr_lagged_matrix_v4,
    crosscorr_topedge_profiles_v5,
    manifold_ph_knn_profile_v5,
    manifold_ph_stratified_lifetime_v7,
    pca_reconstruction_product_v6,
    trajectory_occupancy_velocity_v4,
    trajectory_path_features_v3,
)


ScoreFn = Callable[[np.ndarray, np.ndarray], dict]
CorruptionFn = Callable[[np.ndarray, float, int], np.ndarray]


DEFAULT_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _finite_float(value: float) -> float:
    return float(value) if np.isfinite(value) else np.nan


def _extract_score(result: dict) -> float:
    if not isinstance(result, dict):
        return np.nan
    value = result.get("score", np.nan)
    return _finite_float(value)


def _weighted_mean_available(values: dict[str, float], weights: dict[str, float]) -> float:
    valid = [(key, values[key], weights[key]) for key in values if np.isfinite(values[key]) and weights.get(key, 0.0) > 0]
    if not valid:
        return np.nan
    total_weight = float(np.sum([weight for _, _, weight in valid]))
    if total_weight <= 0:
        return np.nan
    return float(np.sum([value * weight for _, value, weight in valid]) / total_weight)


def _distance_from_score(score: float) -> float:
    if not np.isfinite(score) or score <= 0:
        return np.nan
    return float((1.0 / score) - 1.0)


def _metric_display_name(metric_key: str) -> str:
    names = {
        "PCA_score01": "PCA",
        "AUTO_score01": "Autocorr",
        "CC_score01": "CrossCorr",
        "MANI_score01": "Manifold",
        "BP_score01": "Bandpower",
        "TRJDIST_score01": "Trajectory",
    }
    return names.get(metric_key, metric_key)


def _build_corruption_df(
    gt_arr: np.ndarray,
    score_fn: ScoreFn,
    corruption_fn: CorruptionFn,
    *,
    score_key: str,
    levels: tuple[float, ...] = DEFAULT_LEVELS,
    family_label: str,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for level in levels:
        corrupted = corruption_fn(gt_arr, float(level), seed=seed)
        score = _extract_score(score_fn(gt_arr, corrupted))
        rows.append(
            {
                "family": family_label if level > 0 else "baseline",
                "magnitude": float(level),
                score_key: score,
            }
        )
    return pd.DataFrame(rows)


def _plot_corruption_df(
    df: pd.DataFrame,
    *,
    score_key: str,
    title: str,
    enable_plots: bool,
) -> None:
    if not enable_plots or df.empty or score_key not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(df["magnitude"], df[score_key], marker="o", linewidth=2.2, color="#1f77b4")
    ax.set_xlabel("Corruption magnitude")
    ax.set_ylabel("score (0-1)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()


def _display_summary(
    *,
    label: str,
    score_key: str,
    score: float,
    description: str,
    corruption_df: pd.DataFrame,
) -> None:
    print(f"=== {label} replacement metric ===")
    print(description)
    print(f"{score_key}: {score:.6f}" if np.isfinite(score) else f"{score_key}: NaN")
    if corruption_df.empty:
        return
    cols = ["family", "magnitude", score_key]
    print("\nCorruption sweep:")
    print(corruption_df[cols].to_string(index=False))


def _wrap_population_metric(
    *,
    score_key: str,
    mean_key: str,
    score: float,
    description: str,
    candidate_name: str,
    extra_scores: dict[str, float] | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> dict:
    scores = {
        score_key: _finite_float(score),
        mean_key: _finite_float(score),
    }
    if extra_scores:
        for key, value in extra_scores.items():
            scores[key] = _finite_float(value)
    out = {
        "candidate_name": candidate_name,
        "description": description,
        "scores": scores,
    }
    if extra_arrays:
        out.update(extra_arrays)
    return out


def _final_crosscorr_score(gt_arr: np.ndarray, pred_arr: np.ndarray) -> dict:
    lagged = _extract_score(crosscorr_lagged_matrix_v4(gt_arr, pred_arr))
    topedge = _extract_score(crosscorr_topedge_profiles_v5(gt_arr, pred_arr))
    return {
        "score": _weighted_mean_available(
            {"lagged": lagged, "topedge": topedge},
            {"lagged": 0.75, "topedge": 0.25},
        )
    }


def _final_trajectory_score(gt_arr: np.ndarray, pred_arr: np.ndarray) -> dict:
    occupancy = _extract_score(trajectory_occupancy_velocity_v4(gt_arr, pred_arr))
    path = _extract_score(trajectory_path_features_v3(gt_arr, pred_arr))
    return {
        "score": _weighted_mean_available(
            {"occupancy": occupancy, "path": path},
            {"occupancy": 0.60, "path": 0.40},
        )
    }


def _final_manifold_score(gt_arr: np.ndarray, pred_arr: np.ndarray) -> dict:
    topology = _extract_score(manifold_ph_stratified_lifetime_v7(gt_arr, pred_arr))
    local = _extract_score(manifold_ph_knn_profile_v5(gt_arr, pred_arr))
    return {
        "score": _weighted_mean_available(
            {"topology": topology, "local": local},
            {"topology": 0.75, "local": 0.25},
        )
    }


def compute_pca_replacement(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    *,
    enable_plots: bool = True,
) -> tuple[dict, pd.DataFrame]:
    description = (
        "Squared GT-basis reconstruction-transfer score. "
        "It compares how well prediction variance is captured by the GT PCA basis "
        "and expands the dynamic range by squaring the agreement score."
    )
    score = _extract_score(pca_reconstruction_product_v6(gt_arr, pred_arr))
    pca_simple = _wrap_population_metric(
        score_key="PCA_score01",
        mean_key="PCA_mean",
        score=score,
        description=description,
        candidate_name="pca_reconstruction_product_v6",
    )
    pca_corr_df = _build_corruption_df(
        gt_arr,
        pca_reconstruction_product_v6,
        corruption_region_permute_blend,
        score_key="PCA_score01",
        family_label="region_permute_blend",
    )
    _display_summary(
        label="PCA",
        score_key="PCA_score01",
        score=score,
        description=description,
        corruption_df=pca_corr_df,
    )
    _plot_corruption_df(
        pca_corr_df,
        score_key="PCA_score01",
        title="PCA replacement degradation",
        enable_plots=enable_plots,
    )
    return pca_simple, pca_corr_df


def compute_autocorr_replacement(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    *,
    enable_plots: bool = True,
) -> tuple[dict, pd.DataFrame]:
    description = (
        "Weighted early-lag autocorrelation agreement with extra dynamic-range "
        "sensitivity from squaring the baseline agreement score."
    )
    score = _extract_score(autocorr_weighted_rmse_power_v6(gt_arr, pred_arr))
    distance = _distance_from_score(score)
    auto_sensitive = _wrap_population_metric(
        score_key="AUTO_score01",
        mean_key="AUTO_mean",
        score=score,
        description=description,
        candidate_name="autocorr_weighted_rmse_power_v6",
        extra_arrays={"d_core_sr": np.asarray([distance], dtype=np.float64)},
    )
    auto_corr_df = _build_corruption_df(
        gt_arr,
        autocorr_weighted_rmse_power_v6,
        corruption_time_shuffle_blend,
        score_key="AUTO_score01",
        family_label="time_shuffle_blend",
    )
    _display_summary(
        label="Autocorr",
        score_key="AUTO_score01",
        score=score,
        description=description,
        corruption_df=auto_corr_df,
    )
    _plot_corruption_df(
        auto_corr_df,
        score_key="AUTO_score01",
        title="Autocorr replacement degradation",
        enable_plots=enable_plots,
    )
    return auto_sensitive, auto_corr_df


def compute_crosscorr_replacement(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    *,
    enable_plots: bool = True,
) -> tuple[dict, pd.DataFrame]:
    description = (
        "Population lagged correlation-matrix agreement stabilized by strong-edge profile agreement. "
        "It combines lag-0/lag-1 coupling structure with a top-edge cross-correlation profile term."
    )
    score = _extract_score(_final_crosscorr_score(gt_arr, pred_arr))
    distance = _distance_from_score(score)
    cc_sensitive = _wrap_population_metric(
        score_key="CC_score01",
        mean_key="CC_mean",
        score=score,
        description=description,
        candidate_name="final_crosscorr_score",
        extra_arrays={"d_core_sp": np.asarray([distance], dtype=np.float64)},
    )
    cc_corr_df = _build_corruption_df(
        gt_arr,
        _final_crosscorr_score,
        corruption_region_shift_desync,
        score_key="CC_score01",
        family_label="region_shift_desync",
    )
    _display_summary(
        label="CrossCorr",
        score_key="CC_score01",
        score=score,
        description=description,
        corruption_df=cc_corr_df,
    )
    _plot_corruption_df(
        cc_corr_df,
        score_key="CC_score01",
        title="CrossCorr replacement degradation",
        enable_plots=enable_plots,
    )
    return cc_sensitive, cc_corr_df


def compute_bandpower_replacement(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    *,
    enable_plots: bool = True,
) -> tuple[dict, pd.DataFrame]:
    description = (
        "Data-driven band-fraction agreement. "
        "The GT mean spectrum is split into equal-power bands and the score compares "
        "how prediction redistributes power across those bands."
    )
    score = _extract_score(bandpower_band_fraction_v2(gt_arr, pred_arr))
    bandpower_simple = _wrap_population_metric(
        score_key="BP_score01",
        mean_key="BP_mean",
        score=score,
        description=description,
        candidate_name="bandpower_band_fraction_v2",
    )
    bp_corr_df = _build_corruption_df(
        gt_arr,
        bandpower_band_fraction_v2,
        corruption_time_shuffle_blend,
        score_key="BP_score01",
        family_label="time_shuffle_blend",
    )
    _display_summary(
        label="Bandpower",
        score_key="BP_score01",
        score=score,
        description=description,
        corruption_df=bp_corr_df,
    )
    _plot_corruption_df(
        bp_corr_df,
        score_key="BP_score01",
        title="Bandpower replacement degradation",
        enable_plots=enable_plots,
    )
    return bandpower_simple, bp_corr_df


def compute_trajectory_replacement(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    *,
    enable_plots: bool = True,
) -> tuple[dict, pd.DataFrame]:
    description = (
        "Pooled latent occupancy and velocity-distribution agreement stabilized by latent path features. "
        "It scores occupancy, speed, turning, and sequence-level path structure in a shared GT PCA space."
    )
    score = _extract_score(_final_trajectory_score(gt_arr, pred_arr))
    trajectory_dist_simple = _wrap_population_metric(
        score_key="TRJDIST_score01",
        mean_key="TRJDIST_seq_mean",
        score=score,
        description=description,
        candidate_name="final_trajectory_score",
    )
    trjdist_corr_df = _build_corruption_df(
        gt_arr,
        _final_trajectory_score,
        corruption_time_shuffle_blend,
        score_key="TRJDIST_score01",
        family_label="time_shuffle_blend",
    )
    _display_summary(
        label="Trajectory",
        score_key="TRJDIST_score01",
        score=score,
        description=description,
        corruption_df=trjdist_corr_df,
    )
    _plot_corruption_df(
        trjdist_corr_df,
        score_key="TRJDIST_score01",
        title="Trajectory replacement degradation",
        enable_plots=enable_plots,
    )
    return trajectory_dist_simple, trjdist_corr_df


def compute_manifold_replacement(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    *,
    variant: str = "final_manifold_score",
    enable_plots: bool = True,
) -> tuple[dict, pd.DataFrame]:
    description = (
        "Persistent-homology lifetime agreement stabilized by local-neighborhood geometry. "
        "It combines a stratified PH topology term with a local k-nearest-neighbor geometry term."
    )
    score = _extract_score(_final_manifold_score(gt_arr, pred_arr))
    mani_simple = _wrap_population_metric(
        score_key="MANI_score01",
        mean_key="MANI_mean",
        score=score,
        description=description,
        candidate_name="final_manifold_score",
    )
    mani_corr_df = _build_corruption_df(
        gt_arr,
        _final_manifold_score,
        corruption_region_permute_blend,
        score_key="MANI_score01",
        family_label="region_permute_blend",
    )
    _display_summary(
        label="Manifold",
        score_key="MANI_score01",
        score=score,
        description=description,
        corruption_df=mani_corr_df,
    )
    _plot_corruption_df(
        mani_corr_df,
        score_key="MANI_score01",
        title=f"{_metric_display_name('MANI_score01')} replacement degradation",
        enable_plots=enable_plots,
    )
    return mani_simple, mani_corr_df
