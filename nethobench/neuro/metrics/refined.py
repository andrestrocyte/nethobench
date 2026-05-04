from __future__ import annotations

import logging
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kurtosis, skew, spearmanr

logger = logging.getLogger(__name__)
from nethobench.utils.calculation import _align_arrays, weighted_mean_available, EPS
from nethobench.neuro.metrics.sensitive import (
    _finite_rows,
    _safe_corrcoef,
    _score_from_distance,
    autocorr_weighted_rmse_power,
    bandpower_band_fraction,
    corruption_region_permute_blend,
    corruption_region_shift_desync,
    corruption_time_shuffle_blend,
    crosscorr_lagged_matrix,
    crosscorr_topedge_profiles,
    manifold_ph_knn_profile,
    manifold_ph_stratified_lifetime,
    pca_reconstruction_product,
    trajectory_occupancy_velocity,
    trajectory_path_features,
)
from nethobench.utils.evaluation_constants import (
    WEIGHT_MOMENT_VAR,
    WEIGHT_MOMENT_SKEW,
    WEIGHT_MOMENT_KURT,
    WEIGHT_GRAPH_JACCARD,
    WEIGHT_GRAPH_WEIGHT,
    WEIGHT_GRAPH_DEGREE,
    WEIGHT_GRAPH_CLUSTER,
    WEIGHT_CROSSCORR_LAGGED,
    WEIGHT_CROSSCORR_TOPEDGE,
    WEIGHT_TRAJECTORY_OCCUPANCY,
    WEIGHT_TRAJECTORY_PATH,
    WEIGHT_MANIFOLD_TOPOLOGY,
    WEIGHT_MANIFOLD_LOCAL,
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



def _distance_from_score(score: float) -> float:
    if not np.isfinite(score) or score <= 0:
        return np.nan
    return float((1.0 / score) - 1.0)


def _metric_display_name(metric_key: str) -> str:
    names = {
        "MOM_score01": "Moments",
        "GRAPH_score01": "Graph",
        "PCA_score01": "PCA",
        "AUTO_score01": "Autocorr",
        "CC_score01": "CrossCorr",
        "MANI_score01": "Manifold",
        "BP_score01": "Bandpower",
        "TRJDIST_score01": "Trajectory",
    }
    return names.get(metric_key, metric_key)



def _safe_spearman01(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xx = x[mask]
    yy = y[mask]
    if np.nanstd(xx) < 1e-12 or np.nanstd(yy) < 1e-12:
        return np.nan
    corr = spearmanr(xx, yy).correlation
    if corr is None or not np.isfinite(corr):
        return np.nan
    return float(np.clip(0.5 * (corr + 1.0), 0.0, 1.0))




def _score_from_components(
    components: dict[str, tuple[float, float]],
    stat_weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for name, (corr_score, rmse_score) in components.items():
        scores[name] = weighted_mean_available(
            {"corr": corr_score, "rmse": rmse_score},
            {"corr": 0.5, "rmse": 0.5},
        )
    final_score = weighted_mean_available(scores, stat_weights)
    return final_score, scores


def _moment_feature_arrays(arr: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "var": np.nanvar(arr, axis=1, ddof=1),
        "skew": skew(arr, axis=1, bias=False, nan_policy="omit"),
        "kurt": kurtosis(arr, axis=1, fisher=True, bias=False, nan_policy="omit"),
    }


def _one_sided_spikes_corruption(
    data: np.ndarray, level: float, seed: int
) -> np.ndarray:
    level = float(level)
    out = np.asarray(data, dtype=np.float64).copy()
    if level <= 0:
        return out
    rng = np.random.default_rng(seed)
    flat = out.reshape(-1, out.shape[-1])
    for region_idx in range(flat.shape[1]):
        values = flat[:, region_idx]
        valid_idx = np.flatnonzero(np.isfinite(values))
        if valid_idx.size < 8:
            continue
        valid_values = values[valid_idx]
        q25, q75 = np.quantile(valid_values, [0.25, 0.75])
        iqr = max(float(q75 - q25), 1e-6)
        tail_threshold = float(np.quantile(valid_values, 0.9))
        candidate_idx = valid_idx[values[valid_idx] >= tail_threshold]
        if candidate_idx.size == 0:
            candidate_idx = valid_idx
        n_spikes = max(1, int(np.ceil(level * 0.20 * candidate_idx.size)))
        chosen = rng.choice(
            candidate_idx, size=n_spikes, replace=n_spikes > candidate_idx.size
        )
        spike_mag = (0.75 + 0.75 * rng.random(n_spikes)) * level * iqr
        values[chosen] = values[chosen] + spike_mag
        flat[:, region_idx] = values
    return out


def corruption_gain_scaling_blend(
    data: np.ndarray, level: float, seed: int
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if level <= 0:
        return data.copy()
    rng = np.random.default_rng(seed)
    region_gain = rng.lognormal(mean=0.0, sigma=0.55, size=(1, 1, data.shape[-1]))
    scaled = data * region_gain
    return ((1.0 - level) * data) + (level * scaled)


def _corrcoef_from_flat(data: np.ndarray) -> np.ndarray | None:
    flat = np.asarray(data, dtype=np.float64).reshape(-1, data.shape[-1])
    finite_rows = np.isfinite(flat).all(axis=1)
    flat = flat[finite_rows]
    if flat.shape[0] < 5 or flat.shape[1] < 2:
        return None
    keep_cols = np.nanstd(flat, axis=0) > 1e-9
    flat = flat[:, keep_cols]
    if flat.shape[1] < 2:
        return None
    corr = np.corrcoef(flat, rowvar=False)
    corr = np.asarray(corr, dtype=np.float64)
    np.fill_diagonal(corr, 0.0)
    return corr


def _top_edge_index(abs_weights: np.ndarray, frac: float = 0.2) -> np.ndarray:
    n_edges = abs_weights.size
    if n_edges == 0:
        return np.asarray([], dtype=int)
    k = max(1, int(np.ceil(frac * n_edges)))
    order = np.argsort(abs_weights)
    return order[-k:]


def _local_clustering_from_adj(adj: np.ndarray) -> np.ndarray:
    if adj.size == 0:
        return np.asarray([], dtype=np.float64)
    A = np.asarray(adj, dtype=np.float64)
    A = ((A > 0).astype(np.float64) + (A.T > 0).astype(np.float64)) > 0
    A = A.astype(np.float64)
    np.fill_diagonal(A, 0.0)
    degree = A.sum(axis=1)
    tri = np.diag(A @ A @ A)
    out = np.full(A.shape[0], np.nan, dtype=np.float64)
    valid = degree >= 2
    out[valid] = tri[valid] / np.maximum(degree[valid] * (degree[valid] - 1.0), 1e-9)
    out[(degree < 2) & np.isfinite(degree)] = 0.0
    return out


def _binary_topology_from_corr(C: np.ndarray, frac: float = 0.15) -> np.ndarray:
    C = np.abs(np.asarray(C, dtype=np.float64))
    np.fill_diagonal(C, 0.0)
    iu = np.triu_indices_from(C, k=1)
    n_edges = len(iu[0])
    topk = max(3, int(np.ceil(frac * n_edges)))
    topk = min(topk, n_edges)
    idx = np.argsort(C[iu])[-topk:]
    A = np.zeros_like(C, dtype=np.int64)
    A[iu[0][idx], iu[1][idx]] = 1
    A = A + A.T
    return A


def _binary_clustering(adj: np.ndarray) -> np.ndarray:
    adj = np.asarray(adj, dtype=np.int64)
    n = adj.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        nbr = np.flatnonzero(adj[i])
        k = nbr.size
        if k < 2:
            continue
        sub = adj[np.ix_(nbr, nbr)]
        edges = float(np.sum(sub) / 2.0)
        out[i] = (2.0 * edges) / (k * (k - 1))
    return out


def _final_moment_score(gt_arr: np.ndarray, pred_arr: np.ndarray) -> dict:
    gt, pred = _align_arrays(gt_arr, pred_arr)
    gt_feats = _moment_feature_arrays(gt)
    pred_feats = _moment_feature_arrays(pred)
    components = {}
    for name in ["var", "skew", "kurt"]:
        gt_values = gt_feats[name].ravel()
        pred_values = pred_feats[name].ravel()
        components[name] = (
            _safe_spearman01(gt_values, pred_values),
            _rmse_similarity(gt_values, pred_values),
        )
    score, per_stat = _score_from_components(
        components,
        {"var": WEIGHT_MOMENT_VAR, "skew": WEIGHT_MOMENT_SKEW, "kurt": WEIGHT_MOMENT_KURT},
    )
    return {
        "score": score,
        "var_score": per_stat.get("var", np.nan),
        "skew_score": per_stat.get("skew", np.nan),
        "kurt_score": per_stat.get("kurt", np.nan),
    }


def _perfected_moment_score_legacy(gt_arr: np.ndarray, pred_arr: np.ndarray) -> dict:
    gt, pred = _align_arrays(gt_arr, pred_arr)
    scores = []
    for region in range(gt.shape[-1]):
        g = gt[:, :, region].reshape(-1)
        p = pred[:, :, region].reshape(-1)
        mask = np.isfinite(g) & np.isfinite(p)
        g = g[mask]
        p = p[mask]
        if g.size < 24:
            continue
        var_g = float(np.var(g)) + EPS
        var_p = float(np.var(p)) + EPS
        skew_g = float(stats.skew(g, bias=False))
        skew_p = float(stats.skew(p, bias=False))
        kurt_g = float(stats.kurtosis(g, fisher=True, bias=False))
        kurt_p = float(stats.kurtosis(p, fisher=True, bias=False))
        dist = (
            abs(np.log(var_p / var_g))
            + 0.50 * abs(skew_p - skew_g)
            + 0.25 * abs(kurt_p - kurt_g)
        )
        scores.append(_score_from_distance(dist))
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return {"score": float(np.mean(arr)) if arr.size else np.nan}


def _final_graph_score(gt_arr: np.ndarray, pred_arr: np.ndarray) -> dict:
    gt, pred = _align_arrays(gt_arr, pred_arr)
    corr_gt = _corrcoef_from_flat(gt)
    corr_pred = _corrcoef_from_flat(pred)
    if corr_gt is None or corr_pred is None or corr_gt.shape != corr_pred.shape:
        return {
            "score": np.nan,
            "jaccard": np.nan,
            "weight_score": np.nan,
            "degree_score": np.nan,
            "cluster_score": np.nan,
        }
    tri = np.triu_indices(corr_gt.shape[0], k=1)
    gt_edges = corr_gt[tri]
    pred_edges = corr_pred[tri]
    abs_gt = np.abs(gt_edges)
    abs_pred = np.abs(pred_edges)
    idx_gt = _top_edge_index(abs_gt, frac=0.20)
    idx_pred = _top_edge_index(abs_pred, frac=0.20)
    set_gt = set(idx_gt.tolist())
    set_pred = set(idx_pred.tolist())
    union_idx = np.asarray(sorted(set_gt | set_pred), dtype=int)
    inter_size = len(set_gt & set_pred)
    union_size = len(set_gt | set_pred)
    jaccard = float(inter_size / union_size) if union_size else 1.0
    if union_idx.size:
        weight_score = _safe_spearman01(gt_edges[union_idx], pred_edges[union_idx])
    else:
        weight_score = np.nan
    degree_score = _safe_spearman01(
        np.sum(np.abs(corr_gt), axis=1), np.sum(np.abs(corr_pred), axis=1)
    )
    adj_gt = np.zeros_like(corr_gt, dtype=np.float64)
    adj_pred = np.zeros_like(corr_pred, dtype=np.float64)
    if idx_gt.size:
        adj_gt[tri[0][idx_gt], tri[1][idx_gt]] = 1.0
    if idx_pred.size:
        adj_pred[tri[0][idx_pred], tri[1][idx_pred]] = 1.0
    adj_gt = adj_gt + adj_gt.T
    adj_pred = adj_pred + adj_pred.T
    cluster_score = _rmse_similarity(
        _local_clustering_from_adj(adj_gt), _local_clustering_from_adj(adj_pred)
    )
    score = weighted_mean_available(
        {
            "jaccard": jaccard,
            "weight": weight_score,
            "degree": degree_score,
            "cluster": cluster_score,
        },
        {
            "jaccard": WEIGHT_GRAPH_JACCARD,
            "weight": WEIGHT_GRAPH_WEIGHT,
            "degree": WEIGHT_GRAPH_DEGREE,
            "cluster": WEIGHT_GRAPH_CLUSTER,
        },
    )
    return {
        "score": score,
        "jaccard": jaccard,
        "weight_score": weight_score,
        "degree_score": degree_score,
        "cluster_score": cluster_score,
    }


def _perfected_graph_score_legacy(gt_arr: np.ndarray, pred_arr: np.ndarray) -> dict:
    gt, pred = _align_arrays(gt_arr, pred_arr)
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = _finite_rows(Xg, Xp)
    Cg = _safe_corrcoef(Xg)
    Cp = _safe_corrcoef(Xp)
    if Cg is None or Cp is None:
        return {
            "score": np.nan,
            "jaccard": np.nan,
            "degree_score": np.nan,
            "clustering_score": np.nan,
        }
    Ag = _binary_topology_from_corr(Cg)
    Ap = _binary_topology_from_corr(Cp)
    edges_g = set(zip(*np.where(np.triu(Ag, k=1) > 0)))
    edges_p = set(zip(*np.where(np.triu(Ap, k=1) > 0)))
    union = edges_g | edges_p
    jaccard = float(len(edges_g & edges_p) / max(len(union), 1))
    deg_g = np.sum(np.abs(Cg), axis=0)
    deg_p = np.sum(np.abs(Cp), axis=0)
    degree_score = _score_from_distance(
        float(np.mean(np.abs(deg_p - deg_g)) / (np.mean(np.abs(deg_g)) + EPS))
    )
    clust_g = _binary_clustering(Ag)
    clust_p = _binary_clustering(Ap)
    clustering_score = _finite_float(1.0 - float(np.mean(np.abs(clust_p - clust_g))))
    pieces = [
        value
        for value in [jaccard, degree_score, clustering_score]
        if np.isfinite(value)
    ]
    if not pieces:
        score = np.nan
    else:
        score = _finite_float(
            float(np.exp(np.mean(np.log(np.clip(pieces, 1e-6, 1.0)))))
        )
    return {
        "score": score,
        "jaccard": jaccard,
        "degree_score": degree_score,
        "clustering_score": clustering_score,
    }


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
    logger.info(f"=== {label} replacement metric ===")
    logger.info(description)
    logger.info(f"{score_key}: {score:.6f}" if np.isfinite(score) else f"{score_key}: NaN")
    if corruption_df.empty:
        return
    cols = ["family", "magnitude", score_key]
    logger.info("\nCorruption sweep:")
    logger.info(corruption_df[cols].to_string(index=False))


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
    lagged = _extract_score(crosscorr_lagged_matrix(gt_arr, pred_arr))
    topedge = _extract_score(crosscorr_topedge_profiles(gt_arr, pred_arr))
    return {
        "score": weighted_mean_available(
            {"lagged": lagged, "topedge": topedge},
            {"lagged": WEIGHT_CROSSCORR_LAGGED, "topedge": WEIGHT_CROSSCORR_TOPEDGE},
        )
    }


def _final_trajectory_score(gt_arr: np.ndarray, pred_arr: np.ndarray) -> dict:
    occupancy = _extract_score(trajectory_occupancy_velocity(gt_arr, pred_arr))
    path = _extract_score(trajectory_path_features(gt_arr, pred_arr))
    return {
        "score": weighted_mean_available(
            {"occupancy": occupancy, "path": path},
            {"occupancy": WEIGHT_TRAJECTORY_OCCUPANCY, "path": WEIGHT_TRAJECTORY_PATH},
        )
    }


def _final_manifold_score(gt_arr: np.ndarray, pred_arr: np.ndarray) -> dict:
    topology = _extract_score(manifold_ph_stratified_lifetime(gt_arr, pred_arr))
    local = _extract_score(manifold_ph_knn_profile(gt_arr, pred_arr))
    return {
        "score": weighted_mean_available(
            {"topology": topology, "local": local},
            {"topology": WEIGHT_MANIFOLD_TOPOLOGY, "local": WEIGHT_MANIFOLD_LOCAL},
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
    score = _extract_score(pca_reconstruction_product(gt_arr, pred_arr))
    pca_simple = _wrap_population_metric(
        score_key="PCA_score01",
        mean_key="PCA_mean",
        score=score,
        description=description,
        candidate_name="pca_reconstruction_product",
    )
    pca_corr_df = _build_corruption_df(
        gt_arr,
        pca_reconstruction_product,
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


def compute_moment_replacement(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    *,
    enable_plots: bool = True,
) -> tuple[dict, pd.DataFrame]:
    description = "Variance, skewness, and kurtosis agreement."
    result = _perfected_moment_score_legacy(gt_arr, pred_arr)
    score = _extract_score(result)
    mom_simple = _wrap_population_metric(
        score_key="MOM_score01",
        mean_key="MOM_mean",
        score=score,
        description=description,
        candidate_name="perfected_moment_score",
    )
    mom_corr_df = _build_corruption_df(
        gt_arr,
        _perfected_moment_score_legacy,
        corruption_gain_scaling_blend,
        score_key="MOM_score01",
        family_label="gain_scaling_blend",
    )
    _display_summary(
        label="Moments",
        score_key="MOM_score01",
        score=score,
        description=description,
        corruption_df=mom_corr_df,
    )
    _plot_corruption_df(
        mom_corr_df,
        score_key="MOM_score01",
        title="Moments replacement degradation",
        enable_plots=enable_plots,
    )
    return mom_simple, mom_corr_df


def compute_graph_replacement(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    *,
    enable_plots: bool = True,
) -> tuple[dict, pd.DataFrame]:
    description = "Top-edge topology, weighted degree, and clustering agreement."
    result = _perfected_graph_score_legacy(gt_arr, pred_arr)
    score = _extract_score(result)
    graph_simple = _wrap_population_metric(
        score_key="GRAPH_score01",
        mean_key="GRAPH_mean",
        score=score,
        description=description,
        candidate_name="perfected_graph_score",
        extra_scores={
            "GRAPH_jacc_mean": result.get("jaccard", np.nan),
            "GRAPH_deg_mean": result.get("degree_score", np.nan),
            "GRAPH_cluster_mean": result.get("clustering_score", np.nan),
        },
    )
    graph_corr_df = _build_corruption_df(
        gt_arr,
        _perfected_graph_score_legacy,
        corruption_region_permute_blend,
        score_key="GRAPH_score01",
        family_label="region_permute_blend",
    )
    _display_summary(
        label="Graph",
        score_key="GRAPH_score01",
        score=score,
        description=description,
        corruption_df=graph_corr_df,
    )
    _plot_corruption_df(
        graph_corr_df,
        score_key="GRAPH_score01",
        title="Graph replacement degradation",
        enable_plots=enable_plots,
    )
    return graph_simple, graph_corr_df


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
    score = _extract_score(autocorr_weighted_rmse_power(gt_arr, pred_arr))
    distance = _distance_from_score(score)
    auto_sensitive = _wrap_population_metric(
        score_key="AUTO_score01",
        mean_key="AUTO_mean",
        score=score,
        description=description,
        candidate_name="autocorr_weighted_rmse_power",
        extra_arrays={"d_core_sr": np.asarray([distance], dtype=np.float64)},
    )
    auto_corr_df = _build_corruption_df(
        gt_arr,
        autocorr_weighted_rmse_power,
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
    score = _extract_score(bandpower_band_fraction(gt_arr, pred_arr))
    bandpower_simple = _wrap_population_metric(
        score_key="BP_score01",
        mean_key="BP_mean",
        score=score,
        description=description,
        candidate_name="bandpower_band_fraction",
    )
    bp_corr_df = _build_corruption_df(
        gt_arr,
        bandpower_band_fraction,
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
