from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_regression

from nethobench.neuro.metrics.definitions import (
    compute_neuro_composite,
    compute_fidelity_composite,
    build_neuro_metrics_df,
    build_neuro_families_df,
)
from nethobench.neuro.metrics.direct import (
    compute_moment_score,
    compute_graph_score,
    compute_manifold_score,
    compute_trajectory_score,
)
from nethobench.neuro.metrics.additional import (
    compute_additional_structural_metrics,
)
from nethobench.utils.calculation import iqr_robust
from nethobench.utils.helpers import load_and_align
from nethobench.neuro.reporting import generate_full_neuro_report
from nethobench.utils.evaluation_constants import (
    MIN_SAMPLES_KL,
    MIN_SAMPLES_ERROR,
    MIN_SAMPLES_MI,
    MAX_TIME_STEPS_SUBSAMPLE,
    QUANTILE_10P,
    TAIL_TRIM_QUANTILE_LO,
    TAIL_TRIM_QUANTILE_HI,
    QUANTILE_IQR_LO,
    QUANTILE_IQR_HI,
    MAD_TO_STD_CONSTANT,
    MI_N_NEIGHBORS,
)



EPS = 1e-12


def _topq_mean(x: np.ndarray, q: float = 0.25) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    k = max(1, int(np.ceil(q * x.size)))
    return float(np.mean(np.sort(x)[-k:]))


def _tail_binned_hist(
    gt_vals: np.ndarray,
    pr_vals: np.ndarray,
    bins: int = 60,
    support_q: tuple = (TAIL_TRIM_QUANTILE_LO, TAIL_TRIM_QUANTILE_HI),
    eps: float = 1e-12,
):
    gt_vals = np.asarray(gt_vals, dtype=np.float64)
    pr_vals = np.asarray(pr_vals, dtype=np.float64)
    gt_vals = gt_vals[np.isfinite(gt_vals)]
    pr_vals = pr_vals[np.isfinite(pr_vals)]

    if gt_vals.size < MIN_SAMPLES_KL or pr_vals.size < MIN_SAMPLES_KL:
        return None, None

    pool = np.concatenate([gt_vals, pr_vals])
    lo = np.quantile(pool, support_q[0])
    hi = np.quantile(pool, support_q[1])

    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.min(pool))
        hi = float(np.max(pool))
        if lo >= hi:
            hi = lo + 1e-6

    interior_edges = np.linspace(lo, hi, bins + 1)
    gt_counts = [np.sum(gt_vals < lo)]
    pr_counts = [np.sum(pr_vals < lo)]
    h_gt, _ = np.histogram(gt_vals, bins=interior_edges)
    h_pr, _ = np.histogram(pr_vals, bins=interior_edges)

    gt_counts += list(h_gt)
    pr_counts += list(h_pr)
    gt_counts += [np.sum(gt_vals > hi)]
    pr_counts += [np.sum(pr_vals > hi)]

    gt_counts = np.asarray(gt_counts, dtype=np.float64)
    pr_counts = np.asarray(pr_counts, dtype=np.float64)
    p = (gt_counts + eps) / (gt_counts + eps).sum()
    q = (pr_counts + eps) / (pr_counts + eps).sum()
    return p, q


def compute_kl_score(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    bins: int = 60,
    support_q: tuple = (0.001, 0.999),
) -> float:
    n_seq, _, n_reg = gt_arr.shape

    kl_sym_sr = np.full((n_seq, n_reg), np.nan, dtype=np.float64)
    for s in range(n_seq):
        for r in range(n_reg):
            p, q = _tail_binned_hist(
                gt_arr[s, :, r], pred_arr[s, :, r], bins=bins, support_q=support_q
            )
            if p is None:
                continue
            kl_sym_sr[s, r] = 0.5 * (entropy(p, q) + entropy(q, p))

    kl_geo_seq = np.full(n_seq, np.nan, dtype=np.float64)
    for s in range(n_seq):
        row = kl_sym_sr[s, :]
        row = row[np.isfinite(row)]
        if row.size == 0:
            continue
        sim = 1.0 / (1.0 + row)
        sim = np.clip(sim, 1e-12, 1.0)
        kl_geo_seq[s] = float(np.exp(np.mean(np.log(sim))))

    valid = kl_geo_seq[np.isfinite(kl_geo_seq)]
    if valid.size:
        kl_mean = float(np.mean(valid))
        kl_q10 = float(np.quantile(valid, QUANTILE_10P))
        return 0.5 * (kl_mean + kl_q10)
    return np.nan


def compute_mean_score_top10(gt_arr: np.ndarray, pred_arr: np.ndarray) -> float:
    n_seq, _, n_reg = gt_arr.shape

    mu_gt = np.nanmean(gt_arr, axis=1)
    mu_pr = np.nanmean(pred_arr, axis=1)

    iqr_gt = np.array(
        [iqr_robust(gt_arr[:, :, r].reshape(-1)) for r in range(n_reg)],
        dtype=np.float64,
    )
    d = np.abs(mu_gt - mu_pr) / (iqr_gt[None, :] + EPS)

    K = max(1, int(np.ceil(QUANTILE_10P * n_reg)))

    D_top10_seq = np.full(n_seq, np.nan, dtype=np.float64)
    for i in range(n_seq):
        row = d[i, :]
        row = row[np.isfinite(row)]
        if row.size == 0:
            continue
        kk = min(K, row.size)
        topk = np.partition(row, row.size - kk)[-kk:]
        D_top10_seq[i] = float(np.mean(topk))

    D = float(np.nanmean(D_top10_seq)) if np.isfinite(D_top10_seq).any() else np.nan
    return float(1.0 / (1.0 + D)) if np.isfinite(D) else np.nan


def compute_quantile_score_simple(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    q_lo: float = 0.01,
    q_hi: float = 0.99,
    n_q: int = 99,
    tail_lo: float = QUANTILE_10P,
    tail_hi: float = 0.90,
    min_samples: int = MIN_SAMPLES_MI,
    max_time: int = MAX_TIME_STEPS_SUBSAMPLE,
    top_q_regions: float = QUANTILE_IQR_LO,
    rng_seed: int = 0,
) -> float:
    n_seq, _, n_reg = gt_arr.shape
    rng = np.random.default_rng(rng_seed)

    quantiles = np.linspace(q_lo, q_hi, int(n_q))
    tail_mask = (quantiles <= tail_lo) | (quantiles >= tail_hi)

    iqr_gt = np.array(
        [iqr_robust(gt_arr[:, :, r].reshape(-1)) for r in range(n_reg)],
        dtype=np.float64,
    )
    d_tail_sr = np.full((n_seq, n_reg), np.nan, dtype=np.float64)

    for i in range(n_seq):
        for r in range(n_reg):
            x = np.asarray(gt_arr[i, :, r], dtype=np.float64)
            y = np.asarray(pred_arr[i, :, r], dtype=np.float64)

            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]

            if x.size < min_samples or y.size < min_samples:
                continue
            if x.size > max_time:
                x = x[rng.choice(x.size, size=max_time, replace=False)]
            if y.size > max_time:
                y = y[rng.choice(y.size, size=max_time, replace=False)]

            qx = np.quantile(x, quantiles)
            qy = np.quantile(y, quantiles)

            dq = np.abs(qx - qy) / (iqr_gt[r] + EPS)
            d_tail_sr[i, r] = float(np.mean(dq[tail_mask]))

    D_seq = np.full(n_seq, np.nan, dtype=np.float64)
    for i in range(n_seq):
        D_seq[i] = _topq_mean(d_tail_sr[i], q=top_q_regions)

    D = float(np.nanmean(D_seq)) if np.isfinite(D_seq).any() else np.nan
    return float(1.0 / (1.0 + D)) if np.isfinite(D) else np.nan


def calculate_neuro_composites(gt_arr: np.ndarray, pred_arr: np.ndarray) -> dict:
    """
    Computes all standard neuro metrics and aggregates them into families and a final composite score.
    Returns a dictionary designed to directly interface with Nethobench's CLI reporting.
    """
    # 1. Distribution Family
    kl_score = compute_kl_score(gt_arr, pred_arr)
    mean_score = compute_mean_score_top10(gt_arr, pred_arr)
    qnt_score = compute_quantile_score_simple(gt_arr, pred_arr)

    mom_out = compute_moment_score(gt_arr, pred_arr)
    mom_score = mom_out["scores"].get("MOM_score", np.nan)

    # 2. Relational & Geometry Families (Direct Imports)
    graph_out = compute_graph_score(gt_arr, pred_arr)
    graph_score = graph_out["scores"].get("GRAPH_score", np.nan)

    mani_out = compute_manifold_score(gt_arr, pred_arr)
    mani_score = mani_out["scores"].get("MANI_score", np.nan)

    trj_out = compute_trajectory_score(gt_arr, pred_arr)
    trj_score = trj_out["scores"].get("TRJDIST_score", np.nan)

    # 3. Additional Structural Metrics
    add_out = compute_additional_structural_metrics(gt_arr, pred_arr)
    all_extra_scores = add_out.get("scores", {})

    # The notebook explicitly whitelisted only these specific extra metrics
    extra_metric_names = [
        "CrossRegionMI_score",
        "SubspaceAngle_score",
        "LaggedCovariance_score",
        "ImpulseResponse_score",
        "LatentStateOccupancyK11_score",
        "LatentStateOccupancyK12_score",
        "LatentStateTransitionLag1K11_score",
        "LatentStateTransitionLag2K11_score",
        "LatentStateTransitionLag3K11_score",
    ]

    # Construct the master SCORES dictionary
    SCORES = {
        "KL_or_JSD_score": kl_score,
        "Mean_score": mean_score,
        "QNT_score": qnt_score,
        "MOM_score": mom_score,
        "GRAPH_score": graph_score,
        "MANI_score": mani_score,
        "TRJDIST_score": trj_score,
    }

    # Merge ONLY the whitelisted structural scores
    for k in extra_metric_names:
        SCORES[k] = float(all_extra_scores.get(k, np.nan))

    # Build DFs and calculate composites
    metrics_df = build_neuro_metrics_df(SCORES)
    families_df = build_neuro_families_df(SCORES)
    final_composite = compute_neuro_composite(SCORES)

    # 4. Enforce exact legacy dictionary order
    flat_scores = {}

    for _, row in metrics_df.iterrows():
        flat_scores[str(row["metric"])] = (
            float(row["value"]) if np.isfinite(row["value"]) else np.nan
        )

    for _, row in families_df.iterrows():
        flat_scores[f"family_{row['family']}"] = (
            float(row["value"]) if np.isfinite(row["value"]) else np.nan
        )

    flat_scores["composite_score"] = final_composite
    flat_scores["FINAL_COMPOSITE_SCORE"] = final_composite
    flat_scores["FINAL_NEURO_COMPOSITE_SCORE"] = final_composite

    return flat_scores


def _robust_median_mad(x: np.ndarray) -> tuple[float, float]:
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    s = MAD_TO_STD_CONSTANT * mad
    if not np.isfinite(s) or s <= 1e-12:
        s = float(np.nanstd(x))
    if not np.isfinite(s) or s <= 1e-12:
        s = 1.0
    return med, s


def compute_error_score(gt_arr: np.ndarray, pred_arr: np.ndarray) -> float:
    n_seq, _, n_reg = gt_arr.shape
    iqr_gt = np.array([iqr_robust(gt_arr[:, :, r].reshape(-1)) for r in range(n_reg)])
    nrmse_sr = np.full((n_seq, n_reg), np.nan)

    for i in range(n_seq):
        for r in range(n_reg):
            x = gt_arr[i, :, r]
            y = pred_arr[i, :, r]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < MIN_SAMPLES_ERROR:
                continue
            rmse = np.sqrt(np.mean((y[m] - x[m]) ** 2))
            nrmse_sr[i, r] = rmse / (iqr_gt[r] + 1e-12)

    D_seq = np.full(n_seq, np.nan)
    for i in range(n_seq):
        row = nrmse_sr[i]
        row = row[np.isfinite(row)]
        if row.size == 0:
            continue
        k = max(1, int(np.ceil(QUANTILE_IQR_LO * row.size)))
        D_seq[i] = np.mean(np.sort(row)[-k:])

    D = np.nanmean(D_seq)
    return float(1.0 / (1.0 + D)) if np.isfinite(D) else np.nan


def compute_mi_score(gt_arr: np.ndarray, pred_arr: np.ndarray) -> float:
    n_seq, _, n_reg = gt_arr.shape
    rng = np.random.default_rng(0)
    mi_all = np.full((n_seq, n_reg), np.nan)

    for i in range(n_seq):
        for r in range(n_reg):
            x = gt_arr[i, :, r]
            y = pred_arr[i, :, r]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < MIN_SAMPLES_MI:
                continue
            x, y = x[m], y[m]

            if x.size > MAX_TIME_STEPS_SUBSAMPLE:
                idx = rng.choice(x.size, size=MAX_TIME_STEPS_SUBSAMPLE, replace=False)
                x, y = x[idx], y[idx]

            mx, sx = _robust_median_mad(x)
            my, sy = _robust_median_mad(y)

            x = (x - mx) / (sx + 1e-12)
            y = (y - my) / (sy + 1e-12)

            mi_val = mutual_info_regression(
                x.reshape(-1, 1),
                y,
                discrete_features=False,
                n_neighbors=MI_N_NEIGHBORS,
                random_state=0,
            )[0]
            if np.isfinite(mi_val):
                mi_all[i, r] = max(0.0, float(mi_val))


    seq_worst = np.full(n_seq, np.nan)
    for i in range(n_seq):
        row = mi_all[i]
        row = row[np.isfinite(row)]
        if row.size == 0:
            continue
        seq_worst[i] = float(np.quantile(row, 0.10))

    strict = (
        float(np.nanquantile(seq_worst, 0.10))
        if np.isfinite(seq_worst).any()
        else np.nan
    )
    return float(strict / (1.0 + strict)) if np.isfinite(strict) else np.nan


# --- Core Scoring Functions ---


def load_and_run_neuro_full_analysis(
    gt_arr: np.ndarray, pred_arr: np.ndarray
) -> Dict[str, float]:
    """Computes legacy composites directly from in-memory arrays."""

    if gt_arr.ndim != 3 or pred_arr.ndim != 3:
        raise ValueError(
            "Expected gt_arr/pred_arr arrays with shape [n_seq, n_time, n_reg]."
        )
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"Shape mismatch: {gt_arr.shape} vs {pred_arr.shape}")

    flat_scores = calculate_neuro_composites(gt_arr, pred_arr)

    # Calculate fidelity implementations locally
    flat_scores["Error_score"] = compute_error_score(gt_arr, pred_arr)
    flat_scores["MI_score"] = compute_mi_score(gt_arr, pred_arr)

    # Compute master composites
    fid_composite = compute_fidelity_composite(flat_scores)
    final_composite = compute_neuro_composite(flat_scores)

    flat_scores["family_fidelity"] = fid_composite
    flat_scores["FINAL_COMPOSITE_SCORE"] = final_composite
    flat_scores["FINAL_NEURO_COMPOSITE_SCORE"] = final_composite
    flat_scores["composite_score"] = final_composite

    return flat_scores


def compute_composite_scores(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    per_sequence_stats: bool = False,
    neuro_cols: Optional[list[str]] = None,
    ddconfig_path: Optional[Path] = None,  # Parameter kept for signature compatibility
) -> Dict[str, float]:

    if per_sequence_stats:
        raise ValueError(
            "per_sequence_stats is not supported for legacy notebook-based neuro scores."
        )

    gt, pred, _ = load_and_align(
        Path(predictions_csv),
        Path(ground_truth_csv),
        neuro_cols=neuro_cols,
    )

    return load_and_run_neuro_full_analysis(gt, pred)
