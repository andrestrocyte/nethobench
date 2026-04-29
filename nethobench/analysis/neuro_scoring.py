from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy

from nethobench.analysis.score_definitions import (
    build_neuro_families_df,
    build_neuro_metrics_df,
    compute_neuro_composite,
)
from nethobench.analysis.direct_neuro_metrics import (
    compute_moment_score01,
    compute_graph_score01,
    compute_manifold_score01,
    compute_trajectory_score01,
)
from nethobench.analysis.additional_neuro_metrics import (
    compute_additional_structural_metrics,
)

EPS = 1e-12

# ---------------------------------------------------------
# Alignment & Robust Statistics Helpers
# ---------------------------------------------------------


def _iqr_robust(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 10:
        s = np.nanstd(x)
        return float(s if np.isfinite(s) and s > 0 else 1.0)
    q25, q75 = np.nanquantile(x, [0.25, 0.75])
    s = float(q75 - q25)
    if not np.isfinite(s) or s <= 0:
        s = float(np.nanstd(x))
    return float(s if np.isfinite(s) and s > 0 else 1.0)


def _topq_mean(x: np.ndarray, q: float = 0.25) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    k = max(1, int(np.ceil(q * x.size)))
    return float(np.mean(np.sort(x)[-k:]))


# ---------------------------------------------------------
# KL/JSD Distribution Realism
# ---------------------------------------------------------


def _tail_binned_hist(
    gt_vals: np.ndarray,
    pr_vals: np.ndarray,
    bins: int = 60,
    support_q: tuple = (0.001, 0.999),
    eps: float = 1e-12,
):
    gt_vals = np.asarray(gt_vals, dtype=np.float64)
    pr_vals = np.asarray(pr_vals, dtype=np.float64)
    gt_vals = gt_vals[np.isfinite(gt_vals)]
    pr_vals = pr_vals[np.isfinite(pr_vals)]

    if gt_vals.size < 10 or pr_vals.size < 10:
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
        kl_q10 = float(np.quantile(valid, 0.10))
        return 0.5 * (kl_mean + kl_q10)
    return np.nan


# ---------------------------------------------------------
# Mean-Shift Realism
# ---------------------------------------------------------


def compute_mean_score01_top10(gt_arr: np.ndarray, pred_arr: np.ndarray) -> float:
    n_seq, _, n_reg = gt_arr.shape

    mu_gt = np.nanmean(gt_arr, axis=1)
    mu_pr = np.nanmean(pred_arr, axis=1)

    iqr_gt = np.array(
        [_iqr_robust(gt_arr[:, :, r].reshape(-1)) for r in range(n_reg)],
        dtype=np.float64,
    )
    d = np.abs(mu_gt - mu_pr) / (iqr_gt[None, :] + EPS)

    K = max(1, int(np.ceil(0.1 * n_reg)))

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


# ---------------------------------------------------------
# Quantile/Tail Realism
# ---------------------------------------------------------


def compute_quantile_score01_simple(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    q_lo: float = 0.01,
    q_hi: float = 0.99,
    n_q: int = 99,
    tail_lo: float = 0.10,
    tail_hi: float = 0.90,
    min_samples: int = 80,
    max_time: int = 1200,
    top_q_regions: float = 0.25,
    rng_seed: int = 0,
) -> float:
    n_seq, _, n_reg = gt_arr.shape
    rng = np.random.default_rng(rng_seed)

    quantiles = np.linspace(q_lo, q_hi, int(n_q))
    tail_mask = (quantiles <= tail_lo) | (quantiles >= tail_hi)

    iqr_gt = np.array(
        [_iqr_robust(gt_arr[:, :, r].reshape(-1)) for r in range(n_reg)],
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


# ---------------------------------------------------------
# Master Metric Aggregator
# ---------------------------------------------------------
def calculate_neuro_composites(gt_arr: np.ndarray, pred_arr: np.ndarray) -> dict:
    """
    Computes all standard neuro metrics and aggregates them into families and a final composite score.
    Returns a dictionary designed to directly interface with Nethobench's CLI reporting.
    """
    # 1. Distribution Family
    kl_score = compute_kl_score(gt_arr, pred_arr)
    mean_score = compute_mean_score01_top10(gt_arr, pred_arr)
    qnt_score = compute_quantile_score01_simple(gt_arr, pred_arr)

    mom_out = compute_moment_score01(gt_arr, pred_arr)
    mom_score = mom_out["scores"].get("MOM_score01", np.nan)

    # 2. Relational & Geometry Families (Direct Imports)
    graph_out = compute_graph_score01(gt_arr, pred_arr)
    graph_score = graph_out["scores"].get("GRAPH_score01", np.nan)

    mani_out = compute_manifold_score01(gt_arr, pred_arr)
    mani_score = mani_out["scores"].get("MANI_score01", np.nan)

    trj_out = compute_trajectory_score01(gt_arr, pred_arr)
    trj_score = trj_out["scores"].get("TRJDIST_score01", np.nan)

    # 3. Additional Structural Metrics
    add_out = compute_additional_structural_metrics(gt_arr, pred_arr)
    all_extra_scores = add_out.get("scores", {})

    # The notebook explicitly whitelisted only these specific extra metrics
    extra_metric_names = [
        "CrossRegionMI_score01",
        "SubspaceAngle_score01",
        "LaggedCovariance_score01",
        "ImpulseResponse_score01",
        "LatentStateOccupancyK11_score01",
        "LatentStateOccupancyK12_score01",
        "LatentStateTransitionLag1K11_score01",
        "LatentStateTransitionLag2K11_score01",
        "LatentStateTransitionLag3K11_score01",
    ]

    # Construct the master SCORES dictionary
    SCORES = {
        "KL_or_JSD_score01": kl_score,
        "Mean_score01": mean_score,
        "QNT_score01": qnt_score,
        "MOM_score01": mom_score,
        "GRAPH_score01": graph_score,
        "MANI_score01": mani_score,
        "TRJDIST_score01": trj_score,
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
