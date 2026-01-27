# Auto-generated from notebooks/final_implementation_benchmark.ipynb
# NOTE: This script is executed via runpy with globals injected by the caller.
# Required globals: preds_fname, gt_fname
# Optional globals: ddconfig_path, MODE, SKIP_CROSSCORR, SKIP_CCA, ENABLE_PLOTS


MODE = globals().get('MODE', 'full')
SKIP_CROSSCORR = bool(globals().get('SKIP_CROSSCORR', False))
SKIP_CCA = bool(globals().get('SKIP_CCA', False))
ENABLE_PLOTS = bool(globals().get('ENABLE_PLOTS', False))

# Disable interactive display by default
try:
    import matplotlib
    if not ENABLE_PLOTS:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not ENABLE_PLOTS:
        plt.show = lambda *args, **kwargs: None
except Exception:
    pass

# Fallback display stub
try:
    display
except NameError:
    def display(*args, **kwargs):
        return None


import pandas as pd
import numpy as np
import json

preds_fname = globals().get('preds_fname')
gt_fname = globals().get('gt_fname')
ddconfig_path = globals().get('ddconfig_path')

if preds_fname is None or gt_fname is None:
    raise ValueError('preds_fname and gt_fname must be provided')

# === Predicciones ===
preds_df = pd.read_csv(preds_fname, index_col=0)
pred_region_names = preds_df.columns.tolist()
pred_arr = preds_df.to_numpy()
n_pred_trials = preds_df.index.max() + 1
n_pred_time = pred_arr.shape[0] // n_pred_trials
n_pred_dims = pred_arr.shape[1]
data_predicted = pred_arr.reshape(n_pred_trials, n_pred_time, n_pred_dims)

print("Pred region names:", pred_region_names)
print("Pred shape:", data_predicted.shape)

# === Ground truth ===
gt_df = pd.read_csv(gt_fname)
gt_df = gt_df.sort_values(["sequenceId", "itemPosition"]).reset_index(drop=True)
gt_regions = [c for c in gt_df.columns if c not in {"sequenceId", "itemPosition"}]
seq_lengths = gt_df.groupby("sequenceId").size()
if seq_lengths.nunique() != 1:
    raise ValueError(f"Secuencias con distinta longitud: {seq_lengths.describe()}")
n_seq = seq_lengths.size
n_time = int(seq_lengths.iloc[0])
n_dims = len(gt_regions)
gt_array = gt_df[gt_regions].to_numpy(dtype=np.float32).reshape(n_seq, n_time, n_dims)

if ddconfig_path:
    with open(ddconfig_path, "r") as fh:
        stats = json.load(fh).get("selected_columns_statistics", {})
else:
    stats = {}

# === Ground truth stays in raw scale (skip denormalization) ===
gt_array_denorm = gt_array.astype(np.float64, copy=True)

print("GT region names:", gt_regions)
print("GT shape (raw):", gt_array.shape)
print("GT copy shape:", gt_array_denorm.shape)

aligned_time = min(data_predicted.shape[1], gt_array.shape[1])
if aligned_time < gt_array.shape[1]:
    gt_array = gt_array[:, :aligned_time, :]
    gt_array_denorm = gt_array_denorm[:, :aligned_time, :]
if aligned_time < data_predicted.shape[1]:
    data_predicted = data_predicted[:, :aligned_time, :]

print("Aligned timesteps:", aligned_time)
print("Pred shape (aligned):", data_predicted.shape)
print("GT shape (aligned):", gt_array.shape)
print("GT raw copy shape (aligned):", gt_array_denorm.shape)


# === Distributions + Symmetric KL/JSD + W1n (TAIL-BINNED; PER-SEQUENCE + mismatch-corrected) ===
# Fixes:
#  1) mismatch leak removed (always j != i)
#  2) robust region naming (works if pred_region_names missing / mismatch length)
#  3) score mapping slope is "softness" (divide), consistent with your other cells

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, wasserstein_distance

EPS = 1e-12

# ----------------------------
# 0) Align helper
# ----------------------------
def _resample_pred_to_gt(gt_arr, pred_arr):
    gt_len = gt_arr.shape[1]
    pr_len = pred_arr.shape[1]
    if pr_len != gt_len and pr_len % gt_len == 0:
        factor = pr_len // gt_len
        pred_res = pred_arr.reshape(pred_arr.shape[0], gt_len, factor, pred_arr.shape[2]).mean(axis=2)
        gt_cmp = gt_arr
    elif pr_len != gt_len:
        min_len = min(gt_len, pr_len)
        gt_cmp = gt_arr[:, :min_len, :]
        pred_res = pred_arr[:, :min_len, :]
    else:
        gt_cmp = gt_arr
        pred_res = pred_arr
    return gt_cmp, pred_res


# ----------------------------
# 1) Robust utilities
# ----------------------------
def _robust_scale_iqr(pooled):
    pooled = np.asarray(pooled, dtype=np.float64)
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size < 10:
        s = np.nanstd(pooled)
        return float(s if np.isfinite(s) and s > 0 else 1.0)
    q25, q75 = np.quantile(pooled, [0.25, 0.75])
    s = float(q75 - q25)
    if not np.isfinite(s) or s <= 0:
        s = float(np.nanstd(pooled))
    return float(s if np.isfinite(s) and s > 0 else 1.0)

def _robust_scale_mad(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 5:
        s = np.nanstd(x)
        return float(s if np.isfinite(s) and s > 0 else 1.0)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    s = 1.4826 * mad
    if np.isfinite(s) and s > 1e-12:
        return float(s)
    # fallback IQR
    q25, q75 = np.nanpercentile(x, [25, 75])
    iqr = q75 - q25
    s = iqr / 1.349 if np.isfinite(iqr) else np.nan
    if np.isfinite(s) and s > 1e-12:
        return float(s)
    s = np.nanstd(x)
    return float(s if np.isfinite(s) and s > 1e-12 else 1.0)

def _sigmoid_stable(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))

def _tail_binned_pq(gt_vals, pr_vals, bins=60, support_q=(0.001, 0.999), eps=1e-12):
    """
    Tail-preserving histogram:
      [(-inf, lo), interior bins between lo..hi, (hi, +inf)]
    interior edges defined by quantiles of pooled values.
    """
    gt_vals = np.asarray(gt_vals, dtype=np.float64)
    pr_vals = np.asarray(pr_vals, dtype=np.float64)
    gt_vals = gt_vals[np.isfinite(gt_vals)]
    pr_vals = pr_vals[np.isfinite(pr_vals)]
    if gt_vals.size < 5 or pr_vals.size < 5:
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


# ----------------------------
# 2) Main metric computation (per-seq, per-region)
# ----------------------------
def compute_distribution_metrics_per_seq(
    gt_denorm,
    pred,
    region_names=None,
    bins=60,
    support_q=(0.001, 0.999),
    worst_q=0.99,
    mismatch_M=12,
    normalize_w1=True,
    mapping_floor=0.05,
    mapping_cap=0.95,
    z0=0.75,
    slope=1.25,      # softness (divide), larger => gentler
    rng_seed=0,
    use_resampled=True,
):
    """
    Per-seq scalars:
      jsd_worst[s] (lower better),
      w1_mean[s]   (lower better),
      kl_geo[s]    (higher better; geo mean of 1/(1+KLsym) across regions)

    Plus mismatch-corrected per-seq z and score01 for JSD worst and W1 mean.
    """
    gt_denorm = np.asarray(gt_denorm, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if use_resampled:
        gt_denorm, pred = _resample_pred_to_gt(gt_denorm, pred)

    if gt_denorm.ndim != 3 or pred.ndim != 3:
        raise ValueError("Expected gt/pred as [n_seq, T, R].")
    if gt_denorm.shape != pred.shape:
        raise ValueError(f"Aligned mismatch: {gt_denorm.shape} vs {pred.shape}")

    n_seq, T, n_reg = gt_denorm.shape

    # robust region naming
    if region_names is None:
        region_names = [f"R{i}" for i in range(n_reg)]
    else:
        region_names = list(region_names)
        if len(region_names) != n_reg:
            region_names = [f"R{i}" for i in range(n_reg)]

    # per-seq per-region divergences
    jsd_sr = np.full((n_seq, n_reg), np.nan, dtype=np.float64)
    kl_sr  = np.full((n_seq, n_reg), np.nan, dtype=np.float64)
    w1_sr  = np.full((n_seq, n_reg), np.nan, dtype=np.float64)

    for s in range(n_seq):
        for r in range(n_reg):
            gt_vals = gt_denorm[s, :, r]
            pr_vals = pred[s, :, r]
            gt_vals = gt_vals[np.isfinite(gt_vals)]
            pr_vals = pr_vals[np.isfinite(pr_vals)]
            if gt_vals.size < 10 or pr_vals.size < 10:
                continue

            p, q = _tail_binned_pq(gt_vals, pr_vals, bins=bins, support_q=support_q, eps=EPS)
            if p is None:
                continue

            kl_sym = 0.5 * (entropy(p, q) + entropy(q, p))
            kl_sr[s, r] = float(kl_sym)

            m = 0.5 * (p + q)
            jsd = 0.5 * (entropy(p, m) + entropy(q, m)) / np.log(2.0)
            jsd_sr[s, r] = float(jsd)

            w = wasserstein_distance(gt_vals, pr_vals)
            if normalize_w1:
                sc = _robust_scale_iqr(np.concatenate([gt_vals, pr_vals]))
                w = w / (sc + 1e-12)
            w1_sr[s, r] = float(w)

    # ---- per-seq reductions across regions
    jsd_worst = np.full(n_seq, np.nan, dtype=np.float64)
    w1_mean   = np.full(n_seq, np.nan, dtype=np.float64)
    kl_geo    = np.full(n_seq, np.nan, dtype=np.float64)

    for s in range(n_seq):
        a = jsd_sr[s]
        b = w1_sr[s]
        c = kl_sr[s]

        if np.isfinite(a).any():
            jsd_worst[s] = float(np.nanquantile(a[np.isfinite(a)], worst_q))
        if np.isfinite(b).any():
            w1_mean[s] = float(np.nanmean(b[np.isfinite(b)]))
        if np.isfinite(c).any():
            sreg = 1.0 / (1.0 + c[np.isfinite(c)])  # higher better
            sreg = np.clip(sreg, 1e-12, 1.0)
            kl_geo[s] = float(np.exp(np.mean(np.log(sreg))))

    # ---- mismatch controls: for each seq s, sample mismatch_M indices j != s
    rng = np.random.default_rng(rng_seed)
    mis_jsd = np.full((mismatch_M, n_seq), np.nan, dtype=np.float64)
    mis_w1  = np.full((mismatch_M, n_seq), np.nan, dtype=np.float64)

    if n_seq < 2:
        raise ValueError("Need n_seq >= 2 for mismatch baseline.")

    all_idx = np.arange(n_seq)
    for s in range(n_seq):
        pool = all_idx[all_idx != s]
        M = int(min(mismatch_M, pool.size))
        # sample without replacement; if mismatch_M > pool.size, reuse with replacement for remaining
        js = rng.choice(pool, size=M, replace=False)
        if M < mismatch_M:
            js2 = rng.choice(pool, size=(mismatch_M - M), replace=True)
            js = np.concatenate([js, js2])

        for m in range(mismatch_M):
            sp = int(js[m])
            jsd_list = []
            w1_list = []
            for r in range(n_reg):
                gt_vals = gt_denorm[s, :, r]
                pr_vals = pred[sp, :, r]
                gt_vals = gt_vals[np.isfinite(gt_vals)]
                pr_vals = pr_vals[np.isfinite(pr_vals)]
                if gt_vals.size < 10 or pr_vals.size < 10:
                    continue

                p, q = _tail_binned_pq(gt_vals, pr_vals, bins=bins, support_q=support_q, eps=EPS)
                if p is None:
                    continue

                mm = 0.5 * (p + q)
                jsd = 0.5 * (entropy(p, mm) + entropy(q, mm)) / np.log(2.0)
                jsd_list.append(jsd)

                w = wasserstein_distance(gt_vals, pr_vals)
                if normalize_w1:
                    sc = _robust_scale_iqr(np.concatenate([gt_vals, pr_vals]))
                    w = w / (sc + 1e-12)
                w1_list.append(w)

            if len(jsd_list) >= 4:
                mis_jsd[m, s] = float(np.nanquantile(np.asarray(jsd_list), worst_q))
            if len(w1_list) >= 4:
                mis_w1[m, s] = float(np.nanmean(np.asarray(w1_list)))

    # ---- mismatch-corrected effect sizes (lower divergence is better)
    mis_jsd_med = np.nanmedian(mis_jsd, axis=0)
    mis_w1_med  = np.nanmedian(mis_w1,  axis=0)

    mis_jsd_scale = np.array([_robust_scale_mad(mis_jsd[:, s]) for s in range(n_seq)], dtype=np.float64)
    mis_w1_scale  = np.array([_robust_scale_mad(mis_w1[:, s])  for s in range(n_seq)], dtype=np.float64)

    z_jsd = (mis_jsd_med - jsd_worst) / (mis_jsd_scale + 1e-9)
    z_w1  = (mis_w1_med  - w1_mean)  / (mis_w1_scale  + 1e-9)

    # ---- map to 0–1 (non-saturating; slope is softness)
    floor, cap = float(mapping_floor), float(mapping_cap)
    jsd01 = floor + (cap - floor) * _sigmoid_stable((z_jsd - z0) / (slope + 1e-12))
    w101  = floor + (cap - floor) * _sigmoid_stable((z_w1  - z0) / (slope + 1e-12))

    def _q10(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        return float(np.quantile(x, 0.10)) if x.size else np.nan

    finite_jsd = np.isfinite(jsd_worst)
    finite_w1  = np.isfinite(w1_mean)
    finite_kl  = np.isfinite(kl_geo)

    out = {
        "per_seq": dict(
            jsd_worst=jsd_worst, w1_mean=w1_mean, kl_geo=kl_geo,
            z_jsd=z_jsd, z_w1=z_w1, jsd01=jsd01, w101=w101
        ),
        "per_seq_per_region": dict(jsd=jsd_sr, kl=kl_sr, w1=w1_sr),
        "mismatch": dict(mis_jsd=mis_jsd, mis_w1=mis_w1),
        "summary": {
            "JSD_worstq": dict(
                orig_mean=float(np.nanmean(jsd_worst)),
                orig_median=float(np.nanmedian(jsd_worst)),
                orig_q10_strict=_q10(jsd_worst),
                mis_med_mean=float(np.nanmean(mis_jsd_med)),
                z_mean=float(np.nanmean(z_jsd)),
                score01_mean=float(np.nanmean(jsd01)),
                score01_q10_strict=_q10(jsd01),
            ),
            "W1_mean": dict(
                orig_mean=float(np.nanmean(w1_mean)),
                orig_median=float(np.nanmedian(w1_mean)),
                orig_q10_strict=_q10(w1_mean),
                mis_med_mean=float(np.nanmean(mis_w1_med)),
                z_mean=float(np.nanmean(z_w1)),
                score01_mean=float(np.nanmean(w101)),
                score01_q10_strict=_q10(w101),
            ),
            "KL_geo": dict(
                mean=float(np.nanmean(kl_geo)),
                median=float(np.nanmedian(kl_geo)),
                q10_strict=_q10(kl_geo),
            ),
        },
        "params": dict(
            support_q=support_q, bins=bins, worst_q=worst_q, mismatch_M=mismatch_M,
            normalize_w1=normalize_w1, mapping_floor=mapping_floor, mapping_cap=mapping_cap,
            z0=z0, slope=slope, rng_seed=rng_seed
        ),
        "region_names": region_names,
    }

    print("\n=== Distribution metrics (tail-binned; PER-SEQUENCE; mismatch-corrected) ===")
    print(f"support_q={support_q} | bins={bins} | worst_q={worst_q} | mismatch_M={mismatch_M}")
    print(f"coverage jsd/w1/kl: {finite_jsd.mean()*100:.1f}% / {finite_w1.mean()*100:.1f}% / {finite_kl.mean()*100:.1f}%")

    sJ = out["summary"]["JSD_worstq"]
    sW = out["summary"]["W1_mean"]
    sK = out["summary"]["KL_geo"]

    print(f"\nJSD worst-q (orig): mean={sJ['orig_mean']:.4f} | median={sJ['orig_median']:.4f} | q10(strict)={sJ['orig_q10_strict']:.4f}")
    print(f"            mismatch-med(mean): {sJ['mis_med_mean']:.4f} | z_mean={sJ['z_mean']:.3f} | score01_mean={sJ['score01_mean']:.4f} | score01_q10={sJ['score01_q10_strict']:.4f}")

    print(f"\nW1 mean (orig):     mean={sW['orig_mean']:.4f} | median={sW['orig_median']:.4f} | q10(strict)={sW['orig_q10_strict']:.4f}")
    print(f"            mismatch-med(mean): {sW['mis_med_mean']:.4f} | z_mean={sW['z_mean']:.3f} | score01_mean={sW['score01_mean']:.4f} | score01_q10={sW['score01_q10_strict']:.4f}")

    print(f"\nKL geo (score-like, no mismatch): mean={sK['mean']:.4f} | median={sK['median']:.4f} | q10(strict)={sK['q10_strict']:.4f}")

    scores_for_composite = {
        "JSD_qsupport_worstq": float(sJ["orig_mean"]),           # divergence (lower better) for reporting
        "JSD_worstq_z_score01": float(sJ["score01_q10_strict"]), # strict realism 0–1
        "KL_qsupport_geo": float(sK["q10_strict"]),              # strict geo score 0–1
        "W1n_mean": float(sW["orig_mean"]),                      # distance (lower better) for reporting
        "W1n_mean_z_score01": float(sW["score01_q10_strict"]),   # strict realism 0–1
    }

    print("\n=== Score summary (USE THESE for composite) ===")
    for k in sorted(scores_for_composite.keys()):
        print(f"{k:26s}: {scores_for_composite[k]:.4f}")

    return out, scores_for_composite


# ----------------------------
# 3) Visualization: per-region histograms (viz-trimmed only)
# ----------------------------
def plot_region_histograms_viztrim(
    gt_denorm,
    pred,
    region_names,
    bins=60,
    viz_q=(0.001, 0.999),
    use_resampled=True,
):
    gt_denorm = np.asarray(gt_denorm, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if use_resampled:
        gt_denorm, pred = _resample_pred_to_gt(gt_denorm, pred)

    n_seq, T, n_reg = gt_denorm.shape
    region_names = list(region_names) if region_names is not None else [f"R{i}" for i in range(n_reg)]
    if len(region_names) != n_reg:
        region_names = [f"R{i}" for i in range(n_reg)]

    ncols = 4
    nrows = int(np.ceil(n_reg / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
    axes = axes.ravel()

    for r, name in enumerate(region_names):
        gt_vals = gt_denorm[:, :, r].ravel()
        pr_vals = pred[:, :, r].ravel()
        gt_vals = gt_vals[np.isfinite(gt_vals)]
        pr_vals = pr_vals[np.isfinite(pr_vals)]
        if gt_vals.size == 0 or pr_vals.size == 0:
            axes[r].axis("off")
            continue

        pool = np.concatenate([gt_vals, pr_vals])
        lo = np.quantile(pool, viz_q[0])
        hi = np.quantile(pool, viz_q[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo = float(np.min(pool))
            hi = float(np.max(pool))
            if lo >= hi:
                hi = lo + 1e-6

        edges = np.linspace(lo, hi, bins + 1)
        ax = axes[r]
        ax.hist(gt_vals, bins=edges, alpha=0.5, density=True, label="GT")
        ax.hist(pr_vals, bins=edges, alpha=0.5, density=True, label="Pred")
        ax.set_title(name, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    for ax in axes[n_reg:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("GT vs Pred distributions per region (viz-trimmed; metrics are tail-preserving & per-seq)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()
    return fig



if MODE == 'full':
    # ----------------------------
    # 4) RUN
    # ----------------------------
    _gt_aligned, _pred_aligned = _resample_pred_to_gt(gt_array_denorm, data_predicted)

    # choose region labels robustly
    _region_names = None
    if "pred_region_names" in globals() and globals().get("pred_region_names") is not None:
        _region_names = globals()["pred_region_names"]
    elif "gt_regions" in globals() and globals().get("gt_regions") is not None:
        _region_names = globals()["gt_regions"]

    dist_out, dist_scores = compute_distribution_metrics_per_seq(
        _gt_aligned,
        _pred_aligned,
        region_names=_region_names,
        bins=60,
        support_q=(0.001, 0.999),
        worst_q=0.99,
        mismatch_M=12,
        normalize_w1=True,
        mapping_floor=0.05,
        mapping_cap=0.95,
        z0=0.75,
        slope=1.25,
        rng_seed=0,
        use_resampled=False,
    )

    plot_region_histograms_viztrim(
        _gt_aligned,
        _pred_aligned,
        region_names=dist_out["region_names"],
        bins=60,
        viz_q=(0.001, 0.999),
        use_resampled=False,
    )

# === Mean values of each GT and Predicted distribution (robust + informative + mismatch-corrected) ===
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-12

# ----------------------------
# 0) Minimal alignment helper (fallback only)
# ----------------------------
def _resample_pred_to_gt_fallback(gt_arr, pred_arr):
    gt_len = gt_arr.shape[1]
    pr_len = pred_arr.shape[1]
    if pr_len != gt_len and pr_len % gt_len == 0:
        factor = pr_len // gt_len
        pred_res = pred_arr.reshape(pred_arr.shape[0], gt_len, factor, pred_arr.shape[2]).mean(axis=2)
        return gt_arr, pred_res
    if pr_len != gt_len:
        m = min(gt_len, pr_len)
        return gt_arr[:, :m, :], pred_arr[:, :m, :]
    return gt_arr, pred_arr

def _align_gt_pred(gt_arr, pred_arr, use_resampled=True):
    gt_arr = np.asarray(gt_arr, dtype=np.float64)
    pred_arr = np.asarray(pred_arr, dtype=np.float64)
    if not use_resampled:
        return gt_arr, pred_arr
    # Prefer notebook's helper if it exists (do NOT overwrite it)
    if "_resample_pred_to_gt" in globals() and callable(globals()["_resample_pred_to_gt"]):
        return globals()["_resample_pred_to_gt"](gt_arr, pred_arr)
    return _resample_pred_to_gt_fallback(gt_arr, pred_arr)


# ----------------------------
# 1) Core computation
# ----------------------------
def compute_region_means(
    gt_arr,
    pred_arr,
    region_names=None,
    use_resampled=True,
    top_q=0.10,
    # scoring controls
    floor=0.05,
    cap=0.95,
    z0=0.75,
    slope=1.25,        # softness (divide). larger => gentler
    mismatch_M=25,
    rng_seed=0,
    # normalization choice (IMPORTANT)
    scale_mode="gt",   # "gt" (benchmark-safe default) or "pooled"
    # numerical / reporting
    eps=1e-12,
    tol_offender=1e-10,
    warn_mean_ratio=5.0,   # warn if |mu_pred - mu_gt| / (gt_scale+eps) is huge
):
    """
    Computes:
      (A) GLOBAL per-region means (flattened across all seq/time) + raw/normalized |Δμ|
      (B) PER-SEQUENCE mean shifts + mismatch-corrected realism score01 (0–1)

    Key points:
      - mismatch is done PER-SEQUENCE with j != i always (no fixed-point leakage)
      - score mapping uses division by slope (softness), consistent with other cells
      - normalization uses GT-only IQR by default (prevents variance-inflation loophole)
    """
    gt_arr, pred_arr = _align_gt_pred(gt_arr, pred_arr, use_resampled=use_resampled)

    if gt_arr.ndim != 3 or pred_arr.ndim != 3:
        raise ValueError("Expected gt/pred as [n_seq, T, R].")
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"Aligned mismatch: GT {gt_arr.shape} vs Pred {pred_arr.shape}")

    n_seq, T, n_reg = gt_arr.shape
    if n_seq < 2:
        raise ValueError("Need n_seq >= 2 for mismatch baseline.")

    # robust region naming
    if region_names is None:
        region_names = [f"R{i}" for i in range(n_reg)]
    else:
        region_names = list(region_names)
        if len(region_names) != n_reg:
            region_names = [f"R{i}" for i in range(n_reg)]

    # -------------------------
    # Helpers
    # -------------------------
    def _stable_sigmoid(x):
        x = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _iqr(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size < 10:
            s = np.nanstd(x)
            return float(s if np.isfinite(s) and s > 0 else 1.0)
        q25, q75 = np.nanquantile(x, [0.25, 0.75])
        s = float(q75 - q25)
        if np.isfinite(s) and s > 0:
            return s
        s = float(np.nanstd(x))
        return float(s if np.isfinite(s) and s > 0 else 1.0)

    def _aggregate_stats(vals_1d):
        v = np.asarray(vals_1d, dtype=np.float64)
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            return {"mean": np.nan, "topq": np.nan, "max": np.nan, "median": np.nan}
        mean = float(np.nanmean(finite))
        maxv = float(np.nanmax(finite))
        median = float(np.nanmedian(finite))
        k = max(1, int(np.ceil(top_q * finite.size)))
        topq = float(np.mean(np.sort(finite)[-k:]))
        return {"mean": mean, "topq": topq, "max": maxv, "median": median}

    def _robust_loc_scale(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size < 5:
            med = float(np.nanmedian(x)) if x.size else np.nan
            s = float(np.nanstd(x)) if x.size else np.nan
            if not np.isfinite(s) or s <= 1e-12:
                s = 1.0
            return med, float(s)
        med = float(np.nanmedian(x))
        mad = float(np.nanmedian(np.abs(x - med)))
        s = 1.4826 * mad
        if not np.isfinite(s) or s <= 1e-12:
            q75, q25 = np.nanpercentile(x, [75, 25])
            iqr = float(q75 - q25)
            s = (iqr / 1.349) if np.isfinite(iqr) and iqr > 1e-12 else float(np.nanstd(x))
        if not np.isfinite(s) or s <= 1e-12:
            s = 1.0
        return med, float(s)

    def _score_from_z(z):
        p = _stable_sigmoid((z - z0) / (slope + 1e-12))
        return float(floor + (cap - floor) * p)

    def _print_top_offenders(names, vals, title, tol=1e-10, k=5):
        vals = np.asarray(vals, dtype=np.float64)
        finite = np.isfinite(vals)
        print(f"\n{title}")
        if not np.any(finite):
            print("  (no finite values)")
            return
        vmax = float(np.nanmax(vals[finite]))
        if vmax <= tol:
            print(f"  all values ≤ tol={tol:g} (max={vmax:.3e}) → effectively perfect match at this granularity")
            return
        order = np.argsort(vals[finite])[::-1]
        finite_idx = np.where(finite)[0]
        idx = finite_idx[order]
        shown = 0
        for i in idx:
            if vals[i] > tol:
                print(f"  {names[i]:45s}  {vals[i]:.6g}")
                shown += 1
                if shown >= k:
                    break

    if scale_mode not in ("gt", "pooled"):
        raise ValueError("scale_mode must be 'gt' or 'pooled'")

    # -------------------------
    # A) GLOBAL means (flatten all seq/time)
    # -------------------------
    gt_means_global = np.full(n_reg, np.nan, dtype=np.float64)
    pr_means_global = np.full(n_reg, np.nan, dtype=np.float64)
    delta_mu_global = np.full(n_reg, np.nan, dtype=np.float64)
    delta_mu_norm_global = np.full(n_reg, np.nan, dtype=np.float64)
    scale_region = np.full(n_reg, np.nan, dtype=np.float64)

    signed_diff_global = np.full(n_reg, np.nan, dtype=np.float64)
    signed_diff_norm_global = np.full(n_reg, np.nan, dtype=np.float64)

    print("Per-region GLOBAL mean values (denormalized scale; flattened over all sequences):")
    print("-" * 128)
    header = (
        f"{'Region':45s} | {'GT μ':>22s} | {'Pred μ':>22s} | "
        f"{'|Δμ|':>12s} | {'scale(IQR)':>12s} | {'|Δμ|/scale':>14s}"
    )
    print(header)
    print("-" * 128)

    # also track warning stats
    warn_list = []

    for r, name in enumerate(region_names):
        gt_vals = gt_arr[:, :, r].reshape(-1)
        pr_vals = pred_arr[:, :, r].reshape(-1)
        gt_vals = gt_vals[np.isfinite(gt_vals)]
        pr_vals = pr_vals[np.isfinite(pr_vals)]
        if gt_vals.size < 10 or pr_vals.size < 10:
            continue

        mu_g = float(np.mean(gt_vals))
        mu_p = float(np.mean(pr_vals))
        dmu = float(abs(mu_g - mu_p))

        sc = _iqr(gt_vals) if scale_mode == "gt" else _iqr(np.concatenate([gt_vals, pr_vals]))

        gt_means_global[r] = mu_g
        pr_means_global[r] = mu_p
        delta_mu_global[r] = dmu
        scale_region[r] = sc
        delta_mu_norm_global[r] = float(dmu / (sc + eps))

        signed = mu_g - mu_p
        signed_diff_global[r] = signed
        signed_diff_norm_global[r] = float(signed / (sc + eps))

        # warn if mean offset is extreme in GT units
        ratio = float(dmu / (sc + eps))
        if np.isfinite(ratio) and ratio > warn_mean_ratio:
            warn_list.append((name, ratio, mu_g, mu_p, sc))

        print(f"{name:45s} | {mu_g:22.20f} | {mu_p:22.20f} | {dmu:12.6g} | {sc:12.6g} | {delta_mu_norm_global[r]:14.6g}")

    print("-" * 128)
    print(f"Global GT mean   : {float(np.nanmean(gt_means_global)):.20f}")
    print(f"Global Pred mean : {float(np.nanmean(pr_means_global)):.20f}")
    print(f"scale_mode       : {scale_mode!r} (IQR reference)")

    if len(warn_list):
        print("\n[WARN] Extremely large mean offsets in units of chosen IQR scale (possible denorm mismatch):")
        for (nm, rr, mg, mp, sc) in sorted(warn_list, key=lambda x: -x[1])[:8]:
            print(f"  {nm:45s}  |Δμ|/IQR={rr:.2f}  (GTμ={mg:.3g}, Predμ={mp:.3g}, IQR={sc:.3g})")

    agg_global_raw = _aggregate_stats(delta_mu_global)
    agg_global_norm = _aggregate_stats(delta_mu_norm_global)

    _print_top_offenders(region_names, delta_mu_global, "Top offending regions (GLOBAL raw |Δμ|):", tol=tol_offender)
    _print_top_offenders(region_names, delta_mu_norm_global, f"Top offending regions (GLOBAL normalized |Δμ|/scale) [scale_mode={scale_mode}]:", tol=tol_offender)

    print("\nGLOBAL aggregate divergences (raw):", agg_global_raw)
    print("GLOBAL aggregate divergences (normalized):", agg_global_norm)

    # -------------------------
    # B) PER-SEQUENCE mean shifts (mismatch-sensitive)
    # -------------------------
    mu_gt_seq = np.nanmean(gt_arr, axis=1)    # (n_seq, n_reg)
    mu_pr_seq = np.nanmean(pred_arr, axis=1) # (n_seq, n_reg)

    dmu_seq = np.abs(mu_gt_seq - mu_pr_seq)                          # (n_seq, n_reg)
    dmu_seq_norm = dmu_seq / (scale_region[None, :] + eps)           # (n_seq, n_reg)

    region_seq_med = np.nanmedian(dmu_seq, axis=0)        # per region
    region_seq_med_norm = np.nanmedian(dmu_seq_norm, axis=0)

    agg_seq_raw = _aggregate_stats(region_seq_med)
    agg_seq_norm = _aggregate_stats(region_seq_med_norm)

    _print_top_offenders(region_names, region_seq_med, "Top offending regions (PER-SEQ median raw |Δμ_seq|):", tol=tol_offender)
    _print_top_offenders(region_names, region_seq_med_norm, f"Top offending regions (PER-SEQ median normalized |Δμ_seq|/scale) [scale_mode={scale_mode}]:", tol=tol_offender)

    print("\nPER-SEQ aggregate divergences over regions (raw):", agg_seq_raw)
    print("PER-SEQ aggregate divergences over regions (normalized):", agg_seq_norm)

    # -------------------------
    # Mismatch controls (NO fixed points): for each seq i, sample j != i
    # -------------------------
    rng = np.random.default_rng(rng_seed)
    all_idx = np.arange(n_seq)

    null = {k: [] for k in ["mean", "median", "topq", "max"]}

    for _ in range(int(mismatch_M)):
        mu_pr_mis = np.empty_like(mu_pr_seq)
        for i in range(n_seq):
            pool = all_idx[all_idx != i]
            j = int(rng.choice(pool))
            mu_pr_mis[i] = mu_pr_seq[j]

        dmu_mis = np.abs(mu_gt_seq - mu_pr_mis)
        dmu_mis_norm = dmu_mis / (scale_region[None, :] + eps)

        region_mis_med_norm = np.nanmedian(dmu_mis_norm, axis=0)
        agg_mis = _aggregate_stats(region_mis_med_norm)
        for kk in null.keys():
            null[kk].append(agg_mis[kk])

    z_effect = {}
    score01 = {}
    for kk in null.keys():
        null_arr = np.asarray(null[kk], dtype=np.float64)
        null_med, null_s = _robust_loc_scale(null_arr)
        orig = float(agg_seq_norm[kk])
        z = (null_med - orig) / (null_s + eps)  # lower is better -> positive if better than mismatch
        z_effect[kk] = float(z)
        score01[f"MeanShiftZ_{kk}"] = _score_from_z(z)

    print("\nMismatch-corrected mean-shift scores (0–1; PER-SEQ normalized):")
    for kk in ["mean", "median", "topq", "max"]:
        print(f"  MeanShiftZ_{kk:6s} : {score01[f'MeanShiftZ_{kk}']:.6f}   (z={z_effect[kk]:+.3f})")

    mean_shift_final = score01["MeanShiftZ_mean"]
    print(f"\nRecommended mean-shift realism scalar: MeanShiftZ_mean = {mean_shift_final:.6f}")

    results = {
        "region_names": region_names,
        "per_region": {
            "mu_gt_global": gt_means_global,
            "mu_pr_global": pr_means_global,
            "delta_mu_global": delta_mu_global,
            "delta_mu_norm_global": delta_mu_norm_global,
            "scale_region": scale_region,
            "signed_diff_global": signed_diff_global,
            "signed_diff_norm_global": signed_diff_norm_global,
            "mu_gt_seq": mu_gt_seq,
            "mu_pr_seq": mu_pr_seq,
            "delta_mu_seq": dmu_seq,
            "delta_mu_seq_norm": dmu_seq_norm,
            "region_seq_med": region_seq_med,
            "region_seq_med_norm": region_seq_med_norm,
        },
        "aggregates": {
            "global_raw": agg_global_raw,
            "global_norm": agg_global_norm,
            "seq_raw": agg_seq_raw,
            "seq_norm": agg_seq_norm,
        },
        "scores01": score01,
        "z_effect": z_effect,
        "final_scalar": mean_shift_final,
        "meta": {
            "use_resampled": use_resampled,
            "top_q": top_q,
            "mismatch_M": mismatch_M,
            "floor": floor,
            "cap": cap,
            "z0": z0,
            "slope": slope,
            "scale_mode": scale_mode,
        }
    }

    return gt_means_global, pr_means_global, results


# ----------------------------
# 2) RUN (robust region name selection)
# ----------------------------
_required = ["gt_array_denorm", "data_predicted"]
_missing = [v for v in _required if v not in globals()]
if _missing:
    raise RuntimeError(f"Missing variables: {_missing}")

_region_names = None
if "pred_region_names" in globals() and globals().get("pred_region_names") is not None:
    _region_names = globals()["pred_region_names"]
elif "gt_regions" in globals() and globals().get("gt_regions") is not None:
    _region_names = globals()["gt_regions"]


if MODE in ('full', 'instant'):
    gt_means, pred_means, mean_results = compute_region_means(
        gt_array_denorm,
        data_predicted,
        region_names=_region_names,
        use_resampled=True,
        scale_mode="gt",   # <-- benchmark-safe default
    )

    names = mean_results["region_names"]

    # ----------------------------
    # 3) Plots
    # ----------------------------
    mean_diff = mean_results["per_region"]["signed_diff_global"]
    plt.figure(figsize=(12, 5))
    plt.bar(names, mean_diff, alpha=0.85, edgecolor='black')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xticks(rotation=90)
    plt.ylabel('GT μ − Pred μ', fontsize=12)
    plt.title('GLOBAL Mean Difference per Region (flattened over sequences)', fontsize=14)
    plt.tight_layout()
    plt.show()

    mean_diff_norm = mean_results["per_region"]["signed_diff_norm_global"]
    plt.figure(figsize=(12, 5))
    plt.bar(names, mean_diff_norm, alpha=0.85, edgecolor='black')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xticks(rotation=90)
    plt.ylabel('Normalized (GT μ − Pred μ) / IQR', fontsize=12)
    plt.title(f'GLOBAL Normalized Mean Difference per Region (scale_mode={mean_results["meta"]["scale_mode"]})', fontsize=14)
    plt.tight_layout()
    plt.show()

    per_seq_med = mean_results["per_region"]["region_seq_med"]
    per_seq_med_norm = mean_results["per_region"]["region_seq_med_norm"]

    plt.figure(figsize=(12, 5))
    plt.bar(names, per_seq_med, alpha=0.85, edgecolor='black')
    plt.xticks(rotation=90)
    plt.ylabel('median over seq of |μ_seq(GT) − μ_seq(Pred)|', fontsize=12)
    plt.title('PER-SEQUENCE Mean Shift per Region (median over sequences)', fontsize=14)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.bar(names, per_seq_med_norm, alpha=0.85, edgecolor='black')
    plt.xticks(rotation=90)
    plt.ylabel('median over seq of |Δμ_seq| / IQR', fontsize=12)
    plt.title(f'PER-SEQUENCE Normalized Mean Shift per Region (scale_mode={mean_results["meta"]["scale_mode"]})', fontsize=14)
    plt.tight_layout()
    plt.show()

# --- Mutual information realism (GT vs Pred) — per-sequence, per-region; robust; mismatch-corrected; benchmark-ready ---
# Key fixes vs previous cell:
#   ✅ NO "NMI" (avoids mixing incompatible estimators: kNN-MI vs histogram entropy)
#   ✅ Mismatch baseline (j != i always) -> robust effect z -> bounded 0–1 score
#   ✅ Runtime guards: time subsampling + cap regions used in mismatch
#   ✅ Robust scaling (median/MAD) per (seq,region) before MI (reduces scale sensitivity)
#   ✅ Safer reductions (no nanmin warnings; handles all-NaN rows cleanly)
#
# Requires:
#   - gt_array_denorm: (n_seq, T, n_reg)
#   - data_predicted:  (n_seq, T, n_reg) or resample-able
#   - _resample_pred_to_gt(gt, pred) if you want use_resampled=True (will use yours if present)
# Optional:
#   - pred_region_names or gt_regions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

EPS = 1e-12

# ----------------------------
# 0) Alignment helper (fallback only)
# ----------------------------
def _resample_pred_to_gt_fallback(gt_arr, pred_arr):
    gt_len = gt_arr.shape[1]
    pr_len = pred_arr.shape[1]
    if pr_len != gt_len and pr_len % gt_len == 0:
        factor = pr_len // gt_len
        pred_res = pred_arr.reshape(pred_arr.shape[0], gt_len, factor, pred_arr.shape[2]).mean(axis=2)
        return gt_arr, pred_res
    if pr_len != gt_len:
        m = min(gt_len, pr_len)
        return gt_arr[:, :m, :], pred_arr[:, :m, :]
    return gt_arr, pred_arr

def _align_gt_pred(gt_arr, pred_arr, use_resampled=True):
    gt_arr = np.asarray(gt_arr, dtype=np.float64)
    pred_arr = np.asarray(pred_arr, dtype=np.float64)
    if not use_resampled:
        return gt_arr, pred_arr
    if "_resample_pred_to_gt" in globals() and callable(globals()["_resample_pred_to_gt"]):
        return globals()["_resample_pred_to_gt"](gt_arr, pred_arr)
    return _resample_pred_to_gt_fallback(gt_arr, pred_arr)

# ----------------------------
# 1) Robust utils
# ----------------------------
def _robust_median_mad(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, 1.0
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    s = 1.4826 * mad
    if not np.isfinite(s) or s <= 1e-12:
        s = float(np.nanstd(x))
    if not np.isfinite(s) or s <= 1e-12:
        s = 1.0
    return med, float(s)

def _robust_scale_mad(x):
    _, s = _robust_median_mad(x)
    return float(s)

def _sigmoid_stable(z):
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))

def _safe_nanquantile(x, q):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else np.nan

def _row_reduce(vals_row, worst_q=0.10):
    """Return mean, worstq (low-quantile), min for one row with NaNs handled."""
    v = np.asarray(vals_row, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(v))
    worst = float(np.quantile(v, worst_q))
    mn = float(np.min(v))
    return mean, worst, mn

# ----------------------------
# 2) Core computation
# ----------------------------
def compute_mi_realism_benchmark(
    gt_arr,
    pred_arr,
    region_names=None,
    use_resampled=True,
    # MI estimator
    n_neighbors=5,
    # data / stability controls
    min_samples=80,
    max_time=1200,          # cap timepoints per (seq,region) for MI
    standardize=True,        # robust z-score each x,y before MI
    # mismatch baseline controls
    mismatch_M=10,
    mismatch_regions_max=10, # cap regions used inside mismatch loop (bounds runtime)
    rng_seed=0,
    # score mapping (bounded 0–1)
    floor=0.05,
    cap=0.95,
    z0=0.75,
    slope=1.25,              # softness (divide), larger => gentler
    # reductions
    worst_q_regions=0.10,     # per-seq worst-q across regions (low MI is worse)
):
    gt_arr, pred_arr = _align_gt_pred(gt_arr, pred_arr, use_resampled=use_resampled)
    if gt_arr.ndim != 3 or pred_arr.ndim != 3:
        raise ValueError("Expected gt/pred as [n_seq, T, R].")
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"Aligned mismatch: GT {gt_arr.shape} vs Pred {pred_arr.shape}")

    n_seq, T, n_reg = gt_arr.shape
    if n_seq < 2:
        raise ValueError("Need n_seq >= 2 for mismatch baseline.")

    # robust region naming
    if region_names is None:
        region_names = [f"R{i}" for i in range(n_reg)]
    else:
        region_names = list(region_names)
        if len(region_names) != n_reg:
            region_names = [f"R{i}" for i in range(n_reg)]

    rng = np.random.default_rng(rng_seed)

    # -------------------------
    # A) Observed MI per (seq,region)
    # -------------------------
    mi_sr = np.full((n_seq, n_reg), np.nan, dtype=np.float64)
    valid_sr = np.zeros((n_seq, n_reg), dtype=bool)

    for s in range(n_seq):
        for r in range(n_reg):
            x = gt_arr[s, :, r]
            y = pred_arr[s, :, r]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < min_samples:
                continue
            x = x[m]; y = y[m]

            # time cap
            if x.size > max_time:
                idx = rng.choice(x.size, size=max_time, replace=False)
                x = x[idx]; y = y[idx]

            # robust standardization (important for kNN MI)
            if standardize:
                mx, sx = _robust_median_mad(x)
                my, sy = _robust_median_mad(y)
                x = (x - mx) / (sx + EPS)
                y = (y - my) / (sy + EPS)

            try:
                mi = mutual_info_regression(
                    x.reshape(-1, 1),
                    y,
                    discrete_features=False,
                    n_neighbors=int(n_neighbors),
                    random_state=int(rng_seed),
                )[0]
                # MI should be >= 0, but tiny negatives can happen numerically
                if np.isfinite(mi):
                    mi_sr[s, r] = float(max(0.0, mi))
                    valid_sr[s, r] = True
            except Exception:
                pass

    # Per-seq observed reductions
    mi_seq_mean  = np.full(n_seq, np.nan, dtype=np.float64)
    mi_seq_worst = np.full(n_seq, np.nan, dtype=np.float64)
    mi_seq_min   = np.full(n_seq, np.nan, dtype=np.float64)
    n_valid_reg  = np.zeros(n_seq, dtype=int)

    for s in range(n_seq):
        v = mi_sr[s]
        v = v[np.isfinite(v)]
        n_valid_reg[s] = int(v.size)
        if v.size == 0:
            continue
        mi_seq_mean[s]  = float(np.mean(v))
        mi_seq_worst[s] = float(np.quantile(v, worst_q_regions))
        mi_seq_min[s]   = float(np.min(v))

    # -------------------------
    # B) Mismatch baseline (PER-SEQUENCE, j != s), bounded runtime
    # -------------------------
    mis_mean  = np.full((mismatch_M, n_seq), np.nan, dtype=np.float64)
    mis_worst = np.full((mismatch_M, n_seq), np.nan, dtype=np.float64)
    mis_min   = np.full((mismatch_M, n_seq), np.nan, dtype=np.float64)

    all_idx = np.arange(n_seq)
    for s in range(n_seq):
        # choose mismatch seq indices j != s
        pool = all_idx[all_idx != s]
        M = int(min(mismatch_M, pool.size))
        js = rng.choice(pool, size=M, replace=False)
        if M < mismatch_M:
            js2 = rng.choice(pool, size=(mismatch_M - M), replace=True)
            js = np.concatenate([js, js2])

        # choose regions for mismatch (cap cost)
        valid_regions = np.where(valid_sr[s])[0]
        if valid_regions.size == 0:
            continue
        if valid_regions.size > mismatch_regions_max:
            rr = rng.choice(valid_regions, size=mismatch_regions_max, replace=False)
        else:
            rr = valid_regions

        for m_i in range(mismatch_M):
            j = int(js[m_i])
            mi_list = []
            for r in rr:
                x = gt_arr[s, :, r]
                y = pred_arr[j, :, r]
                mm = np.isfinite(x) & np.isfinite(y)
                if mm.sum() < min_samples:
                    continue
                x = x[mm]; y = y[mm]

                if x.size > max_time:
                    idx = rng.choice(x.size, size=max_time, replace=False)
                    x = x[idx]; y = y[idx]

                if standardize:
                    mx, sx = _robust_median_mad(x)
                    my, sy = _robust_median_mad(y)
                    x = (x - mx) / (sx + EPS)
                    y = (y - my) / (sy + EPS)

                try:
                    mi = mutual_info_regression(
                        x.reshape(-1, 1),
                        y,
                        discrete_features=False,
                        n_neighbors=int(n_neighbors),
                        random_state=int(rng_seed),
                    )[0]
                    if np.isfinite(mi):
                        mi_list.append(float(max(0.0, mi)))
                except Exception:
                    pass

            if len(mi_list) == 0:
                continue
            mi_list = np.asarray(mi_list, dtype=np.float64)
            mis_mean[m_i, s]  = float(np.mean(mi_list))
            mis_worst[m_i, s] = float(np.quantile(mi_list, worst_q_regions))
            mis_min[m_i, s]   = float(np.min(mi_list))

    # -------------------------
    # C) Effect sizes + bounded 0–1 scores
    #     Higher MI is better => z = (obs - mismatch_med)/scale
    # -------------------------
    def _z_and_score(obs, mis_mat):
        mis_med = np.nanmedian(mis_mat, axis=0)
        mis_s   = np.array([_robust_scale_mad(mis_mat[:, s]) for s in range(n_seq)], dtype=np.float64)
        z = (obs - mis_med) / (mis_s + EPS)
        score01 = floor + (cap - floor) * _sigmoid_stable((z - z0) / (slope + EPS))
        return z, score01, mis_med

    z_mean,  score_mean,  mis_med_mean  = _z_and_score(mi_seq_mean,  mis_mean)
    z_worst, score_worst, mis_med_worst = _z_and_score(mi_seq_worst, mis_worst)
    z_min,   score_min,   mis_med_min   = _z_and_score(mi_seq_min,   mis_min)

    # -------------------------
    # D) Summaries (benchmark scalars)
    # -------------------------
    coverage = float(np.mean(np.isfinite(mi_seq_mean)))

    scores_for_composite = {
        # STRICT (recommended): q10 over sequences of mismatch-corrected 0–1 score
        "MI_mean_z_score01_q10":  _safe_nanquantile(score_mean,  0.10),
        "MI_worst_z_score01_q10": _safe_nanquantile(score_worst, 0.10),
        "MI_min_z_score01_q10":   _safe_nanquantile(score_min,   0.10),
        # Also provide means (less strict)
        "MI_mean_z_score01_mean":  float(np.nanmean(score_mean)),
        "MI_worst_z_score01_mean": float(np.nanmean(score_worst)),
        "MI_min_z_score01_mean":   float(np.nanmean(score_min)),
        "MI_coverage_seq":         coverage,
    }

    print("\n=== MI realism (kNN MI; PER-SEQUENCE; mismatch-corrected; bounded 0–1) ===")
    print(f"n_seq={n_seq} | n_reg={n_reg} | coverage(seq finite mean)={coverage*100:.1f}%")
    print(f"min_samples={min_samples} | max_time={max_time} | n_neighbors={n_neighbors} | standardize={standardize}")
    print(f"mismatch_M={mismatch_M} | mismatch_regions_max={mismatch_regions_max} | worst_q_regions={worst_q_regions}")
    print(f"mapping: floor={floor} cap={cap} | z0={z0} | slope(softness)={slope}")

    print("\n--- Benchmark scalars (USE THESE for composite) ---")
    for k in sorted(scores_for_composite.keys()):
        v = scores_for_composite[k]
        print(f"{k:26s}: {v:.4f}" if np.isfinite(v) else f"{k:26s}: nan")

    # Offenders
    per_region_df = pd.DataFrame({
        "region": region_names,
        "mi_mean": np.nanmean(mi_sr, axis=0),
        "mi_q10":  np.nanquantile(mi_sr, 0.10, axis=0),
        "valid_frac": np.mean(np.isfinite(mi_sr), axis=0),
    }).sort_values("mi_mean")

    per_seq_df = pd.DataFrame({
        "seq": np.arange(n_seq),
        "n_valid_regions": n_valid_reg,
        "mi_mean": mi_seq_mean,
        "mi_worstq": mi_seq_worst,
        "mi_min": mi_seq_min,
        "z_mean": z_mean,
        "score01_mean": score_mean,
        "z_worst": z_worst,
        "score01_worst": score_worst,
        "z_min": z_min,
        "score01_min": score_min,
    }).sort_values("score01_mean")

    print("\nTop offending regions (low MI_mean):")
    for _, row in per_region_df.head(5).iterrows():
        print(f"  {row['region']:30s}  mi_mean={row['mi_mean']:.4f}  mi_q10={row['mi_q10']:.4f}  valid={row['valid_frac']:.2f}")

    print("\nWorst sequences by score01_mean:")
    for _, row in per_seq_df.head(5).iterrows():
        print(f"  seq={int(row['seq'])}: score01_mean={row['score01_mean']:.4f}  mi_mean={row['mi_mean']:.4f}  nreg={int(row['n_valid_regions'])}")

    return {
        "mi_sr": mi_sr,
        "valid_sr": valid_sr,
        "per_region": per_region_df,
        "per_seq": per_seq_df,
        "mismatch": {
            "mis_mean": mis_mean,
            "mis_worst": mis_worst,
            "mis_min": mis_min,
            "mis_med_mean": mis_med_mean,
            "mis_med_worst": mis_med_worst,
            "mis_med_min": mis_med_min,
        },
        "scores01": {
            "score_mean": score_mean,
            "score_worst": score_worst,
            "score_min": score_min,
        },
        "scores_for_composite": scores_for_composite,
        "params": dict(
            n_neighbors=n_neighbors, min_samples=min_samples, max_time=max_time, standardize=standardize,
            mismatch_M=mismatch_M, mismatch_regions_max=mismatch_regions_max, worst_q_regions=worst_q_regions,
            floor=floor, cap=cap, z0=z0, slope=slope, rng_seed=rng_seed, use_resampled=use_resampled
        ),
        "region_names": region_names,
    }

# ----------------------------
# 3) RUN (robust region name selection)
# ----------------------------
_required = ["gt_array_denorm", "data_predicted"]
_missing = [v for v in _required if v not in globals()]
if _missing:
    raise RuntimeError(f"Missing variables: {_missing}")

_region_names = None
if "pred_region_names" in globals() and globals().get("pred_region_names") is not None:
    _region_names = globals()["pred_region_names"]
elif "gt_regions" in globals() and globals().get("gt_regions") is not None:
    _region_names = globals()["gt_regions"]


if MODE == 'full':
    results_mi = compute_mi_realism_benchmark(
        gt_array_denorm,
        data_predicted,
        region_names=_region_names,
        use_resampled=True,
        # sensible defaults; tweak if too slow
        n_neighbors=5,
        min_samples=80,
        max_time=1200,
        mismatch_M=10,
        mismatch_regions_max=10,
        worst_q_regions=0.10,
        standardize=True,
        rng_seed=0,
    )

    # ----------------------------
    # 4) Plots (lightweight)
    # ----------------------------
    names = results_mi["region_names"]
    per_region = results_mi["per_region"].set_index("region").reindex(names)

    plt.figure(figsize=(12, 4))
    plt.bar(names, per_region["mi_mean"].values, alpha=0.85, edgecolor="k")
    plt.xticks(rotation=90)
    plt.ylabel("Mean MI (per region)")
    plt.title("GT vs Pred dependence: mean MI per region (kNN MI)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 3))
    plt.hist(results_mi["scores01"]["score_mean"][np.isfinite(results_mi["scores01"]["score_mean"])], bins=20, alpha=0.8, edgecolor="k")
    plt.xlabel("score01_mean (mismatch-corrected, bounded)")
    plt.ylabel("Count")
    plt.title("Per-sequence MI realism scores (mean over regions)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    print("\nInterpretation:")
    print("- MI is a dependence metric (does Pred carry information about GT?), NOT a distribution-matching metric.")
    print("- The benchmark-ready scalar is mismatch-corrected (better-than-mismatch) and bounded in [floor,cap].")
    print("- Use MI alongside KL/JSD/Wasserstein; it complements realism metrics by probing coupling rather than marginals.")

# === Error realism (GT vs Pred) — nRMSE/nMAE + mismatch-corrected 0–1 (benchmark v1; FIXED) ===

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS = 1e-12

# ----------------------------
# 0) Alignment helper (fallback only)
# ----------------------------
def _resample_pred_to_gt_fallback(gt_arr, pred_arr):
    gt_len = gt_arr.shape[1]
    pr_len = pred_arr.shape[1]
    if pr_len != gt_len and pr_len % gt_len == 0:
        factor = pr_len // gt_len
        pred_res = pred_arr.reshape(pred_arr.shape[0], gt_len, factor, pred_arr.shape[2]).mean(axis=2)
        return gt_arr, pred_res
    if pr_len != gt_len:
        m = min(gt_len, pr_len)
        return gt_arr[:, :m, :], pred_arr[:, :m, :]
    return gt_arr, pred_arr

def _align_gt_pred(gt_arr, pred_arr, use_resampled=True):
    gt_arr = np.asarray(gt_arr, dtype=np.float64)
    pred_arr = np.asarray(pred_arr, dtype=np.float64)
    if not use_resampled:
        return gt_arr, pred_arr
    if "_resample_pred_to_gt" in globals() and callable(globals()["_resample_pred_to_gt"]):
        return globals()["_resample_pred_to_gt"](gt_arr, pred_arr)
    return _resample_pred_to_gt_fallback(gt_arr, pred_arr)

# ----------------------------
# 1) Robust utilities
# ----------------------------
def _robust_scale_iqr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 10:
        s = float(np.nanstd(x))
        return float(s if np.isfinite(s) and s > 1e-12 else 1.0)
    q25, q75 = np.nanquantile(x, [0.25, 0.75])
    s = float(q75 - q25)
    if not np.isfinite(s) or s <= 1e-12:
        s = float(np.nanstd(x))
    return float(s if np.isfinite(s) and s > 1e-12 else 1.0)

def _robust_mad_scale(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 5:
        s = float(np.nanstd(x))
        return float(s if np.isfinite(s) and s > 1e-12 else 1.0)
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    s = float(1.4826 * mad)
    if not np.isfinite(s) or s <= 1e-12:
        s = float(_robust_scale_iqr(x) / 1.349)  # rough
    return float(s if np.isfinite(s) and s > 1e-12 else 1.0)

def _sigmoid_stable(z):
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))

def _safe_quantile(x, q):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else np.nan

def _worst_topq_mean(vals, top_q=0.25):
    """Mean of the largest top_q fraction (i.e., WORST regions)."""
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    k = max(1, int(np.ceil(top_q * v.size)))
    return float(np.mean(np.sort(v)[-k:]))

def _compute_rmse_mae_1d(gt_1d, pr_1d, min_samples=50):
    """Compute RMSE/MAE on the SAME validity mask: finite(gt) & finite(pred)."""
    gt_1d = np.asarray(gt_1d, dtype=np.float64)
    pr_1d = np.asarray(pr_1d, dtype=np.float64)
    m = np.isfinite(gt_1d) & np.isfinite(pr_1d)
    n = int(m.sum())
    if n < min_samples:
        return np.nan, np.nan, n
    d = pr_1d[m] - gt_1d[m]
    rmse = float(np.sqrt(np.mean(d * d)))
    mae  = float(np.mean(np.abs(d)))
    return rmse, mae, n

# ----------------------------
# 2) Main benchmark computation
# ----------------------------
def compute_error_realism_benchmark_v1(
    gt_arr,
    pred_arr,
    region_names=None,
    use_resampled=True,
    # validity
    min_samples=50,              # per (seq,region) minimum valid timepoints
    # region reduction
    top_q_regions=0.25,           # WORST-topq over regions (largest errors)
    min_regions_for_reduce=4,     # require enough valid regions for per-seq reduction
    # mismatch
    mismatch_M=10,
    mismatch_regions_max=None,    # optional cap for mismatch runtime (None = no cap)
    rng_seed=0,
    # normalization
    scale_mode="gt_iqr",          # benchmark-safe default: GT-only IQR per region
    # mapping to 0–1 (better-than-mismatch)
    floor=0.05,
    cap=0.95,
    z0=0.75,
    slope=1.25,                  # softness (divide): larger = gentler
):
    gt_arr, pred_arr = _align_gt_pred(gt_arr, pred_arr, use_resampled=use_resampled)
    if gt_arr.ndim != 3 or pred_arr.ndim != 3:
        raise ValueError("Expected gt/pred as [n_seq, T, R].")
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"Aligned mismatch: GT {gt_arr.shape} vs Pred {pred_arr.shape}")

    n_seq, T, n_reg = gt_arr.shape
    if n_seq < 2:
        raise ValueError("Need n_seq >= 2 for mismatch baseline.")

    # robust region naming
    if region_names is None:
        region_names = [f"R{i}" for i in range(n_reg)]
    else:
        region_names = list(region_names)
        if len(region_names) != n_reg:
            region_names = [f"R{i}" for i in range(n_reg)]

    if scale_mode != "gt_iqr":
        raise ValueError("scale_mode currently supports only 'gt_iqr' (benchmark-safe).")

    rng = np.random.default_rng(int(rng_seed))

    # ---- per-region scale from GT only (finite)
    scale_r = np.full(n_reg, np.nan, dtype=np.float64)
    for r in range(n_reg):
        x = gt_arr[:, :, r].ravel()
        x = x[np.isfinite(x)]
        if x.size < 10:
            scale_r[r] = np.nan
        else:
            scale_r[r] = _robust_scale_iqr(x)

    # -------------------------
    # A) Observed errors per (seq, region) with consistent NaN mask
    # -------------------------
    rmse_sr = np.full((n_seq, n_reg), np.nan, dtype=np.float64)
    mae_sr  = np.full((n_seq, n_reg), np.nan, dtype=np.float64)
    nvalid_sr = np.zeros((n_seq, n_reg), dtype=int)

    for s in range(n_seq):
        for r in range(n_reg):
            rmse, mae, n = _compute_rmse_mae_1d(gt_arr[s, :, r], pred_arr[s, :, r], min_samples=min_samples)
            rmse_sr[s, r] = rmse
            mae_sr[s, r]  = mae
            nvalid_sr[s, r] = n

    nrmse_sr = rmse_sr / (scale_r[None, :] + EPS)
    nmae_sr  = mae_sr  / (scale_r[None, :] + EPS)

    # observed per-seq reduction across regions (WORST-topq)
    nrmse_topq_seq = np.full(n_seq, np.nan, dtype=np.float64)
    nmae_topq_seq  = np.full(n_seq, np.nan, dtype=np.float64)
    n_valid_regions_seq = np.zeros(n_seq, dtype=int)

    valid_region_mask_obs = np.isfinite(nrmse_sr) & np.isfinite(nmae_sr) & np.isfinite(scale_r[None, :])

    for s in range(n_seq):
        rr = np.where(valid_region_mask_obs[s])[0]
        n_valid_regions_seq[s] = int(rr.size)
        if rr.size < min_regions_for_reduce:
            continue
        nrmse_topq_seq[s] = _worst_topq_mean(nrmse_sr[s, rr], top_q=top_q_regions)
        nmae_topq_seq[s]  = _worst_topq_mean(nmae_sr[s, rr],  top_q=top_q_regions)

    # also region-wise aggregates (mean over sequences) for reporting
    per_region_df = pd.DataFrame({
        "region": region_names,
        "scale_IQR": scale_r,
        "nRMSE_mean": np.nanmean(nrmse_sr, axis=0),
        "nMAE_mean":  np.nanmean(nmae_sr, axis=0),
        "valid_seq_frac": np.mean(np.isfinite(nrmse_sr) & np.isfinite(nmae_sr), axis=0),
    }).sort_values("nRMSE_mean", ascending=False)

    # -------------------------
    # B) Mismatch baseline (matches observed region-valid mask per sequence)
    # -------------------------
    mis_nrmse_topq = np.full((int(mismatch_M), n_seq), np.nan, dtype=np.float64)
    mis_nmae_topq  = np.full((int(mismatch_M), n_seq), np.nan, dtype=np.float64)

    all_idx = np.arange(n_seq)
    for s in range(n_seq):
        rr_obs = np.where(valid_region_mask_obs[s])[0]  # SAME region-validity mask as observed
        if rr_obs.size < min_regions_for_reduce:
            continue
        if mismatch_regions_max is not None and rr_obs.size > int(mismatch_regions_max):
            rr_obs = rng.choice(rr_obs, size=int(mismatch_regions_max), replace=False)

        pool = all_idx[all_idx != s]
        M = int(min(int(mismatch_M), pool.size))
        js = rng.choice(pool, size=M, replace=False)
        if M < int(mismatch_M):
            js2 = rng.choice(pool, size=(int(mismatch_M) - M), replace=True)
            js = np.concatenate([js, js2])

        for m_i in range(int(mismatch_M)):
            j = int(js[m_i])

            # compute mismatch errors on SAME region set rr_obs, with SAME timepoint masking rule (finite(gt)&finite(pred_j))
            nrmse_list = []
            nmae_list  = []
            for r in rr_obs:
                rmse, mae, n = _compute_rmse_mae_1d(gt_arr[s, :, r], pred_arr[j, :, r], min_samples=min_samples)
                if not np.isfinite(rmse) or not np.isfinite(mae) or not np.isfinite(scale_r[r]):
                    continue
                nrmse_list.append(rmse / (scale_r[r] + EPS))
                nmae_list.append(mae  / (scale_r[r] + EPS))

            if len(nrmse_list) < min_regions_for_reduce:
                continue

            mis_nrmse_topq[m_i, s] = _worst_topq_mean(np.asarray(nrmse_list), top_q=top_q_regions)
            mis_nmae_topq[m_i, s]  = _worst_topq_mean(np.asarray(nmae_list),  top_q=top_q_regions)

    # -------------------------
    # C) Mismatch-corrected z and bounded 0–1
    #     Lower error is better => z = (mis_med - obs)/scale
    # -------------------------
    def _z_and_score(obs_seq, mis_mat):
        mis_med = np.nanmedian(mis_mat, axis=0)
        mis_s = np.array([_robust_mad_scale(mis_mat[:, s]) for s in range(n_seq)], dtype=np.float64)
        z = (mis_med - obs_seq) / (mis_s + EPS)
        score01 = floor + (cap - floor) * _sigmoid_stable((z - z0) / (slope + EPS))
        return z, score01, mis_med

    z_nrmse, score01_nrmse, mis_med_nrmse = _z_and_score(nrmse_topq_seq, mis_nrmse_topq)
    z_nmae,  score01_nmae,  mis_med_nmae  = _z_and_score(nmae_topq_seq,  mis_nmae_topq)

    # -------------------------
    # D) Benchmark scalars (correct labels + true 0–1)
    # -------------------------
    coverage_seq = float(np.mean(np.isfinite(nrmse_topq_seq) & np.isfinite(nmae_topq_seq)))
    coverage_regions = float(np.mean(np.isfinite(scale_r)))

    # IMPORTANT: these are "mean over sequences of per-seq WORST-topq over regions"
    ERR_nRMSE_topq_mean = float(np.nanmean(nrmse_topq_seq))
    ERR_nMAE_topq_mean  = float(np.nanmean(nmae_topq_seq))

    scores_for_composite = {
        "ERR_coverage_regions": coverage_regions,
        "ERR_coverage_seq": coverage_seq,

        # raw (lower better) — properly labeled
        "ERR_nRMSE_topq_mean": ERR_nRMSE_topq_mean,
        "ERR_nMAE_topq_mean":  ERR_nMAE_topq_mean,

        # mismatch-corrected 0–1 (STRICT: q10 over sequences)
        "ERR_nRMSE_topq_z_score01_mean": float(np.nanmean(score01_nrmse)),
        "ERR_nRMSE_topq_z_score01_q10":  _safe_quantile(score01_nrmse, 0.10),
        "ERR_nMAE_topq_z_score01_mean":  float(np.nanmean(score01_nmae)),
        "ERR_nMAE_topq_z_score01_q10":   _safe_quantile(score01_nmae, 0.10),
    }

    print("\n=== Error realism (GT vs Pred) — nRMSE/nMAE + mismatch-corrected 0–1 (benchmark v1; FIXED) ===")
    print(f"n_seq={n_seq} | n_reg={n_reg} | coverage(regions)={coverage_regions*100:.1f}% | coverage(seq)={coverage_seq*100:.1f}%")
    print(f"region reduction: WORST-top_q_regions={top_q_regions} | min_regions_for_reduce={min_regions_for_reduce} | mismatch_M={mismatch_M}")
    print(f"validity: min_samples={min_samples} per (seq,region)")
    print(f"mapping: floor={floor} cap={cap} | z0={z0} | slope(softness)={slope} | scale_mode={scale_mode}")

    print("\n--- Benchmark scalars (USE THESE for composite) ---")
    for k in sorted(scores_for_composite.keys()):
        v = scores_for_composite[k]
        print(f"{k:30s}: {v:.4f}" if np.isfinite(v) else f"{k:30s}: nan")

    # Offenders: regions by mean nRMSE
    print("\nTop offending regions by nRMSE_mean (highest):")
    for _, row in per_region_df.head(5).iterrows():
        print(f"  {row['region'][:40]:40s}  nRMSE_mean={row['nRMSE_mean']:.4f}  nMAE_mean={row['nMAE_mean']:.4f}  valid={row['valid_seq_frac']:.2f}")

    # Offenders: sequences by 0–1 score (lower is worse)
    per_seq_df = pd.DataFrame({
        "seq": np.arange(n_seq),
        "n_valid_regions": n_valid_regions_seq,
        "nRMSE_topq": nrmse_topq_seq,
        "nMAE_topq": nmae_topq_seq,
        "z_nrmse": z_nrmse,
        "score01_nrmse_topq": score01_nrmse,
        "z_nmae": z_nmae,
        "score01_nmae_topq": score01_nmae,
    }).sort_values("score01_nrmse_topq")

    print("\nWorst sequences by score01_nrmse_topq (lowest):")
    for _, row in per_seq_df.head(5).iterrows():
        print(f"  seq={int(row['seq']):4d}  score01={row['score01_nrmse_topq']:.4f}  nRMSE_topq={row['nRMSE_topq']:.4f}  nreg={int(row['n_valid_regions'])}")

    results = {
        "per_region": per_region_df,
        "per_seq": per_seq_df,
        "raw": {
            "rmse_sr": rmse_sr,
            "mae_sr": mae_sr,
            "nrmse_sr": nrmse_sr,
            "nmae_sr": nmae_sr,
            "nvalid_sr": nvalid_sr,
            "scale_r": scale_r,
            "nrmse_topq_seq": nrmse_topq_seq,
            "nmae_topq_seq": nmae_topq_seq,
        },
        "mismatch": {
            "mis_nrmse_topq": mis_nrmse_topq,
            "mis_nmae_topq": mis_nmae_topq,
            "mis_med_nrmse": mis_med_nrmse,
            "mis_med_nmae": mis_med_nmae,
        },
        "scores01": {
            "score01_nrmse_topq": score01_nrmse,
            "score01_nmae_topq": score01_nmae,
        },
        "scores_for_composite": scores_for_composite,
        "params": dict(
            min_samples=min_samples, top_q_regions=top_q_regions, min_regions_for_reduce=min_regions_for_reduce,
            mismatch_M=mismatch_M, mismatch_regions_max=mismatch_regions_max, rng_seed=rng_seed,
            scale_mode=scale_mode, floor=floor, cap=cap, z0=z0, slope=slope, use_resampled=use_resampled,
        ),
        "region_names": region_names,
    }
    return results


# ----------------------------
# 3) RUN (robust region name selection)
# ----------------------------
_required = ["gt_array_denorm", "data_predicted"]
_missing = [v for v in _required if v not in globals()]
if _missing:
    raise RuntimeError(f"Missing variables: {_missing}")

_region_names = None
if "pred_region_names" in globals() and globals().get("pred_region_names") is not None:
    _region_names = globals()["pred_region_names"]
elif "gt_regions" in globals() and globals().get("gt_regions") is not None:
    _region_names = globals()["gt_regions"]


if MODE in ('full', 'instant'):
    err_out = compute_error_realism_benchmark_v1(
        gt_array_denorm,
        data_predicted,
        region_names=_region_names,
        use_resampled=True,
        min_samples=50,
        top_q_regions=0.25,
        min_regions_for_reduce=4,
        mismatch_M=10,
        mismatch_regions_max=None,   # set e.g. 10 if you need speed
        rng_seed=0,
        scale_mode="gt_iqr",
        floor=0.05,
        cap=0.95,
        z0=0.75,
        slope=1.25,
    )

    # ----------------------------
    # 4) Plots (lightweight; no tight_layout warning)
    # ----------------------------
    names = err_out["region_names"]
    per_region = err_out["per_region"].set_index("region").reindex(names)

    plt.figure(figsize=(12, 5))
    plt.bar(names, per_region["nRMSE_mean"].values, alpha=0.85, edgecolor="k")
    plt.xticks(rotation=90)
    plt.ylabel("Mean nRMSE (GT-IQR normalized)")
    plt.title("Error realism: mean nRMSE per region")
    plt.grid(axis="y", alpha=0.25)
    plt.gcf().subplots_adjust(bottom=0.35)
    plt.show()

    plt.figure(figsize=(6.5, 3.5))
    sc = err_out["scores01"]["score01_nrmse_topq"]
    sc = sc[np.isfinite(sc)]
    plt.hist(sc, bins=20, alpha=0.85, edgecolor="k")
    plt.xlabel("score01_nrmse_topq (mismatch-corrected, bounded)")
    plt.ylabel("Count")
    plt.title("Per-sequence error realism scores (nRMSE WORST-topq over regions)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    # Optional: show tables
    print("\nPer-region table (head):")
    print(err_out["per_region"].head(8).to_string(index=False))

    print("\nPer-seq table (head):")
    print(err_out["per_seq"].head(8)[["seq","n_valid_regions","nRMSE_topq","score01_nrmse_topq","nMAE_topq","score01_nmae_topq"]].to_string(index=False))

# --- Quantile / Tail realism (GT vs Pred) — PER-SEQUENCE; per-region quantiles; mismatch-corrected; bounded 0–1 (benchmark v1) ---
# What this does (benchmark-style):
#   ✅ Per (seq, region): compare GT vs Pred *quantile functions* (Q(q), q=0.01..0.99)
#   ✅ Normalize by GT scale per region (GT IQR by default)  -> unitless distances
#   ✅ Reduce per sequence across regions using WORST-top_q (largest errors)
#   ✅ Mismatch baseline: compare GT(seq i) vs Pred(seq j!=i) using SAME region-validity mask as observed for seq i
#   ✅ Robust effect z (median/MAD) -> bounded score01 in [floor, cap]
#   ✅ Mapping is intentionally *non-saturating*: larger slope + wider floor/cap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS = 1e-12

# ----------------------------
# 0) Alignment helper (fallback only)
# ----------------------------
def _resample_pred_to_gt_fallback(gt_arr, pred_arr):
    gt_len = gt_arr.shape[1]
    pr_len = pred_arr.shape[1]
    if pr_len != gt_len and pr_len % gt_len == 0:
        factor = pr_len // gt_len
        pred_res = pred_arr.reshape(pred_arr.shape[0], gt_len, factor, pred_arr.shape[2]).mean(axis=2)
        return gt_arr, pred_res
    if pr_len != gt_len:
        m = min(gt_len, pr_len)
        return gt_arr[:, :m, :], pred_arr[:, :m, :]
    return gt_arr, pred_arr

def _align_gt_pred(gt_arr, pred_arr, use_resampled=True):
    gt_arr = np.asarray(gt_arr, dtype=np.float64)
    pred_arr = np.asarray(pred_arr, dtype=np.float64)
    if not use_resampled:
        return gt_arr, pred_arr
    if "_resample_pred_to_gt" in globals() and callable(globals()["_resample_pred_to_gt"]):
        return globals()["_resample_pred_to_gt"](gt_arr, pred_arr)
    return _resample_pred_to_gt_fallback(gt_arr, pred_arr)

# ----------------------------
# 1) Robust utils
# ----------------------------
def _robust_loc_scale(x):
    """Median + MAD scale (fallback to std); returns (median, scale>0)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, 1.0
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    s = 1.4826 * mad
    if not np.isfinite(s) or s <= 1e-12:
        s = float(np.nanstd(x))
    if not np.isfinite(s) or s <= 1e-12:
        s = 1.0
    return med, float(s)

def _stable_sigmoid(z):
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))

def _safe_nanquantile(x, q):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else np.nan

def _topq_mean_worst(v, top_q):
    """Mean of largest top_q fraction (worst errors)."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    k = max(1, int(np.ceil(top_q * v.size)))
    return float(np.mean(np.sort(v)[-k:]))

def _iqr_scale(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 10:
        s = float(np.nanstd(x))
        return s if np.isfinite(s) and s > 0 else 1.0
    q25, q75 = np.nanquantile(x, [0.25, 0.75])
    s = float(q75 - q25)
    if np.isfinite(s) and s > 0:
        return s
    s = float(np.nanstd(x))
    return s if np.isfinite(s) and s > 0 else 1.0

# ----------------------------
# 2) Core computation
# ----------------------------
def compute_quantile_realism_benchmark_v1(
    gt_arr,
    pred_arr,
    region_names=None,
    use_resampled=True,
    # quantile grid
    q_lo=0.01,
    q_hi=0.99,
    n_q=99,
    tail_lo=0.10,
    tail_hi=0.90,
    # validity / runtime guards
    min_samples=80,      # per (seq,region) after finite masking (for quantiles)
    max_time=1200,       # subsample points used to estimate quantiles
    rng_seed=0,
    # scaling (benchmark-safe)
    scale_mode="gt_iqr", # "gt_iqr" (default) or "gt_mad"
    # reductions
    top_q_regions=0.25,        # worst-top_q across regions per sequence
    min_regions_for_reduce=4,  # require at least this many valid regions to score a sequence
    # mismatch
    mismatch_M=10,       # number of mismatch draws per seq (j!=i)
    # score mapping (bounded 0–1, *non-saturating*)
    floor=0.02,
    cap=0.98,
    z0=0.50,
    slope=3.00,          # larger => gentler, less saturation
):
    gt_arr, pred_arr = _align_gt_pred(gt_arr, pred_arr, use_resampled=use_resampled)
    if gt_arr.ndim != 3 or pred_arr.ndim != 3:
        raise ValueError("Expected gt/pred as [n_seq, T, R].")
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"Aligned mismatch: GT {gt_arr.shape} vs Pred {pred_arr.shape}")

    n_seq, T, n_reg = gt_arr.shape
    if n_seq < 2:
        raise ValueError("Need n_seq >= 2 for mismatch baseline.")

    # region names
    if region_names is None:
        region_names = [f"R{i}" for i in range(n_reg)]
    else:
        region_names = list(region_names)
        if len(region_names) != n_reg:
            region_names = [f"R{i}" for i in range(n_reg)]

    if scale_mode not in ("gt_iqr", "gt_mad"):
        raise ValueError("scale_mode must be 'gt_iqr' or 'gt_mad'")

    rng = np.random.default_rng(rng_seed)

    # ---- quantile grid ----
    quantiles = np.linspace(q_lo, q_hi, int(n_q))
    tail_mask = (quantiles <= tail_lo) | (quantiles >= tail_hi)

    # ---- per-region GT scale (global across all seq/time) ----
    scales = np.full(n_reg, np.nan, dtype=np.float64)
    for r in range(n_reg):
        x = gt_arr[:, :, r].ravel()
        x = x[np.isfinite(x)]
        if x.size < 10:
            continue
        if scale_mode == "gt_iqr":
            scales[r] = _iqr_scale(x)
        else:
            _, s = _robust_loc_scale(x)
            scales[r] = s
    scales[~np.isfinite(scales) | (scales <= 0)] = 1.0  # hard fallback (benchmark safety)

    # ---- helper: compute quantiles for a 1D array with subsampling ----
    def _quantile_vec(v):
        v = np.asarray(v, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size < min_samples:
            return np.full(quantiles.shape, np.nan, dtype=np.float64)
        if v.size > max_time:
            idx = rng.choice(v.size, size=max_time, replace=False)
            v = v[idx]
        # np.quantile is fine; we avoid interpolation args for broad compatibility
        return np.quantile(v, quantiles).astype(np.float64)

    # ---- precompute quantile functions for GT and Pred separately ----
    gtQ = np.full((n_seq, n_reg, quantiles.size), np.nan, dtype=np.float64)
    prQ = np.full_like(gtQ, np.nan)
    valid_gtQ = np.zeros((n_seq, n_reg), dtype=bool)
    valid_prQ = np.zeros((n_seq, n_reg), dtype=bool)

    for s in range(n_seq):
        for r in range(n_reg):
            gtq = _quantile_vec(gt_arr[s, :, r])
            prq = _quantile_vec(pred_arr[s, :, r])
            gtQ[s, r] = gtq
            prQ[s, r] = prq
            valid_gtQ[s, r] = np.isfinite(gtq).all()
            valid_prQ[s, r] = np.isfinite(prq).all()

    # Observed validity mask per (seq,region) — used to *lock* which regions contribute per sequence
    valid_obs_sr = valid_gtQ & valid_prQ

    # ---- observed distances per (seq,region) ----
    scale_b = scales[None, :, None] + EPS
    abs_q_obs = np.abs(prQ - gtQ) / scale_b               # (n_seq, n_reg, n_q)
    D_full_obs_sr = np.nanmean(abs_q_obs, axis=2)         # (n_seq, n_reg)
    D_tail_obs_sr = np.nanmean(abs_q_obs[:, :, tail_mask], axis=2)
    D_qmax_obs_sr = np.nanmax(abs_q_obs, axis=2)

    # enforce validity (NaN handling consistent)
    D_full_obs_sr[~valid_obs_sr] = np.nan
    D_tail_obs_sr[~valid_obs_sr] = np.nan
    D_qmax_obs_sr[~valid_obs_sr] = np.nan

    # ---- per-seq region reductions (WORST-top_q across regions) ----
    n_valid_regions = np.sum(valid_obs_sr, axis=1).astype(int)
    D_full_topq_seq = np.full(n_seq, np.nan, dtype=np.float64)
    D_tail_topq_seq = np.full(n_seq, np.nan, dtype=np.float64)
    D_qmax_topq_seq = np.full(n_seq, np.nan, dtype=np.float64)

    for s in range(n_seq):
        if n_valid_regions[s] < min_regions_for_reduce:
            continue
        D_full_topq_seq[s] = _topq_mean_worst(D_full_obs_sr[s], top_q_regions)
        D_tail_topq_seq[s] = _topq_mean_worst(D_tail_obs_sr[s], top_q_regions)
        D_qmax_topq_seq[s] = _topq_mean_worst(D_qmax_obs_sr[s], top_q_regions)

    coverage_seq = float(np.mean(np.isfinite(D_tail_topq_seq)))  # same coverage pattern for all reductions
    coverage_regions = float(np.mean(np.isfinite(np.nanmean(D_tail_obs_sr, axis=0))))

    # ---- mismatch baseline (j != s), using SAME region-validity mask as observed for seq s ----
    mis_tail = np.full((mismatch_M, n_seq), np.nan, dtype=np.float64)
    mis_full = np.full((mismatch_M, n_seq), np.nan, dtype=np.float64)

    all_idx = np.arange(n_seq)
    for s in range(n_seq):
        if not np.isfinite(D_tail_topq_seq[s]):
            continue

        # lock contributing regions to observed-valid regions for this sequence
        rr = np.where(valid_obs_sr[s])[0]
        if rr.size < min_regions_for_reduce:
            continue

        pool = all_idx[all_idx != s]
        # draw mismatch indices (allow replace if needed)
        if pool.size >= mismatch_M:
            js = rng.choice(pool, size=mismatch_M, replace=False)
        else:
            js = rng.choice(pool, size=mismatch_M, replace=True)

        # compute mismatch distances via quantile functions (fast)
        gtq = gtQ[s, rr, :]  # (n_rr, n_q)
        for m_i, j in enumerate(js):
            prq = prQ[j, rr, :]

            # IMPORTANT: require pred quantiles valid for those regions too; otherwise drop that region for this mismatch draw
            ok = np.isfinite(prq).all(axis=1) & np.isfinite(gtq).all(axis=1)
            if ok.sum() < min_regions_for_reduce:
                continue

            dq = np.abs(prq[ok] - gtq[ok]) / (scales[rr[ok], None] + EPS)
            d_full_r = np.mean(dq, axis=1)
            d_tail_r = np.mean(dq[:, tail_mask], axis=1)

            mis_full[m_i, s] = _topq_mean_worst(d_full_r, top_q_regions)
            mis_tail[m_i, s] = _topq_mean_worst(d_tail_r, top_q_regions)

    # ---- effect sizes + bounded score01 (lower distance is better) ----
    def _z_and_score(obs_vec, mis_mat):
        mis_med = np.nanmedian(mis_mat, axis=0)  # (n_seq,)
        mis_s = np.full(n_seq, np.nan, dtype=np.float64)
        for s in range(n_seq):
            _, ss = _robust_loc_scale(mis_mat[:, s])
            mis_s[s] = ss
        z = (mis_med - obs_vec) / (mis_s + EPS)  # positive if better-than-mismatch
        score01 = floor + (cap - floor) * _stable_sigmoid((z - z0) / (slope + EPS))
        return z.astype(np.float64), score01.astype(np.float64), mis_med.astype(np.float64), mis_s.astype(np.float64)

    z_tail, score01_tail, mis_med_tail, mis_s_tail = _z_and_score(D_tail_topq_seq, mis_tail)
    z_full, score01_full, mis_med_full, mis_s_full = _z_and_score(D_full_topq_seq, mis_full)

    # ---- per-region summaries (for offenders) ----
    per_region = pd.DataFrame({
        "region": region_names,
        "scale": scales,
        "D_full_mean": np.nanmean(D_full_obs_sr, axis=0),
        "D_tail_mean": np.nanmean(D_tail_obs_sr, axis=0),
        "D_qmax_mean": np.nanmean(D_qmax_obs_sr, axis=0),
        "valid_frac": np.mean(valid_obs_sr, axis=0).astype(np.float64),
    }).sort_values("D_tail_mean", ascending=False)  # tail offenders first

    # ---- per-seq table ----
    per_seq = pd.DataFrame({
        "seq": np.arange(n_seq),
        "n_valid_regions": n_valid_regions,
        "D_tail_topq": D_tail_topq_seq,
        "D_full_topq": D_full_topq_seq,
        "z_tail_topq": z_tail,
        "score01_tail_topq": score01_tail,
        "z_full_topq": z_full,
        "score01_full_topq": score01_full,
        "mis_med_tail": mis_med_tail,
        "mis_s_tail": mis_s_tail,
    }).sort_values("score01_tail_topq")

    # ---- benchmark scalars (USE THESE) ----
    scores_for_composite = {
        "QNT_coverage_regions": coverage_regions,
        "QNT_coverage_seq": coverage_seq,

        # raw distances (unitless; lower is better)
        "QNT_tail_topq_mean": float(np.nanmean(D_tail_topq_seq)),
        "QNT_full_topq_mean": float(np.nanmean(D_full_topq_seq)),

        # bounded 0–1 realism (higher is better) — strict + mean
        "QNT_tail_topq_z_score01_mean": float(np.nanmean(score01_tail)),
        "QNT_tail_topq_z_score01_q10":  _safe_nanquantile(score01_tail, 0.10),

        "QNT_full_topq_z_score01_mean": float(np.nanmean(score01_full)),
        "QNT_full_topq_z_score01_q10":  _safe_nanquantile(score01_full, 0.10),
    }

    print("\n=== Quantile realism (GT vs Pred) — tail/full quantile functions + mismatch-corrected 0–1 (benchmark v1) ===")
    print(f"n_seq={n_seq} | n_reg={n_reg} | coverage(regions)={coverage_regions*100:.1f}% | coverage(seq)={coverage_seq*100:.1f}%")
    print(f"quantiles: q in [{q_lo:.2f},{q_hi:.2f}] (n_q={n_q}); tail=({tail_lo:.2f} & {tail_hi:.2f})")
    print(f"validity: min_samples={min_samples} per (seq,region) | max_time={max_time}")
    print(f"region reduction: WORST-top_q_regions={top_q_regions} | min_regions_for_reduce={min_regions_for_reduce} | mismatch_M={mismatch_M}")
    print(f"mapping (non-saturating): floor={floor} cap={cap} | z0={z0} | slope={slope} | scale_mode={scale_mode}")

    print("\n--- Benchmark scalars (USE THESE for composite) ---")
    for k in sorted(scores_for_composite.keys()):
        v = scores_for_composite[k]
        print(f"{k:30s}: {v:.4f}" if np.isfinite(v) else f"{k:30s}: nan")

    print("\nTop offending regions by D_tail_mean (highest):")
    for _, row in per_region.head(5).iterrows():
        nm = str(row["region"])
        print(f"  {nm[:38]:38s}  D_tail_mean={row['D_tail_mean']:.4f}  D_full_mean={row['D_full_mean']:.4f}  valid={row['valid_frac']:.2f}")

    print("\nWorst sequences by score01_tail_topq (lowest):")
    for _, row in per_seq.head(5).iterrows():
        s = int(row["seq"])
        print(f"  seq={s:4d}  score01={row['score01_tail_topq']:.4f}  D_tail_topq={row['D_tail_topq']:.4f}  nreg={int(row['n_valid_regions'])}")

    return {
        "quantiles": quantiles,
        "tail_mask": tail_mask,
        "per_region": per_region,
        "per_seq": per_seq,
        "scores_for_composite": scores_for_composite,
        "arrays": {
            "D_tail_obs_sr": D_tail_obs_sr,
            "D_full_obs_sr": D_full_obs_sr,
            "valid_obs_sr": valid_obs_sr,
            "D_tail_topq_seq": D_tail_topq_seq,
            "D_full_topq_seq": D_full_topq_seq,
            "score01_tail_topq": score01_tail,
            "score01_full_topq": score01_full,
        },
        "params": dict(
            q_lo=q_lo, q_hi=q_hi, n_q=n_q, tail_lo=tail_lo, tail_hi=tail_hi,
            min_samples=min_samples, max_time=max_time, rng_seed=rng_seed,
            scale_mode=scale_mode, top_q_regions=top_q_regions, min_regions_for_reduce=min_regions_for_reduce,
            mismatch_M=mismatch_M, floor=floor, cap=cap, z0=z0, slope=slope, use_resampled=use_resampled
        ),
        "region_names": region_names,
    }

# ----------------------------
# 3) RUN (robust region name selection)
# ----------------------------
_required = ["gt_array_denorm", "data_predicted"]
_missing = [v for v in _required if v not in globals()]
if _missing:
    raise RuntimeError(f"Missing variables: {_missing}")

_region_names = None
if "pred_region_names" in globals() and globals().get("pred_region_names") is not None:
    _region_names = globals()["pred_region_names"]
elif "gt_regions" in globals() and globals().get("gt_regions") is not None:
    _region_names = globals()["gt_regions"]


if MODE in ('full', 'instant'):
    results_qnt = compute_quantile_realism_benchmark_v1(
        gt_array_denorm,
        data_predicted,
        region_names=_region_names,
        use_resampled=True,
        # sensible defaults (non-saturating)
        min_samples=80,
        max_time=1200,
        top_q_regions=0.25,
        min_regions_for_reduce=4,
        mismatch_M=10,
        scale_mode="gt_iqr",
        floor=0.02,
        cap=0.98,
        z0=0.50,
        slope=3.00,
        rng_seed=0,
    )

    # ----------------------------
    # 4) Lightweight plots (benchmark-friendly)
    # ----------------------------
    names = results_qnt["region_names"]
    per_region = results_qnt["per_region"].set_index("region").reindex(names)

    plt.figure(figsize=(12, 4))
    plt.bar(names, per_region["D_tail_mean"].values, alpha=0.85, edgecolor="k")
    plt.xticks(rotation=90)
    plt.ylabel("Mean tail quantile distance (|ΔQ| / scale)")
    plt.title("Tail-shape discrepancy per region (lower is better)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()

    score_tail = results_qnt["arrays"]["score01_tail_topq"]
    plt.figure(figsize=(6, 3))
    plt.hist(score_tail[np.isfinite(score_tail)], bins=20, alpha=0.85, edgecolor="k")
    plt.xlabel("score01_tail_topq (mismatch-corrected, bounded)")
    plt.ylabel("Count")
    plt.title("Per-sequence quantile realism scores (tail, WORST-topq regions)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    print("\nInterpretation:")
    print("- This probes *shape* via quantile functions, not correlation/dependence.")
    print("- The benchmark scalars are mismatch-corrected and bounded (0–1), using a gentle mapping to avoid fast saturation.")
    print("- Use alongside KL/JSD/W1 and MI: quantiles are a robust complementary view of distributional realism.")

# === FC REALISM (GT vs Pred) — per-seq + per-window; ddof-consistent; mismatch-corrected 0–1 (benchmark v1) ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS = 1e-12

# ---------------------------
# 0) Alignment
# ---------------------------
gt_arr, pr_arr = _resample_pred_to_gt(gt_array_denorm, data_predicted)
gt_arr = np.asarray(gt_arr, dtype=np.float64)
pr_arr = np.asarray(pr_arr, dtype=np.float64)
if gt_arr.ndim != 3 or pr_arr.ndim != 3:
    raise ValueError("Expected gt/pred arrays with shape [n_seq, n_time, n_reg].")
if gt_arr.shape != pr_arr.shape:
    raise ValueError(f"Shape mismatch after resample: {gt_arr.shape} vs {pr_arr.shape}")

n_seq, n_time, n_reg = gt_arr.shape
region_names = list(gt_regions) if "gt_regions" in globals() else [f"R{i}" for i in range(n_reg)]
n_edges = n_reg * (n_reg - 1) // 2
iu = np.triu_indices(n_reg, k=1)

# ---------------------------
# 1) Config (edit here)
# ---------------------------
window_size = 60
window_step = 10
min_samples = 50                # per (seq,window) after joint mask

mismatch_M = 10                 # mismatch samples per seq
alpha = 0.6                     # static vs dynamic blending (core = alpha*static + (1-alpha)*dynamic)
agg = "mean"                    # "mean" or "median" for reductions

# strictness within level: blend agg with q10 (lower tail)
beta = 0.5

# bounded score mapping (non-saturating)
floor = 0.02
cap   = 0.98
z0    = 0.5
slope = 3.0                     # larger => gentler, less saturation

rng_seed = 0
plot = True

# ---------------------------
# 2) Utilities (ddof-consistent)
# ---------------------------
def _agg1(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return float(np.mean(v)) if agg == "mean" else float(np.median(v))

def _q(v, q):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, q)) if v.size else np.nan

def _robust_med_mad(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    s = 1.4826 * mad
    if not np.isfinite(s) or s <= 1e-12:
        s = float(np.std(x))
    if not np.isfinite(s) or s <= 1e-12:
        s = 1.0
    return med, s

def _sigmoid(z):
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))

def _score_from_z(z):
    # bounded, gentle mapping; avoids quick saturation
    return float(floor + (cap - floor) * _sigmoid((z - z0) / (slope + EPS)))

def _valid_rows_pair(X, Y):
    m = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    return X[m], Y[m]

def _zscore_cols_ddof1(X):
    # ddof=1 to match covariance /(n-1)
    X = np.asarray(X, dtype=np.float64)
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, ddof=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd

def _corr_from_zscored(Z):
    # if Z is ddof=1 standardized, then (Z^T Z)/(n-1) is correlation
    n = Z.shape[0]
    if n < 2:
        return None
    C = (Z.T @ Z) / float(n - 1)
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    return C

def _vec_upper(C):
    return C[iu]

def _safe_pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    aa, bb = a[m], b[m]
    if np.std(aa) == 0.0 or np.std(bb) == 0.0:
        return np.nan
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    return float((aa @ bb) / (np.sqrt((aa @ aa) * (bb @ bb)) + EPS))

# ---------------------------
# 3) Precompute FC edge-vectors
#    - static: one vector per seq
#    - dynamic: one vector per (seq,window)
#    Everything uses the same joint mask + ddof-consistent zscore/corr
# ---------------------------
starts = np.arange(0, n_time - window_size + 1, window_step, dtype=int)
centers = starts + window_size / 2.0
n_win = starts.size

vg_static = np.full((n_seq, n_edges), np.nan, dtype=np.float32)
vp_static = np.full((n_seq, n_edges), np.nan, dtype=np.float32)

vg_sw = np.full((n_seq, n_win, n_edges), np.nan, dtype=np.float32)
vp_sw = np.full((n_seq, n_win, n_edges), np.nan, dtype=np.float32)

# static
for s in range(n_seq):
    Xg, Xp = _valid_rows_pair(gt_arr[s], pr_arr[s])
    if Xg.shape[0] < min_samples:
        continue
    Cg = _corr_from_zscored(_zscore_cols_ddof1(Xg))
    Cp = _corr_from_zscored(_zscore_cols_ddof1(Xp))
    if Cg is None or Cp is None:
        continue
    vg_static[s] = _vec_upper(Cg).astype(np.float32)
    vp_static[s] = _vec_upper(Cp).astype(np.float32)

# dynamic
for wi, st in enumerate(starts):
    en = st + window_size
    for s in range(n_seq):
        Xg, Xp = _valid_rows_pair(gt_arr[s, st:en, :], pr_arr[s, st:en, :])
        if Xg.shape[0] < min_samples:
            continue
        Cg = _corr_from_zscored(_zscore_cols_ddof1(Xg))
        Cp = _corr_from_zscored(_zscore_cols_ddof1(Xp))
        if Cg is None or Cp is None:
            continue
        vg_sw[s, wi] = _vec_upper(Cg).astype(np.float32)
        vp_sw[s, wi] = _vec_upper(Cp).astype(np.float32)

# ---------------------------
# 4) Observed similarities (per seq)
# ---------------------------
# static r per sequence
r_static_seq = np.full(n_seq, np.nan, dtype=np.float64)
for s in range(n_seq):
    r_static_seq[s] = _safe_pearson(vg_static[s], vp_static[s])

# dynamic: r per (seq,win) then aggregate over windows per seq
r_dyn_sw = np.full((n_seq, n_win), np.nan, dtype=np.float64)
for s in range(n_seq):
    for wi in range(n_win):
        r_dyn_sw[s, wi] = _safe_pearson(vg_sw[s, wi], vp_sw[s, wi])

r_dyn_seq = np.array([_agg1(r_dyn_sw[s]) for s in range(n_seq)], dtype=np.float64)

# strict versions within each level
static_agg = _agg1(r_static_seq)
static_q10 = _q(r_static_seq, 0.10)
static_strict = (1 - beta) * static_agg + beta * static_q10 if np.isfinite(static_agg) and np.isfinite(static_q10) else np.nan

dyn_agg = _agg1(r_dyn_seq)
dyn_q10 = _q(r_dyn_seq, 0.10)
dyn_strict = (1 - beta) * dyn_agg + beta * dyn_q10 if np.isfinite(dyn_agg) and np.isfinite(dyn_q10) else np.nan

# ---------------------------
# 5) Mismatch baselines (per seq, j != s), WITH SAME validity objects
#     - static mismatch: compare vg_static[s] vs vp_static[j]
#     - dynamic mismatch: compare vg_sw[s,wi] vs vp_sw[j,wi], then aggregate over windows
# ---------------------------
rng = np.random.default_rng(rng_seed)
all_idx = np.arange(n_seq)

mis_static = np.full((mismatch_M, n_seq), np.nan, dtype=np.float64)
mis_dyn    = np.full((mismatch_M, n_seq), np.nan, dtype=np.float64)

for s in range(n_seq):
    pool = all_idx[all_idx != s]
    if pool.size == 0:
        continue
    js = rng.choice(pool, size=mismatch_M, replace=(pool.size < mismatch_M))

    # pre-check: need GT vectors at least
    if not np.isfinite(vg_static[s]).any():
        continue

    for m_i, j in enumerate(js):
        # static mismatch
        mis_static[m_i, s] = _safe_pearson(vg_static[s], vp_static[j])

        # dynamic mismatch: aggregate across windows for this mismatched pair
        r_tmp = np.full(n_win, np.nan, dtype=np.float64)
        for wi in range(n_win):
            r_tmp[wi] = _safe_pearson(vg_sw[s, wi], vp_sw[j, wi])
        mis_dyn[m_i, s] = _agg1(r_tmp)

# ---------------------------
# 6) Convert to mismatch-corrected z and bounded 0–1 score
# ---------------------------
def _z_and_score(obs_seq, mis_mat):
    mis_med = np.nanmedian(mis_mat, axis=0)
    mis_s   = np.array([_robust_med_mad(mis_mat[:, s])[1] for s in range(n_seq)], dtype=np.float64)
    z = (obs_seq - mis_med) / (mis_s + EPS)
    score01 = np.array([_score_from_z(zz) if np.isfinite(zz) else np.nan for zz in z], dtype=np.float64)
    return z, score01

z_static_seq, score_static_seq = _z_and_score(r_static_seq, mis_static)
z_dyn_seq,    score_dyn_seq    = _z_and_score(r_dyn_seq,    mis_dyn)

# core r-space (still in r units) and score-space (recommended for composite)
core_r_seq = alpha * r_static_seq + (1 - alpha) * r_dyn_seq
z_core_seq, score_core_seq = _z_and_score(core_r_seq, alpha * mis_static + (1 - alpha) * mis_dyn)

# strict versions (apply strictness after score01, like your MI/error/quantile style)
def _strict_from_scores(score_seq):
    a = _agg1(score_seq)
    q10 = _q(score_seq, 0.10)
    return (1 - beta) * a + beta * q10 if np.isfinite(a) and np.isfinite(q10) else np.nan

score_static_mean = _agg1(score_static_seq)
score_static_strict = _strict_from_scores(score_static_seq)
score_dyn_mean = _agg1(score_dyn_seq)
score_dyn_strict = _strict_from_scores(score_dyn_seq)
score_core_mean = _agg1(score_core_seq)
score_core_strict = _strict_from_scores(score_core_seq)

cov_static = float(np.mean(np.isfinite(r_static_seq)))
cov_dyn    = float(np.mean(np.isfinite(r_dyn_seq)))
cov_core   = float(np.mean(np.isfinite(core_r_seq)))

# ---------------------------
# 7) Benchmark scalars (USE THESE)
# ---------------------------
scores_for_composite = {
    "FC_coverage_static_seq": cov_static,
    "FC_coverage_dyn_seq": cov_dyn,
    "FC_coverage_core_seq": cov_core,

    "FC_static_score01_mean": score_static_mean,
    "FC_static_score01_q10":  _q(score_static_seq, 0.10),
    "FC_dyn_score01_mean":    score_dyn_mean,
    "FC_dyn_score01_q10":     _q(score_dyn_seq, 0.10),
    "FC_core_score01_mean":   score_core_mean,
    "FC_core_score01_q10":    _q(score_core_seq, 0.10),

    # (optional, for debugging) r-space aggregates
    "FC_static_r_mean": _agg1(r_static_seq),
    "FC_dyn_r_mean":    _agg1(r_dyn_seq),
    "FC_core_r_mean":   _agg1(core_r_seq),
}

print("\n=== FC realism (GT vs Pred) — per-seq static+dynamic; mismatch-corrected 0–1 (benchmark v1) ===")
print(f"n_seq={n_seq} | n_reg={n_reg} | n_edges={n_edges} | n_win={n_win}")
print(f"window_size={window_size} | step={window_step} | min_samples={min_samples}")
print(f"mismatch_M={mismatch_M} | alpha={alpha} | beta={beta} | agg={agg}")
print(f"mapping: floor={floor} cap={cap} | z0={z0} | slope={slope} | ddof(z)=1")

print("\n--- Benchmark scalars (USE THESE for composite) ---")
for k in sorted(scores_for_composite.keys()):
    v = scores_for_composite[k]
    print(f"{k:28s}: {v:.4f}" if np.isfinite(v) else f"{k:28s}: nan")

# Offenders
df_seq = pd.DataFrame({
    "seq": np.arange(n_seq),
    "r_static": r_static_seq,
    "r_dyn": r_dyn_seq,
    "r_core": core_r_seq,
    "score01_static": score_static_seq,
    "score01_dyn": score_dyn_seq,
    "score01_core": score_core_seq,
}).sort_values("score01_core")

print("\nWorst sequences by score01_core (lowest):")
for _, row in df_seq.head(5).iterrows():
    print(f"  seq={int(row['seq']):4d}  score01_core={row['score01_core']:.4f}  r_core={row['r_core']:.3f}  r_static={row['r_static']:.3f}  r_dyn={row['r_dyn']:.3f}")

# ---------------------------
# 8) Lightweight plots (optional)
# ---------------------------
if plot:
    plt.figure(figsize=(6, 3.5))
    plt.hist(df_seq["score01_core"].dropna().values, bins=20, alpha=0.8, edgecolor="k")
    plt.xlabel("score01_core (mismatch-corrected, bounded)")
    plt.ylabel("count")
    plt.title("FC realism: per-sequence core score distribution")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 3.5))
    plt.plot(centers, np.nanmean(r_dyn_sw, axis=0), marker="o", ms=3)
    plt.xlabel("window center (timestep)")
    plt.ylabel("mean r over sequences")
    plt.title("Dynamic FC similarity over time (mean over sequences)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

# Return a compact results dict

if MODE in ('full', 'instant'):
    fc_realism_summary = dict(
        scores_for_composite=scores_for_composite,
        per_seq=df_seq,
        params=dict(
            window_size=window_size, window_step=window_step, min_samples=min_samples,
            mismatch_M=mismatch_M, alpha=alpha, beta=beta, agg=agg,
            floor=floor, cap=cap, z0=z0, slope=slope, rng_seed=rng_seed,
            ddof_zscore=1
        ),
    )
    fc_realism_summary

# === PCA realism (GT vs Pred) — GT-anchored PCA + mismatch-corrected 0–1 (benchmark v1; NO whitening) ===
# What this cell fixes vs typical PCA-sim cells:
#   ✅ Per-sequence (no flatten)
#   ✅ GT-anchored standardization (GT mean/std used for BOTH GT+Pred)
#   ✅ Same validity mask logic for observed + mismatch baseline (pairwise finite rows)
#   ✅ Same k choice (from GT EVR target) used for observed + mismatch comparisons
#   ✅ Mismatch baseline (j != s) -> robust z -> bounded 0–1 (non-saturating mapping)
#   ✅ Runtime guards: min_samples + max_time subsampling
#
# Requires:
#   - gt_array_denorm: (n_seq, T, n_reg)
#   - data_predicted:  (n_seq, T, n_reg) OR resample-able
#   - _resample_pred_to_gt(gt, pred) recommended (will use if present)
# Optional:
#   - gt_regions or pred_region_names for labels

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

EPS = 1e-12

# ----------------------------
# 0) Alignment helper (fallback only)
# ----------------------------
def _resample_pred_to_gt_fallback(gt_arr, pred_arr):
    gt_len = gt_arr.shape[1]
    pr_len = pred_arr.shape[1]
    if pr_len != gt_len and pr_len % gt_len == 0:
        factor = pr_len // gt_len
        pred_res = pred_arr.reshape(pred_arr.shape[0], gt_len, factor, pred_arr.shape[2]).mean(axis=2)
        return gt_arr, pred_res
    if pr_len != gt_len:
        m = min(gt_len, pr_len)
        return gt_arr[:, :m, :], pred_arr[:, :m, :]
    return gt_arr, pred_arr

def _align_gt_pred(gt_arr, pred_arr, use_resampled=True):
    gt_arr = np.asarray(gt_arr, dtype=np.float64)
    pred_arr = np.asarray(pred_arr, dtype=np.float64)
    if not use_resampled:
        return gt_arr, pred_arr
    if "_resample_pred_to_gt" in globals() and callable(globals()["_resample_pred_to_gt"]):
        return globals()["_resample_pred_to_gt"](gt_arr, pred_arr)
    return _resample_pred_to_gt_fallback(gt_arr, pred_arr)

# ----------------------------
# 1) Robust utils
# ----------------------------
def _robust_median_mad(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, 1.0
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    s = 1.4826 * mad
    if not np.isfinite(s) or s <= 1e-12:
        s = float(np.nanstd(x))
    if not np.isfinite(s) or s <= 1e-12:
        s = 1.0
    return med, float(s)

def _robust_scale_mad(x):
    _, s = _robust_median_mad(x)
    return float(s)

def _sigmoid_stable(z):
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))

def _safe_nanquantile(x, q):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else np.nan

def _nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan

# ----------------------------
# 2) PCA math helpers (NO whitening)
# ----------------------------
def _pair_valid_rows(Xg, Xp):
    """Return rows where BOTH GT and Pred are fully finite across regions."""
    m = np.isfinite(Xg).all(axis=1) & np.isfinite(Xp).all(axis=1)
    return Xg[m], Xp[m]

def _gt_standardize(Zg, Zp, eps=1e-8):
    """Standardize both using GT mean/std (per region)."""
    mu = np.nanmean(Zg, axis=0, keepdims=True)
    sd = np.nanstd(Zg, axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (Zg - mu) / sd, (Zp - mu) / sd

def _orth_rows(A):
    """Orthonormalize ROW vectors in A (k x R)."""
    A = np.asarray(A, dtype=np.float64)
    Q, _ = np.linalg.qr(A.T)     # QR on columns (R x k)
    return Q.T                  # (k x R), rows orthonormal

def _principal_cos(U, V):
    """Cosines of principal angles between row-subspaces span(U) and span(V)."""
    # U,V: (k x R) with orthonormal rows
    s = np.linalg.svd(U @ V.T, compute_uv=False)
    return np.clip(s, 0.0, 1.0)

def _cov(Z):
    Z = np.asarray(Z, dtype=np.float64)
    denom = max(1, Z.shape[0] - 1)
    return (Z.T @ Z) / float(denom)

def _proj_fraction(Sigma, U):
    """Fraction of variance (trace) captured by subspace U (rows orthonormal)."""
    tot = float(np.trace(Sigma))
    if not np.isfinite(tot) or tot <= 1e-12:
        return np.nan
    return float(np.trace(U @ Sigma @ U.T) / tot)

def _choose_k_by_gt_evr(evr, var_target=0.90, k_min=5, k_max=20):
    evr = np.asarray(evr, dtype=np.float64)
    if evr.size == 0 or not np.isfinite(evr).any():
        return None
    c = np.cumsum(np.nan_to_num(evr, nan=0.0))
    k = int(np.searchsorted(c, var_target) + 1)
    k = int(np.clip(k, k_min, min(k_max, evr.size)))
    return k

# ----------------------------
# 3) Core benchmark computation
# ----------------------------
def compute_pca_realism_benchmark_v1(
    gt_arr,
    pred_arr,
    region_names=None,
    use_resampled=True,
    # validity / runtime
    min_samples=80,
    max_time=1200,
    rng_seed=0,
    # PCA config (GT-anchored)
    var_target=0.90,
    k_min=5,
    k_max=20,
    # composite
    w_proj=1/3,
    w_mean=1/3,
    w_min=1/3,
    min_exp=2.0,
    # mismatch baseline
    mismatch_M=10,
    # score mapping (non-saturating)
    floor=0.02,
    cap=0.98,
    z0=0.5,
    slope=3.0,
):
    gt_arr, pred_arr = _align_gt_pred(gt_arr, pred_arr, use_resampled=use_resampled)
    if gt_arr.ndim != 3 or pred_arr.ndim != 3:
        raise ValueError("Expected gt/pred as [n_seq, T, R].")
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"Aligned mismatch: GT {gt_arr.shape} vs Pred {pred_arr.shape}")

    n_seq, T, R = gt_arr.shape
    if n_seq < 2:
        raise ValueError("Need n_seq >= 2 for mismatch baseline.")

    if region_names is None:
        region_names = [f"R{i}" for i in range(R)]
    else:
        region_names = list(region_names)
        if len(region_names) != R:
            region_names = [f"R{i}" for i in range(R)]

    rng = np.random.default_rng(rng_seed)

    # -------------------------
    # A) OBSERVED per-seq metrics
    # -------------------------
    k_used     = np.full(n_seq, np.nan, dtype=np.float64)
    proj_obs   = np.full(n_seq, np.nan, dtype=np.float64)   # Pred variance captured by GT subspace
    mean_cos   = np.full(n_seq, np.nan, dtype=np.float64)   # mean principal cosine
    min_cos    = np.full(n_seq, np.nan, dtype=np.float64)   # min principal cosine
    comp_obs   = np.full(n_seq, np.nan, dtype=np.float64)   # composite in [0,1]

    # store GT PCA subspace per seq (to reuse in mismatch)
    Ug_store   = [None] * n_seq
    evr_gt_agg = np.full((n_seq, k_max), np.nan, dtype=np.float64)

    for s in range(n_seq):
        Xg, Xp = _pair_valid_rows(gt_arr[s], pred_arr[s])
        if Xg.shape[0] < min_samples:
            continue

        # time cap (subsample matched rows)
        if Xg.shape[0] > max_time:
            idx = rng.choice(Xg.shape[0], size=max_time, replace=False)
            Xg = Xg[idx]
            Xp = Xp[idx]

        # GT-anchored standardization
        Zg, Zp = _gt_standardize(Xg, Xp, eps=1e-8)

        # fit PCA (no whitening)
        k_fit = int(min(k_max, R, Zg.shape[0] - 1))
        if k_fit < 2:
            continue

        pca_g = PCA(n_components=k_fit, svd_solver="full", random_state=rng_seed).fit(Zg)
        pca_p = PCA(n_components=k_fit, svd_solver="full", random_state=rng_seed).fit(Zp)

        k = _choose_k_by_gt_evr(pca_g.explained_variance_ratio_, var_target=var_target, k_min=k_min, k_max=k_fit)
        if k is None or k < 2:
            continue

        k_used[s] = float(k)
        evr_gt_agg[s, :k_fit] = pca_g.explained_variance_ratio_[:k_fit]

        Ug = _orth_rows(pca_g.components_[:k])   # (k x R)
        Up = _orth_rows(pca_p.components_[:k])   # (k x R)
        Ug_store[s] = Ug

        Sig_p = _cov(Zp)

        proj = _proj_fraction(Sig_p, Ug)
        cosv = _principal_cos(Ug, Up)
        mc   = float(np.mean(cosv))
        mic  = float(np.min(cosv))

        proj_obs[s] = proj
        mean_cos[s] = mc
        min_cos[s]  = mic

        comp_obs[s] = float(
            w_proj * proj +
            w_mean * mc +
            w_min  * (mic ** float(min_exp))
        )

    # coverage over sequences (composite defined)
    coverage_seq = float(np.mean(np.isfinite(comp_obs)))

    # -------------------------
    # B) MISMATCH baseline per seq (j != s), SAME validity mask logic
    # -------------------------
    mis_comp = np.full((mismatch_M, n_seq), np.nan, dtype=np.float64)

    all_idx = np.arange(n_seq)
    for s in range(n_seq):
        if Ug_store[s] is None or not np.isfinite(comp_obs[s]):
            continue

        pool = all_idx[all_idx != s]
        M = int(min(mismatch_M, pool.size))
        js = rng.choice(pool, size=M, replace=False)
        if M < mismatch_M:
            js2 = rng.choice(pool, size=(mismatch_M - M), replace=True)
            js = np.concatenate([js, js2])

        Ug = Ug_store[s]
        k  = int(k_used[s])

        for m_i in range(mismatch_M):
            j = int(js[m_i])

            Xg, Xp = _pair_valid_rows(gt_arr[s], pred_arr[j])   # IMPORTANT: pairwise mask uses pred[j]
            if Xg.shape[0] < min_samples:
                continue

            if Xg.shape[0] > max_time:
                idx = rng.choice(Xg.shape[0], size=max_time, replace=False)
                Xg = Xg[idx]
                Xp = Xp[idx]

            Zg, Zp = _gt_standardize(Xg, Xp, eps=1e-8)

            k_fit = int(min(k_max, R, Zg.shape[0] - 1))
            if k_fit < 2:
                continue

            # Use SAME k derived from GT(s), but clamp if not feasible for this pair
            k_use = int(min(k, k_fit))
            if k_use < 2:
                continue

            # Need Pred PCA subspace at k_use for angle-based terms
            pca_p = PCA(n_components=k_fit, svd_solver="full", random_state=rng_seed).fit(Zp)
            Up = _orth_rows(pca_p.components_[:k_use])

            Sig_p = _cov(Zp)

            proj = _proj_fraction(Sig_p, Ug[:k_use])
            cosv = _principal_cos(Ug[:k_use], Up)
            mc   = float(np.mean(cosv))
            mic  = float(np.min(cosv))

            mis_comp[m_i, s] = float(
                w_proj * proj +
                w_mean * mc +
                w_min  * (mic ** float(min_exp))
            )

    # -------------------------
    # C) Robust z + non-saturating 0–1 score
    # -------------------------
    mis_med = np.nanmedian(mis_comp, axis=0)  # (n_seq,)
    mis_s   = np.array([_robust_scale_mad(mis_comp[:, s]) for s in range(n_seq)], dtype=np.float64)
    z_comp  = (comp_obs - mis_med) / (mis_s + EPS)

    score01 = floor + (cap - floor) * _sigmoid_stable((z_comp - z0) / (slope + EPS))

    # -------------------------
    # D) Benchmark scalars
    # -------------------------
    scores_for_composite = {
        "PCA_coverage_seq":          coverage_seq,
        "PCA_comp_raw_mean":         float(np.nanmean(comp_obs)),
        "PCA_comp_raw_q10":          _safe_nanquantile(comp_obs, 0.10),
        "PCA_comp_z_score01_mean":   float(np.nanmean(score01)),
        "PCA_comp_z_score01_q10":    _safe_nanquantile(score01, 0.10),
        # optional diagnostics (still useful)
        "PCA_proj_mean":             float(np.nanmean(proj_obs)),
        "PCA_mean_cos_mean":         float(np.nanmean(mean_cos)),
        "PCA_min_cos_mean":          float(np.nanmean(min_cos)),
        "PCA_k_used_mean":           float(np.nanmean(k_used)),
    }

    print("\n=== PCA realism (GT vs Pred) — GT-anchored PCA + mismatch-corrected 0–1 (benchmark v1) ===")
    print(f"n_seq={n_seq} | n_reg={R} | coverage(seq)={coverage_seq*100:.1f}%")
    print(f"validity: min_samples={min_samples} | max_time={max_time} | var_target={var_target:.2f}")
    print(f"k: k_min={k_min} | k_max={k_max} | min_exp={min_exp}")
    print(f"mismatch_M={mismatch_M}")
    print(f"mapping (non-saturating): floor={floor} cap={cap} | z0={z0} | slope={slope}")

    print("\n--- Benchmark scalars (USE THESE for composite) ---")
    for k in sorted(scores_for_composite.keys()):
        v = scores_for_composite[k]
        print(f"{k:28s}: {v:.4f}" if np.isfinite(v) else f"{k:28s}: nan")

    # Per-seq table
    per_seq_df = pd.DataFrame({
        "seq": np.arange(n_seq),
        "k_used": k_used,
        "proj_pred_on_gt": proj_obs,
        "mean_cos": mean_cos,
        "min_cos": min_cos,
        "comp_raw": comp_obs,
        "mis_med": mis_med,
        "z_comp": z_comp,
        "score01": score01,
    }).sort_values("score01")

    print("\nWorst sequences by PCA score01 (lowest):")
    for _, row in per_seq_df.head(5).iterrows():
        print(f"  seq={int(row['seq']):4d}  score01={row['score01']:.4f}  comp_raw={row['comp_raw']:.4f}  z={row['z_comp']:.3f}  k={int(row['k_used']) if np.isfinite(row['k_used']) else -1}")

    return {
        "per_seq": per_seq_df,
        "scores_for_composite": scores_for_composite,
        "arrays": dict(
            k_used=k_used,
            proj_pred_on_gt=proj_obs,
            mean_cos=mean_cos,
            min_cos=min_cos,
            comp_raw=comp_obs,
            z_comp=z_comp,
            score01=score01,
            mismatch_comp=mis_comp,
            mismatch_med=mis_med,
            mismatch_scale=mis_s,
            evr_gt=evr_gt_agg,
        ),
        "params": dict(
            min_samples=min_samples, max_time=max_time, rng_seed=rng_seed,
            var_target=var_target, k_min=k_min, k_max=k_max,
            w_proj=w_proj, w_mean=w_mean, w_min=w_min, min_exp=min_exp,
            mismatch_M=mismatch_M,
            floor=floor, cap=cap, z0=z0, slope=slope,
            use_resampled=use_resampled,
        ),
        "region_names": region_names,
    }

# ----------------------------
# 4) RUN
# ----------------------------
_required = ["gt_array_denorm", "data_predicted"]
_missing = [v for v in _required if v not in globals()]
if _missing:
    raise RuntimeError(f"Missing variables: {_missing}")

_region_names = None
if "pred_region_names" in globals() and globals().get("pred_region_names") is not None:
    _region_names = globals()["pred_region_names"]
elif "gt_regions" in globals() and globals().get("gt_regions") is not None:
    _region_names = globals()["gt_regions"]


if MODE in ('full', 'instant'):
    pca_realism_v1 = compute_pca_realism_benchmark_v1(
        gt_array_denorm,
        data_predicted,
        region_names=_region_names,
        use_resampled=True,
        # defaults chosen to match your other v1 cells
        min_samples=80,
        max_time=1200,
        rng_seed=0,
        var_target=0.90,
        k_min=5,
        k_max=20,
        mismatch_M=10,
        # non-saturating mapping
        floor=0.02,
        cap=0.98,
        z0=0.5,
        slope=3.0,
    )

    # ----------------------------
    # 5) Lightweight plots (benchmark-friendly)
    # ----------------------------
    score01 = pca_realism_v1["arrays"]["score01"]
    comp_raw = pca_realism_v1["arrays"]["comp_raw"]
    k_used = pca_realism_v1["arrays"]["k_used"]

    plt.figure(figsize=(6, 3.2))
    plt.hist(score01[np.isfinite(score01)], bins=20, alpha=0.8, edgecolor="k")
    plt.xlabel("PCA score01 (mismatch-corrected, bounded)")
    plt.ylabel("Count")
    plt.title("Per-seq PCA realism scores")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 3.2))
    plt.scatter(comp_raw[np.isfinite(comp_raw)], score01[np.isfinite(score01)], alpha=0.6)
    plt.xlabel("comp_raw (in [0,1])")
    plt.ylabel("score01 (mismatch-corrected)")
    plt.title("Raw composite vs mismatch-corrected score")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 3.2))
    plt.hist(k_used[np.isfinite(k_used)], bins=np.arange(1, 22) - 0.5, alpha=0.8, edgecolor="k")
    plt.xlabel("k used (chosen from GT EVR target)")
    plt.ylabel("Count")
    plt.title("Distribution of k across sequences")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    pca_realism_v1

# === AUTOCORR realism (GT vs Pred) — short+long; mismatch-corrected 0–1 (benchmark v1; BENCHMARK-CONSISTENT + PLOTS) ===
# FIXES (per your review):
#  1) Mismatch baseline uses the *same observed region-validity mask* per seq (rr_obs_short/long).
#     We only draw mismatch sequences j that are valid on *all* regions in rr_obs_{short,long} for that seq s.
#  2) Mapping uses the benchmark-consistent form:
#        score = floor + (cap-floor) * sigmoid((z - z0) / (slope + eps))
#     (larger slope => gentler / less saturation).
#  3) nRMSE denom is floored:
#        denom = max(rms(GT_autocorr), rms_floor)
#     applied consistently in observed and mismatch computations.
#
# ALSO: plots are integrated inside the benchmark function (toggle plot=True/False).

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.stats import pearsonr

# ---------------------------
# Benchmark function
# ---------------------------
def run_autocorr_realism_benchmark_v1(
    gt_array_denorm,
    data_predicted,
    region_names=None,
    # alignment
    use_resampled=True,
    # validity + truncation
    min_seg_len=80,
    max_time=1200,
    min_lag=1,
    # lags
    max_lag_short=60,
    max_lag_long=600,
    # mismatch calibration
    mismatch_M=10,
    # core blending + strictness
    alpha_short=0.60,      # short vs long
    beta_tail=0.50,        # strict tail penalty for printed scalars (q10/q90)
    lam_shape=0.60,        # z blend: shape vs magnitude
    # scale safety
    rms_floor=0.05,        # floors GT autocorr RMS denom for nRMSE
    # mismatch scale pooling
    pooled_scale=True,
    scale_floor_r=0.05,    # pooled scale floors (z-space calibration)
    scale_floor_e=0.10,
    # mapping (benchmark-consistent divide form)
    floor=0.02,
    cap=0.98,
    z0=0.5,
    slope=3.0,             # larger => gentler (less saturation)
    ddof_z=1,
    # plotting
    plot=True,
    show_long_heatmaps=False,
    bins=25,
    eps=1e-12,
    random_seed=0,
):
    # ---------------------------
    # helpers
    # ---------------------------
    def _nanmean(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else np.nan

    def _nanq(x, q):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return float(np.quantile(x, q)) if x.size else np.nan

    def _safe_pearson(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 5:
            return np.nan
        aa, bb = a[m], b[m]
        if np.std(aa) == 0.0 or np.std(bb) == 0.0:
            return np.nan
        return float(pearsonr(aa, bb)[0])

    def _rmse(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 5:
            return np.nan
        d = a[m] - b[m]
        return float(np.sqrt(np.mean(d * d)))

    def _rms(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size < 5:
            return np.nan
        return float(np.sqrt(np.mean(x * x)))

    def _autocorr_1d(x, L):
        """Returns autocorr[0:L] for finite segment; z-scored; normalized by N (not by lag count)."""
        x = np.asarray(x, float)
        m = np.isfinite(x)
        if int(m.sum()) < int(min_seg_len):
            return None
        x = x[m]
        mu = float(np.mean(x))
        sd = float(np.std(x))
        if not np.isfinite(sd) or sd < 1e-12:
            return None
        x = (x - mu) / sd
        c = correlate(x, x, mode="full", method="auto") / float(len(x))
        mid = len(c) // 2
        L_eff = min(L, len(c) - mid)
        if L_eff <= 1:
            return None
        return c[mid:mid + L_eff].astype(np.float64, copy=False)

    def _autocorr_tensor(data, L):
        nS, nT, nR = data.shape
        L_eff = min(L, nT)
        out = np.full((nS, nR, L_eff), np.nan, dtype=np.float64)
        valid = np.zeros((nS, nR), dtype=bool)
        for s in range(nS):
            for r in range(nR):
                ac = _autocorr_1d(data[s, :, r], L_eff)
                if ac is None:
                    continue
                out[s, r, :ac.size] = ac
                # validity: enough finite lags after min_lag mask
                vv = np.isfinite(ac[min_lag:]).sum()
                valid[s, r] = (vv >= 5)
        return out, valid

    def _nrmse_ac(gt_ac_vec, pr_ac_vec):
        """nRMSE over lag>=min_lag with denom=max(rms(gt), rms_floor)."""
        a = np.asarray(gt_ac_vec, float)
        b = np.asarray(pr_ac_vec, float)
        use = slice(int(min_lag), None)
        denom = _rms(a[use])
        if not np.isfinite(denom):
            return np.nan
        denom = max(float(denom), float(rms_floor))
        err = _rmse(a[use], b[use])
        if not np.isfinite(err):
            return np.nan
        return float(err / (denom + eps))

    def _map_z_to_01(z):
        """Benchmark-consistent: larger slope => gentler."""
        if not np.isfinite(z):
            return np.nan
        u = 1.0 / (1.0 + np.exp(-(float(z) - float(z0)) / (float(slope) + eps)))
        return float(floor + (cap - floor) * u)

    def _robust_scale(v):
        """MAD -> robust std."""
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size < 10:
            return np.nan
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        sc = 1.4826 * mad
        return float(sc) if np.isfinite(sc) else np.nan

    rng = np.random.default_rng(int(random_seed))

    # ---------------------------
    # align + truncate
    # ---------------------------
    gt = np.asarray(gt_array_denorm, dtype=np.float64)
    pr = np.asarray(data_predicted, dtype=np.float64)

    if use_resampled and "_resample_pred_to_gt" in globals():
        gt, pr = _resample_pred_to_gt(gt, pr)

    if gt.shape != pr.shape or gt.ndim != 3:
        raise ValueError(f"Expected aligned gt/pred [n_seq,T,R]. Got {gt.shape} vs {pr.shape}")

    n_seq, T, n_reg = gt.shape

    # choose effective time used (ensure long lag possible when available, but don't exceed max_time)
    max_time_eff = int(min(T, int(max_time)))
    gt = gt[:, :max_time_eff, :]
    pr = pr[:, :max_time_eff, :]

    # effective lags (need <= max_time_eff - 1)
    Ls = int(min(max_lag_short, max_time_eff))
    Ll = int(min(max_lag_long,  max_time_eff))
    # we will report "short_lag = Ls-1" etc
    if Ls < (min_lag + 6) or Ll < (min_lag + 6):
        raise ValueError(f"Not enough timepoints after truncation: max_time_eff={max_time_eff}, Ls={Ls}, Ll={Ll}")

    if region_names is None:
        if "gt_regions" in globals():
            region_names = list(gt_regions)
        else:
            region_names = [f"R{i}" for i in range(n_reg)]
    region_names = list(region_names)

    # ---------------------------
    # autocorr tensors + per-seq/per-reg validity (GT-only and Pred-only)
    # ---------------------------
    gt_s, gt_valid_s = _autocorr_tensor(gt, Ls)
    pr_s, pr_valid_s = _autocorr_tensor(pr, Ls)

    gt_l, gt_valid_l = _autocorr_tensor(gt, Ll)
    pr_l, pr_valid_l = _autocorr_tensor(pr, Ll)

    # observed region validity masks per seq (FIX #1)
    valid_obs_sr_short = gt_valid_s & pr_valid_s
    valid_obs_sr_long  = gt_valid_l & pr_valid_l

    # ---------------------------
    # observed per-(seq,reg) metrics on SHORT/LONG
    # ---------------------------
    r_sr_s    = np.full((n_seq, n_reg), np.nan)
    nrmse_sr_s = np.full((n_seq, n_reg), np.nan)

    r_sr_l    = np.full((n_seq, n_reg), np.nan)
    nrmse_sr_l = np.full((n_seq, n_reg), np.nan)

    for s in range(n_seq):
        for r in range(n_reg):
            if valid_obs_sr_short[s, r]:
                r_sr_s[s, r] = _safe_pearson(gt_s[s, r, min_lag:], pr_s[s, r, min_lag:])
                nrmse_sr_s[s, r] = _nrmse_ac(gt_s[s, r, :], pr_s[s, r, :])
            if valid_obs_sr_long[s, r]:
                r_sr_l[s, r] = _safe_pearson(gt_l[s, r, min_lag:], pr_l[s, r, min_lag:])
                nrmse_sr_l[s, r] = _nrmse_ac(gt_l[s, r, :], pr_l[s, r, :])

    # per-seq observed (avg over *observed-valid regions only*)
    r_seq_s     = np.full(n_seq, np.nan)
    nrmse_seq_s = np.full(n_seq, np.nan)
    r_seq_l     = np.full(n_seq, np.nan)
    nrmse_seq_l = np.full(n_seq, np.nan)

    nreg_obs_s  = np.zeros(n_seq, dtype=int)
    nreg_obs_l  = np.zeros(n_seq, dtype=int)

    for s in range(n_seq):
        rr = np.where(valid_obs_sr_short[s])[0]
        nreg_obs_s[s] = rr.size
        if rr.size:
            r_seq_s[s]     = float(np.nanmean(r_sr_s[s, rr]))
            nrmse_seq_s[s] = float(np.nanmean(nrmse_sr_s[s, rr]))

        rr = np.where(valid_obs_sr_long[s])[0]
        nreg_obs_l[s] = rr.size
        if rr.size:
            r_seq_l[s]     = float(np.nanmean(r_sr_l[s, rr]))
            nrmse_seq_l[s] = float(np.nanmean(nrmse_sr_l[s, rr]))

    # core observed (short+long)
    r_core_obs     = alpha_short * r_seq_s     + (1 - alpha_short) * r_seq_l
    nrmse_core_obs = alpha_short * nrmse_seq_s + (1 - alpha_short) * nrmse_seq_l

    # coverage (seq-level)
    coverage_seq = float(np.mean(np.isfinite(r_core_obs) & np.isfinite(nrmse_core_obs))) if n_seq else 0.0

    # ---------------------------
    # mismatch baseline (FIX #1): for each seq s, draw js valid on rr_obs_{short,long}
    # ---------------------------
    # precompute pred-valid masks (independent) for fast eligibility checks
    pred_valid_short = pr_valid_s
    pred_valid_long  = pr_valid_l

    r_core_mis     = np.full((n_seq, mismatch_M), np.nan)
    nrmse_core_mis = np.full((n_seq, mismatch_M), np.nan)

    for s in range(n_seq):
        rr_s = np.where(valid_obs_sr_short[s])[0]
        rr_l = np.where(valid_obs_sr_long[s])[0]

        # require at least a few regions to be meaningful; else skip seq
        if rr_s.size < 1 or rr_l.size < 1:
            continue

        # eligible mismatch js must be valid on all observed regions for that seq & level
        # (keeps region set identical for mismatches)
        eligible = []
        for j in range(n_seq):
            if j == s:
                continue
            if rr_s.size and not np.all(pred_valid_short[j, rr_s]):
                continue
            if rr_l.size and not np.all(pred_valid_long[j, rr_l]):
                continue
            eligible.append(j)

        if len(eligible) == 0:
            continue

        # sample mismatch js (with replacement if needed)
        js = rng.choice(eligible, size=mismatch_M, replace=(len(eligible) < mismatch_M))

        for mi, j in enumerate(js):
            # SHORT mismatch: GT(s) vs Pred(j) over rr_s
            r_m_s = np.nan
            e_m_s = np.nan
            if rr_s.size:
                # per-region values (same region set rr_s; all valid by construction)
                rr = rr_s
                r_vals = []
                e_vals = []
                for r in rr:
                    r_vals.append(_safe_pearson(gt_s[s, r, min_lag:], pr_s[j, r, min_lag:]))
                    e_vals.append(_nrmse_ac(gt_s[s, r, :], pr_s[j, r, :]))
                r_m_s = float(np.nanmean(r_vals))
                e_m_s = float(np.nanmean(e_vals))

            # LONG mismatch: GT(s) vs Pred(j) over rr_l
            r_m_l = np.nan
            e_m_l = np.nan
            if rr_l.size:
                rr = rr_l
                r_vals = []
                e_vals = []
                for r in rr:
                    r_vals.append(_safe_pearson(gt_l[s, r, min_lag:], pr_l[j, r, min_lag:]))
                    e_vals.append(_nrmse_ac(gt_l[s, r, :], pr_l[j, r, :]))
                r_m_l = float(np.nanmean(r_vals))
                e_m_l = float(np.nanmean(e_vals))

            # core mismatch
            r_core_mis[s, mi]     = alpha_short * r_m_s + (1 - alpha_short) * r_m_l
            nrmse_core_mis[s, mi] = alpha_short * e_m_s + (1 - alpha_short) * e_m_l

    # mismatch mean per seq (for centering)
    mis_r_mean_seq = np.nanmean(r_core_mis, axis=1)
    mis_e_mean_seq = np.nanmean(nrmse_core_mis, axis=1)

    # pooled (or per-seq) scales for z (with floors)
    if pooled_scale:
        sc_r = _robust_scale(r_core_mis.ravel())
        sc_e = _robust_scale(nrmse_core_mis.ravel())
        if not np.isfinite(sc_r): sc_r = np.nanstd(r_core_mis.ravel())
        if not np.isfinite(sc_e): sc_e = np.nanstd(nrmse_core_mis.ravel())
        sc_r = float(max(scale_floor_r, sc_r if np.isfinite(sc_r) else scale_floor_r))
        sc_e = float(max(scale_floor_e, sc_e if np.isfinite(sc_e) else scale_floor_e))
        sc_r_seq = np.full(n_seq, sc_r, dtype=float)
        sc_e_seq = np.full(n_seq, sc_e, dtype=float)
    else:
        sc_r_seq = np.full(n_seq, np.nan)
        sc_e_seq = np.full(n_seq, np.nan)
        for s in range(n_seq):
            sc_r_s = _robust_scale(r_core_mis[s])
            sc_e_s = _robust_scale(nrmse_core_mis[s])
            if not np.isfinite(sc_r_s): sc_r_s = np.nanstd(r_core_mis[s])
            if not np.isfinite(sc_e_s): sc_e_s = np.nanstd(nrmse_core_mis[s])
            sc_r_seq[s] = float(max(scale_floor_r, sc_r_s if np.isfinite(sc_r_s) else scale_floor_r))
            sc_e_seq[s] = float(max(scale_floor_e, sc_e_s if np.isfinite(sc_e_s) else scale_floor_e))

    # z scores: higher = better
    # shape: compare to mismatch mean
    z_r = (r_core_obs - mis_r_mean_seq) / (sc_r_seq + eps)

    # error: lower is better => invert
    z_e = (mis_e_mean_seq - nrmse_core_obs) / (sc_e_seq + eps)

    # combined
    z_core = lam_shape * z_r + (1 - lam_shape) * z_e

    # map to [0,1] (FIX #2)
    score01_core = np.array([_map_z_to_01(z) for z in z_core], dtype=np.float64)

    # strict summaries (tail penalties) for printed scalars
    AUTO_core_r_mean  = _nanmean(r_core_obs)
    AUTO_core_r_q10   = _nanq(r_core_obs, 0.10)
    AUTO_core_e_mean  = _nanmean(nrmse_core_obs)
    AUTO_core_e_q90   = _nanq(nrmse_core_obs, 0.90)

    AUTO_score01_mean = _nanmean(score01_core)
    AUTO_score01_q10  = _nanq(score01_core, 0.10)

    scalars = dict(
        AUTO_coverage_seq=coverage_seq,
        AUTO_core_r_mean=AUTO_core_r_mean,
        AUTO_core_r_q10=AUTO_core_r_q10,
        AUTO_core_nRMSE_mean=AUTO_core_e_mean,
        AUTO_core_nRMSE_q90=AUTO_core_e_q90,
        AUTO_core_score01_mean=AUTO_score01_mean,
        AUTO_core_score01_q10=AUTO_score01_q10,
    )

    # ---------------------------
    # PRINT SUMMARY (benchmark-style)
    # ---------------------------
    print(f"=== AUTOCORR realism (GT vs Pred) — short+long; mismatch-corrected 0–1 (benchmark v1; FIXED) ===")
    print(f"n_seq={n_seq} | n_reg={n_reg} | min_seg_len={min_seg_len} | min_lag={min_lag} | max_time={max_time_eff}")
    print(f"short_lag={Ls-1} | long_lag={Ll-1} | mismatch_M={mismatch_M}")
    print(f"alpha_short={alpha_short:.2f} | beta_tail={beta_tail:.2f} | lam_shape={lam_shape:.2f}")
    print(f"mapping (benchmark-consistent): floor={floor} cap={cap} | z0={z0} | slope={slope} | ddof={ddof_z}")
    if pooled_scale:
        print(f"pooled scales: sc_r={float(sc_r_seq[0]):.4f} (floor {scale_floor_r}) | sc_e={float(sc_e_seq[0]):.4f} (floor {scale_floor_e}) | rms_floor={rms_floor}")
    else:
        print(f"per-seq scales: r_floor={scale_floor_r} | e_floor={scale_floor_e} | rms_floor={rms_floor}")

    print("\n--- Benchmark scalars (USE THESE for composite) ---")
    for k in sorted(scalars.keys()):
        print(f"{k:26s}: {scalars[k]:.4f}")

    # worst sequences by score01
    idx = np.argsort(score01_core)
    print("\nWorst sequences by AUTO_core_score01 (lowest):")
    shown = 0
    for s in idx[:10]:
        if not np.isfinite(score01_core[s]):
            continue
        print(f"  seq={s:4d}  score01={score01_core[s]:.4f}  z_r={z_r[s]:+.2f} z_e={z_e[s]:+.2f}  r={r_core_obs[s]:.3f}  nRMSE={nrmse_core_obs[s]:.3f}  nreg_short={nreg_obs_s[s]} nreg_long={nreg_obs_l[s]}")
        shown += 1
        if shown >= 5:
            break

    # ---------------------------
    # PLOTS (integrated)
    # ---------------------------
    if plot:
        # 1) score distribution + mapping curve
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))

        x = score01_core[np.isfinite(score01_core)]
        ax[0].hist(x, bins=bins, alpha=0.8, edgecolor="k")
        ax[0].axvline(_nanmean(x), ls="--", lw=1, label=f"mean={_nanmean(x):.3f}")
        ax[0].axvline(_nanq(x, 0.10), ls=":", lw=1, label=f"q10={_nanq(x,0.10):.3f}")
        ax[0].set_title("AUTO core score01 distribution")
        ax[0].set_xlabel("score01"); ax[0].set_ylabel("count")
        ax[0].grid(alpha=0.25); ax[0].legend(frameon=False)

        # z distribution
        zf = z_core[np.isfinite(z_core)]
        ax[1].hist(zf, bins=bins, alpha=0.8, edgecolor="k")
        ax[1].axvline(_nanmean(zf), ls="--", lw=1, label=f"mean={_nanmean(zf):.2f}")
        ax[1].axvline(z0, ls=":", lw=1, label=f"z0={z0}")
        ax[1].set_title("Mismatch-corrected z (core)")
        ax[1].set_xlabel("z"); ax[1].set_ylabel("count")
        ax[1].grid(alpha=0.25); ax[1].legend(frameon=False)

        # mapping curve (FIX #2 divide form)
        zs = np.linspace(-4, 4, 401)
        scores = np.array([_map_z_to_01(zz) for zz in zs])
        ax[2].plot(zs, scores, lw=2)
        ax[2].axvline(z0, ls="--", lw=1)
        ax[2].set_ylim(0, 1)
        ax[2].set_title("z → score01 mapping")
        ax[2].set_xlabel("z"); ax[2].set_ylabel("score01")
        ax[2].grid(alpha=0.25)

        plt.tight_layout()
        plt.show()

        # 2) per-seq scatter: r vs nRMSE colored by score
        r = r_core_obs.copy()
        e = nrmse_core_obs.copy()
        m = np.isfinite(r) & np.isfinite(e) & np.isfinite(score01_core)
        if np.any(m):
            plt.figure(figsize=(6.5, 5))
            sc = plt.scatter(r[m], e[m], c=score01_core[m], s=18, alpha=0.85)
            cb = plt.colorbar(sc)
            cb.set_label("score01")
            plt.xlabel("core r (shape)")
            plt.ylabel("core nRMSE (magnitude)")
            plt.title("Per-seq core: shape vs magnitude (colored by score01)")
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.show()

        # 3) per-seq r and nRMSE distributions
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        rr = r_core_obs[np.isfinite(r_core_obs)]
        ee = nrmse_core_obs[np.isfinite(nrmse_core_obs)]

        ax[0].hist(rr, bins=bins, alpha=0.8, edgecolor="k")
        ax[0].axvline(_nanmean(rr), ls="--", lw=1)
        ax[0].axvline(_nanq(rr, 0.10), ls=":", lw=1)
        ax[0].set_title("core r distribution")
        ax[0].set_xlabel("r"); ax[0].set_ylabel("count")
        ax[0].grid(alpha=0.25)

        ax[1].hist(ee, bins=bins, alpha=0.8, edgecolor="k")
        ax[1].axvline(_nanmean(ee), ls="--", lw=1)
        ax[1].axvline(_nanq(ee, 0.90), ls=":", lw=1)
        ax[1].set_title("core nRMSE distribution")
        ax[1].set_xlabel("nRMSE"); ax[1].set_ylabel("count")
        ax[1].grid(alpha=0.25)

        plt.tight_layout()
        plt.show()

        # 4) mean autocorr curves (short & long)
        def _mean_ac(ac_tensor):
            # mean over seq+reg
            return np.nanmean(ac_tensor, axis=(0, 1))

        m_gt_s = _mean_ac(gt_s); m_pr_s = _mean_ac(pr_s)
        lags_s = np.arange(m_gt_s.size)

        plt.figure(figsize=(7, 4))
        plt.plot(lags_s, m_gt_s, label="GT")
        plt.plot(lags_s, m_pr_s, label="Pred")
        plt.axvline(min_lag, ls=":", alpha=0.6)
        plt.title("Mean autocorr (SHORT)")
        plt.xlabel("lag"); plt.ylabel("autocorr")
        plt.grid(alpha=0.25); plt.legend(frameon=False); plt.tight_layout()
        plt.show()

        m_gt_l = _mean_ac(gt_l); m_pr_l = _mean_ac(pr_l)
        lags_l = np.arange(m_gt_l.size)

        plt.figure(figsize=(7, 4))
        plt.plot(lags_l, m_gt_l, label="GT")
        plt.plot(lags_l, m_pr_l, label="Pred")
        plt.axvline(min_lag, ls=":", alpha=0.6)
        plt.title("Mean autocorr (LONG)")
        plt.xlabel("lag"); plt.ylabel("autocorr")
        plt.grid(alpha=0.25); plt.legend(frameon=False); plt.tight_layout()
        plt.show()

        # 5) region×lag heatmaps (SHORT; LONG optional)
        reg_gt_s = np.nanmean(gt_s, axis=0)  # [reg, lag]
        reg_pr_s = np.nanmean(pr_s, axis=0)
        reg_df_s = reg_pr_s - reg_gt_s

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        im0 = axes[0].imshow(reg_gt_s, aspect="auto")
        axes[0].set_title("GT mean autocorr (reg×lag, SHORT)")
        axes[0].set_xlabel("lag"); axes[0].set_ylabel("region")

        im1 = axes[1].imshow(reg_pr_s, aspect="auto")
        axes[1].set_title("Pred mean autocorr (reg×lag, SHORT)")
        axes[1].set_xlabel("lag")

        im2 = axes[2].imshow(reg_df_s, aspect="auto")
        axes[2].set_title("Pred − GT (reg×lag, SHORT)")
        axes[2].set_xlabel("lag")

        axes[0].set_yticks(np.arange(n_reg))
        axes[0].set_yticklabels(region_names, fontsize=7)

        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

        if show_long_heatmaps:
            reg_gt_l = np.nanmean(gt_l, axis=0)
            reg_pr_l = np.nanmean(pr_l, axis=0)
            reg_df_l = reg_pr_l - reg_gt_l

            fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
            im0 = axes[0].imshow(reg_gt_l, aspect="auto")
            axes[0].set_title("GT mean autocorr (reg×lag, LONG)")
            axes[0].set_xlabel("lag"); axes[0].set_ylabel("region")

            im1 = axes[1].imshow(reg_pr_l, aspect="auto")
            axes[1].set_title("Pred mean autocorr (reg×lag, LONG)")
            axes[1].set_xlabel("lag")

            im2 = axes[2].imshow(reg_df_l, aspect="auto")
            axes[2].set_title("Pred − GT (reg×lag, LONG)")
            axes[2].set_xlabel("lag")

            axes[0].set_yticks(np.arange(n_reg))
            axes[0].set_yticklabels(region_names, fontsize=7)

            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

    # ---------------------------
    # return
    # ---------------------------
    out = dict(
        scalars=scalars,
        # observed
        r_core_obs=r_core_obs,
        nrmse_core_obs=nrmse_core_obs,
        z_r=z_r,
        z_e=z_e,
        z_core=z_core,
        score01_core=score01_core,
        # mismatch
        r_core_mis=r_core_mis,
        nrmse_core_mis=nrmse_core_mis,
        mis_r_mean_seq=mis_r_mean_seq,
        mis_e_mean_seq=mis_e_mean_seq,
        sc_r_seq=sc_r_seq,
        sc_e_seq=sc_e_seq,
        # validity masks (FIX #1)
        valid_obs_sr_short=valid_obs_sr_short,
        valid_obs_sr_long=valid_obs_sr_long,
        nreg_obs_short=nreg_obs_s,
        nreg_obs_long=nreg_obs_l,
        # tensors (optional downstream)
        gt_ac_short=gt_s,
        pr_ac_short=pr_s,
        gt_ac_long=gt_l,
        pr_ac_long=pr_l,
        params=dict(
            min_seg_len=min_seg_len,
            max_time=max_time_eff,
            min_lag=min_lag,
            max_lag_short=Ls,
            max_lag_long=Ll,
            mismatch_M=mismatch_M,
            alpha_short=alpha_short,
            beta_tail=beta_tail,
            lam_shape=lam_shape,
            rms_floor=rms_floor,
            pooled_scale=pooled_scale,
            scale_floor_r=scale_floor_r,
            scale_floor_e=scale_floor_e,
            floor=floor,
            cap=cap,
            z0=z0,
            slope=slope,
            ddof_z=ddof_z,
            random_seed=random_seed,
        ),
    )
    return out


# ---------------------------
# RUN (single-call benchmark cell)
# ---------------------------

if MODE == 'full':
    autocorr_realism_summary = run_autocorr_realism_benchmark_v1(
        gt_array_denorm=gt_array_denorm,
        data_predicted=data_predicted,
        region_names=(gt_regions if "gt_regions" in globals() else None),
        use_resampled=True,
        # tune these if needed
        min_seg_len=80,
        max_time=1200,
        min_lag=1,
        max_lag_short=60,
        max_lag_long=600,
        mismatch_M=10,
        alpha_short=0.60,
        beta_tail=0.50,
        lam_shape=0.60,
        rms_floor=0.05,
        pooled_scale=True,
        scale_floor_r=0.05,
        scale_floor_e=0.10,
        floor=0.02,
        cap=0.98,
        z0=0.5,
        slope=3.0,     # larger => gentler (consistent with other v1 cells)
        ddof_z=1,
        plot=True,
        show_long_heatmaps=False,
        bins=25,
        random_seed=0,
    )

    autocorr_realism_summary

# === CROSSCORR REALISM (GT vs Pred) — mismatch-corrected 0–1 (benchmark v1; FIXED MASKS) ===
# Fixes vs prior version:
#  1) Observed (s,s) and mismatch (s,j) use the intended JOINT time mask:
#       - observed:   mask = finite(GT[s,i], GT[s,k], Pred[s,i], Pred[s,k])
#       - mismatch:   mask = finite(GT[s,i], GT[s,k], Pred[j,i], Pred[j,k])   ✅ (no GT[j] anywhere)
#  2) Mismatch baseline uses the SAME observed-valid pair set per seq s (shape and peak can differ).
#  3) Benchmark-consistent z->score mapping: sigmoid((z - z0)/(slope+eps)) with floor/cap.
#  4) Scale-safe pooled mismatch scales with floors (prevents tiny scale blow-ups).
#
# Outputs (benchmark scalars):
#   CC_core_score01_mean, CC_core_score01_q10, CC_coverage_seq,
#   CC_shape_r_mean/q10, CC_rmse_mean/q90, CC_dlag_mean/q90
#
# Requires in notebook:
#   - gt_array_denorm: [n_seq, T, R]
#   - data_predicted:  [n_seq, T, R]
#   - (optional) gt_regions: list[str] length R
#   - (optional) _resample_pred_to_gt(gt, pred)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.stats import pearsonr

# ---------------------------
# CONFIG (edit here)
# ---------------------------
max_lag = 12

ignore_lag0 = True
min_abs_lag = 1 if ignore_lag0 else 0

# validity
min_seg_len   = 80          # min joint-finite timepoints to compute a curve
min_lags_shape = 5          # min finite lags (after lag-mask) for shape/rmse
min_lags_peak  = 3          # min finite points on full curve for peak lag
max_time = 601              # optional cap for speed; set None to disable

# mismatch baseline
mismatch_M   = 10           # number of mismatched Pred sequences per GT seq
pooled_scale = True
ddof         = 1

# strictness
beta_tail = 0.5             # tail penalty weight inside summary stats (q10/q90)

# component -> core blend
# (these are *component weights* for the final core score in score-space after z-map)
w_r, w_e, w_l = 0.45, 0.35, 0.20  # must sum to 1

# z->score map (benchmark-consistent)
floor01 = 0.02
cap01   = 0.98
z0      = 0.5
slope   = 3.0                # larger => gentler (less saturation)

# mismatch scale floors
sc_r_floor = 0.05
sc_e_floor = 0.10
sc_l_floor = 0.20

# plotting
plot = True
bins = 30
region_grid_shape = (4, 4)   # assumes R=16; adjust if needed
show_region_grid = True
show_mapping_curves = True
show_seq_examples = False
n_seq_examples = 3

eps = 1e-9

# ---------------------------
# INPUTS / ALIGN
# ---------------------------
gt_arr = np.asarray(gt_array_denorm, dtype=np.float64)
pr_arr = np.asarray(data_predicted, dtype=np.float64)

if "_resample_pred_to_gt" in globals():
    gt_arr, pr_arr = _resample_pred_to_gt(gt_arr, pr_arr)

if gt_arr.shape != pr_arr.shape or gt_arr.ndim != 3:
    raise ValueError(f"Expected aligned gt/pred with shape [n_seq, T, R]. Got {gt_arr.shape} vs {pr_arr.shape}")

n_seq, T, R = gt_arr.shape
region_names = list(gt_regions) if "gt_regions" in globals() else [f"R{i}" for i in range(R)]

if max_time is not None:
    T_use = min(T, int(max_time))
    gt_arr = gt_arr[:, :T_use, :]
    pr_arr = pr_arr[:, :T_use, :]
    T = T_use

pairs = [(i, j) for i in range(R) for j in range(i + 1, R)]
n_pairs = len(pairs)

# lag axis length (same for all curves)
span = min(int(max_lag), T // 2)
lag_axis = np.arange(-span, span + 1, dtype=int)
L = lag_axis.size

# ---------------------------
# HELPERS
# ---------------------------
def _nanmean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan

def _nanmedian(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else np.nan

def _nanq(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else np.nan

def _safe_pearson_vec(a, b, min_n=5):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < int(min_n):
        return np.nan
    aa, bb = a[m], b[m]
    if np.std(aa) == 0.0 or np.std(bb) == 0.0:
        return np.nan
    return float(pearsonr(aa, bb)[0])

def _rmse(a, b, min_n=5):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < int(min_n):
        return np.nan
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d * d)))

def _mae(a, b, min_n=5):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < int(min_n):
        return np.nan
    d = a[m] - b[m]
    return float(np.mean(np.abs(d)))

def _zscore_1d(x, eps=1e-9):
    x = np.asarray(x, dtype=np.float64)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < eps:
        return None
    return (x - mu) / (sd + eps)

def _xcorr(zx, zy, span):
    corr = correlate(zx, zy, mode="full", method="auto")
    n = len(zx)
    corr = corr / max(1, n)
    mid = len(corr) // 2
    return corr[mid - span: mid + span + 1]

def _cross_corr_pair_1d(xg, yg, xp, yp, span, eps=1e-9):
    """
    Compute GT and Pred cross-corr curves under JOINT mask of all four series.
    IMPORTANT: For mismatch, pass (GT[s], Pred[j]) so GT[j] never appears in the mask.
    """
    xg = np.asarray(xg, dtype=np.float64)
    yg = np.asarray(yg, dtype=np.float64)
    xp = np.asarray(xp, dtype=np.float64)
    yp = np.asarray(yp, dtype=np.float64)

    m = np.isfinite(xg) & np.isfinite(yg) & np.isfinite(xp) & np.isfinite(yp)
    if m.sum() < int(min_seg_len):
        return None, None, int(m.sum())

    xg = xg[m]; yg = yg[m]
    xp = xp[m]; yp = yp[m]

    zxg = _zscore_1d(xg, eps=eps)
    zyg = _zscore_1d(yg, eps=eps)
    zxp = _zscore_1d(xp, eps=eps)
    zyp = _zscore_1d(yp, eps=eps)
    if zxg is None or zyg is None or zxp is None or zyp is None:
        return None, None, int(m.sum())

    ccg = _xcorr(zxg, zyg, span=span)
    ccp = _xcorr(zxp, zyp, span=span)
    return ccg, ccp, int(m.sum())

def _peak_lag_and_val(curve, lags):
    curve = np.asarray(curve, dtype=np.float64)
    m = np.isfinite(curve)
    if m.sum() < int(min_lags_peak):
        return np.nan, np.nan
    c = curve[m]
    l = lags[m]
    idx = int(np.argmax(np.abs(c)))
    return float(l[idx]), float(c[idx])

def _map_z_to_01(z, floor=floor01, cap=cap01, z0=z0, slope=slope, eps=1e-12):
    """
    Benchmark-consistent: floor + (cap-floor) * sigmoid((z - z0)/(slope+eps))
    where larger slope => gentler.
    """
    if not np.isfinite(z):
        return np.nan
    x = (float(z) - float(z0)) / (float(slope) + eps)
    sig = 1.0 / (1.0 + np.exp(-x))
    return float(floor + (cap - floor) * sig)

def _choose_mismatch_js(s, M, n_seq, rng):
    # sample without replacement from {0..n_seq-1}\{s}
    cand = np.arange(n_seq, dtype=int)
    cand = cand[cand != s]
    if cand.size == 0:
        return np.array([], dtype=int)
    M_eff = min(int(M), int(cand.size))
    return rng.choice(cand, size=M_eff, replace=False)

# ---------------------------
# OBSERVED CURVES + OBS-VALID PAIR SETS
# ---------------------------
use_shape = np.ones_like(lag_axis, dtype=bool)
if min_abs_lag > 0:
    use_shape &= (np.abs(lag_axis) >= int(min_abs_lag))

cc_gt_obs = np.full((n_seq, n_pairs, L), np.nan, dtype=np.float64)
cc_pr_obs = np.full((n_seq, n_pairs, L), np.nan, dtype=np.float64)

obs_valid_shape_sp = np.zeros((n_seq, n_pairs), dtype=bool)
obs_valid_peak_sp  = np.zeros((n_seq, n_pairs), dtype=bool)

# per-(s,p) peak lags for diagnostics
peak_lag_gt = np.full((n_seq, n_pairs), np.nan, dtype=np.float64)
peak_lag_pr = np.full((n_seq, n_pairs), np.nan, dtype=np.float64)

for s in range(n_seq):
    for p_idx, (i, j) in enumerate(pairs):
        ccg, ccp, n_joint = _cross_corr_pair_1d(
            gt_arr[s, :, i], gt_arr[s, :, j],
            pr_arr[s, :, i], pr_arr[s, :, j],
            span=span, eps=eps
        )
        if ccg is None:
            continue

        cc_gt_obs[s, p_idx, :] = ccg
        cc_pr_obs[s, p_idx, :] = ccp

        # observed-valid for shape: enough finite lags under shape mask
        a = np.asarray(ccg, dtype=np.float64)[use_shape]
        b = np.asarray(ccp, dtype=np.float64)[use_shape]
        if (np.isfinite(a) & np.isfinite(b)).sum() >= int(min_lags_shape):
            obs_valid_shape_sp[s, p_idx] = True

        # observed-valid for peak: enough finite points on full curve
        if np.isfinite(ccg).sum() >= int(min_lags_peak) and np.isfinite(ccp).sum() >= int(min_lags_peak):
            obs_valid_peak_sp[s, p_idx] = True
            lg, _ = _peak_lag_and_val(ccg, lag_axis)
            lp, _ = _peak_lag_and_val(ccp, lag_axis)
            peak_lag_gt[s, p_idx] = lg
            peak_lag_pr[s, p_idx] = lp

# ---------------------------
# OBSERVED PER-SEQ METRICS (aggregate over obs-valid pairs)
# ---------------------------
r_seq    = np.full(n_seq, np.nan, dtype=np.float64)
rmse_seq = np.full(n_seq, np.nan, dtype=np.float64)
dlag_seq = np.full(n_seq, np.nan, dtype=np.float64)

nPairs_shape_seq = np.zeros(n_seq, dtype=int)
nPairs_peak_seq  = np.zeros(n_seq, dtype=int)

for s in range(n_seq):
    shp = np.where(obs_valid_shape_sp[s])[0]
    pk  = np.where(obs_valid_peak_sp[s])[0]
    nPairs_shape_seq[s] = int(shp.size)
    nPairs_peak_seq[s]  = int(pk.size)

    if shp.size > 0:
        rs = []
        es = []
        for p in shp:
            a = cc_gt_obs[s, p, use_shape]
            b = cc_pr_obs[s, p, use_shape]
            rs.append(_safe_pearson_vec(a, b, min_n=min_lags_shape))
            es.append(_rmse(a, b, min_n=min_lags_shape))
        r_seq[s]    = float(np.nanmean(rs)) if np.isfinite(rs).any() else float(np.nanmean(rs))
        rmse_seq[s] = float(np.nanmean(es)) if np.isfinite(es).any() else float(np.nanmean(es))

    if pk.size > 0:
        ds = []
        for p in pk:
            lg = peak_lag_gt[s, p]
            lp = peak_lag_pr[s, p]
            if np.isfinite(lg) and np.isfinite(lp):
                ds.append(abs(lg - lp))
        dlag_seq[s] = float(np.nanmean(ds)) if len(ds) else np.nan

coverage_seq = float(np.mean(np.isfinite(r_seq))) if r_seq.size else 0.0

# strict summaries
r_mean   = _nanmean(r_seq)
r_q10    = _nanq(r_seq, 0.10)
r_strict = (1 - beta_tail) * r_mean + beta_tail * r_q10 if np.isfinite(r_mean) and np.isfinite(r_q10) else np.nan

rmse_mean   = _nanmean(rmse_seq)
rmse_q90    = _nanq(rmse_seq, 0.90)
rmse_strict = (1 - beta_tail) * rmse_mean + beta_tail * rmse_q90 if np.isfinite(rmse_mean) and np.isfinite(rmse_q90) else np.nan

dlag_mean   = _nanmean(dlag_seq)
dlag_q90    = _nanq(dlag_seq, 0.90)
dlag_strict = (1 - beta_tail) * dlag_mean + beta_tail * dlag_q90 if np.isfinite(dlag_mean) and np.isfinite(dlag_q90) else np.nan

# ---------------------------
# MISMATCH BASELINE (recompute curves under GT[s] + Pred[j] joint mask)
# ---------------------------
rng = np.random.default_rng(0)

def _mismatch_seq_stats(s, js):
    shp_idx = np.where(obs_valid_shape_sp[s])[0]
    pk_idx  = np.where(obs_valid_peak_sp[s])[0]

    r_j    = np.full(len(js), np.nan, dtype=np.float64)
    rmse_j = np.full(len(js), np.nan, dtype=np.float64)
    dlag_j = np.full(len(js), np.nan, dtype=np.float64)

    # quick local access for speed
    pi = np.array([ij[0] for ij in pairs], dtype=int)
    pj = np.array([ij[1] for ij in pairs], dtype=int)

    for t, j in enumerate(js):
        # shape/rmse mismatch for seq s vs pred j using ONLY observed-valid shape pairs of s
        if shp_idx.size > 0:
            rs = []
            es = []
            for p in shp_idx:
                i = pi[p]; k = pj[p]
                ccg, ccp, _ = _cross_corr_pair_1d(
                    gt_arr[s, :, i], gt_arr[s, :, k],
                    pr_arr[j, :, i], pr_arr[j, :, k],
                    span=span, eps=eps
                )
                if ccg is None:
                    continue
                a = np.asarray(ccg, dtype=np.float64)[use_shape]
                b = np.asarray(ccp, dtype=np.float64)[use_shape]
                if (np.isfinite(a) & np.isfinite(b)).sum() < int(min_lags_shape):
                    continue
                rs.append(_safe_pearson_vec(a, b, min_n=min_lags_shape))
                es.append(_rmse(a, b, min_n=min_lags_shape))
            if len(rs):
                r_j[t] = float(np.nanmean(rs))
            if len(es):
                rmse_j[t] = float(np.nanmean(es))

        # peak-lag mismatch for seq s vs pred j using ONLY observed-valid peak pairs of s
        if pk_idx.size > 0:
            ds = []
            for p in pk_idx:
                i = pi[p]; k = pj[p]
                ccg, ccp, _ = _cross_corr_pair_1d(
                    gt_arr[s, :, i], gt_arr[s, :, k],
                    pr_arr[j, :, i], pr_arr[j, :, k],
                    span=span, eps=eps
                )
                if ccg is None:
                    continue
                lg, _ = _peak_lag_and_val(ccg, lag_axis)
                lp, _ = _peak_lag_and_val(ccp, lag_axis)
                if np.isfinite(lg) and np.isfinite(lp):
                    ds.append(abs(lg - lp))
            if len(ds):
                dlag_j[t] = float(np.nanmean(ds))

    return r_j, rmse_j, dlag_j

# collect mismatch distributions (per seq) then pool scales
mis_mu_r = np.full(n_seq, np.nan)
mis_mu_e = np.full(n_seq, np.nan)
mis_mu_l = np.full(n_seq, np.nan)

mis_sc_r = np.full(n_seq, np.nan)
mis_sc_e = np.full(n_seq, np.nan)
mis_sc_l = np.full(n_seq, np.nan)

for s in range(n_seq):
    js = _choose_mismatch_js(s, mismatch_M, n_seq, rng)
    if js.size == 0:
        continue
    rj, ej, lj = _mismatch_seq_stats(s, js)

    # per-seq mismatch mean/std
    rjf = rj[np.isfinite(rj)]
    ejf = ej[np.isfinite(ej)]
    ljf = lj[np.isfinite(lj)]

    if rjf.size:
        mis_mu_r[s] = float(np.mean(rjf))
        mis_sc_r[s] = float(np.std(rjf, ddof=ddof)) if rjf.size > ddof else np.nan
    if ejf.size:
        mis_mu_e[s] = float(np.mean(ejf))
        mis_sc_e[s] = float(np.std(ejf, ddof=ddof)) if ejf.size > ddof else np.nan
    if ljf.size:
        mis_mu_l[s] = float(np.mean(ljf))
        mis_sc_l[s] = float(np.std(ljf, ddof=ddof)) if ljf.size > ddof else np.nan

# pooled scales + floors
def _pooled_scale(sc_arr, floor):
    v = sc_arr[np.isfinite(sc_arr)]
    if v.size == 0:
        return float(floor)
    return float(max(float(np.nanmean(v)), float(floor)))

sc_r = _pooled_scale(mis_sc_r, sc_r_floor) if pooled_scale else np.where(np.isfinite(mis_sc_r), np.maximum(mis_sc_r, sc_r_floor), sc_r_floor)
sc_e = _pooled_scale(mis_sc_e, sc_e_floor) if pooled_scale else np.where(np.isfinite(mis_sc_e), np.maximum(mis_sc_e, sc_e_floor), sc_e_floor)
sc_l = _pooled_scale(mis_sc_l, sc_l_floor) if pooled_scale else np.where(np.isfinite(mis_sc_l), np.maximum(mis_sc_l, sc_l_floor), sc_l_floor)

# ---------------------------
# Z-SCORES (higher is better) per seq
#  - r:    want higher than mismatch mean
#  - rmse: want lower than mismatch mean  => negate
#  - dlag: want lower than mismatch mean  => negate
# ---------------------------
z_r = (r_seq - mis_mu_r) / (sc_r + eps)
z_e = (mis_mu_e - rmse_seq) / (sc_e + eps)
z_l = (mis_mu_l - dlag_seq) / (sc_l + eps)

# component scores in [0,1]
s_r = np.array([_map_z_to_01(z) for z in z_r], dtype=np.float64)
s_e = np.array([_map_z_to_01(z) for z in z_e], dtype=np.float64)
s_l = np.array([_map_z_to_01(z) for z in z_l], dtype=np.float64)

# core score per seq
score_core = w_r * s_r + w_e * s_e + w_l * s_l

# aggregate benchmark scalars
CC_core_score01_mean = _nanmean(score_core)
CC_core_score01_q10  = _nanq(score_core, 0.10)

CC_shape_r_mean = r_mean
CC_shape_r_q10  = r_q10
CC_rmse_mean    = rmse_mean
CC_rmse_q90     = rmse_q90
CC_dlag_mean    = dlag_mean
CC_dlag_q90     = dlag_q90

# ---------------------------
# PRINT SUMMARY
# ---------------------------
print("=== CROSSCORR realism (GT vs Pred) — mismatch-corrected 0–1 (benchmark v1; FIXED MASKS) ===")
print(f"n_seq={n_seq} | n_reg={R} | n_pairs={n_pairs} | n_lag={L} | max_lag={max_lag}")
print(f"validity: min_seg_len={min_seg_len} | min_lags_shape={min_lags_shape} | min_lags_peak={min_lags_peak} | min_abs_lag={min_abs_lag} | max_time={T}")
print(f"mismatch_M={mismatch_M} | pooled_scale={pooled_scale} | ddof={ddof}")
print(f"pooled scales: sc_r={float(sc_r):.4f} (floor {sc_r_floor}) | sc_e={float(sc_e):.4f} (floor {sc_e_floor}) | sc_l={float(sc_l):.4f} (floor {sc_l_floor})")
print(f"mapping (benchmark-consistent): floor={floor01} cap={cap01} | z0={z0} | slope={slope}")
print("")
print("--- Benchmark scalars (USE THESE for composite) ---")
scalars = {
    "CC_core_score01_mean": CC_core_score01_mean,
    "CC_core_score01_q10":  CC_core_score01_q10,
    "CC_coverage_seq":      coverage_seq,
    "CC_shape_r_mean":      CC_shape_r_mean,
    "CC_shape_r_q10":       CC_shape_r_q10,
    "CC_rmse_mean":         CC_rmse_mean,
    "CC_rmse_q90":          CC_rmse_q90,
    "CC_dlag_mean":         CC_dlag_mean,
    "CC_dlag_q90":          CC_dlag_q90,
}
for k in sorted(scalars.keys()):
    v = scalars[k]
    print(f"{k:22s}: {v:.4f}" if np.isfinite(v) else f"{k:22s}: nan")

# worst sequences
k_show = 5
order = np.argsort(score_core)
print("\nWorst sequences by CC_core_score01 (lowest):")
for idx in order[:k_show]:
    print(f"  seq={idx:4d}  score01={score_core[idx]:.4f}  z_r={z_r[idx]:+.2f} z_e={z_e[idx]:+.2f} z_l={z_l[idx]:+.2f}  "
          f"r={r_seq[idx]:+.3f}  rmse={rmse_seq[idx]:.3f}  dlag={dlag_seq[idx]:.3f}  "
          f"nPairs_shape={nPairs_shape_seq[idx]} nPairs_peak={nPairs_peak_seq[idx]}")

# ---------------------------
# PLOTS (integrated)
# ---------------------------
if plot:
    # 1) mean curve across all seq & pairs (obs only)
    mean_gt = np.nanmean(cc_gt_obs, axis=(0, 1))
    std_gt  = np.nanstd(cc_gt_obs, axis=(0, 1))
    mean_pr = np.nanmean(cc_pr_obs, axis=(0, 1))
    std_pr  = np.nanstd(cc_pr_obs, axis=(0, 1))

    plt.figure(figsize=(6, 4))
    plt.plot(lag_axis, mean_gt, label="GT")
    plt.fill_between(lag_axis, mean_gt - std_gt, mean_gt + std_gt, alpha=0.2)
    plt.plot(lag_axis, mean_pr, label="Pred")
    plt.fill_between(lag_axis, mean_pr - std_pr, mean_pr + std_pr, alpha=0.2)
    plt.axvline(0, color="k", linestyle="--", lw=1, alpha=0.7)
    plt.title("Mean cross-corr across pairs (±1 std)")
    plt.xlabel("Lag (steps)"); plt.ylabel("Cross-corr")
    plt.grid(alpha=0.3); plt.legend(frameon=False); plt.tight_layout(); plt.show()

    # 2) per-seq histograms: r, rmse, dlag
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].hist(r_seq[np.isfinite(r_seq)], bins=bins, edgecolor="k", alpha=0.75)
    ax[0].axvline(r_mean, linestyle="--"); ax[0].axvline(r_q10, linestyle=":")
    ax[0].set_title("Per-seq shape r"); ax[0].set_xlabel("r"); ax[0].set_ylabel("count")

    ax[1].hist(rmse_seq[np.isfinite(rmse_seq)], bins=bins, edgecolor="k", alpha=0.75)
    ax[1].axvline(rmse_mean, linestyle="--"); ax[1].axvline(rmse_q90, linestyle=":")
    ax[1].set_title("Per-seq RMSE"); ax[1].set_xlabel("RMSE")

    ax[2].hist(dlag_seq[np.isfinite(dlag_seq)], bins=bins, edgecolor="k", alpha=0.75)
    ax[2].axvline(dlag_mean, linestyle="--"); ax[2].axvline(dlag_q90, linestyle=":")
    ax[2].set_title("Per-seq |Δ peak-lag|"); ax[2].set_xlabel("Δlag (steps)")
    plt.tight_layout(); plt.show()

    # 3) peak-lag heatmaps (GT/Pred/Diff), averaged over seq
    mean_pl_gt = nps = np.nanmean(peak_lag_gt, axis=0)  # [pair]
    mean_pl_pr = np.nanmean(peak_lag_pr, axis=0)

    PL_gt = np.full((R, R), np.nan)
    PL_pr = np.full((R, R), np.nan)
    PL_df = np.full((R, R), np.nan)

    for p_idx, (i, j) in enumerate(pairs):
        PL_gt[i, j] = mean_pl_gt[p_idx]; PL_gt[j, i] = mean_pl_gt[p_idx]
        PL_pr[i, j] = mean_pl_pr[p_idx]; PL_pr[j, i] = mean_pl_pr[p_idx]
        if np.isfinite(mean_pl_gt[p_idx]) and np.isfinite(mean_pl_pr[p_idx]):
            PL_df[i, j] = mean_pl_pr[p_idx] - mean_pl_gt[p_idx]
            PL_df[j, i] = PL_df[i, j]

    np.fill_diagonal(PL_gt, 0.0)
    np.fill_diagonal(PL_pr, 0.0)
    np.fill_diagonal(PL_df, 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    im0 = axes[0].imshow(PL_gt, vmin=lag_axis.min(), vmax=lag_axis.max())
    axes[0].set_title("GT mean peak-lag (pairwise)")
    axes[0].set_xlabel("region"); axes[0].set_ylabel("region")

    im1 = axes[1].imshow(PL_pr, vmin=lag_axis.min(), vmax=lag_axis.max())
    axes[1].set_title("Pred mean peak-lag (pairwise)")
    axes[1].set_xlabel("region")

    vmax_df = np.nanmax(np.abs(PL_df)) if np.isfinite(PL_df).any() else 1.0
    im2 = axes[2].imshow(PL_df, vmin=-vmax_df, vmax=vmax_df)
    axes[2].set_title("Pred - GT peak-lag (pairwise)")
    axes[2].set_xlabel("region")

    for axx in axes:
        axx.set_xticks(range(R))
        axx.set_yticks(range(R))
        axx.set_xticklabels(region_names, rotation=45, ha="right", fontsize=7)
        axx.set_yticklabels(region_names, fontsize=7)

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.show()

    # 4) score bars + mapping curves
    plt.figure(figsize=(10, 3.6))
    plt.subplot(1, 2, 1)
    plt.bar(["s_r", "s_e", "s_l"], [_nanmean(s_r), _nanmean(s_e), _nanmean(s_l)])
    plt.ylim([0, 1]); plt.title("Mean component scores"); plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(["core_mean", "core_q10"], [CC_core_score01_mean, CC_core_score01_q10])
    plt.ylim([0, 1]); plt.title("Core score"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    if show_mapping_curves:
        zz = np.linspace(-5, 5, 401)
        yy = np.array([_map_z_to_01(z) for z in zz])
        plt.figure(figsize=(6, 4))
        plt.plot(zz, yy)
        plt.axvline(z0, linestyle="--")
        plt.xlabel("z"); plt.ylabel("score01")
        plt.title("z -> score01 mapping (benchmark-consistent)")
        plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# ---------------------------
# RETURN dict for logging
# ---------------------------

if MODE == 'full' and not SKIP_CROSSCORR:
    crosscorr_realism_summary = dict(
        scalars=scalars,
        score_core=score_core,
        z_r=z_r, z_e=z_e, z_l=z_l,
        s_r=s_r, s_e=s_e, s_l=s_l,
        r_seq=r_seq, rmse_seq=rmse_seq, dlag_seq=dlag_seq,
        nPairs_shape_seq=nPairs_shape_seq,
        nPairs_peak_seq=nPairs_peak_seq,
        pairs=pairs,
        lag_axis=lag_axis,
        cc_gt_obs=cc_gt_obs,
        cc_pr_obs=cc_pr_obs,
        peak_lag_gt=peak_lag_gt,
        peak_lag_pr=peak_lag_pr,
        params=dict(
            max_lag=max_lag,
            ignore_lag0=ignore_lag0,
            min_abs_lag=min_abs_lag,
            min_seg_len=min_seg_len,
            min_lags_shape=min_lags_shape,
            min_lags_peak=min_lags_peak,
            mismatch_M=mismatch_M,
            pooled_scale=pooled_scale,
            ddof=ddof,
            beta_tail=beta_tail,
            weights=dict(w_r=w_r, w_e=w_e, w_l=w_l),
            mapping=dict(floor=floor01, cap=cap01, z0=z0, slope=slope),
            scale_floors=dict(sc_r=sc_r_floor, sc_e=sc_e_floor, sc_l=sc_l_floor),
            max_time=T,
        ),
    )

    crosscorr_realism_summary

# === Higher-order moments realism (Skew + Kurt) — mismatch-corrected 0–1 (benchmark v1; NO LEAK; FIXED RR) ===
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, spearmanr

def run_moments_realism_benchmark_v1(
    gt_array_denorm,
    data_predicted,
    region_names=None,
    fisher=True,              # kurtosis: True -> excess kurt (0 for Gaussian)
    min_seg_len=80,           # min joint-finite timepoints per (seq,reg) to compute moments

    # blending
    alpha_moment=0.50,        # skew vs kurt (core blend)
    lam_shape=0.60,           # Spearman vs nRMSE within each moment

    # mismatch baseline
    mismatch_M=10,
    pooled_scale=True,
    ddof=1,

    # denom floors for nRMSE stability (moment-space)
    rms_floor_skew=0.05,
    rms_floor_kurt=0.10,

    # z-scale floors
    sc_floor_r=0.05,
    sc_floor_e=0.10,

    # mapping (v1-consistent)
    floor01=0.02,
    cap01=0.98,
    z0=0.50,
    slope=3.0,                # larger => gentler (less saturation)

    # mismatch strictness: FIXED region-set per seq (benchmark default)
    drop_mismatch_if_any_region_invalid=True,

    # plotting
    plot=True,
    bins=30,
    random_seed=0,
    eps=1e-12,
):
    gt = np.asarray(gt_array_denorm, dtype=np.float64)
    pr = np.asarray(data_predicted, dtype=np.float64)

    if "_resample_pred_to_gt" in globals():
        gt, pr = _resample_pred_to_gt(gt, pr)

    if gt.shape != pr.shape or gt.ndim != 3:
        raise ValueError(f"Expected aligned gt/pred [n_seq,T,R]. Got {gt.shape} vs {pr.shape}")

    n_seq, T, R = gt.shape
    if region_names is None:
        region_names = list(gt_regions) if "gt_regions" in globals() else [f"R{i}" for i in range(R)]
    region_names = list(region_names)

    rng = np.random.default_rng(int(random_seed))
    all_idx = np.arange(n_seq, dtype=int)

    # -----------------------
    # helpers
    # -----------------------
    def _nanmean(x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else np.nan

    def _nanq(x, q):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        return float(np.quantile(x, q)) if x.size else np.nan

    def _rms(x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        if x.size < 3:
            return np.nan
        return float(np.sqrt(np.mean(x*x)))

    def _rmse(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return np.nan
        d = x[m] - y[m]
        return float(np.sqrt(np.mean(d*d)))

    def _safe_spearman(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return np.nan
        xx, yy = x[m], y[m]
        if np.nanstd(xx) < 1e-12 or np.nanstd(yy) < 1e-12:
            return np.nan
        r = spearmanr(xx, yy).correlation
        return float(r) if r is not None else np.nan

    def _nrmse_vec(gt_vec, pr_vec, rms_floor):
        denom = _rms(gt_vec)
        if not np.isfinite(denom):
            return np.nan
        denom = max(float(denom), float(rms_floor))
        e = _rmse(gt_vec, pr_vec)
        if not np.isfinite(e):
            return np.nan
        return float(e / (denom + eps))

    def _map_z_to_01(z):
        if not np.isfinite(z):
            return np.nan
        x = (float(z) - float(z0)) / (float(slope) + eps)  # v1-consistent divide
        x = float(np.clip(x, -50.0, 50.0))
        sig = 1.0 / (1.0 + np.exp(-x))
        return float(floor01 + (cap01 - floor01) * sig)

    def _pooled_scale(sc_arr, floor):
        v = np.asarray(sc_arr, float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float(floor)
        return float(max(float(np.nanmedian(v)), float(floor)))

    def _moments_from_vectors(g, p):
        sg = skew(g, bias=False)
        sp = skew(p, bias=False)
        kg = kurtosis(g, fisher=fisher, bias=False)
        kp = kurtosis(p, fisher=fisher, bias=False)
        if not (np.isfinite(sg) and np.isfinite(sp) and np.isfinite(kg) and np.isfinite(kp)):
            return None
        return float(sg), float(sp), float(kg), float(kp)

    def _mu_sd(v):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        if v.size < 2:
            return np.nan, np.nan
        sd = float(np.std(v, ddof=int(ddof))) if v.size > ddof else np.nan
        return float(np.mean(v)), sd

    # -----------------------
    # 1) observed moments (for observed metric only)
    #    mask_obs(s,r) = finite(gt[s,r]) & finite(pr[s,r])
    # -----------------------
    skew_gt_obs = np.full((n_seq, R), np.nan)
    skew_pr_obs = np.full((n_seq, R), np.nan)
    kurt_gt_obs = np.full((n_seq, R), np.nan)
    kurt_pr_obs = np.full((n_seq, R), np.nan)

    valid_obs_sr = np.zeros((n_seq, R), dtype=bool)

    for s in range(n_seq):
        for r in range(R):
            m = np.isfinite(gt[s, :, r]) & np.isfinite(pr[s, :, r])
            if int(m.sum()) < int(min_seg_len):
                continue
            out = _moments_from_vectors(gt[s, m, r], pr[s, m, r])
            if out is None:
                continue
            sg, sp, kg, kp = out
            skew_gt_obs[s, r] = sg
            skew_pr_obs[s, r] = sp
            kurt_gt_obs[s, r] = kg
            kurt_pr_obs[s, r] = kp
            valid_obs_sr[s, r] = True

    # -----------------------
    # 2) observed per-seq metrics across FIXED rr_obs(s)
    # -----------------------
    r_sk_obs   = np.full(n_seq, np.nan)
    e_sk_obs   = np.full(n_seq, np.nan)
    r_ku_obs   = np.full(n_seq, np.nan)
    e_ku_obs   = np.full(n_seq, np.nan)
    nreg_obs   = np.zeros(n_seq, dtype=int)

    for s in range(n_seq):
        rr = np.where(valid_obs_sr[s])[0]
        nreg_obs[s] = int(rr.size)
        if rr.size < 3:
            continue
        r_sk_obs[s] = _safe_spearman(skew_gt_obs[s, rr], skew_pr_obs[s, rr])
        r_ku_obs[s] = _safe_spearman(kurt_gt_obs[s, rr], kurt_pr_obs[s, rr])
        e_sk_obs[s] = _nrmse_vec(skew_gt_obs[s, rr], skew_pr_obs[s, rr], rms_floor=rms_floor_skew)
        e_ku_obs[s] = _nrmse_vec(kurt_gt_obs[s, rr], kurt_pr_obs[s, rr], rms_floor=rms_floor_kurt)

    # -----------------------
    # 3) mismatch baseline per seq (NO LEAK):
    #    For each mismatch (s,j,r), recompute BOTH GT and Pred moments under:
    #      mask_mis(s,j,r)=finite(gt[s,r]) & finite(pr[j,r])
    #    And enforce FIXED region set rr_obs(s) by dropping j if any r fails.
    # -----------------------
    mis_mu_r_sk = np.full(n_seq, np.nan); mis_sc_r_sk = np.full(n_seq, np.nan)
    mis_mu_e_sk = np.full(n_seq, np.nan); mis_sc_e_sk = np.full(n_seq, np.nan)
    mis_mu_r_ku = np.full(n_seq, np.nan); mis_sc_r_ku = np.full(n_seq, np.nan)
    mis_mu_e_ku = np.full(n_seq, np.nan); mis_sc_e_ku = np.full(n_seq, np.nan)

    mis_used = np.zeros(n_seq, dtype=int)

    for s in range(n_seq):
        rr = np.where(valid_obs_sr[s])[0]  # fixed region set for seq s
        if rr.size < 3:
            continue

        cand = all_idx[all_idx != s]
        if cand.size == 0:
            continue
        js = rng.choice(cand, size=min(int(mismatch_M), int(cand.size)), replace=(cand.size < mismatch_M))

        rsk_list, esk_list, rku_list, eku_list = [], [], [], []
        used = 0

        for j in js:
            # recompute mismatch GT+Pred moments under mask(gt[s], pr[j]) for ALL r in rr
            sk_gt_mis = np.empty(rr.size, dtype=np.float64)
            sk_pr_mis = np.empty(rr.size, dtype=np.float64)
            ku_gt_mis = np.empty(rr.size, dtype=np.float64)
            ku_pr_mis = np.empty(rr.size, dtype=np.float64)

            ok = True
            for k, r in enumerate(rr):
                m = np.isfinite(gt[s, :, r]) & np.isfinite(pr[j, :, r])
                if int(m.sum()) < int(min_seg_len):
                    ok = False
                    if drop_mismatch_if_any_region_invalid:
                        break
                    else:
                        sk_gt_mis[k] = np.nan; sk_pr_mis[k] = np.nan
                        ku_gt_mis[k] = np.nan; ku_pr_mis[k] = np.nan
                        continue

                out = _moments_from_vectors(gt[s, m, r], pr[j, m, r])
                if out is None:
                    ok = False
                    if drop_mismatch_if_any_region_invalid:
                        break
                    else:
                        sk_gt_mis[k] = np.nan; sk_pr_mis[k] = np.nan
                        ku_gt_mis[k] = np.nan; ku_pr_mis[k] = np.nan
                        continue

                sg, sp, kg, kp = out
                sk_gt_mis[k] = sg; sk_pr_mis[k] = sp
                ku_gt_mis[k] = kg; ku_pr_mis[k] = kp

            if drop_mismatch_if_any_region_invalid and (not ok):
                continue  # drop entire mismatch sample j to keep fixed rr space

            # (optional) if not dropping, still require enough finite entries
            if not drop_mismatch_if_any_region_invalid:
                if np.isfinite(sk_gt_mis).sum() < 3 or np.isfinite(ku_gt_mis).sum() < 3:
                    continue

            # mismatch metrics computed entirely in mismatch space
            rsk_list.append(_safe_spearman(sk_gt_mis, sk_pr_mis))
            rku_list.append(_safe_spearman(ku_gt_mis, ku_pr_mis))
            esk_list.append(_nrmse_vec(sk_gt_mis, sk_pr_mis, rms_floor=rms_floor_skew))
            eku_list.append(_nrmse_vec(ku_gt_mis, ku_pr_mis, rms_floor=rms_floor_kurt))

            used += 1

        mis_used[s] = used

        mis_mu_r_sk[s], mis_sc_r_sk[s] = _mu_sd(rsk_list)
        mis_mu_e_sk[s], mis_sc_e_sk[s] = _mu_sd(esk_list)
        mis_mu_r_ku[s], mis_sc_r_ku[s] = _mu_sd(rku_list)
        mis_mu_e_ku[s], mis_sc_e_ku[s] = _mu_sd(eku_list)

    # pooled scales (v1-style) + floors
    if pooled_scale:
        sc_r = _pooled_scale(np.r_[mis_sc_r_sk, mis_sc_r_ku], sc_floor_r)
        sc_e = _pooled_scale(np.r_[mis_sc_e_sk, mis_sc_e_ku], sc_floor_e)
        sc_r_sk = sc_r_ku = sc_r
        sc_e_sk = sc_e_ku = sc_e
    else:
        sc_r_sk = np.where(np.isfinite(mis_sc_r_sk), np.maximum(mis_sc_r_sk, sc_floor_r), sc_floor_r)
        sc_e_sk = np.where(np.isfinite(mis_sc_e_sk), np.maximum(mis_sc_e_sk, sc_floor_e), sc_floor_e)
        sc_r_ku = np.where(np.isfinite(mis_sc_r_ku), np.maximum(mis_sc_r_ku, sc_floor_r), sc_floor_r)
        sc_e_ku = np.where(np.isfinite(mis_sc_e_ku), np.maximum(mis_sc_e_ku, sc_floor_e), sc_floor_e)

    # -----------------------
    # 4) z scores (higher better) + map to 0–1, then core blend
    # -----------------------
    z_r_sk = (r_sk_obs - mis_mu_r_sk) / (sc_r_sk + eps)
    z_r_ku = (r_ku_obs - mis_mu_r_ku) / (sc_r_ku + eps)

    z_e_sk = (mis_mu_e_sk - e_sk_obs) / (sc_e_sk + eps)  # invert error
    z_e_ku = (mis_mu_e_ku - e_ku_obs) / (sc_e_ku + eps)

    z_sk = lam_shape * z_r_sk + (1 - lam_shape) * z_e_sk
    z_ku = lam_shape * z_r_ku + (1 - lam_shape) * z_e_ku

    s_sk = np.array([_map_z_to_01(z) for z in z_sk], dtype=np.float64)
    s_ku = np.array([_map_z_to_01(z) for z in z_ku], dtype=np.float64)

    score_core = alpha_moment * s_sk + (1 - alpha_moment) * s_ku

    # -----------------------
    # 5) benchmark scalars + prints
    # -----------------------
    coverage_seq = float(np.mean(np.isfinite(score_core))) if score_core.size else 0.0

    scalars = dict(
        MOM_coverage_seq=coverage_seq,
        MOM_core_score01_mean=_nanmean(score_core),
        MOM_core_score01_q10=_nanq(score_core, 0.10),

        MOM_skew_r_mean=_nanmean(r_sk_obs),
        MOM_skew_r_q10=_nanq(r_sk_obs, 0.10),
        MOM_skew_nRMSE_mean=_nanmean(e_sk_obs),
        MOM_skew_nRMSE_q90=_nanq(e_sk_obs, 0.90),

        MOM_kurt_r_mean=_nanmean(r_ku_obs),
        MOM_kurt_r_q10=_nanq(r_ku_obs, 0.10),
        MOM_kurt_nRMSE_mean=_nanmean(e_ku_obs),
        MOM_kurt_nRMSE_q90=_nanq(e_ku_obs, 0.90),

        MOM_mismatch_used_mean=float(np.mean(mis_used[np.isfinite(score_core)])) if np.any(np.isfinite(score_core)) else np.nan,
        MOM_mismatch_used_min=float(np.min(mis_used[np.isfinite(score_core)])) if np.any(np.isfinite(score_core)) else np.nan,
    )

    print("=== Higher-order moments realism (Skew+Kurt) — mismatch-corrected 0–1 (benchmark v1; NO LEAK; FIXED RR) ===")
    print(f"n_seq={n_seq} | n_reg={R} | min_seg_len={min_seg_len} | mismatch_M={mismatch_M} | pooled_scale={pooled_scale} | ddof={ddof}")
    print(f"moments: fisher(excess_kurt)={fisher} | alpha_moment={alpha_moment:.2f} | lam_shape={lam_shape:.2f}")
    print(f"mismatch rr policy: drop_if_any_region_invalid={drop_mismatch_if_any_region_invalid} (fixed rr_obs per seq)")
    if pooled_scale:
        print(f"pooled scales: sc_r={float(sc_r):.4f} (floor {sc_floor_r}) | sc_e={float(sc_e):.4f} (floor {sc_floor_e})")
    print(f"mapping (benchmark-consistent): floor={floor01} cap={cap01} | z0={z0} | slope={slope}")

    print("\n--- Benchmark scalars (USE THESE for composite) ---")
    for k in sorted(scalars.keys()):
        v = scalars[k]
        print(f"{k:26s}: {v:.4f}" if np.isfinite(v) else f"{k:26s}: nan")

    idx = np.argsort(score_core)
    print("\nWorst sequences by MOM_core_score01 (lowest):")
    shown = 0
    for s in idx[:15]:
        if not np.isfinite(score_core[s]):
            continue
        print(f"  seq={s:4d} score01={score_core[s]:.4f}  "
              f"skew(z={z_sk[s]:+.2f}, r={r_sk_obs[s]:+.3f}, nRMSE={e_sk_obs[s]:.3f})  "
              f"kurt(z={z_ku[s]:+.2f}, r={r_ku_obs[s]:+.3f}, nRMSE={e_ku_obs[s]:.3f})  "
              f"nreg_obs={nreg_obs[s]} mis_used={mis_used[s]}")
        shown += 1
        if shown >= 5:
            break

    # -----------------------
    # plots
    # -----------------------
    if plot:
        plt.figure(figsize=(6,4))
        x = score_core[np.isfinite(score_core)]
        plt.hist(x, bins=bins, edgecolor="k", alpha=0.8)
        plt.axvline(_nanmean(x), ls="--", lw=1, label=f"mean={_nanmean(x):.3f}")
        plt.axvline(_nanq(x,0.10), ls=":", lw=1, label=f"q10={_nanq(x,0.10):.3f}")
        plt.title("MOM core score01 distribution")
        plt.xlabel("score01"); plt.ylabel("count")
        plt.grid(alpha=0.25); plt.legend(frameon=False)
        plt.tight_layout(); plt.show()

        plt.figure(figsize=(6,4))
        plt.hist(mis_used, bins=np.arange(-0.5, mismatch_M+1.5, 1), edgecolor="k", alpha=0.8)
        plt.axvline(np.mean(mis_used), ls="--", lw=1, label=f"mean used={np.mean(mis_used):.2f}")
        plt.title("Mismatch samples actually used per seq (fixed rr + no-leak)")
        plt.xlabel("# used (out of mismatch_M)"); plt.ylabel("count")
        plt.grid(alpha=0.25); plt.legend(frameon=False)
        plt.tight_layout(); plt.show()

        zz = np.linspace(-6, 6, 401)
        yy = np.array([_map_z_to_01(z) for z in zz])
        plt.figure(figsize=(6,4))
        plt.plot(zz, yy)
        plt.axvline(z0, ls="--", lw=1)
        plt.title("z → score01 mapping (v1)")
        plt.xlabel("z"); plt.ylabel("score01")
        plt.grid(alpha=0.25)
        plt.tight_layout(); plt.show()

        m = np.isfinite(r_sk_obs) & np.isfinite(e_sk_obs) & np.isfinite(score_core)
        if np.any(m):
            plt.figure(figsize=(6,5))
            scp = plt.scatter(r_sk_obs[m], e_sk_obs[m], c=score_core[m], s=18, alpha=0.85)
            cb = plt.colorbar(scp); cb.set_label("core score01")
            plt.xlabel("skew Spearman (shape)")
            plt.ylabel("skew nRMSE (magnitude)")
            plt.title("Skew: shape vs magnitude")
            plt.grid(alpha=0.25)
            plt.tight_layout(); plt.show()

        m = np.isfinite(r_ku_obs) & np.isfinite(e_ku_obs) & np.isfinite(score_core)
        if np.any(m):
            plt.figure(figsize=(6,5))
            scp = plt.scatter(r_ku_obs[m], e_ku_obs[m], c=score_core[m], s=18, alpha=0.85)
            cb = plt.colorbar(scp); cb.set_label("core score01")
            plt.xlabel("kurt Spearman (shape)")
            plt.ylabel("kurt nRMSE (magnitude)")
            plt.title("Kurt: shape vs magnitude")
            plt.grid(alpha=0.25)
            plt.tight_layout(); plt.show()

    return dict(
        scalars=scalars,
        score_core=score_core,

        # observed (for diagnostics)
        r_sk_obs=r_sk_obs, e_sk_obs=e_sk_obs,
        r_ku_obs=r_ku_obs, e_ku_obs=e_ku_obs,
        valid_obs_sr=valid_obs_sr,
        nreg_obs=nreg_obs,

        # mismatch diagnostics
        mismatch=dict(
            mis_mu_r_sk=mis_mu_r_sk, mis_mu_e_sk=mis_mu_e_sk,
            mis_mu_r_ku=mis_mu_r_ku, mis_mu_e_ku=mis_mu_e_ku,
            mis_sc_r_sk=mis_sc_r_sk, mis_sc_e_sk=mis_sc_e_sk,
            mis_sc_r_ku=mis_sc_r_ku, mis_sc_e_ku=mis_sc_e_ku,
            used=mis_used,
            pooled_scale=pooled_scale,
            floors=dict(sc_floor_r=sc_floor_r, sc_floor_e=sc_floor_e),
            rr_policy=dict(drop_if_any_region_invalid=drop_mismatch_if_any_region_invalid),
        ),
        params=dict(
            fisher=fisher, min_seg_len=min_seg_len,
            mismatch_M=mismatch_M, pooled_scale=pooled_scale, ddof=ddof,
            alpha_moment=alpha_moment, lam_shape=lam_shape,
            rms_floors=dict(skew=rms_floor_skew, kurt=rms_floor_kurt),
            mapping=dict(floor=floor01, cap=cap01, z0=z0, slope=slope),
            seed=random_seed,
        ),
        moments_observed=dict(
            skew_gt_obs=skew_gt_obs, skew_pr_obs=skew_pr_obs,
            kurt_gt_obs=kurt_gt_obs, kurt_pr_obs=kurt_pr_obs,
        ),
    )

# ---- RUN (single-call cell) ----

if MODE == 'full':
    moments_realism_summary = run_moments_realism_benchmark_v1(
        gt_array_denorm=gt_array_denorm,
        data_predicted=data_predicted,
        region_names=gt_regions if "gt_regions" in globals() else None,
        fisher=True,
        min_seg_len=80,
        mismatch_M=10,
        pooled_scale=True,
        ddof=1,
        alpha_moment=0.50,
        lam_shape=0.60,
        rms_floor_skew=0.05,
        rms_floor_kurt=0.10,
        sc_floor_r=0.05,
        sc_floor_e=0.10,
        floor01=0.02,
        cap01=0.98,
        z0=0.50,
        slope=3.0,
        drop_mismatch_if_any_region_invalid=True,  # fixed rr per seq
        plot=True,
        random_seed=0,
    )

    moments_realism_summary

# === Corr-connectivity GRAPH realism (TOP-K / WEIGHTED) — mismatch-corrected 0–1 (benchmark v1; FIXED TOPK + OVERSAMPLE) ===
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import spearmanr

def run_graph_realism_benchmark_v1(
    gt_array_denorm,
    data_predicted,
    gt_regions,
    pred_region_names=None,

    # validity
    min_seg_len=80,         # min joint-finite timepoints for corr computation
    max_time=1200,          # cap timepoints for speed/stability (0/None to disable)

    # top-k edge selection
    topk_mode="k",          # "k" or "frac"
    topk_k=30,
    topk_frac=0.25,
    min_abs_corr=0.0,       # prefilter abs(corr) < min_abs_corr as invalid before top-k
    require_exact_k=True,   # ✅ if True: drop sample when valid edges < k (so top-k truly fixed)

    # component weights
    w_jacc=0.35,
    w_wgt=0.35,
    w_deg=0.20,
    w_str=0.10,

    # mismatch baseline
    mismatch_M=10,
    pooled_scale=True,
    ddof=1,

    # mismatch oversampling/early-stop
    mismatch_oversample_factor=8,     # how many candidates to try vs M (without replacement)
    mismatch_allow_replacement=True,  # fallback with replacement if not enough valid samples
    mismatch_max_extra_draws=200,     # extra replacement draws cap

    # scale floors
    sc_floor_sim=0.05,      # for similarity-like metrics (jacc, spearman r's)
    sc_floor_err=0.10,      # for error-like metric (struct_err)

    # mapping (benchmark v1 consistent)
    floor01=0.02,
    cap01=0.98,
    z0=0.50,
    slope=3.0,              # larger => gentler (less saturation)

    # plotting
    plot=True,
    bins=25,
    random_seed=0,

    eps=1e-12,
):
    gt = np.asarray(gt_array_denorm, dtype=np.float64)
    pr = np.asarray(data_predicted, dtype=np.float64)

    if "_resample_pred_to_gt" in globals():
        gt, pr = _resample_pred_to_gt(gt, pr)

    if gt.shape != pr.shape or gt.ndim != 3:
        raise ValueError(f"Expected aligned gt/pred [n_seq,T,R]. Got {gt.shape} vs {pr.shape}")

    n_seq, T, R0 = gt.shape
    gt_regions = list(gt_regions)

    if pred_region_names is None:
        pred_region_names = gt_regions if "pred_region_names" not in globals() else globals().get("pred_region_names")

    # --- region alignment (overlap only, keep GT order) ---
    if pred_region_names is None or list(pred_region_names) == gt_regions:
        reg_labels = gt_regions
        gt_al = gt
        pr_al = pr
    else:
        pred_region_names = list(pred_region_names)
        overlap = [r for r in gt_regions if r in pred_region_names]
        if len(overlap) == 0:
            raise ValueError("No overlapping regions between GT and Pred.")
        gt_idx = [gt_regions.index(r) for r in overlap]
        pr_idx = [pred_region_names.index(r) for r in overlap]
        gt_al = gt[..., gt_idx]
        pr_al = pr[..., pr_idx]
        reg_labels = overlap

    n_seq, T, R = gt_al.shape

    # --- time cap ---
    if max_time is not None and int(max_time) > 0 and T > int(max_time):
        gt_al = gt_al[:, :int(max_time), :]
        pr_al = pr_al[:, :int(max_time), :]
        T = gt_al.shape[1]

    # top-k edges count
    E = R * (R - 1) // 2
    k_edges = int(np.round(topk_frac * E)) if topk_mode == "frac" else int(topk_k)
    k_edges = int(np.clip(k_edges, 1, E))

    rng = np.random.default_rng(int(random_seed))
    all_idx = np.arange(n_seq, dtype=int)

    # ----------------------------
    # helpers
    # ----------------------------
    def _upper_tri_indices(n):
        return np.triu_indices(n, k=1)

    iu = _upper_tri_indices(R)

    def _safe_spearman(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return np.nan
        if np.nanstd(x[m]) <= 1e-12 or np.nanstd(y[m]) <= 1e-12:
            return np.nan
        r = spearmanr(x[m], y[m]).correlation
        return float(r) if r is not None else np.nan

    def _corrcoef_cols(X):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.corrcoef(X, rowvar=False)

    def _topk_mask_from_corr(C, k_edges, min_abs=0.0, require_exact_k=True):
        w = C[iu].astype(np.float64)
        valid = np.isfinite(w)
        if min_abs > 0:
            valid &= (np.abs(w) >= float(min_abs))

        n_valid = int(valid.sum())
        if n_valid == 0:
            return None, None, 0

        k = int(k_edges)
        if require_exact_k and n_valid < k:
            # ✅ strict: cannot form a fixed-size top-k
            return None, None, n_valid

        k_eff = int(min(k, n_valid))
        absw = np.abs(w)
        absw_sel = absw.copy()
        absw_sel[~valid] = -np.inf
        idx = np.argpartition(absw_sel, -k_eff)[-k_eff:]
        mask = np.zeros_like(w, dtype=bool)
        mask[idx] = True
        mask &= valid
        return mask, w, n_valid

    def _mask_to_adj(mask_ut, w_ut, n):
        A = np.zeros((n, n), dtype=np.float64)
        A[iu] = np.where(mask_ut, w_ut, 0.0)
        A = A + A.T
        np.fill_diagonal(A, 0.0)
        return A

    def _node_strength_abs(A):
        return np.sum(np.abs(A), axis=1)

    def _graph_struct_metrics_from_adj(A):
        G = nx.Graph()
        n = A.shape[0]
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i+1, n):
                w = A[i, j]
                if w != 0 and np.isfinite(w):
                    G.add_edge(i, j, weight=float(w))
        n_edges = G.number_of_edges()
        density = nx.density(G)
        n_comp = nx.number_connected_components(G)
        if n_edges == 0:
            clustering = 0.0
            trans = 0.0
        else:
            clustering = nx.average_clustering(G, weight=None)
            trans = nx.transitivity(G)
        return dict(
            density=float(density),
            n_components=int(n_comp),
            clustering=float(clustering),
            transitivity=float(trans),
            n_edges=int(n_edges),
        )

    def _struct_err(mg, mp):
        dens = abs(mg["density"] - mp["density"])
        comp = abs(mg["n_components"] - mp["n_components"]) / max(1, R)
        clus = abs(mg["clustering"] - mp["clustering"])
        return float(np.mean([dens, comp, clus]))

    def _nanmean(x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else np.nan

    def _nanq(x, q):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        return float(np.quantile(x, q)) if x.size else np.nan

    def _pooled_scale(vals, floor):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float(floor)
        return float(max(float(np.nanmedian(v)), float(floor)))

    def _map_z_to_01(z):
        if not np.isfinite(z): return np.nan
        x = (float(z) - float(z0)) / (float(slope) + eps)   # ✅ v1 divide-form
        x = float(np.clip(x, -50.0, 50.0))
        sig = 1.0 / (1.0 + np.exp(-x))
        return float(floor01 + (cap01 - floor01) * sig)

    # ----------------------------
    # metric computation for a given (s, j) under JOINT mask across all regions
    # ----------------------------
    def _metrics_under_joint_mask(s, j):
        Xg = gt_al[s].astype(np.float64)
        Xp = pr_al[j].astype(np.float64)

        # fixed region set: require all regions finite per timepoint
        m = np.isfinite(Xg).all(axis=1) & np.isfinite(Xp).all(axis=1)
        if int(m.sum()) < int(min_seg_len):
            return None

        Xg = Xg[m]
        Xp = Xp[m]

        Cg = _corrcoef_cols(Xg)
        Cp = _corrcoef_cols(Xp)

        mask_g, wut_g, nvalid_g = _topk_mask_from_corr(
            Cg, k_edges=k_edges, min_abs=min_abs_corr, require_exact_k=require_exact_k
        )
        mask_p, wut_p, nvalid_p = _topk_mask_from_corr(
            Cp, k_edges=k_edges, min_abs=min_abs_corr, require_exact_k=require_exact_k
        )

        if mask_g is None or mask_p is None:
            # cannot guarantee fixed top-k
            return None

        # Jaccard on fixed-size top-k sets
        inter = np.logical_and(mask_g, mask_p).sum()
        uni = np.logical_or(mask_g, mask_p).sum()
        jacc = float(inter / uni) if uni > 0 else np.nan

        # weighted agreement on union of top-k edges
        union_mask = mask_g | mask_p
        wg_u = wut_g[union_mask]
        wp_u = wut_p[union_mask]
        w_r = _safe_spearman(wg_u, wp_u)

        # node strength agreement
        Ag = _mask_to_adj(mask_g, wut_g, R)
        Ap = _mask_to_adj(mask_p, wut_p, R)
        sg = _node_strength_abs(Ag)
        sp = _node_strength_abs(Ap)
        d_r = _safe_spearman(sg, sp)

        # structure error
        mg = _graph_struct_metrics_from_adj(Ag)
        mp = _graph_struct_metrics_from_adj(Ap)
        serr = _struct_err(mg, mp)

        return dict(
            jacc=jacc,
            w_r=w_r,
            d_r=d_r,
            serr=serr,
            n_time=int(m.sum()),
            n_edges_gt=int(mask_g.sum()),
            n_edges_pr=int(mask_p.sum()),
            n_valid_edges_gt=int(nvalid_g),
            n_valid_edges_pr=int(nvalid_p),
        )

    # ----------------------------
    # 1) observed metrics per seq (s,s)
    # ----------------------------
    obs = [None] * n_seq
    for s in range(n_seq):
        obs[s] = _metrics_under_joint_mask(s, s)

    jacc_obs = np.array([o["jacc"] if o is not None else np.nan for o in obs], dtype=np.float64)
    w_r_obs  = np.array([o["w_r"]  if o is not None else np.nan for o in obs], dtype=np.float64)
    d_r_obs  = np.array([o["d_r"]  if o is not None else np.nan for o in obs], dtype=np.float64)
    serr_obs = np.array([o["serr"] if o is not None else np.nan for o in obs], dtype=np.float64)

    ntime_obs = np.array([o["n_time"] if o is not None else 0 for o in obs], dtype=int)
    nedges_gt = np.array([o["n_edges_gt"] if o is not None else 0 for o in obs], dtype=int)
    nedges_pr = np.array([o["n_edges_pr"] if o is not None else 0 for o in obs], dtype=int)

    # ----------------------------
    # 2) mismatch baseline per seq with oversampling + early-stop
    # ----------------------------
    mis_mu_j = np.full(n_seq, np.nan); mis_sc_j = np.full(n_seq, np.nan)
    mis_mu_wr = np.full(n_seq, np.nan); mis_sc_wr = np.full(n_seq, np.nan)
    mis_mu_dr = np.full(n_seq, np.nan); mis_sc_dr = np.full(n_seq, np.nan)
    mis_mu_se = np.full(n_seq, np.nan); mis_sc_se = np.full(n_seq, np.nan)

    mis_used = np.zeros(n_seq, dtype=int)
    mis_attempted = np.zeros(n_seq, dtype=int)

    for s in range(n_seq):
        if obs[s] is None:
            continue

        cand = all_idx[all_idx != s]
        if cand.size == 0:
            continue

        # try a lot without replacement first
        n_try = int(min(cand.size, max(mismatch_M, mismatch_oversample_factor * mismatch_M)))
        cand_shuf = cand.copy()
        rng.shuffle(cand_shuf)
        cand_try = cand_shuf[:n_try]

        vals_j, vals_wr, vals_dr, vals_se = [], [], [], []

        for j in cand_try:
            mis_attempted[s] += 1
            mm = _metrics_under_joint_mask(s, j)
            if mm is None:
                continue
            vals_j.append(mm["jacc"])
            vals_wr.append(mm["w_r"])
            vals_dr.append(mm["d_r"])
            vals_se.append(mm["serr"])
            if len(vals_j) >= int(mismatch_M):
                break

        # fallback with replacement if still short
        if mismatch_allow_replacement and len(vals_j) < int(mismatch_M):
            extra = 0
            while len(vals_j) < int(mismatch_M) and extra < int(mismatch_max_extra_draws):
                j = int(rng.choice(cand))
                mis_attempted[s] += 1
                extra += 1
                mm = _metrics_under_joint_mask(s, j)
                if mm is None:
                    continue
                vals_j.append(mm["jacc"])
                vals_wr.append(mm["w_r"])
                vals_dr.append(mm["d_r"])
                vals_se.append(mm["serr"])

        mis_used[s] = int(len(vals_j))

        def _mu_sd(v):
            v = np.asarray(v, float); v = v[np.isfinite(v)]
            if v.size < 2:
                return np.nan, np.nan
            mu = float(np.mean(v))
            sd = float(np.std(v, ddof=int(ddof))) if v.size > ddof else np.nan
            return mu, sd

        mis_mu_j[s],  mis_sc_j[s]  = _mu_sd(vals_j)
        mis_mu_wr[s], mis_sc_wr[s] = _mu_sd(vals_wr)
        mis_mu_dr[s], mis_sc_dr[s] = _mu_sd(vals_dr)
        mis_mu_se[s], mis_sc_se[s] = _mu_sd(vals_se)

    # pooled scales + floors
    if pooled_scale:
        sc_sim = _pooled_scale(np.r_[mis_sc_j, mis_sc_wr, mis_sc_dr], sc_floor_sim)
        sc_err = _pooled_scale(mis_sc_se, sc_floor_err)
        sc_j = sc_wr = sc_dr = sc_sim
        sc_se = sc_err
    else:
        sc_j  = np.where(np.isfinite(mis_sc_j),  np.maximum(mis_sc_j,  sc_floor_sim), sc_floor_sim)
        sc_wr = np.where(np.isfinite(mis_sc_wr), np.maximum(mis_sc_wr, sc_floor_sim), sc_floor_sim)
        sc_dr = np.where(np.isfinite(mis_sc_dr), np.maximum(mis_sc_dr, sc_floor_sim), sc_floor_sim)
        sc_se = np.where(np.isfinite(mis_sc_se), np.maximum(mis_sc_se, sc_floor_err), sc_floor_err)

    # z-scores
    z_j  = (jacc_obs - mis_mu_j) / (sc_j  + eps)
    z_wr = (w_r_obs  - mis_mu_wr) / (sc_wr + eps)
    z_dr = (d_r_obs  - mis_mu_dr) / (sc_dr + eps)
    z_se = (mis_mu_se - serr_obs) / (sc_se + eps)   # err lower better

    wsum = float(w_jacc + w_wgt + w_deg + w_str)
    if wsum <= 0:
        raise ValueError("Weights must sum to > 0.")
    ww_j, ww_wr, ww_dr, ww_se = (w_jacc/wsum, w_wgt/wsum, w_deg/wsum, w_str/wsum)

    z_core = ww_j*z_j + ww_wr*z_wr + ww_dr*z_dr + ww_se*z_se
    score01 = np.array([_map_z_to_01(z) for z in z_core], dtype=np.float64)

    coverage = float(np.mean(np.isfinite(score01))) if score01.size else 0.0

    scalars = dict(
        GRAPH_coverage_seq=coverage,
        GRAPH_core_score01_mean=_nanmean(score01),
        GRAPH_core_score01_q10=_nanq(score01, 0.10),

        GRAPH_jacc_mean=_nanmean(jacc_obs),
        GRAPH_jacc_q10=_nanq(jacc_obs, 0.10),

        GRAPH_wgt_r_mean=_nanmean(w_r_obs),
        GRAPH_wgt_r_q10=_nanq(w_r_obs, 0.10),

        GRAPH_deg_r_mean=_nanmean(d_r_obs),
        GRAPH_deg_r_q10=_nanq(d_r_obs, 0.10),

        GRAPH_structErr_mean=_nanmean(serr_obs),
        GRAPH_structErr_q90=_nanq(serr_obs, 0.90),

        GRAPH_mismatch_used_mean=_nanmean(mis_used),
        GRAPH_mismatch_used_min=float(np.min(mis_used)) if mis_used.size else np.nan,
        GRAPH_mismatch_attempted_mean=_nanmean(mis_attempted),
    )

    print("=== Corr-connectivity GRAPH realism (GT vs Pred) — mismatch-corrected 0–1 (benchmark v1; FIXED TOPK + OVERSAMPLE) ===")
    print(f"n_seq={n_seq} | n_reg={R} | E={E} | topk={k_edges} | require_exact_k={require_exact_k}")
    print(f"validity: min_seg_len={min_seg_len} | max_time={T}")
    print(f"mismatch_M={mismatch_M} | oversample_factor={mismatch_oversample_factor} | repl={mismatch_allow_replacement} | pooled_scale={pooled_scale} | ddof={ddof}")
    if pooled_scale:
        print(f"pooled scales: sc_sim={float(sc_sim):.4f} (floor {sc_floor_sim}) | sc_err={float(sc_err):.4f} (floor {sc_floor_err})")
    print(f"mapping (benchmark-consistent): floor={floor01} cap={cap01} | z0={z0} | slope={slope}")
    print(f"weights (normalized): jacc={ww_j:.2f} | wgt={ww_wr:.2f} | deg={ww_dr:.2f} | struct={ww_se:.2f}")

    print("\n--- Benchmark scalars (USE THESE for composite) ---")
    for k in sorted(scalars.keys()):
        v = scalars[k]
        print(f"{k:30s}: {v:.4f}" if np.isfinite(v) else f"{k:30s}: nan")

    print("\nWorst sequences by GRAPH_core_score01 (lowest):")
    idx = np.argsort(score01)
    shown = 0
    for s in idx[:10]:
        if not np.isfinite(score01[s]):
            continue
        print(f"  seq={s:4d} score01={score01[s]:.4f}  z_core={z_core[s]:+.2f}  "
              f"z(j,w,d,se)=({z_j[s]:+.2f},{z_wr[s]:+.2f},{z_dr[s]:+.2f},{z_se[s]:+.2f})  "
              f"jacc={jacc_obs[s]:.3f} w_r={w_r_obs[s]:+.3f} d_r={d_r_obs[s]:+.3f} serr={serr_obs[s]:.3f} "
              f"Tobs={ntime_obs[s]} edges(gt/pr)={nedges_gt[s]}/{nedges_pr[s]} mis_used={mis_used[s]} tried={mis_attempted[s]}")
        shown += 1
        if shown >= 5:
            break

    if plot:
        # score distribution
        vals = score01[np.isfinite(score01)]
        plt.figure(figsize=(7,4))
        plt.hist(vals, bins=bins, edgecolor="k", alpha=0.8)
        plt.axvline(_nanmean(vals), ls="--", lw=1, label=f"mean={_nanmean(vals):.3f}")
        plt.axvline(_nanq(vals,0.10), ls=":", lw=1, label=f"q10={_nanq(vals,0.10):.3f}")
        plt.title("GRAPH core score01 distribution")
        plt.xlabel("score01"); plt.ylabel("count")
        plt.grid(alpha=0.25); plt.legend(frameon=False)
        plt.tight_layout(); plt.show()

        # mapping curve
        zz = np.linspace(-6, 6, 401)
        yy = np.array([_map_z_to_01(z) for z in zz])
        plt.figure(figsize=(6,4))
        plt.plot(zz, yy)
        plt.axvline(z0, ls="--", lw=1)
        plt.title("z → score01 mapping (v1)")
        plt.xlabel("z"); plt.ylabel("score01")
        plt.grid(alpha=0.25)
        plt.tight_layout(); plt.show()

        # representative sequence near median (finite)
        fin = np.where(np.isfinite(score01))[0]
        if fin.size:
            med = float(np.nanmedian(score01[fin]))
            rep = fin[int(np.argmin(np.abs(score01[fin] - med)))]
            print(f"\n[plot] Representative seq={rep} score01={score01[rep]:.3f}")

            # rebuild graphs for representative (s,s)
            Xg = gt_al[rep].astype(np.float64)
            Xp = pr_al[rep].astype(np.float64)
            m = np.isfinite(Xg).all(axis=1) & np.isfinite(Xp).all(axis=1)
            Xg = Xg[m]; Xp = Xp[m]
            Cg = _corrcoef_cols(Xg); Cp = _corrcoef_cols(Xp)

            mask_g, wut_g, _ = _topk_mask_from_corr(Cg, k_edges=k_edges, min_abs=min_abs_corr, require_exact_k=require_exact_k)
            mask_p, wut_p, _ = _topk_mask_from_corr(Cp, k_edges=k_edges, min_abs=min_abs_corr, require_exact_k=require_exact_k)

            if mask_g is not None and mask_p is not None:
                Ag = _mask_to_adj(mask_g, wut_g, R)
                Ap = _mask_to_adj(mask_p, wut_p, R)

                def _nx_from_adj(A):
                    G = nx.Graph()
                    G.add_nodes_from(range(R))
                    for i in range(R):
                        for j in range(i+1, R):
                            w = A[i,j]
                            if w != 0 and np.isfinite(w):
                                G.add_edge(i,j, weight=float(w), absweight=float(abs(w)))
                    return G

                Gg = _nx_from_adj(Ag)
                Gp = _nx_from_adj(Ap)
                pos = nx.spring_layout(Gg, seed=42)

                def _edge_widths(G, scale=6.0, minw=0.5):
                    ws = np.array([d.get("absweight", 0.0) for _,_,d in G.edges(data=True)], dtype=np.float64)
                    if ws.size == 0 or np.nanmax(ws) <= 1e-12:
                        return []
                    ws = ws / np.nanmax(ws)
                    return list(minw + scale*ws)

                labels = {i: reg_labels[i] for i in range(R)}
                fig, axes = plt.subplots(1,2, figsize=(14,6))
                nx.draw_networkx(Gg, pos=pos, ax=axes[0], labels=labels, with_labels=True,
                                 node_size=650, font_size=7, width=_edge_widths(Gg))
                axes[0].set_title(f"GT top-k graph (seq={rep}, k={k_edges})"); axes[0].axis("off")

                nx.draw_networkx(Gp, pos=pos, ax=axes[1], labels=labels, with_labels=True,
                                 node_size=650, font_size=7, width=_edge_widths(Gp))
                axes[1].set_title(f"Pred top-k graph (seq={rep}, k={k_edges})"); axes[1].axis("off")
                plt.tight_layout(); plt.show()

    return dict(
        scalars=scalars,
        score01=score01,
        z_core=z_core,
        components=dict(jacc=jacc_obs, w_r=w_r_obs, d_r=d_r_obs, struct_err=serr_obs),
        z_components=dict(z_j=z_j, z_w=z_wr, z_d=z_dr, z_se=z_se),
        mismatch=dict(
            mu=dict(j=mis_mu_j, w=mis_mu_wr, d=mis_mu_dr, se=mis_mu_se),
            sc=dict(j=mis_sc_j, w=mis_sc_wr, d=mis_sc_dr, se=mis_sc_se),
            pooled_scale=pooled_scale,
            pooled=dict(sc_sim=float(sc_sim) if pooled_scale else None, sc_err=float(sc_err) if pooled_scale else None),
            used=mis_used,
            attempted=mis_attempted,
        ),
        diagnostics=dict(ntime_obs=ntime_obs, nedges_gt=nedges_gt, nedges_pr=nedges_pr),
        params=dict(
            min_seg_len=min_seg_len, max_time=T,
            topk_mode=topk_mode, topk_k=topk_k, topk_frac=topk_frac, k_edges=k_edges,
            min_abs_corr=min_abs_corr, require_exact_k=require_exact_k,
            weights=dict(w_jacc=w_jacc, w_wgt=w_wgt, w_deg=w_deg, w_str=w_str),
            mismatch_M=mismatch_M, ddof=ddof,
            mismatch_sampling=dict(oversample_factor=mismatch_oversample_factor,
                                   allow_replacement=mismatch_allow_replacement,
                                   max_extra_draws=mismatch_max_extra_draws),
            floors=dict(sc_floor_sim=sc_floor_sim, sc_floor_err=sc_floor_err),
            mapping=dict(floor=floor01, cap=cap01, z0=z0, slope=slope),
            seed=random_seed,
        ),
        region_labels=reg_labels,
    )

# ---- RUN (single-call cell) ----

if MODE in ('full', 'instant'):
    graph_results = run_graph_realism_benchmark_v1(
        gt_array_denorm=gt_array_denorm,
        data_predicted=data_predicted,
        gt_regions=gt_regions,
        pred_region_names=globals().get("pred_region_names", None),

        min_seg_len=80,
        max_time=1200,

        topk_mode="k",
        topk_k=30,
        topk_frac=0.25,
        min_abs_corr=0.0,
        require_exact_k=True,                 # ✅ strict fixed top-k

        w_jacc=0.35,
        w_wgt=0.35,
        w_deg=0.20,
        w_str=0.10,

        mismatch_M=10,
        pooled_scale=True,
        ddof=1,

        mismatch_oversample_factor=8,         # ✅ oversampling
        mismatch_allow_replacement=True,      # ✅ fallback
        mismatch_max_extra_draws=200,

        sc_floor_sim=0.05,
        sc_floor_err=0.10,

        floor01=0.02,
        cap01=0.98,
        z0=0.50,
        slope=3.0,

        plot=True,
        random_seed=0,
    )

    graph_results

# === CCA REALISM (GT vs Pred) — proper CV-on-TEST + mismatch-corrected 0–1 (benchmark v1; FIXED) ===
# Key properties (benchmark v1):
#   ✅ Proper CV-CCA: standardize on TRAIN only, evaluate canonical correlations on TEST only
#   ✅ Mismatch baseline per seq: compare GT[s] vs Pred[j] under the joint mask of (GT[s], Pred[j])
#   ✅ Oversampling + early-stop so mismatch baseline never collapses to NaN
#   ✅ Pooled mismatch scale + floors (v1-consistent; avoids tiny scales)
#   ✅ v1-consistent mapping: score01 = floor + (cap-floor)*sigmoid((z - z0)/slope)
#   ✅ Plots integrated (distributions, mismatch examples, score hist, mapping curve, representative canonical traces)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold

# ----------------------------
# Single-call benchmark
# ----------------------------
def run_cca_realism_benchmark_v1(
    gt_array_denorm,
    data_predicted,
    region_names=None,
    # validity / truncation
    min_seg_len=80,
    max_time=601,
    # CV-CCA
    n_comp=5,
    folds=5,
    max_iter=2000,
    # mismatch baseline
    mismatch_M=25,
    oversample_factor=8,          # try up to oversample_factor * mismatch_M draws
    allow_replace=True,           # allow replace if candidate set small
    pooled_scale=True,
    ddof=1,
    sc_floor_r=0.05,
    # mapping (benchmark-consistent)
    floor01=0.02,
    cap01=0.98,
    z0=0.50,
    slope=3.0,                   # larger => gentler (less saturation)
    # plotting
    plot=True,
    bins=25,
    show_mismatch_examples=True,
    n_mismatch_examples=6,
    show_rep_traces=True,
    rep_n_plot=3,
    # misc
    random_seed=0,
    eps=1e-9,
):
    gt = np.asarray(gt_array_denorm, dtype=np.float64)
    pr = np.asarray(data_predicted, dtype=np.float64)

    # align time if helper exists
    if "_resample_pred_to_gt" in globals():
        gt, pr = _resample_pred_to_gt(gt, pr)

    if gt.shape != pr.shape or gt.ndim != 3:
        raise ValueError(f"Expected aligned gt/pred [n_seq,T,R]. Got {gt.shape} vs {pr.shape}")

    n_seq, T, R = gt.shape
    Tuse = int(min(int(max_time), int(T)))
    gt = gt[:, :Tuse, :]
    pr = pr[:, :Tuse, :]
    T = Tuse

    if region_names is None:
        region_names = list(gt_regions) if "gt_regions" in globals() else [f"R{i}" for i in range(R)]
    region_names = list(region_names)

    rng = np.random.default_rng(int(random_seed))
    all_idx = np.arange(n_seq, dtype=int)

    # ----------------------------
    # helpers
    # ----------------------------
    def _finite_row_mask(X, Y):
        return np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)

    def _zscore_train_apply(Xtr, Xte):
        mu = np.mean(Xtr, axis=0, keepdims=True)
        sd = np.std(Xtr, axis=0, keepdims=True)
        sd = np.where(sd < eps, 1.0, sd)  # avoid blowups on constant columns
        return (Xtr - mu) / sd, (Xte - mu) / sd

    def _corr_safe(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.size < 3 or b.size < 3:
            return np.nan
        if np.std(a) < eps or np.std(b) < eps:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    def _nanmean(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else np.nan

    def _nanq(x, q):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        return float(np.quantile(x, q)) if x.size else np.nan

    def _map_z_to_01(z):
        if not np.isfinite(z):
            return np.nan
        x = (float(z) - float(z0)) / (float(slope) + eps)
        x = float(np.clip(x, -50.0, 50.0))
        sig = 1.0 / (1.0 + np.exp(-x))
        return float(floor01 + (cap01 - floor01) * sig)

    def _mu_sd(v):
        v = np.asarray(v, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size < 2:
            return np.nan, np.nan
        mu = float(np.mean(v))
        sd = float(np.std(v, ddof=int(ddof))) if v.size > ddof else np.nan
        return mu, sd

    def _pooled_scale(sc_arr, floor):
        v = np.asarray(sc_arr, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float(floor)
        return float(max(float(np.median(v)), float(floor)))

    def _cv_cca_testcorr(X, Y):
        """
        Proper CV-CCA:
          - joint finite row mask
          - per fold: z-score on TRAIN only; apply to TEST
          - fit on TRAIN; evaluate canonical correlations on TEST
          - return mean(test corr over components and folds)
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        m = _finite_row_mask(X, Y)
        if int(m.sum()) < int(min_seg_len):
            return np.nan

        X = X[m]
        Y = Y[m]
        n = X.shape[0]
        if n < max(12, folds * 3):
            return np.nan

        k = int(min(n_comp, X.shape[1], Y.shape[1], n - 2))
        if k < 1:
            return np.nan

        kf = KFold(n_splits=int(folds), shuffle=True, random_state=int(random_seed))
        fold_scores = []

        for tr, te in kf.split(X):
            Xtr, Xte = X[tr], X[te]
            Ytr, Yte = Y[tr], Y[te]

            Xtr_z, Xte_z = _zscore_train_apply(Xtr, Xte)
            Ytr_z, Yte_z = _zscore_train_apply(Ytr, Yte)

            cca = CCA(n_components=k, max_iter=int(max_iter))
            try:
                cca.fit(Xtr_z, Ytr_z)
                Xc, Yc = cca.transform(Xte_z, Yte_z)
            except Exception:
                fold_scores.append(np.nan)
                continue

            corrs = []
            for c in range(k):
                r = _corr_safe(Xc[:, c], Yc[:, c])
                if np.isfinite(r):
                    corrs.append(r)
            fold_scores.append(float(np.mean(corrs)) if len(corrs) else np.nan)

        fs = np.asarray(fold_scores, dtype=np.float64)
        fs = fs[np.isfinite(fs)]
        return float(np.mean(fs)) if fs.size else np.nan

    # ----------------------------
    # main per-seq: observed + row-perm + mismatch baseline
    # ----------------------------
    r_obs   = np.full(n_seq, np.nan)
    r_perm  = np.full(n_seq, np.nan)

    mis_mu  = np.full(n_seq, np.nan)
    mis_sc  = np.full(n_seq, np.nan)
    mis_used = np.zeros(n_seq, dtype=int)
    mis_tried = np.zeros(n_seq, dtype=int)

    z_r     = np.full(n_seq, np.nan)
    score01 = np.full(n_seq, np.nan)

    # store some mismatch distributions for plots
    mismatch_store = []
    mismatch_store_ids = []

    for s in range(n_seq):
        Xs = gt[s]
        Ys = pr[s]

        # observed
        r_s = _cv_cca_testcorr(Xs, Ys)
        r_obs[s] = r_s

        # row-perm control (destroys within-seq temporal alignment)
        perm = rng.permutation(T)
        r_perm[s] = _cv_cca_testcorr(Xs, Ys[perm])

        # mismatch baseline: GT[s] vs Pred[j], recomputed under joint mask of (GT[s], Pred[j])
        cand = all_idx[all_idx != s]
        if cand.size == 0:
            continue

        # oversampling + early stop (benchmark-style)
        target = int(mismatch_M)
        max_attempts = int(min(cand.size if not allow_replace else (oversample_factor * target),
                               oversample_factor * target))
        replace = bool(allow_replace and cand.size < max_attempts)

        # draw candidates up front (cheap) then evaluate until we collect target
        js = rng.choice(cand, size=max_attempts, replace=replace)

        mis_vals = []
        for j in js:
            mis_tried[s] += 1
            r_m = _cv_cca_testcorr(Xs, pr[j])
            if np.isfinite(r_m):
                mis_vals.append(r_m)
                if len(mis_vals) >= target:
                    break

        mis_vals = np.asarray(mis_vals, dtype=np.float64)
        mis_used[s] = int(mis_vals.size)

        if np.isfinite(r_s) and mis_vals.size >= max(5, min(10, target)):
            mu, sd = _mu_sd(mis_vals)
            mis_mu[s] = mu
            mis_sc[s] = sd

            # store mismatches for a subset
            if (s % max(1, n_seq // 8) == 0) and mis_vals.size:
                mismatch_store.append(mis_vals)
                mismatch_store_ids.append(s)

    # pooled scale + floors
    if pooled_scale:
        sc_r = _pooled_scale(mis_sc, sc_floor_r)
        sc_use = np.full(n_seq, sc_r, dtype=np.float64)
    else:
        sc_use = np.where(np.isfinite(mis_sc), np.maximum(mis_sc, sc_floor_r), sc_floor_r)

    # z + score
    for s in range(n_seq):
        if np.isfinite(r_obs[s]) and np.isfinite(mis_mu[s]):
            z = (r_obs[s] - mis_mu[s]) / (sc_use[s] + eps)
            z_r[s] = float(z)
            score01[s] = _map_z_to_01(z)

    # ----------------------------
    # scalars + print
    # ----------------------------
    coverage_seq = float(np.mean(np.isfinite(score01))) if n_seq else 0.0

    scalars = dict(
        CCA_coverage_seq=coverage_seq,
        CCA_r_mean=_nanmean(r_obs),
        CCA_r_q10=_nanq(r_obs, 0.10),
        CCA_rowperm_r_mean=_nanmean(r_perm),
        CCA_rowperm_r_q90=_nanq(r_perm, 0.90),
        CCA_core_score01_mean=_nanmean(score01),
        CCA_core_score01_q10=_nanq(score01, 0.10),
        CCA_mismatch_used_mean=float(np.mean(mis_used)) if mis_used.size else np.nan,
        CCA_mismatch_used_min=float(np.min(mis_used)) if mis_used.size else np.nan,
        CCA_mismatch_attempted_mean=float(np.mean(mis_tried)) if mis_tried.size else np.nan,
    )

    print("=== CCA REALISM (GT vs Pred) — proper CV-on-TEST + mismatch-corrected 0–1 (benchmark v1) ===")
    print(f"n_seq={n_seq} | n_reg={R} | max_time={T} | min_seg_len={min_seg_len}")
    print(f"CV: n_comp={n_comp} | folds={folds} | max_iter={max_iter}")
    print(f"mismatch_M={mismatch_M} | oversample_factor={oversample_factor} | repl={allow_replace} | pooled_scale={pooled_scale} | ddof={ddof}")
    if pooled_scale:
        print(f"pooled scale: sc_r={float(sc_r):.4f} (floor {sc_floor_r})")
    print(f"mapping (benchmark-consistent): floor={floor01} cap={cap01} | z0={z0} | slope={slope}")

    print("\n--- Benchmark scalars (USE THESE for composite) ---")
    for k in sorted(scalars.keys()):
        v = scalars[k]
        print(f"{k:28s}: {v:.4f}" if np.isfinite(v) else f"{k:28s}: nan")

    # worst sequences
    idx = np.argsort(score01)
    print("\nWorst sequences by CCA_core_score01 (lowest):")
    shown = 0
    for s in idx[:20]:
        if not np.isfinite(score01[s]):
            continue
        print(f"  seq={s:4d} score01={score01[s]:.4f}  z_r={z_r[s]:+.2f}  "
              f"r_obs={r_obs[s]:+.3f}  mis_mu={mis_mu[s]:+.3f}  mis_used={mis_used[s]:2d}/{mismatch_M}  tried={mis_tried[s]:2d}  "
              f"rowperm={r_perm[s]:+.3f}")
        shown += 1
        if shown >= 5:
            break

    # ----------------------------
    # dataframe (handy)
    # ----------------------------
    cca_df = pd.DataFrame({
        "sequence_id": np.arange(n_seq),
        "cca_r_obs_cvtest": r_obs,
        "cca_r_rowperm_cvtest": r_perm,
        "mis_mu": mis_mu,
        "mis_sc": mis_sc,
        "z_r": z_r,
        "score01": score01,
        "mis_used": mis_used,
        "mis_tried": mis_tried,
    }).round(4)

    display(cca_df)

    # ----------------------------
    # plots
    # ----------------------------
    if plot:
        # 1) distributions
        plt.figure(figsize=(12, 3.6))
        plt.subplot(1, 3, 1)
        x = r_obs[np.isfinite(r_obs)]
        plt.hist(x, bins=bins, alpha=0.8, edgecolor="k")
        plt.axvline(_nanmean(x), ls="--", lw=1)
        plt.title("Matched CV-CCA (TEST)\nper-seq mean canonical corr")
        plt.xlabel("r_obs"); plt.ylabel("count")
        plt.grid(alpha=0.25)

        plt.subplot(1, 3, 2)
        x = r_perm[np.isfinite(r_perm)]
        plt.hist(x, bins=bins, alpha=0.8, edgecolor="k")
        plt.axvline(_nanmean(x), ls="--", lw=1)
        plt.title("Row-perm control\n(should be low)")
        plt.xlabel("r_rowperm")
        plt.grid(alpha=0.25)

        plt.subplot(1, 3, 3)
        x = z_r[np.isfinite(z_r)]
        plt.hist(x, bins=bins, alpha=0.8, edgecolor="k")
        plt.axvline(_nanmean(x), ls="--", lw=1)
        plt.title("Effect z vs mismatch\nz = (r_obs - mu_mis)/sc")
        plt.xlabel("z_r")
        plt.grid(alpha=0.25)

        plt.tight_layout()
        plt.show()

        # 2) score hist + scatter
        plt.figure(figsize=(12, 3.6))
        plt.subplot(1, 3, 1)
        x = score01[np.isfinite(score01)]
        plt.hist(x, bins=bins, alpha=0.8, edgecolor="k")
        plt.axvline(_nanmean(x), ls="--", lw=1)
        plt.axvline(_nanq(x, 0.10), ls=":", lw=1)
        plt.title("CCA score01 distribution")
        plt.xlabel("score01"); plt.ylabel("count")
        plt.grid(alpha=0.25)

        plt.subplot(1, 3, 2)
        m = np.isfinite(r_obs) & np.isfinite(score01)
        plt.scatter(r_obs[m], score01[m], s=18, alpha=0.35, edgecolor="none")
        plt.title("score01 vs r_obs")
        plt.xlabel("r_obs"); plt.ylabel("score01")
        plt.grid(alpha=0.25)

        plt.subplot(1, 3, 3)
        m = np.isfinite(z_r) & np.isfinite(score01)
        plt.scatter(z_r[m], score01[m], s=18, alpha=0.35, edgecolor="none")
        plt.title("score01 vs z_r")
        plt.xlabel("z_r"); plt.ylabel("score01")
        plt.grid(alpha=0.25)

        plt.tight_layout()
        plt.show()

        # 3) mapping curve sanity
        zz = np.linspace(-6, 6, 401)
        yy = np.array([_map_z_to_01(z) for z in zz], dtype=np.float64)
        plt.figure(figsize=(6, 4))
        plt.plot(zz, yy)
        plt.axvline(z0, ls="--", lw=1)
        plt.title("z → score01 mapping (v1)")
        plt.xlabel("z"); plt.ylabel("score01")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

        # 4) mismatch examples (subset)
        if show_mismatch_examples and mismatch_store:
            n_show = int(min(int(n_mismatch_examples), len(mismatch_store)))
            nrows = 2
            ncols = int(np.ceil(n_show / nrows))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 5))
            axes = np.ravel(axes)

            for k in range(n_show):
                s = mismatch_store_ids[k]
                mis = mismatch_store[k]
                ax = axes[k]
                ax.hist(mis[np.isfinite(mis)], bins=15, alpha=0.75, edgecolor="k")
                if np.isfinite(r_obs[s]):
                    ax.axvline(r_obs[s], ls="--", lw=2, label="matched")
                ax.set_title(f"seq {s} | matched={r_obs[s]:.3f}\nmis_mu={np.mean(mis):.3f}  mis_med={np.median(mis):.3f}")
                ax.set_xlabel("CV-CCA r"); ax.set_ylabel("count")
                ax.grid(alpha=0.25)
                ax.legend(frameon=False)

            for ax in axes[n_show:]:
                ax.axis("off")

            fig.suptitle("Mismatch baseline vs matched (subset)", y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

        # 5) representative canonical traces (viz only; fit on full masked data for display)
        if show_rep_traces:
            valid = np.where(np.isfinite(score01))[0]
            if valid.size:
                rep = valid[np.argsort(score01[valid])[len(valid)//2]]
                X = gt[rep]
                Y = pr[rep]
                m = _finite_row_mask(X, Y)
                X = X[m]
                Y = Y[m]
                if X.shape[0] >= min_seg_len and X.shape[0] >= 10:
                    # zscore on full (viz only)
                    X = (X - np.mean(X, axis=0, keepdims=True)) / (np.std(X, axis=0, keepdims=True) + eps)
                    Y = (Y - np.mean(Y, axis=0, keepdims=True)) / (np.std(Y, axis=0, keepdims=True) + eps)

                    k = int(min(n_comp, R, X.shape[0]-2))
                    k = max(1, k)
                    cca = CCA(n_components=k, max_iter=int(max_iter))
                    try:
                        cca.fit(X, Y)
                        Xc, Yc = cca.transform(X, Y)
                        plt.figure(figsize=(10, 4))
                        nplot = int(min(rep_n_plot, k))
                        for c in range(nplot):
                            plt.plot(Xc[:, c], alpha=0.85, label=f"GT c{c+1}")
                            plt.plot(Yc[:, c], alpha=0.85, ls="--", label=f"Pred c{c+1}")
                        plt.title(f"Representative seq {rep} | score01={score01[rep]:.3f} | r_obs={r_obs[rep]:.3f}")
                        plt.xlabel("time (joint-finite rows)")
                        plt.ylabel("canonical component")
                        plt.grid(alpha=0.25)
                        plt.legend(ncol=3, fontsize=8, frameon=False)
                        plt.tight_layout()
                        plt.show()
                    except Exception:
                        pass

    return dict(
        scalars=scalars,
        score01=score01,
        z_r=z_r,
        r_obs=r_obs,
        r_rowperm=r_perm,
        mismatch=dict(mis_mu=mis_mu, mis_sc=mis_sc, mis_used=mis_used, mis_tried=mis_tried,
                      pooled_scale=pooled_scale, sc_floor_r=sc_floor_r),
        params=dict(
            min_seg_len=min_seg_len, max_time=max_time,
            n_comp=n_comp, folds=folds, max_iter=max_iter,
            mismatch_M=mismatch_M, oversample_factor=oversample_factor, allow_replace=allow_replace,
            pooled_scale=pooled_scale, ddof=ddof,
            mapping=dict(floor=floor01, cap=cap01, z0=z0, slope=slope),
            seed=random_seed,
        ),
        df=cca_df,
    )

# ---- RUN (single-call cell) ----

if MODE == 'full' and not SKIP_CCA:
    cca_realism_summary = run_cca_realism_benchmark_v1(
        gt_array_denorm=gt_array_denorm,
        data_predicted=data_predicted,
        region_names=gt_regions if "gt_regions" in globals() else None,
        min_seg_len=80,
        max_time=601,
        n_comp=5,
        folds=5,
        mismatch_M=25,
        oversample_factor=8,
        allow_replace=True,
        pooled_scale=True,
        ddof=1,
        sc_floor_r=0.05,
        floor01=0.02,
        cap01=0.98,
        z0=0.50,
        slope=3.0,
        plot=True,
        random_seed=0,
    )

    cca_realism_summary

if MODE == 'full':
    # === MANIFOLD realism (v1) — per-seq manifold alignment, mismatch-corrected 0–1 (benchmark-ready) ===
    # Metrics (per sequence):
    #   1) kNN graph overlap (Jaccard)          [higher is better]
    #   2) Laplacian spectrum similarity (RMSE) [lower is better]
    #   3) Procrustes residual in GT-PCA space  [lower is better]
    #   4) Geodesic dist distribution (W1)     [lower is better]
    #
    # v1 features:
    #   - deterministic per-seq RNG (_v1_rng_for_seq)
    #   - mismatch oversample + early-stop until each component has mismatch_M finite
    #   - mismatch_min gating: if any component has < mismatch_min mismatches => score01 = NaN
    #   - pooled_scale=True (default): z uses pooled robust scales (median of per-seq MAD scales)
    #   - uniform artifact contract: scalars, score01 (len n_seq), params, diagnostics, df

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA
    from scipy.sparse import csgraph
    from scipy.linalg import orthogonal_procrustes
    from scipy.stats import wasserstein_distance

    # -----------------------------
    # Prereqs / alignment
    # -----------------------------
    _required = ["gt_array_denorm", "data_predicted"]
    _missing = [k for k in _required if k not in globals()]
    if _missing:
        raise RuntimeError(f"Missing prerequisites: {_missing}")

    def _resample_pred_to_gt_fallback(gt_arr, pred_arr):
        """Simple time alignment fallback."""
        gt_len = gt_arr.shape[1]
        pr_len = pred_arr.shape[1]
        if pr_len == gt_len:
            return gt_arr, pred_arr
        if pr_len % gt_len == 0:
            factor = pr_len // gt_len
            pred_eq = pred_arr.reshape(pred_arr.shape[0], gt_len, factor, pred_arr.shape[2]).mean(axis=2)
            return gt_arr, pred_eq
        min_len = min(gt_len, pr_len)
        return gt_arr[:, :min_len, :], pred_arr[:, :min_len, :]

    if "_resample_pred_to_gt" in globals():
        _gt_aligned, _pred_aligned = _resample_pred_to_gt(gt_array_denorm, data_predicted)
    else:
        _gt_aligned, _pred_aligned = _resample_pred_to_gt_fallback(gt_array_denorm, data_predicted)

    if _gt_aligned.shape != _pred_aligned.shape:
        raise RuntimeError(f"Aligned mismatch: GT={_gt_aligned.shape} vs Pred={_pred_aligned.shape}")

    n_seq, T, n_reg = _gt_aligned.shape
    print(f"Aligned shapes: GT={_gt_aligned.shape} | Pred={_pred_aligned.shape}")

    # -----------------------------
    # v1 helpers (self-contained)
    # -----------------------------
    def _v1_rng_for_seq(seed: int, seq_id: int) -> np.random.Generator:
        # stable even if you drop sequences or change ordering
        return np.random.default_rng(int(seed) + 1000003 * int(seq_id))

    def _v1_strict_q(x, q=0.10):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        return float(np.quantile(x, q)) if x.size else np.nan

    def _v1_zscore_cols(X, eps=1e-9):
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        return (X - mu) / (sd + eps)

    def _v1_robust_mad_scale(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size < 3:
            return np.nan
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return float(1.4826 * mad)

    def _v1_sigmoid_score01(z, mapping):
        """score in [floor, cap] via sigmoid((z - z0)/slope)."""
        if not np.isfinite(z):
            return np.nan
        t = (z - mapping["z0"]) / (mapping["slope"] + 1e-12)
        t = np.clip(t, -10.0, 10.0)
        s = 1.0 / (1.0 + np.exp(-t))
        return float(mapping["floor"] + (mapping["cap"] - mapping["floor"]) * s)

    def _finite_rows_joint(X, Y):
        return np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)

    def _sample_seq_points_seqrng(X, Y, rng, time_step, max_pts, min_pts):
        """Strict joint finite mask, then time subsample, then random row subset."""
        Xs = X[::time_step]
        Ys = Y[::time_step]
        m = _finite_rows_joint(Xs, Ys)
        Xs, Ys = Xs[m], Ys[m]
        if Xs.shape[0] < min_pts:
            return None, None, 0
        if Xs.shape[0] > max_pts:
            idx = rng.choice(Xs.shape[0], size=max_pts, replace=False)
            Xs, Ys = Xs[idx], Ys[idx]
        Xs = _v1_zscore_cols(Xs)
        Ys = _v1_zscore_cols(Ys)
        return Xs.astype(np.float64), Ys.astype(np.float64), int(Xs.shape[0])

    def _knn_adj(X, k, weighted=False):
        nn = NearestNeighbors(n_neighbors=min(k + 1, X.shape[0]), metric="euclidean")
        nn.fit(X)
        dists, idx = nn.kneighbors(X)
        n = X.shape[0]
        A = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for jpos in range(1, idx.shape[1]):  # skip self
                j = int(idx[i, jpos])
                if j == i:
                    continue
                val = float(dists[i, jpos]) if weighted else 1.0
                A[i, j] = val
                A[j, i] = val
        return A

    def _jaccard_from_adj(A, B):
        Au = np.triu(A > 0, k=1)
        Bu = np.triu(B > 0, k=1)
        inter = np.logical_and(Au, Bu).sum()
        union = np.logical_or(Au, Bu).sum()
        return float(inter / union) if union else np.nan

    def _laplacian_eigs_from_weighted_adj(W, top):
        vals = W[W > 0]
        if vals.size == 0:
            return None
        s = np.median(vals) + 1e-12
        S = np.zeros_like(W)
        S[W > 0] = np.exp(-W[W > 0] / s)  # distance -> similarity
        L = csgraph.laplacian(S, normed=True)
        try:
            evals = np.linalg.eigvalsh(L)
        except Exception:
            return None
        evals = np.sort(evals)
        return evals[:min(top, evals.size)]

    def _spectral_rmse(e1, e2):
        if e1 is None or e2 is None:
            return np.nan
        m = min(len(e1), len(e2))
        if m < 2:
            return np.nan
        return float(np.sqrt(np.mean((e1[:m] - e2[:m]) ** 2)))

    def _procrustes_resid_in_gt_pca(X, Y, pca_dim=8):
        """PCA on X, project both, orthogonal Procrustes (optionally isotropic scale α)."""
        n = X.shape[0]
        k = int(min(pca_dim, X.shape[1], n - 1))
        if k < 2:
            return np.nan, None, None, None, None

        pca = PCA(n_components=k, random_state=0)
        Zx = pca.fit_transform(X)
        Zy = pca.transform(Y)

        # Center in PCA space (important for Procrustes stability)
        Zx0 = Zx - Zx.mean(axis=0, keepdims=True)
        Zy0 = Zy - Zy.mean(axis=0, keepdims=True)

        try:
            R, _ = orthogonal_procrustes(Zy0, Zx0)
        except Exception:
            return np.nan, Zx, Zy, None, pca

        ZyR = Zy0 @ R

        # Optional isotropic scaling α (this is the *correct* way if you want scale)
        denom = (ZyR**2).sum() + 1e-12
        alpha = float((Zx0 * ZyR).sum() / denom)
        Zy_al = alpha * ZyR

        resid = np.linalg.norm(Zx0 - Zy_al, ord="fro")
        base  = np.linalg.norm(Zx0, ord="fro") + 1e-12
        return float(resid / base), Zx0, Zy0, Zy_al, pca

    def _geodesic_w1(X, Y, k, pair_samples, rng, tries_mult=20, min_pairs=200):
        Wx = _knn_adj(X, k=k, weighted=True)
        Wy = _knn_adj(Y, k=k, weighted=True)
        Dx = csgraph.dijkstra(Wx, directed=False)
        Dy = csgraph.dijkstra(Wy, directed=False)

        n = X.shape[0]
        max_pairs = n * (n - 1) // 2
        target = int(min(pair_samples, max_pairs))
        if target <= 0:
            return np.nan, None, None, 0, 0

        di, dj = [], []
        tries = 0
        while len(di) < target and tries < target * tries_mult:
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if j <= i:
                tries += 1
                continue
            dx = Dx[i, j]
            dy = Dy[i, j]
            if np.isfinite(dx) and np.isfinite(dy) and dx > 0 and dy > 0:
                di.append(dx)
                dj.append(dy)
            tries += 1

        used = len(di)
        if used < min_pairs:
            return np.nan, None, None, used, tries

        di = np.asarray(di, dtype=np.float64)
        dj = np.asarray(dj, dtype=np.float64)
        w1 = float(wasserstein_distance(di, dj))
        return w1, di, dj, used, tries

    # -----------------------------
    # v1 params
    # -----------------------------
    params_mani = dict(
        min_seg_len=80,
        max_time=601,
        time_step=10,
        max_pts_per_seq=220,
        min_pts_per_seq=30,        # additional hard minimum post-filtering
        k_nn=10,
        top_eigs=12,
        geo_pair_samples=6000,

        mismatch_M=12,
        mismatch_min=8,
        oversample_factor=8,
        allow_replace=True,

        pooled_scale=True,
        ddof=1,  # not used here; kept for audit consistency

        mapping=dict(floor=0.02, cap=0.98, z0=0.50, slope=3.0),
        seed=42,

        weights=dict(knn=0.30, spec=0.25, proc=0.25, geo=0.20),

        # scale floors (prevents infinite z when mismatch spread tiny)
        sc_floor=dict(knn=0.02, spec=1e-3, proc=1e-3, geo=1e-3),
    )

    # optionally cap time like other cells
    if params_mani["max_time"] is not None:
        Tmax = int(min(T, params_mani["max_time"]))
        _gt_aligned = _gt_aligned[:, :Tmax, :]
        _pred_aligned = _pred_aligned[:, :Tmax, :]
        T = Tmax

    print(f"validity gates: min_pts_per_seq={params_mani['min_pts_per_seq']} | time_step={params_mani['time_step']} | max_pts={params_mani['max_pts_per_seq']} | max_time={params_mani['max_time']} (used T={T})")
    print(f"mismatch: M={params_mani['mismatch_M']} | mismatch_min={params_mani['mismatch_min']} | oversample_factor={params_mani['oversample_factor']} | repl={params_mani['allow_replace']}")
    print(f"mapping: {params_mani['mapping']}")
    print(f"weights: {params_mani['weights']}")

    # -----------------------------
    # Pre-sample per sequence deterministically
    # -----------------------------
    X_list = [None] * n_seq
    Y_list = [None] * n_seq
    valid_pts = np.full(n_seq, np.nan, dtype=np.float64)
    valid_seq = []

    for i in range(n_seq):
        rng_i = _v1_rng_for_seq(params_mani["seed"], i)
        Xs, Ys, npts = _sample_seq_points_seqrng(
            _gt_aligned[i], _pred_aligned[i], rng_i,
            params_mani["time_step"], params_mani["max_pts_per_seq"],
            min_pts=max(params_mani["min_pts_per_seq"], params_mani["k_nn"] + 5),
        )
        if Xs is None:
            continue
        X_list[i] = Xs
        Y_list[i] = Ys
        valid_pts[i] = npts
        valid_seq.append(i)

    valid_seq = sorted(valid_seq)
    if len(valid_seq) < 10:
        raise RuntimeError(f"Too few valid sequences after filtering: {len(valid_seq)}/{n_seq}")

    print(f"Valid sequences: {len(valid_seq)}/{n_seq} | valid_pts mean={np.nanmean(valid_pts):.1f} | q10={_v1_strict_q(valid_pts, 0.10):.1f}")

    # -----------------------------
    # Per-seq computation with mismatch oversample+early-stop
    # -----------------------------
    # arrays over all n_seq (NaN for invalid)
    knn_jacc = np.full(n_seq, np.nan)
    spec_rmse = np.full(n_seq, np.nan)
    proc_resid = np.full(n_seq, np.nan)
    geo_w1 = np.full(n_seq, np.nan)

    mis_med = {k: np.full(n_seq, np.nan) for k in ["knn", "spec", "proc", "geo"]}
    mis_sc  = {k: np.full(n_seq, np.nan) for k in ["knn", "spec", "proc", "geo"]}
    mis_used = {k: np.zeros(n_seq, dtype=int) for k in ["knn", "spec", "proc", "geo"]}
    mis_tried = np.zeros(n_seq, dtype=int)

    geo_used_pairs = np.full(n_seq, np.nan)
    geo_pair_tries = np.full(n_seq, np.nan)

    # per-component score01 + composite
    s_knn  = np.full(n_seq, np.nan)
    s_spec = np.full(n_seq, np.nan)
    s_proc = np.full(n_seq, np.nan)
    s_geo  = np.full(n_seq, np.nan)
    score01 = np.full(n_seq, np.nan)

    w = params_mani["weights"]
    mapping = params_mani["mapping"]

    def _compute_all_metrics(X, Y, rng_pair):
        # 1) kNN Jaccard
        Axb = _knn_adj(X, k=params_mani["k_nn"], weighted=False)
        Ayb = _knn_adj(Y, k=params_mani["k_nn"], weighted=False)
        j = _jaccard_from_adj(Axb, Ayb)

        # 2) Laplacian spectrum RMSE
        Wx = _knn_adj(X, k=params_mani["k_nn"], weighted=True)
        Wy = _knn_adj(Y, k=params_mani["k_nn"], weighted=True)
        ex = _laplacian_eigs_from_weighted_adj(Wx, top=params_mani["top_eigs"])
        ey = _laplacian_eigs_from_weighted_adj(Wy, top=params_mani["top_eigs"])
        sr = _spectral_rmse(ex, ey)

        # 3) Procrustes residual in GT PCA space
        pr, _, _, _, _ = _procrustes_resid_in_gt_pca(X, Y, pca_dim=min(8, n_reg))

        # 4) Geodesic W1
        gw1, _, _, used_pairs, tries = _geodesic_w1(
            X, Y,
            k=params_mani["k_nn"],
            pair_samples=params_mani["geo_pair_samples"],
            rng=rng_pair,
        )

        return j, sr, pr, gw1, used_pairs, tries

    for i in valid_seq:
        Xi = X_list[i]
        Yi = Y_list[i]

        # MATCH metrics
        rng_i = _v1_rng_for_seq(params_mani["seed"], i)
        j, sr, pr, gw1, used_pairs, tries = _compute_all_metrics(Xi, Yi, rng_i)

        knn_jacc[i] = j
        spec_rmse[i] = sr
        proc_resid[i] = pr
        geo_w1[i] = gw1
        geo_used_pairs[i] = used_pairs
        geo_pair_tries[i] = tries

        # MISMATCH: sample candidates deterministically
        pool = np.array([s for s in valid_seq if s != i], dtype=int)
        if pool.size == 0:
            continue

        # We early-stop once EACH component has >= mismatch_M finite values
        target = int(params_mani["mismatch_M"])
        oversamp = int(params_mani["oversample_factor"])
        allow_replace = bool(params_mani["allow_replace"])

        mis_vals = {k: [] for k in ["knn", "spec", "proc", "geo"]}

        # deterministic candidate stream, but order independent of external list mutation
        # we may need multiple rounds if many NaNs
        max_rounds = 6
        for r in range(max_rounds):
            if all(len(mis_vals[k]) >= target for k in mis_vals):
                break

            need = max(target - min(len(mis_vals[k]) for k in mis_vals), 1)
            draw_n = int(min(pool.size, need * oversamp)) if not allow_replace else int(need * oversamp)

            if allow_replace:
                js = rng_i.choice(pool, size=draw_n, replace=True)
            else:
                draw_n = min(draw_n, pool.size)
                js = rng_i.choice(pool, size=draw_n, replace=False)

            for j_idx in js:
                mis_tried[i] += 1
                Xj = Xi

                Yj = Y_list[int(j_idx)]
                if Yj is None:
                    continue

                # equalize cloud size for comparability
                m = int(min(Xj.shape[0], Yj.shape[0]))
                if m < max(params_mani["min_pts_per_seq"], params_mani["k_nn"] + 5):
                    continue

                # deterministic per (i,j) RNG for subsampling/geo pairing
                rng_ij = np.random.default_rng(params_mani["seed"] + 1000003 * int(i) + 9176 * int(j_idx))
                idx_x = rng_ij.choice(Xj.shape[0], size=m, replace=False)
                idx_y = rng_ij.choice(Yj.shape[0], size=m, replace=False)
                Xm = Xj[idx_x]
                Ym = Yj[idx_y]

                jj, ssr, ppr, ggw1, _, _ = _compute_all_metrics(Xm, Ym, rng_ij)

                if np.isfinite(jj)  and len(mis_vals["knn"])  < target: mis_vals["knn"].append(float(jj))
                if np.isfinite(ssr) and len(mis_vals["spec"]) < target: mis_vals["spec"].append(float(ssr))
                if np.isfinite(ppr) and len(mis_vals["proc"]) < target: mis_vals["proc"].append(float(ppr))
                if np.isfinite(ggw1)and len(mis_vals["geo"])  < target: mis_vals["geo"].append(float(ggw1))

                if all(len(mis_vals[k]) >= target for k in mis_vals):
                    break

        # per-component mismatch summaries
        for k in ["knn", "spec", "proc", "geo"]:
            mis_used[k][i] = len(mis_vals[k])
            if len(mis_vals[k]) >= 5:
                mis_med[k][i] = float(np.median(mis_vals[k]))
                sc = _v1_robust_mad_scale(mis_vals[k])
                mis_sc[k][i] = sc if np.isfinite(sc) else np.nan

    # -----------------------------
    # pooled scales (v1 default) + z->score mapping
    # -----------------------------
    pooled_sc = {}
    for k in ["knn", "spec", "proc", "geo"]:
        scs = mis_sc[k][np.isfinite(mis_sc[k])]
        pooled = float(np.median(scs)) if scs.size else np.nan
        floor = float(params_mani["sc_floor"][k])
        if not np.isfinite(pooled) or pooled < floor:
            pooled = floor
        pooled_sc[k] = pooled

    print("\n--- pooled mismatch scales (used for z if pooled_scale=True) ---")
    for k in pooled_sc:
        print(f"  sc_{k} = {pooled_sc[k]:.6g} (floor {params_mani['sc_floor'][k]})")

    def _effect_z(match, med, sc, high_is_good: bool):
        if not (np.isfinite(match) and np.isfinite(med) and np.isfinite(sc) and sc > 0):
            return np.nan
        return float((match - med) / sc) if high_is_good else float((med - match) / sc)

    z_knn = np.full(n_seq, np.nan)
    z_spec = np.full(n_seq, np.nan)
    z_proc = np.full(n_seq, np.nan)
    z_geo = np.full(n_seq, np.nan)

    for i in valid_seq:
        # mismatch_min gating per component
        ok = all(mis_used[k][i] >= params_mani["mismatch_min"] for k in ["knn","spec","proc","geo"])
        if not ok:
            continue

        # choose scale source
        if params_mani["pooled_scale"]:
            sc_knn  = pooled_sc["knn"]
            sc_spec = pooled_sc["spec"]
            sc_proc = pooled_sc["proc"]
            sc_geo  = pooled_sc["geo"]
        else:
            sc_knn  = max(mis_sc["knn"][i],  params_mani["sc_floor"]["knn"])  if np.isfinite(mis_sc["knn"][i])  else np.nan
            sc_spec = max(mis_sc["spec"][i], params_mani["sc_floor"]["spec"]) if np.isfinite(mis_sc["spec"][i]) else np.nan
            sc_proc = max(mis_sc["proc"][i], params_mani["sc_floor"]["proc"]) if np.isfinite(mis_sc["proc"][i]) else np.nan
            sc_geo  = max(mis_sc["geo"][i],  params_mani["sc_floor"]["geo"])  if np.isfinite(mis_sc["geo"][i])  else np.nan

        z_knn[i]  = _effect_z(knn_jacc[i],  mis_med["knn"][i],  sc_knn,  high_is_good=True)
        z_spec[i] = _effect_z(spec_rmse[i], mis_med["spec"][i], sc_spec, high_is_good=False)
        z_proc[i] = _effect_z(proc_resid[i],mis_med["proc"][i], sc_proc, high_is_good=False)
        z_geo[i]  = _effect_z(geo_w1[i],    mis_med["geo"][i],  sc_geo,  high_is_good=False)

        s_knn[i]  = _v1_sigmoid_score01(z_knn[i],  mapping)
        s_spec[i] = _v1_sigmoid_score01(z_spec[i], mapping)
        s_proc[i] = _v1_sigmoid_score01(z_proc[i], mapping)
        s_geo[i]  = _v1_sigmoid_score01(z_geo[i],  mapping)

        if all(np.isfinite(x) for x in [s_knn[i], s_spec[i], s_proc[i], s_geo[i]]):
            score01[i] = float(w["knn"]*s_knn[i] + w["spec"]*s_spec[i] + w["proc"]*s_proc[i] + w["geo"]*s_geo[i])

    # -----------------------------
    # summaries + artifact contract
    # -----------------------------
    coverage_seq = float(np.mean(np.isfinite(score01))) if n_seq else 0.0
    insufficient_frac = float(np.mean([
        not all(mis_used[k][i] >= params_mani["mismatch_min"] for k in ["knn","spec","proc","geo"])
        for i in valid_seq
    ])) if valid_seq else np.nan

    def _mean_or_nan(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else np.nan

    def _min_or_nan(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        return float(np.min(x)) if x.size else np.nan

    scalars = {
        "MANI_coverage_seq": coverage_seq,
        "MANI_core_score01_mean": _mean_or_nan(score01),
        "MANI_core_score01_q10": _v1_strict_q(score01, 0.10),

        "MANI_valid_pts_mean": _mean_or_nan(valid_pts),
        "MANI_valid_pts_q10": _v1_strict_q(valid_pts, 0.10),

        "MANI_mismatch_insufficient_frac": insufficient_frac,
        "MANI_mismatch_attempted_mean": float(np.mean(mis_tried[valid_seq])) if valid_seq else np.nan,

        "MANI_mismatch_used_knn_mean": float(np.mean(mis_used["knn"][valid_seq])) if valid_seq else np.nan,
        "MANI_mismatch_used_spec_mean": float(np.mean(mis_used["spec"][valid_seq])) if valid_seq else np.nan,
        "MANI_mismatch_used_proc_mean": float(np.mean(mis_used["proc"][valid_seq])) if valid_seq else np.nan,
        "MANI_mismatch_used_geo_mean": float(np.mean(mis_used["geo"][valid_seq])) if valid_seq else np.nan,

        "MANI_mismatch_used_knn_min": float(np.min(mis_used["knn"][valid_seq])) if valid_seq else np.nan,
        "MANI_mismatch_used_spec_min": float(np.min(mis_used["spec"][valid_seq])) if valid_seq else np.nan,
        "MANI_mismatch_used_proc_min": float(np.min(mis_used["proc"][valid_seq])) if valid_seq else np.nan,
        "MANI_mismatch_used_geo_min": float(np.min(mis_used["geo"][valid_seq])) if valid_seq else np.nan,

        "MANI_geo_failed_frac": float(np.mean(np.isnan(geo_w1[valid_seq]))) if valid_seq else np.nan,
        "MANI_geo_used_pairs_mean": _mean_or_nan(geo_used_pairs),
        "MANI_geo_used_pairs_q10": _v1_strict_q(geo_used_pairs, 0.10),

        "MANI_pooled_sc_knn": float(pooled_sc["knn"]),
        "MANI_pooled_sc_spec": float(pooled_sc["spec"]),
        "MANI_pooled_sc_proc": float(pooled_sc["proc"]),
        "MANI_pooled_sc_geo": float(pooled_sc["geo"]),
    }

    print("\n--- Benchmark scalars (USE THESE for composite) ---")
    for k in sorted(scalars):
        print(f"{k:28s}: {scalars[k]:.4f}" if np.isfinite(scalars[k]) else f"{k:28s}: nan")

    df = pd.DataFrame({
        "sequence_id": np.arange(n_seq),
        "valid_pts": valid_pts,
        "knn_jaccard": knn_jacc,
        "spec_rmse": spec_rmse,
        "proc_resid": proc_resid,
        "geo_w1": geo_w1,
        "mis_med_knn": mis_med["knn"],
        "mis_med_spec": mis_med["spec"],
        "mis_med_proc": mis_med["proc"],
        "mis_med_geo": mis_med["geo"],
        "mis_sc_knn": mis_sc["knn"],
        "mis_sc_spec": mis_sc["spec"],
        "mis_sc_proc": mis_sc["proc"],
        "mis_sc_geo": mis_sc["geo"],
        "mis_used_knn": mis_used["knn"],
        "mis_used_spec": mis_used["spec"],
        "mis_used_proc": mis_used["proc"],
        "mis_used_geo": mis_used["geo"],
        "mis_tried": mis_tried,
        "z_knn": z_knn, "z_spec": z_spec, "z_proc": z_proc, "z_geo": z_geo,
        "s_knn": s_knn, "s_spec": s_spec, "s_proc": s_proc, "s_geo": s_geo,
        "score01": score01,
    }).round(6)

    # safe display
    try:
        display(df[df["score01"].notna()].sort_values("score01", ascending=False).head(10))
    except NameError:
        print(df[df["score01"].notna()].sort_values("score01", ascending=False).head(10).to_string(index=False))

    mani_out = dict(
        scalars=scalars,
        score01=score01,      # len n_seq with NaNs
        params=params_mani,
        diagnostics=dict(
            valid_seq=np.array(valid_seq, dtype=int),
            valid_pts=valid_pts,
            mis_tried=mis_tried,
            mis_used_knn=mis_used["knn"],
            mis_used_spec=mis_used["spec"],
            mis_used_proc=mis_used["proc"],
            mis_used_geo=mis_used["geo"],
            geo_used_pairs=geo_used_pairs,
            geo_pair_tries=geo_pair_tries,
            pooled_scales=pooled_sc,
        ),
        df=df,
    )

    # -----------------------------
    # Plots (no fixed colors; default matplotlib)
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].hist(df["score01"].dropna(), bins=24, edgecolor="k", alpha=0.7)
    axes[0].set_title("MANI FINAL score01")
    axes[0].set_xlabel("score01"); axes[0].set_ylabel("count"); axes[0].grid(alpha=0.25)

    axes[1].hist(df["s_knn"].dropna(), bins=24, edgecolor="k", alpha=0.7, label="kNN")
    axes[1].hist(df["s_spec"].dropna(), bins=24, edgecolor="k", alpha=0.5, label="spec")
    axes[1].set_title("Component scores (kNN/spec)")
    axes[1].set_xlabel("score01"); axes[1].grid(alpha=0.25); axes[1].legend()

    axes[2].hist(df["s_proc"].dropna(), bins=24, edgecolor="k", alpha=0.7, label="proc")
    axes[2].hist(df["s_geo"].dropna(), bins=24, edgecolor="k", alpha=0.5, label="geo")
    axes[2].set_title("Component scores (proc/geo)")
    axes[2].set_xlabel("score01"); axes[2].grid(alpha=0.25); axes[2].legend()

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Representative sequence diagnostics
    # -----------------------------
    finite_ids = np.where(np.isfinite(score01))[0]
    if finite_ids.size:
        med = np.nanmedian(score01[finite_ids])
        rep_seq = int(finite_ids[np.argmin(np.abs(score01[finite_ids] - med))])
        rep_row = df.iloc[rep_seq]
        print(
            f"\n[rep] seq={rep_seq} | score01={rep_row['score01']:.3f} | "
            f"s_knn={rep_row['s_knn']:.3f} s_spec={rep_row['s_spec']:.3f} s_proc={rep_row['s_proc']:.3f} s_geo={rep_row['s_geo']:.3f}"
        )

        Xr = X_list[rep_seq]
        Yr = Y_list[rep_seq]

        # Procrustes/PCA objects for plotting
        pr, Zx, Zy, Zy_al, _ = _procrustes_resid_in_gt_pca(Xr, Yr, pca_dim=min(8, n_reg))
        Wx_r = _knn_adj(Xr, k=params_mani["k_nn"], weighted=True)
        Wy_r = _knn_adj(Yr, k=params_mani["k_nn"], weighted=True)
        ex_r = _laplacian_eigs_from_weighted_adj(Wx_r, top=params_mani["top_eigs"])
        ey_r = _laplacian_eigs_from_weighted_adj(Wy_r, top=params_mani["top_eigs"])
        rng_rep = _v1_rng_for_seq(params_mani["seed"], rep_seq)
        gw1_r, dgt_r, dpr_r, used_pairs_r, tries_r = _geodesic_w1(
            Xr, Yr, k=params_mani["k_nn"], pair_samples=params_mani["geo_pair_samples"], rng=rng_rep
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

        # PCA2D overlay
        if Zx is not None and Zy_al is not None and Zx.shape[1] >= 2 and Zy_al.shape[1] >= 2:
            axes[0].scatter(Zx[:, 0], Zx[:, 1], s=12, alpha=0.55, label="GT (PCA)")
            axes[0].scatter(Zy_al[:, 0], Zy_al[:, 1], s=12, alpha=0.55, label="Pred (Proj+Proc)")
            axes[0].set_title(f"PCA2D + Procrustes (resid={pr:.3f})")
            axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].grid(alpha=0.25); axes[0].legend()
        else:
            axes[0].text(0.5, 0.5, "PCA/Procrustes failed", ha="center", va="center")
            axes[0].axis("off")

        # Laplacian eigs
        if ex_r is not None and ey_r is not None:
            axes[1].plot(ex_r, marker="o", label="GT")
            axes[1].plot(ey_r, marker="x", label="Pred")
            axes[1].set_title(f"Laplacian eigs (top {min(len(ex_r),len(ey_r))})\nRMSE={_spectral_rmse(ex_r, ey_r):.4g}")
            axes[1].set_xlabel("index"); axes[1].set_ylabel("eigenvalue")
            axes[1].grid(alpha=0.25); axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, "No eigs (graph issue)", ha="center", va="center")
            axes[1].axis("off")

        # Geodesic distance distributions
        if dgt_r is not None and dpr_r is not None:
            axes[2].hist(dgt_r, bins=40, density=True, alpha=0.6, label="GT")
            axes[2].hist(dpr_r, bins=40, density=True, alpha=0.6, label="Pred")
            axes[2].set_title(f"Geodesic dists\nW1={gw1_r:.3f} | used={used_pairs_r}/{tries_r}")
            axes[2].set_xlabel("distance"); axes[2].set_ylabel("density")
            axes[2].grid(alpha=0.25); axes[2].legend()
        else:
            axes[2].text(0.5, 0.5, "No geodesics (disconnected)", ha="center", va="center")
            axes[2].axis("off")

        plt.tight_layout()
        plt.show()
    else:
        print("No finite MANI scores (all sequences failed mismatch_min or metric computation).")

    # Top / bottom quick view
    top = df[df["score01"].notna()].sort_values("score01", ascending=False).head(12)
    bot = df[df["score01"].notna()].sort_values("score01", ascending=True).head(12)
    try:
        display(top[["sequence_id","score01","s_knn","s_spec","s_proc","s_geo","valid_pts","mis_used_knn","mis_used_spec","mis_used_proc","mis_used_geo"]])
        display(bot[["sequence_id","score01","s_knn","s_spec","s_proc","s_geo","valid_pts","mis_used_knn","mis_used_spec","mis_used_proc","mis_used_geo"]])
    except NameError:
        print("\nTop 12:\n", top[["sequence_id","score01","s_knn","s_spec","s_proc","s_geo"]].to_string(index=False))
        print("\nBottom 12:\n", bot[["sequence_id","score01","s_knn","s_spec","s_proc","s_geo"]].to_string(index=False))

# --- Bandpower realism (GLOBAL overall statistics; fine-bins; per-region joint masking; distribution-based; non-saturating 0–1; benchmark-ready) ---
# Benchmark design goals:
#   ✅ deterministic (seeded), no infinite loops, bounded runtime controls
#   ✅ robust to NaNs/constant signals/tiny ranges
#   ✅ shape uses cosine of demeaned GLOBAL mean log-spectrum (Spearman is report-only)
#   ✅ magnitude uses per-bin Wasserstein-1 (W1) on log-power distributions, normalized by GT robust scale (W1_norm)
#   ✅ mapping to [floor, cap] is smooth + non-saturating; uses tau from W1_norm-bin distribution (stable fallback)
#   ✅ null control (time-shuffle Pred) is diagnostic only (NOT used for final score); fast via subset + limited draws
#   ✅ returns a single dict with everything needed for logging
#
# Inputs expected in globals():
#   - gt_array_denorm: np.ndarray (n_seq, T, n_reg)
#   - data_predicted:  np.ndarray (n_seq, T, n_reg_pred or similar; must align after resample)
#   - _resample_pred_to_gt(gt, pred) -> (gt_aligned, pred_aligned) same shape
# Optional:
#   - gt_regions: list/array of region names

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import spearmanr, wasserstein_distance

# ----------------------------
# 0) Prereqs + align data
# ----------------------------
_required = ["gt_array_denorm", "data_predicted", "_resample_pred_to_gt"]
_missing = [x for x in _required if x not in globals()]
if _missing:
    raise RuntimeError(f"Missing prerequisites: {_missing}")

_gt, _pr = _resample_pred_to_gt(gt_array_denorm, data_predicted)
assert _gt.shape == _pr.shape, f"Aligned mismatch: GT {_gt.shape} vs Pred {_pr.shape}"
n_seq, T, n_reg = _gt.shape
region_names = list(gt_regions) if "gt_regions" in globals() else [f"R{i}" for i in range(n_reg)]
print(f"Aligned shapes: GT={_gt.shape} | Pred={_pr.shape}")

# ----------------------------
# 1) Params (SET TRUE fs!)
# ----------------------------
fs = 30.0
nyq = fs / 2.0

# Welch defaults tuned for stability at fs=30Hz; auto-trim if T is short
nperseg = int(min(256, T))
noverlap = int(min(192, max(0, nperseg - 1)))

# Fine bins for scoring (1..Nyquist), last bin clipped
fine_bins = [
    ("b0:1-2",    1.0,  2.0),
    ("b1:2-4",    2.0,  4.0),
    ("b2:4-6",    4.0,  6.0),
    ("b3:6-8",    6.0,  8.0),
    ("b4:8-10",   8.0, 10.0),
    ("b5:10-12", 10.0, 12.0),
    ("b6:12-14", 12.0, 14.0),
    ("b7:14-15", 14.0, min(15.0, nyq - 1e-6)),
]
fine_bins = [(nm, lo, hi) for (nm, lo, hi) in fine_bins if (lo < hi) and (hi <= nyq + 1e-9)]
bin_names = [b[0] for b in fine_bins]
n_bins = len(fine_bins)
if n_bins < 3:
    raise RuntimeError(f"Not enough supported bins for fs={fs} (Nyquist={nyq}). Redefine fine_bins.")

# Score mapping parameters (gentle, non-saturating)
floor, cap = 0.05, 0.95
w_shape, w_mag = 0.55, 0.45

# Shape mapping: cosine->01 uses power gamma (lower gamma = less harsh, higher = harsher)
shape_gamma = 2.0

# Magnitude mapping: exp(- (err/tau)^gamma ) in W1_norm units (tau is auto-fit)
mag_gamma = 1.5

# Minimum data required
min_t_needed = max(32, nperseg)  # per (seq,reg) joint-valid timepoints

# Null diagnostics (NOT used for composite)
null_time_shuffle = True
null_M = 20
null_seed = 0
null_sample_N = 600  # only recompute Welch on this many valid (seq,reg) signals per null draw

# Plot toggles (turn off in automated benchmark runs if needed)
DO_PLOTS = True

print(f"Params: fs={fs:g} Hz | Nyquist={nyq:g} | nperseg={nperseg} | noverlap={noverlap} | min_t_needed={min_t_needed}")
print(f"Bins (fine/score): {bin_names}")
print(f"Null(diag): time-shuffle={null_time_shuffle} | null_M={null_M} | null_sample_N={null_sample_N}")

# ----------------------------
# 2) Helpers (robust + vector-safe)
# ----------------------------
EPS = 1e-12

def _robust_scale_1d(x):
    """Robust scale for 1D arrays that doesn't collapse (MAD -> IQR -> std -> 1)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        s = np.nanstd(x) if x.size else np.nan
        return float(s if np.isfinite(s) and s > 1e-12 else 1.0)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    s_mad = 1.4826 * mad
    if np.isfinite(s_mad) and s_mad > 1e-12:
        return float(s_mad)
    q75, q25 = np.nanpercentile(x, [75, 25])
    iqr = q75 - q25
    s_iqr = iqr / 1.349 if np.isfinite(iqr) else np.nan
    if np.isfinite(s_iqr) and s_iqr > 1e-12:
        return float(s_iqr)
    s = np.nanstd(x)
    return float(s if np.isfinite(s) and s > 1e-12 else 1.0)

def _spearman_safe(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    r = spearmanr(x[m], y[m], nan_policy="omit").correlation
    return float(r) if np.isfinite(r) else np.nan

def _cosine_demeaned(a, b, eps=1e-12):
    """Cosine similarity after demeaning (focuses on spectral shape, ignores mean shift)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    aa = a[m] - np.mean(a[m])
    bb = b[m] - np.mean(b[m])
    na = np.linalg.norm(aa) + eps
    nb = np.linalg.norm(bb) + eps
    return float(np.dot(aa, bb) / (na * nb))

def _soft01_from_cos(cosv, gamma=2.0, floor=0.05, cap=0.95):
    """Map cosine in [-1,1] -> [floor,cap], non-saturating-ish via power gamma."""
    cosv = np.asarray(cosv, float)
    out = np.full_like(cosv, np.nan, dtype=float)
    m = np.isfinite(cosv)
    if np.any(m):
        r01 = np.clip((cosv[m] + 1.0) / 2.0, 0.0, 1.0) ** float(gamma)
        out[m] = float(floor) + (float(cap) - float(floor)) * r01
    return out if out.shape != () else float(out)

def _soft01_from_err(err, tau, gamma=1.5, floor=0.05, cap=0.95):
    """Map err>=0 -> [floor,cap] using exp(- (err/tau)^gamma )."""
    err = np.asarray(err, float)
    out = np.full_like(err, np.nan, dtype=float)
    tau = max(float(tau), 1e-9)
    m = np.isfinite(err)
    if np.any(m):
        e = np.maximum(0.0, err[m])
        s = np.exp(-((e / tau) ** float(gamma)))
        out[m] = float(floor) + (float(cap) - float(floor)) * s
    return out if out.shape != () else float(out)

def _safe_hist(ax, x, bins=25, seed=0, **kwargs):
    """Histogram that won't crash if x has tiny range / constant values."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        ax.text(0.5, 0.5, "no finite data", ha="center", va="center")
        ax.set_axis_off()
        return
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if np.isclose(x_min, x_max):
        jitter = 1e-6 if x_min == 0 else abs(x_min) * 1e-6
        rng = np.random.default_rng(seed)
        x = x + rng.normal(scale=jitter, size=x.shape)
    uniq = np.unique(np.round(x, 12)).size
    b = int(min(bins, max(5, int(np.sqrt(x.size))), max(5, uniq)))
    ax.hist(x, bins=b, **kwargs)

def _welch_bandpower_1d(x, fs, nperseg, noverlap, fine_bins):
    """Return bandpowers for one 1D signal x in fine bins. Returns None if too short."""
    x = np.asarray(x, float)
    if x.size < 16:
        return None
    nps = int(min(nperseg, x.size))
    if nps < 16:
        return None
    nov = int(min(noverlap, max(0, nps - 1)))
    f, Pxx = signal.welch(x, fs=fs, nperseg=nps, noverlap=nov)
    bp = np.full(len(fine_bins), np.nan, float)
    for bi, (_, lo, hi) in enumerate(fine_bins):
        m = (f >= lo) & (f < hi)
        if np.any(m):
            bp[bi] = np.trapezoid(Pxx[m], f[m])
    return bp

# ----------------------------
# 3) Compute bandpower per (seq,reg) with per-region joint masking
# ----------------------------
bp_gt = np.full((n_seq, n_reg, n_bins), np.nan, float)
bp_pr = np.full((n_seq, n_reg, n_bins), np.nan, float)

for s in range(n_seq):
    for r in range(n_reg):
        x = _gt[s, :, r]
        y = _pr[s, :, r]
        mt = np.isfinite(x) & np.isfinite(y)  # JOINT per-region mask
        if mt.sum() < min_t_needed:
            continue
        xr = x[mt].astype(np.float64, copy=False)
        yr = y[mt].astype(np.float64, copy=False)

        bpX = _welch_bandpower_1d(xr, fs, nperseg, noverlap, fine_bins)
        bpY = _welch_bandpower_1d(yr, fs, nperseg, noverlap, fine_bins)
        if bpX is None or bpY is None:
            continue

        # require all bins finite for the (seq,reg) to be valid
        if np.isfinite(bpX).all() and np.isfinite(bpY).all():
            bp_gt[s, r, :] = bpX
            bp_pr[s, r, :] = bpY

ok_sr = np.isfinite(bp_gt).all(axis=2) & np.isfinite(bp_pr).all(axis=2)  # (n_seq,n_reg)
n_ok = int(ok_sr.sum())
print(f"\nValid (seq,region) with full bins: {n_ok}/{n_seq*n_reg} ({n_ok/max(1,n_seq*n_reg)*100:.1f}%)")
if n_ok < 10:
    raise RuntimeError("Too few valid (seq,region) entries. Check NaNs, fs, nperseg/noverlap, bins.")

log_gt = np.log(bp_gt + EPS)
log_pr = np.log(bp_pr + EPS)

# Flatten valid entries: (N, n_bins)
X = log_gt[ok_sr]
Y = log_pr[ok_sr]

# ----------------------------
# 4) GLOBAL metrics (overall realism)
# ----------------------------
# 4a) Shape: global mean spectrum across all valid (seq,reg)
mu_gt = np.nanmean(X, axis=0)
mu_pr = np.nanmean(Y, axis=0)
shape_spearman = _spearman_safe(mu_gt, mu_pr)     # report-only
shape_cos = _cosine_demeaned(mu_gt, mu_pr)        # scoring primitive (less saturating)

# 4b) Magnitude: per-bin distribution gap (W1 on log-power), normalized by GT robust scale
w1_bins = np.full(n_bins, np.nan, float)
for bi in range(n_bins):
    a = X[:, bi]; b = Y[:, bi]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() >= 20:
        w1_bins[bi] = float(wasserstein_distance(a[m], b[m]))

gt_scale_bins = np.array([_robust_scale_1d(X[:, bi]) for bi in range(n_bins)], dtype=float)
w1n_bins = w1_bins / (gt_scale_bins + 1e-9)

mag_w1_mean = float(np.nanmean(w1_bins))
mag_w1n_mean = float(np.nanmean(w1n_bins))

# Offset-corrected diagnostic: remove mean shift per bin and recompute W1_norm
mean_shift_bins = (mu_pr - mu_gt) / (gt_scale_bins + 1e-9)  # in GT-scale units
Y_shifted = Y - (mu_pr - mu_gt)[None, :]
w1n_bins_off = np.full(n_bins, np.nan, float)
for bi in range(n_bins):
    a = X[:, bi]; b = Y_shifted[:, bi]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() >= 20:
        w1n_bins_off[bi] = float(wasserstein_distance(a[m], b[m]) / (gt_scale_bins[bi] + 1e-9))
mag_w1n_off_mean = float(np.nanmean(w1n_bins_off))

# Choose tau for magnitude mapping (stable + non-saturating):
# Use a robust central value of w1n across bins. Fallbacks avoid tau->0.
finite_w1n = w1n_bins[np.isfinite(w1n_bins)]
if finite_w1n.size >= 3:
    tau_mag = float(np.nanmedian(finite_w1n))
    # if distribution is extremely tight, broaden tau a bit to avoid over-penalizing small deviations
    tau_mag = max(tau_mag, 0.25 * float(_robust_scale_1d(finite_w1n)), 1e-3)
else:
    tau_mag = 1.0

# ----------------------------
# 5) Map to 0–1 (FINAL BENCHMARK SCORE)
# ----------------------------
shape01 = float(_soft01_from_cos(shape_cos, gamma=shape_gamma, floor=floor, cap=cap))
mag01   = float(_soft01_from_err(mag_w1n_mean, tau=tau_mag, gamma=mag_gamma, floor=floor, cap=cap))
score01 = float(w_shape * shape01 + w_mag * mag01)

# ----------------------------
# 6) Per-seq diagnostics (not used for global score; for debugging)
# ----------------------------
shape_cos_seq = np.full(n_seq, np.nan, float)
mag_w1n_seq   = np.full(n_seq, np.nan, float)

for s in range(n_seq):
    ok_r = ok_sr[s]
    if ok_r.sum() < 4:
        continue

    mu_s_gt = np.nanmean(log_gt[s, ok_r, :], axis=0)
    mu_s_pr = np.nanmean(log_pr[s, ok_r, :], axis=0)
    shape_cos_seq[s] = _cosine_demeaned(mu_s_gt, mu_s_pr)

    # per-bin W1_norm across regions (distribution across regions at fixed seq)
    w1n_s = []
    for bi in range(n_bins):
        a = log_gt[s, ok_r, bi]
        b = log_pr[s, ok_r, bi]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() >= 4:
            w1n_s.append(float(wasserstein_distance(a[m], b[m]) / (gt_scale_bins[bi] + 1e-9)))
    if len(w1n_s) >= max(3, n_bins // 2):
        mag_w1n_seq[s] = float(np.nanmean(w1n_s))

shape01_seq = _soft01_from_cos(shape_cos_seq, gamma=shape_gamma, floor=floor, cap=cap)
mag01_seq   = _soft01_from_err(mag_w1n_seq, tau=tau_mag, gamma=mag_gamma, floor=floor, cap=cap)
score01_seq = w_shape * shape01_seq + w_mag * mag01_seq

# ----------------------------
# 7) Null control (diagnostic only): time-shuffle Pred and recompute on a subset
# ----------------------------
null_shape_cos = np.full(null_M, np.nan, float)
null_mag_w1n   = np.full(null_M, np.nan, float)
null_score01   = np.full(null_M, np.nan, float)

if null_time_shuffle:
    rng = np.random.default_rng(null_seed)

    sr_idx = np.argwhere(ok_sr)  # (n_ok, 2)
    if sr_idx.shape[0] > null_sample_N:
        sr_idx = sr_idx[rng.choice(sr_idx.shape[0], size=null_sample_N, replace=False)]

    # Fixed GT log bandpowers for sampled (seq,reg)
    X_null = np.vstack([log_gt[s, r, :][None, :] for (s, r) in sr_idx])

    for m in range(null_M):
        Y_rows = []
        for (s, r) in sr_idx:
            x = _gt[s, :, r]
            y = _pr[s, :, r]
            mt = np.isfinite(x) & np.isfinite(y)
            if mt.sum() < min_t_needed:
                Y_rows.append(np.full((1, n_bins), np.nan, float))
                continue

            yr = y[mt].astype(np.float64, copy=False)
            yr_sh = yr[rng.permutation(yr.size)]

            bpY = _welch_bandpower_1d(yr_sh, fs, nperseg, noverlap, fine_bins)
            if bpY is None or not np.isfinite(bpY).all():
                Y_rows.append(np.full((1, n_bins), np.nan, float))
            else:
                Y_rows.append(np.log(bpY + EPS)[None, :])

        Y_null = np.vstack(Y_rows)
        mm = np.isfinite(X_null).all(axis=1) & np.isfinite(Y_null).all(axis=1)
        if mm.sum() < 20:
            continue

        mu_gt_n = np.mean(X_null[mm], axis=0)
        mu_pr_n = np.mean(Y_null[mm], axis=0)

        shape_cos_n = _cosine_demeaned(mu_gt_n, mu_pr_n)

        w1n_list = []
        for bi in range(n_bins):
            a = X_null[mm, bi]
            b = Y_null[mm, bi]
            if a.size >= 20:
                w1n_list.append(float(wasserstein_distance(a, b) / (gt_scale_bins[bi] + 1e-9)))
        if len(w1n_list) < 3:
            continue
        mag_w1n_n = float(np.mean(w1n_list))

        shape01_n = float(_soft01_from_cos(shape_cos_n, gamma=shape_gamma, floor=floor, cap=cap))
        mag01_n   = float(_soft01_from_err(mag_w1n_n, tau=tau_mag, gamma=mag_gamma, floor=floor, cap=cap))
        score01_n = float(w_shape * shape01_n + w_mag * mag01_n)

        null_shape_cos[m] = shape_cos_n
        null_mag_w1n[m]   = mag_w1n_n
        null_score01[m]   = score01_n

# ----------------------------
# 8) Reporting
# ----------------------------
print("\n=== GLOBAL bandpower realism (BENCHMARK SCORE = score01) ===")
print(f"n_seq={n_seq} | n_reg={n_reg} | valid(seq,reg)={n_ok} | fine_bins={n_bins}")
print(f"Shape: Spearman(mean spectra) r={shape_spearman:.3f} (report-only)")
print(f"Shape: cosine(demeaned mean)  cos={shape_cos:.3f} -> shape01={shape01:.3f}")
print(f"Magnitude: mean W1 per bin = {mag_w1_mean:.3f} (raw)")
print(f"Magnitude: mean W1_norm     = {mag_w1n_mean:.3f} | tau_mag={tau_mag:.3f} -> mag01={mag01:.3f}")
print(f"FINAL score01 = {score01:.3f}  (weights shape={w_shape:.2f}, mag={w_mag:.2f})")

if null_time_shuffle:
    nf = np.isfinite(null_score01)
    if nf.any():
        print("\n--- Null diagnostic (time-shuffle Pred; not used for scoring) ---")
        print(f"null score01: mean={float(np.nanmean(null_score01)):.3f} | median={float(np.nanmedian(null_score01)):.3f} | "
              f"q95={float(np.nanpercentile(null_score01[nf],95)):.3f}")

# Per-bin table
bin_df = pd.DataFrame({
    "bin": bin_names,
    "mu_gt_logP": mu_gt,
    "mu_pr_logP": mu_pr,
    "mean_shift_GTscale": mean_shift_bins,
    "w1_logP": w1_bins,
    "gt_scale_logP": gt_scale_bins,
    "w1_norm": w1n_bins,
    "w1_norm_offcorr": w1n_bins_off,
}).round(4)
display(bin_df)

# ----------------------------
# 9) Plots (optional)
# ----------------------------
if DO_PLOTS:
    # Plot 1: global mean spectrum
    plt.figure(figsize=(7, 3))
    plt.plot(mu_gt, marker="o", label="GT mean log-power")
    plt.plot(mu_pr, marker="o", label="Pred mean log-power")
    plt.xticks(range(n_bins), bin_names, rotation=45, ha="right")
    plt.title(f"Global mean spectrum (log-bandpower)\ncos(demean)={shape_cos:.3f} | Spearman={shape_spearman:.3f}")
    plt.ylabel("mean log-power")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: W1 and W1_norm per bin
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
    axes[0].bar(range(n_bins), w1_bins)
    axes[0].set_xticks(range(n_bins)); axes[0].set_xticklabels(bin_names, rotation=45, ha="right")
    axes[0].set_title(f"W1(log-power) per bin | mean={mag_w1_mean:.3f}")
    axes[0].set_ylabel("W1"); axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(range(n_bins), w1n_bins)
    axes[1].set_xticks(range(n_bins)); axes[1].set_xticklabels(bin_names, rotation=45, ha="right")
    axes[1].set_title(f"W1_norm per bin | mean={mag_w1n_mean:.3f}")
    axes[1].set_ylabel("W1 / GT_scale"); axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

    # Plot 3: selected bin distribution overlays (log-power)
    show_bins = list(range(min(4, n_bins))) + (list(range(n_bins - 2, n_bins)) if n_bins > 6 else [])
    show_bins = sorted(set([b for b in show_bins if 0 <= b < n_bins]))
    fig, axes = plt.subplots(1, len(show_bins), figsize=(3.2 * len(show_bins), 2.8), sharey=False)
    if len(show_bins) == 1:
        axes = [axes]
    for ax, bi in zip(axes, show_bins):
        a = X[:, bi]; b = Y[:, bi]
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        ax.hist(a, bins=25, alpha=0.6, density=True, label="GT")
        ax.hist(b, bins=25, alpha=0.6, density=True, label="Pred")
        ax.set_title(f"{bin_names[bi]}\nW1n={w1n_bins[bi]:.2f}")
        ax.set_xlabel("log-power")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Selected bins: GT vs Pred distributions (log-bandpower)", y=1.05)
    plt.tight_layout()
    plt.show()

    # Plot 4: Region × Bin heatmap of W1_norm (per region, across sequences)
    w1n_region_bin = np.full((n_reg, n_bins), np.nan, float)
    for r in range(n_reg):
        ok_s = ok_sr[:, r]
        if ok_s.sum() < 20:
            continue
        A = log_gt[ok_s, r, :]
        B = log_pr[ok_s, r, :]
        for bi in range(n_bins):
            a = A[:, bi]; b = B[:, bi]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() >= 20:
                w1n_region_bin[r, bi] = float(wasserstein_distance(a[m], b[m]) / (gt_scale_bins[bi] + 1e-9))

    plt.figure(figsize=(10, 4))
    plt.imshow(w1n_region_bin, aspect="auto")
    plt.xticks(range(n_bins), bin_names, rotation=45, ha="right")
    plt.yticks(range(n_reg), region_names, fontsize=7)
    plt.title("Region × Bin Wasserstein gap (W1_norm on log-power)")
    plt.colorbar(fraction=0.02, label="W1 / GT_scale")
    plt.tight_layout()
    plt.show()

    # Plot 5: per-seq diagnostic distributions (shape_cos, mag_w1n, implied score01)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2))
    _safe_hist(axes[0], shape_cos_seq, bins=25, alpha=0.75, edgecolor="k")
    axes[0].axvline(shape_cos, linestyle="--", linewidth=2)
    axes[0].set_title("Per-seq shape_cos (mean over regions)")
    axes[0].set_xlabel("shape_cos"); axes[0].grid(alpha=0.25)

    _safe_hist(axes[1], mag_w1n_seq, bins=25, alpha=0.75, edgecolor="k")
    axes[1].axvline(mag_w1n_mean, linestyle="--", linewidth=2)
    axes[1].set_title("Per-seq magnitude gap (W1_norm)")
    axes[1].set_xlabel("W1_norm"); axes[1].grid(alpha=0.25)

    _safe_hist(axes[2], score01_seq, bins=25, alpha=0.75, edgecolor="k")
    axes[2].axvline(score01, linestyle="--", linewidth=2)
    axes[2].set_title("Per-seq implied score01")
    axes[2].set_xlabel("score01"); axes[2].grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    # Plot 6: null distributions (diagnostic)
    if null_time_shuffle:
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.0))
        _safe_hist(axes[0], null_shape_cos, bins=25, alpha=0.75, edgecolor="k")
        axes[0].axvline(shape_cos, linestyle="--", linewidth=2)
        axes[0].set_title("Null (time-shuffle Pred): shape_cos")
        axes[0].set_xlabel("shape_cos"); axes[0].grid(alpha=0.25)

        _safe_hist(axes[1], null_mag_w1n, bins=25, alpha=0.75, edgecolor="k")
        axes[1].axvline(mag_w1n_mean, linestyle="--", linewidth=2)
        axes[1].set_title("Null (time-shuffle Pred): mag W1_norm")
        axes[1].set_xlabel("W1_norm"); axes[1].grid(alpha=0.25)

        _safe_hist(axes[2], null_score01, bins=25, alpha=0.75, edgecolor="k")
        axes[2].axvline(score01, linestyle="--", linewidth=2)
        axes[2].set_title("Null (time-shuffle Pred): score01")
        axes[2].set_xlabel("score01"); axes[2].grid(alpha=0.25)

        plt.tight_layout()
        plt.show()

# ----------------------------
# 10) Summary dict for logging (BENCHMARK SCORE = score01)
# ----------------------------

if MODE in ('full', 'instant'):
    bandpower_global_realism = dict(
        version="bandpower_global_realism_v4_benchmark_score01",
        params=dict(
            fs=float(fs), nyq=float(nyq), nperseg=int(nperseg), noverlap=int(noverlap),
            min_t_needed=int(min_t_needed),
            fine_bins=fine_bins,
            floor=float(floor), cap=float(cap),
            weights=dict(w_shape=float(w_shape), w_mag=float(w_mag)),
            shape_gamma=float(shape_gamma),
            mag_tau=float(tau_mag),
            mag_gamma=float(mag_gamma),
            null_time_shuffle=bool(null_time_shuffle),
            null_M=int(null_M),
            null_seed=int(null_seed),
            null_sample_N=int(null_sample_N),
            do_plots=bool(DO_PLOTS),
        ),
        coverage_valid_seqreg=float(n_ok / max(1, n_seq * n_reg)),
        global_metrics=dict(
            shape_spearman=float(shape_spearman),
            shape_cos=float(shape_cos),
            mag_w1_mean=float(mag_w1_mean),
            mag_w1n_mean=float(mag_w1n_mean),
            mag_w1n_offcorr_mean=float(mag_w1n_off_mean),
            shape01=float(shape01),
            mag01=float(mag01),
            score01=float(score01),
        ),
        per_bin=dict(
            bin_names=list(bin_names),
            mu_gt_logP=mu_gt,
            mu_pr_logP=mu_pr,
            mean_shift_GTscale=mean_shift_bins,
            w1_logP=w1_bins,
            gt_scale_logP=gt_scale_bins,
            w1_norm=w1n_bins,
            w1_norm_offcorr=w1n_bins_off,
        ),
        per_seq_diag=dict(
            shape_cos_seq=shape_cos_seq,
            mag_w1n_seq=mag_w1n_seq,
            shape01_seq=shape01_seq,
            mag01_seq=mag01_seq,
            score01_seq=score01_seq,
        ),
        null=dict(
            null_shape_cos=null_shape_cos,
            null_mag_w1n=null_mag_w1n,
            null_score01=null_score01,
        ) if null_time_shuffle else None,
    )

    bandpower_global_realism
