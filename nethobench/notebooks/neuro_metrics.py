#!/usr/bin/env python
# coding: utf-8

# # Neuro Score
# 
# This notebook defines the active **neural realism score** used by NethoBench. The score is intentionally focused on the metric families that remained most discriminative and scientifically robust across the project:
# 
# - **distribution**
# - **temporal**
# - **relational**
# - **geometry**
# - **state dynamics**
# 
# The direct pointwise fidelity family is no longer folded into the default neural benchmark. It is now exposed separately through the `fidelity-scores` command.
# 
# ## Composite definition
# 
# The active neuro score is
# 
# \[
# S_{neuro}
# = 0.22 D
# + 0.18 T
# + 0.24 R
# + 0.18 G
# + 0.18 S_d,
# \]
# 
# where
# 
# \[
# D = \frac{1}{4}\left(
# s_{KL/JSD}
# + s_{QNT}
# + s_{MOM}
# + s_{Mean}
# \right),
# \]
# 
# \[
# T = s_{TRJDIST},
# \]
# 
# \[
# R = \frac{1}{4}\left(
# s_{GRAPH}
# + s_{CrossRegionMI}
# + s_{LaggedCovariance}
# + s_{ImpulseResponse}
# \right),
# \]
# 
# \[
# G = \frac{1}{2}\left(
# s_{MANI}
# + s_{SubspaceAngle}
# \right),
# \]
# 
# and
# 
# \[
# S_d = \frac{1}{5}\left(
# s_{Occ11}
# + s_{Occ12}
# + s_{Trans1}
# + s_{Trans2}
# + s_{Trans3}
# \right).
# \]
# 
# Every scalar metric is bounded in \([0,1]\), with larger values meaning better agreement between ground truth and prediction.
# 

# ## Setup
# 
# This notebook is runnable both interactively and through the package CLI. The setup cell imports the scientific stack, inserts the repository root into `sys.path` when needed, and defines the default file-path variables patched by the CLI runner.
# 

# In[ ]:


%matplotlib inline

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

_cwd = Path.cwd().resolve()
_candidate_roots = [_cwd, *_cwd.parents, Path.home() / "nethobench", Path.home() / "Desktop" / "nethobench"]
_repo_root = None
for root in _candidate_roots:
    root = root.resolve()
    if (root / "nethobench" / "__init__.py").is_file():
        _repo_root = root
        break
if _repo_root is not None and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from nethobench.neuro import _load_sequences

preds_fname = "predictions.csv"
gt_fname = "ground_truth.csv"
ddconfig_path = "ddconfig.json"

pred_arr_raw, pred_region_names = _load_sequences(Path(preds_fname))
gt_arr_raw, gt_regions = _load_sequences(Path(gt_fname))

shared_regions = [name for name in gt_regions if name in pred_region_names]
if not shared_regions:
    raise ValueError("No overlapping neural regions between GT and predictions.")

gt_idx = [gt_regions.index(name) for name in shared_regions]
pred_idx = [pred_region_names.index(name) for name in shared_regions]

gt_array_denorm = np.asarray(gt_arr_raw[:, :, gt_idx], dtype=np.float64)
data_predicted = np.asarray(pred_arr_raw[:, :, pred_idx], dtype=np.float64)

T = min(gt_array_denorm.shape[1], data_predicted.shape[1])
gt_array_denorm = gt_array_denorm[:, :T, :]
data_predicted = data_predicted[:, :T, :]
pred_region_names = shared_regions

print(f"GT shape           : {gt_array_denorm.shape}")
print(f"Prediction shape   : {data_predicted.shape}")
print(f"Shared region count: {len(shared_regions)}")


# ## Data Loading and Alignment
# 
# This cell loads the prediction and ground-truth CSVs, aligns them by shared neural regions and common horizon, and creates the aligned tensors
# 
# \[
# X^{gt}, X^{pred} \in \mathbb{R}^{N_{seq} \times T \times R}.
# \]
# 
# All downstream metrics operate on these same aligned arrays.
# 

# ## Distribution Realism (Symmetric KL)
# This cell compares the marginal value distribution of each region in ground truth and prediction. For each region, the cell builds matched empirical histograms, computes a symmetric KL-type divergence between the two distributions, aggregates the resulting discrepancies across regions, and maps the final divergence to a bounded realism score in `[0, 1]`, with `1` meaning indistinguishable marginals. The same cell also applies the notebook corruption families and plots how the score degrades as the corruption magnitude increases.

# In[ ]:


# === Distribution realism (KL only): baseline + corruption degradation ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Formula used in this cell
# 1) Per sequence s and region r, build trimmed histograms p_{s,r}, q_{s,r}
# 2) Symmetric KL: KL_sym(s,r) = 0.5 * [KL(p||q) + KL(q||p)]
# 3) Region similarity: sim(s,r) = 1 / (1 + KL_sym(s,r))
# 4) Per-sequence KL score: KL_geo(s) = exp(mean_r(log(sim(s,r))))
# 5) Aggregate scalars:
#       KL_mean = mean_s(KL_geo(s))
#       KL_q10  = q10_s(KL_geo(s))
#       KL_score01_avg = 0.5 * (KL_mean + KL_q10)


def _resample_pred_to_gt_local(gt_arr, pred_arr):
    gt_len = gt_arr.shape[1]
    pred_len = pred_arr.shape[1]
    if pred_len != gt_len and pred_len % gt_len == 0:
        factor = pred_len // gt_len
        pred_res = pred_arr.reshape(pred_arr.shape[0], gt_len, factor, pred_arr.shape[2]).mean(axis=2)
        gt_cmp = gt_arr
    elif pred_len != gt_len:
        min_len = min(gt_len, pred_len)
        gt_cmp = gt_arr[:, :min_len, :]
        pred_res = pred_arr[:, :min_len, :]
    else:
        gt_cmp = gt_arr
        pred_res = pred_arr
    return gt_cmp, pred_res


def _tail_binned_hist(gt_vals, pr_vals, bins=60, support_q=(0.001, 0.999), eps=1e-12):
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


def _compute_kl_metrics(gt_arr, pred_arr, bins=60, support_q=(0.001, 0.999)):
    assert gt_arr.shape == pred_arr.shape
    n_seq, _, n_reg = gt_arr.shape

    kl_sym_sr = np.full((n_seq, n_reg), np.nan, dtype=np.float64)
    for s in range(n_seq):
        for r in range(n_reg):
            p, q = _tail_binned_hist(gt_arr[s, :, r], pred_arr[s, :, r], bins=bins, support_q=support_q)
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
        kl_score01_avg = 0.5 * (kl_mean + kl_q10)
    else:
        kl_mean = np.nan
        kl_q10 = np.nan
        kl_score01_avg = np.nan

    return {
        'kl_sym_per_seq_region': kl_sym_sr,
        'kl_geo_seq': kl_geo_seq,
        'KL_mean': kl_mean,
        'KL_q10': kl_q10,
        'KL_score01_avg': kl_score01_avg,
    }


def _plot_distributions_once(gt_arr, pred_arr, region_names, bins=60, clip_q=(0.001, 0.999)):
    n_reg = gt_arr.shape[2]
    if region_names is None or len(region_names) != n_reg:
        region_names = [f'R{i}' for i in range(n_reg)]

    ncols = 4
    nrows = int(np.ceil(n_reg / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
    axes = axes.ravel()

    for r, name in enumerate(region_names):
        gt_vals = gt_arr[:, :, r].ravel().astype(np.float64)
        pr_vals = pred_arr[:, :, r].ravel().astype(np.float64)
        gt_vals = gt_vals[np.isfinite(gt_vals)]
        pr_vals = pr_vals[np.isfinite(pr_vals)]
        if gt_vals.size == 0 or pr_vals.size == 0:
            axes[r].axis('off')
            continue

        lo = min(np.quantile(gt_vals, clip_q[0]), np.quantile(pr_vals, clip_q[0]))
        hi = max(np.quantile(gt_vals, clip_q[1]), np.quantile(pr_vals, clip_q[1]))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = float(np.min(gt_vals)), float(np.max(gt_vals))
            if lo >= hi:
                hi = lo + 1e-6

        edges = np.linspace(lo, hi, bins)
        ax = axes[r]
        ax.hist(gt_vals, bins=edges, alpha=0.5, label='GT', density=True)
        ax.hist(pr_vals, bins=edges, alpha=0.5, label='Pred', density=True)
        ax.set_title(name, fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    for ax in axes[n_reg:]:
        ax.axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle('GT vs Pred distributions per region (plotted once at baseline)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()


# Align once and keep names compatible with previous sections
_gt_aligned, _pred_aligned = _resample_pred_to_gt_local(gt_array_denorm, data_predicted)
base_gt = np.asarray(_gt_aligned, dtype=np.float64)
base_pred = np.asarray(_pred_aligned, dtype=np.float64)
region_names = globals().get('pred_region_names', None)

# Baseline distribution plot (only once)
_plot_distributions_once(base_gt, base_pred, region_names=region_names, bins=60, clip_q=(0.001, 0.999))

# Baseline KL metrics
baseline = _compute_kl_metrics(base_gt, base_pred, bins=60, support_q=(0.001, 0.999))

# Keep a dist_out-compatible object with KL-only summary for downstream inspection
dist_out = {
    'summary': {
        'KL_geo': {
            'mean': baseline['KL_mean'],
            'q10_strict': baseline['KL_q10'],
        }
    },
    'kl_geo_seq': baseline['kl_geo_seq'],
}
dist_scores = {
    'KL_qsupport_geo': baseline['KL_q10'],
    'KL_score01_avg': baseline['KL_score01_avg'],
}

# Same corruption families as previous sweep
flat_gt = base_gt.reshape(-1, base_gt.shape[-1])
region_q25 = np.nanquantile(flat_gt, 0.25, axis=0)
region_q75 = np.nanquantile(flat_gt, 0.75, axis=0)
region_iqr = np.where(np.isfinite(region_q75 - region_q25) & ((region_q75 - region_q25) > 1e-12), region_q75 - region_q25, 1.0)
region_iqr_3d = region_iqr.reshape(1, 1, -1)

def _corrupt_pred(pred, mode, magnitude, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    corrupted = np.array(pred, copy=True)
    if mode == 'mean_shift':
        corrupted = corrupted + magnitude * region_iqr_3d
    elif mode == 'variance_scale':
        center = np.nanmedian(corrupted, axis=1, keepdims=True)
        corrupted = center + (1.0 + magnitude) * (corrupted - center)
    elif mode == 'tail_spikes':
        spike_prob = min(0.02 + 0.08 * magnitude, 0.35)
        mask = rng.random(corrupted.shape) < spike_prob
        spikes = rng.laplace(loc=0.0, scale=magnitude, size=corrupted.shape) * region_iqr_3d
        corrupted = corrupted + mask * spikes
    else:
        raise ValueError(f'Unknown corruption mode: {mode}')
    return corrupted

magnitudes = [0.10, 0.25, 0.50, 0.75, 1.00]
corruptions = ['mean_shift', 'variance_scale', 'tail_spikes']

rows = [{
    'corruption': 'baseline',
    'magnitude': 0.0,
    'KL_mean': baseline['KL_mean'],
    'KL_q10': baseline['KL_q10'],
    'KL_score01_avg': baseline['KL_score01_avg'],
}]

for mode in corruptions:
    for magnitude in magnitudes:
        pred_corrupt = _corrupt_pred(base_pred, mode, magnitude, rng_seed=1000 + int(100 * magnitude))
        out = _compute_kl_metrics(base_gt, pred_corrupt, bins=60, support_q=(0.001, 0.999))
        rows.append({
            'corruption': mode,
            'magnitude': magnitude,
            'KL_mean': out['KL_mean'],
            'KL_q10': out['KL_q10'],
            'KL_score01_avg': out['KL_score01_avg'],
        })

kl_sweep_df = pd.DataFrame(rows)
kl_sweep_df['delta_mean'] = kl_sweep_df['KL_mean'] - baseline['KL_mean']
kl_sweep_df['delta_q10'] = kl_sweep_df['KL_q10'] - baseline['KL_q10']
kl_sweep_df['delta_avg'] = kl_sweep_df['KL_score01_avg'] - baseline['KL_score01_avg']

print('KL-only corruption sweep (higher is better for KL_mean, KL_q10, KL_score01_avg)')
display(kl_sweep_df.round(4))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
metrics = ['KL_mean', 'KL_q10', 'KL_score01_avg']
titles = ['KL mean', 'KL q10 (strict)', 'KL_score01_avg = 0.5*(mean+q10)']

for ax, metric, title in zip(axes, metrics, titles):
    for mode in corruptions:
        subset = kl_sweep_df[kl_sweep_df['corruption'] == mode]
        ax.plot(subset['magnitude'], subset[metric], marker='o', label=mode)
    ax.axhline(baseline[metric], linestyle='--', color='black', linewidth=1, label='baseline')
    ax.set_title(title)
    ax.set_xlabel('corruption magnitude')
    ax.set_ylabel('score')
    ax.grid(alpha=0.25)

axes[0].legend()
plt.tight_layout()
plt.show()


# ## Mean-Shift Realism
# This cell measures whether predictions preserve the region-wise location statistics of the data. In practice, the code computes region-level mean shifts between `X^{gt}` and `X^{pred}`, emphasizes the most affected regions according to the notebook aggregation rule, and transforms the resulting discrepancy into `Mean_score01 \in [0, 1]`. The corruption sweep then tests whether this score decreases when additive or structured corruptions push the prediction means away from the ground-truth means.

# In[ ]:


# === Mean difference metric (Top-10% regions) ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Formula implemented in this cell:
# mu_gt(i,r)   = mean_t GT[i,t,r]
# mu_pr(i,r)   = mean_t Pred[i,t,r]
# d(i,r)       = |mu_gt(i,r)-mu_pr(i,r)| / (IQR_gt(r) + eps)
# K            = ceil(0.1 * R)
# D_i_top10    = mean of top-K largest d(i,r) over regions r
# D            = mean_i D_i_top10
# Mean_score01 = 1 / (1 + D)


def _align_gt_pred_top10(gt_arr, pred_arr):
    gt_arr = np.asarray(gt_arr, dtype=np.float64)
    pred_arr = np.asarray(pred_arr, dtype=np.float64)

    gt_len = gt_arr.shape[1]
    pr_len = pred_arr.shape[1]
    if pr_len != gt_len and pr_len % gt_len == 0:
        factor = pr_len // gt_len
        pred_arr = pred_arr.reshape(pred_arr.shape[0], gt_len, factor, pred_arr.shape[2]).mean(axis=2)
    elif pr_len != gt_len:
        m = min(gt_len, pr_len)
        gt_arr = gt_arr[:, :m, :]
        pred_arr = pred_arr[:, :m, :]

    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f'Aligned mismatch: {gt_arr.shape} vs {pred_arr.shape}')
    return gt_arr, pred_arr


def _iqr_robust(x):
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


def compute_mean_score01_top10(gt_arr, pred_arr, eps=1e-12):
    gt_arr, pred_arr = _align_gt_pred_top10(gt_arr, pred_arr)
    n_seq, _, n_reg = gt_arr.shape

    mu_gt = np.nanmean(gt_arr, axis=1)  # (N, R)
    mu_pr = np.nanmean(pred_arr, axis=1)  # (N, R)

    iqr_gt = np.array([_iqr_robust(gt_arr[:, :, r].reshape(-1)) for r in range(n_reg)], dtype=np.float64)
    d = np.abs(mu_gt - mu_pr) / (iqr_gt[None, :] + eps)  # (N, R)

    K = int(np.ceil(0.1 * n_reg))
    K = max(1, K)

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
    Mean_score01 = float(1.0 / (1.0 + D)) if np.isfinite(D) else np.nan

    return {
        'D_seq_top10': D_top10_seq,
        'D': D,
        'K': K,
        'R': n_reg,
        'N': n_seq,
        'Mean_score01': Mean_score01,
    }


def _build_mean_corruption(pred, mode, mag, region_iqr, global_iqr, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    pred = np.asarray(pred, dtype=np.float64)
    out = np.array(pred, copy=True)

    if mode == 'global_mean_shift':
        out = out + (mag * global_iqr)
    elif mode == 'region_mean_shift':
        pattern = np.linspace(-1.0, 1.0, pred.shape[2])
        out = out + (mag * region_iqr[None, None, :] * pattern[None, None, :])
    elif mode == 'sequence_mean_shift':
        seq_offsets = rng.normal(loc=0.0, scale=mag * global_iqr, size=(pred.shape[0], 1, 1))
        out = out + seq_offsets
    elif mode == 'slow_drift':
        T = pred.shape[1]
        trend = np.linspace(0.0, 1.0, T, dtype=np.float64).reshape(1, T, 1)
        out = out + (mag * global_iqr) * trend
    else:
        raise ValueError(f'Unknown corruption mode: {mode}')
    return out


# Baseline
base_gt, base_pred = _align_gt_pred_top10(gt_array_denorm, data_predicted)
region_iqr = np.array([_iqr_robust(base_gt[:, :, r].reshape(-1)) for r in range(base_gt.shape[2])], dtype=np.float64)
global_iqr = _iqr_robust(base_gt.reshape(-1))

baseline = compute_mean_score01_top10(base_gt, base_pred)

rows = [{
    'corruption': 'baseline',
    'magnitude': 0.0,
    'D_top10_mean': baseline['D'],
    'Mean_score01': baseline['Mean_score01'],
}]

corruptions = ['global_mean_shift', 'region_mean_shift', 'sequence_mean_shift', 'slow_drift']
magnitudes = [0.10, 0.25, 0.50, 0.75, 1.00]

for mode in corruptions:
    for mag in magnitudes:
        pred_corrupt = _build_mean_corruption(
            base_pred,
            mode=mode,
            mag=mag,
            region_iqr=region_iqr,
            global_iqr=global_iqr,
            rng_seed=1000 + int(mag * 100),
        )
        out = compute_mean_score01_top10(base_gt, pred_corrupt)
        rows.append({
            'corruption': mode,
            'magnitude': mag,
            'D_top10_mean': out['D'],
            'Mean_score01': out['Mean_score01'],
        })

mean_top10_df = pd.DataFrame(rows)
display(mean_top10_df[['corruption', 'magnitude', 'D_top10_mean', 'Mean_score01']].round(4))

# Score degradation plot
plt.figure(figsize=(7.0, 4.5))
for mode in corruptions:
    sub = mean_top10_df[mean_top10_df['corruption'] == mode]
    plt.plot(sub['magnitude'], sub['Mean_score01'], marker='o', label=mode)
plt.axhline(baseline['Mean_score01'], linestyle='--', color='black', linewidth=1, label='baseline')
plt.title('Mean_score01 (Top-10% regions): 1/(1 + D)')
plt.xlabel('corruption magnitude')
plt.ylabel('Mean_score01 (higher better)')
plt.grid(alpha=0.25)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Per-sequence diagnostic (D_i^top10 distribution)
plt.figure(figsize=(6.5, 4.2))
vals = baseline['D_seq_top10']
vals = vals[np.isfinite(vals)]
plt.hist(vals, bins=25, alpha=0.8, edgecolor='k')
if vals.size:
    plt.axvline(float(np.mean(vals)), linestyle='--', linewidth=2, label='mean(D_i^top10)')
plt.title(f'Baseline D_i^top10 distribution (K={baseline["K"]}, R={baseline["R"]})')
plt.xlabel('D_i^top10')
plt.ylabel('count')
plt.grid(alpha=0.25)
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# ## Quantile and Tail-Shape Realism
# This cell asks whether predictions reproduce the distributional shape of each region, not just its mean. It compares matched quantiles or tail summaries between ground truth and predictions, aggregates the resulting discrepancies, and maps them into `QNT_score01`. Because the score is driven by quantile structure, it is sensitive to asymmetric tails, broadening, compression, and other changes that may not be visible in a mean-only metric.

# In[ ]:


# === Quantile / tail realism (simple, strict, benchmark-friendly) ===
# Formula:
#   Q_gt(i,r,q)    = quantile_q of GT[i,:,r]
#   Q_pr(i,r,q)    = quantile_q of Pred[i,:,r]
#   d_tail(i,r)    = mean_{q in tails} |Q_gt(i,r,q) - Q_pr(i,r,q)| / (IQR_GT(r) + eps)
#   D_i            = mean of top-25% largest d_tail(i,r) across regions
#   D              = mean_i D_i
#   QNT_score01    = 1 / (1 + D)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS = 1e-12

# ----------------------------
# 0) Alignment helper
# ----------------------------
def _resample_pred_to_gt_fallback(gt_arr, pred_arr):
    gt_len = gt_arr.shape[1]
    pr_len = pred_arr.shape[1]
    if pr_len != gt_len and pr_len % gt_len == 0:
        factor = pr_len // gt_len
        pred_res = pred_arr.reshape(pred_arr.shape[0], gt_len, factor, pred_arr.shape[2]).mean(axis=2)
        return gt_arr, pred_res
    elif pr_len != gt_len:
        m = min(gt_len, pr_len)
        return gt_arr[:, :m, :], pred_arr[:, :m, :]
    return gt_arr, pred_arr

def _align_gt_pred(gt_arr, pred_arr):
    gt_arr = np.asarray(gt_arr, dtype=np.float64)
    pred_arr = np.asarray(pred_arr, dtype=np.float64)
    if "_resample_pred_to_gt" in globals() and callable(globals()["_resample_pred_to_gt"]):
        return globals()["_resample_pred_to_gt"](gt_arr, pred_arr)
    return _resample_pred_to_gt_fallback(gt_arr, pred_arr)

# ----------------------------
# 1) Robust helpers
# ----------------------------
def _robust_iqr(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 10:
        s = np.nanstd(x)
        return float(s if np.isfinite(s) and s > 1e-12 else 1.0)
    q25, q75 = np.nanquantile(x, [0.25, 0.75])
    s = float(q75 - q25)
    if not np.isfinite(s) or s <= 1e-12:
        s = float(np.nanstd(x))
    return float(s if np.isfinite(s) and s > 1e-12 else 1.0)

def _topq_mean(x, q=0.25):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    k = max(1, int(np.ceil(q * x.size)))
    return float(np.mean(np.sort(x)[-k:]))

def _safe_nanquantile_1d(x, q):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.quantile(x, q))

# ----------------------------
# 2) Main computation
# ----------------------------
def compute_quantile_score01_simple(
    gt_arr,
    pred_arr,
    region_names=None,
    q_lo=0.01,
    q_hi=0.99,
    n_q=99,
    tail_lo=0.10,
    tail_hi=0.90,
    min_samples=80,
    max_time=1200,
    top_q_regions=0.25,
    rng_seed=0,
):
    gt_arr, pred_arr = _align_gt_pred(gt_arr, pred_arr)
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"Aligned mismatch: {gt_arr.shape} vs {pred_arr.shape}")

    n_seq, _, n_reg = gt_arr.shape
    rng = np.random.default_rng(rng_seed)

    if region_names is None:
        region_names = [f"R{i}" for i in range(n_reg)]
    else:
        region_names = list(region_names)
        if len(region_names) != n_reg:
            region_names = [f"R{i}" for i in range(n_reg)]

    quantiles = np.linspace(q_lo, q_hi, int(n_q))
    tail_mask = (quantiles <= tail_lo) | (quantiles >= tail_hi)

    # GT-only scale per region
    iqr_gt = np.array([
        _robust_iqr(gt_arr[:, :, r].reshape(-1))
        for r in range(n_reg)
    ], dtype=np.float64)

    # Per-sequence, per-region tail quantile distance
    d_tail_sr = np.full((n_seq, n_reg), np.nan, dtype=np.float64)
    d_full_sr = np.full((n_seq, n_reg), np.nan, dtype=np.float64)

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
            d_full_sr[i, r] = float(np.mean(dq))
            d_tail_sr[i, r] = float(np.mean(dq[tail_mask]))

    # Per-sequence strict distance: mean of worst top-q regions
    D_seq = np.full(n_seq, np.nan, dtype=np.float64)
    for i in range(n_seq):
        D_seq[i] = _topq_mean(d_tail_sr[i], q=top_q_regions)

    # Final scalar
    D = float(np.nanmean(D_seq)) if np.isfinite(D_seq).any() else np.nan
    QNT_score01 = float(1.0 / (1.0 + D)) if np.isfinite(D) else np.nan

    # Region summaries
    per_region = pd.DataFrame({
        "region": region_names,
        "IQR_GT": iqr_gt,
        "D_tail_mean": np.nanmean(d_tail_sr, axis=0),
        "D_tail_q90": np.nanquantile(d_tail_sr, 0.90, axis=0),
        "D_full_mean": np.nanmean(d_full_sr, axis=0),
        "valid_seq_frac": np.mean(np.isfinite(d_tail_sr), axis=0),
    }).sort_values("D_tail_mean", ascending=False)

    per_seq = pd.DataFrame({
        "seq": np.arange(n_seq),
        "D_seq_topq": D_seq,
        "valid_regions": np.sum(np.isfinite(d_tail_sr), axis=1),
    }).sort_values("D_seq_topq", ascending=False)

    scores = {
        "QNT_tail_topq_mean": D,
        "QNT_score01": QNT_score01,
        "QNT_coverage_seq": float(np.mean(np.isfinite(D_seq))),
        "QNT_coverage_regions": float(np.mean(np.isfinite(iqr_gt))),
    }

    print("\n=== Quantile / tail realism (simple, strict) ===")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}" if np.isfinite(v) else f"  {k}: nan")

    return {
        "d_tail_sr": d_tail_sr,
        "d_full_sr": d_full_sr,
        "D_seq": D_seq,
        "per_region": per_region,
        "per_seq": per_seq,
        "scores": scores,
        "region_names": region_names,
        "params": {
            "q_lo": q_lo,
            "q_hi": q_hi,
            "n_q": n_q,
            "tail_lo": tail_lo,
            "tail_hi": tail_hi,
            "min_samples": min_samples,
            "max_time": max_time,
            "top_q_regions": top_q_regions,
        }
    }

# ----------------------------
# 3) Run
# ----------------------------
_region_names = None
if "pred_region_names" in globals() and globals().get("pred_region_names") is not None:
    _region_names = globals()["pred_region_names"]
elif "gt_regions" in globals() and globals().get("gt_regions") is not None:
    _region_names = globals()["gt_regions"]

qnt_simple = compute_quantile_score01_simple(
    gt_array_denorm,
    data_predicted,
    region_names=_region_names,
    q_lo=0.01,
    q_hi=0.99,
    n_q=99,
    tail_lo=0.10,
    tail_hi=0.90,
    min_samples=80,
    max_time=1200,
    top_q_regions=0.25,
    rng_seed=0,
)

display(qnt_simple["per_region"].round(4))

# ----------------------------
# 4) Plots
# ----------------------------
per_region = qnt_simple["per_region"].set_index("region").reindex(qnt_simple["region_names"])

plt.figure(figsize=(12, 4.5))
plt.bar(per_region.index, per_region["D_tail_mean"], alpha=0.85, edgecolor="k")
plt.xticks(rotation=90)
plt.ylabel("Mean tail quantile distance")
plt.title("Per-region tail quantile discrepancy")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3.5))
vals = qnt_simple["D_seq"]
vals = vals[np.isfinite(vals)]
plt.hist(vals, bins=20, alpha=0.8, edgecolor="k")
if vals.size:
    plt.axvline(np.mean(vals), linestyle="--", linewidth=2, label="mean(D_seq)")
plt.xlabel("Per-sequence top-q tail quantile distance")
plt.ylabel("Count")
plt.title("Distribution of strict per-sequence quantile error")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# Tail corruption sweep (merged)
# ----------------------------
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

if 'compute_quantile_score01_simple' not in globals():
    raise RuntimeError('compute_quantile_score01_simple is not available. Run the tail realism cell above first.')


def _align_local(gt_arr, pred_arr):
    gt_arr = np.asarray(gt_arr, dtype=np.float64)
    pred_arr = np.asarray(pred_arr, dtype=np.float64)
    if '_resample_pred_to_gt' in globals() and callable(globals()['_resample_pred_to_gt']):
        return globals()['_resample_pred_to_gt'](gt_arr, pred_arr)
    gt_len = gt_arr.shape[1]
    pr_len = pred_arr.shape[1]
    if pr_len != gt_len and pr_len % gt_len == 0:
        factor = pr_len // gt_len
        pred_arr = pred_arr.reshape(pred_arr.shape[0], gt_len, factor, pred_arr.shape[2]).mean(axis=2)
    elif pr_len != gt_len:
        m = min(gt_len, pr_len)
        gt_arr = gt_arr[:, :m, :]
        pred_arr = pred_arr[:, :m, :]
    return gt_arr, pred_arr


def _iqr_robust(x):
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


def _tail_scores_quiet(gt_arr, pred_arr, region_names):
    sink = io.StringIO()
    with redirect_stdout(sink):
        out = compute_quantile_score01_simple(
            gt_arr,
            pred_arr,
            region_names=region_names,
            q_lo=0.01,
            q_hi=0.99,
            n_q=99,
            tail_lo=0.10,
            tail_hi=0.90,
            min_samples=80,
            top_q_regions=0.25,
            max_time=1200,
            rng_seed=0,
        )
    scores = out.get('scores', {})
    return {
        'QNT_tail_D': float(scores.get('QNT_tail_topq_mean', np.nan)),
        'QNT_score01': float(scores.get('QNT_score01', np.nan)),
    }


def _variance_scale(pred_arr, a):
    out = np.array(pred_arr, copy=True)
    center = np.nanmedian(out, axis=1, keepdims=True)
    return center + float(a) * (out - center)


def _tail_spikes(pred_arr, p_spike, mag_scale, region_iqr, rng):
    out = np.array(pred_arr, copy=True)
    mask = rng.random(size=out.shape) < float(p_spike)
    spikes = rng.laplace(loc=0.0, scale=1.0, size=out.shape) * (float(mag_scale) * region_iqr[None, None, :])
    return out + mask * spikes


gt_base, pred_base = _align_local(gt_array_denorm, data_predicted)
region_names = globals().get('pred_region_names', globals().get('gt_regions', None))
region_iqr = np.array([_iqr_robust(gt_base[:, :, r].reshape(-1)) for r in range(gt_base.shape[2])], dtype=np.float64)

baseline = _tail_scores_quiet(gt_base, pred_base, region_names)
rows = [{
    'family': 'baseline',
    'level': 0.0,
    'QNT_tail_D': baseline['QNT_tail_D'],
    'QNT_score01': baseline['QNT_score01'],
}]

# 1) Variance scaling sweep (a)
variance_levels = [1.10, 1.25, 1.50, 1.75, 2.00]
for a in variance_levels:
    pred_c = _variance_scale(pred_base, a)
    sc = _tail_scores_quiet(gt_base, pred_c, region_names)
    rows.append({
        'family': 'variance_scale',
        'level': float(a),
        'QNT_tail_D': sc['QNT_tail_D'],
        'QNT_score01': sc['QNT_score01'],
    })

# 2) Tail spikes sweep (progressive sparse corruption)
spike_levels = [
    {'level': 1, 'p': 0.01, 'mag': 0.50},
    {'level': 2, 'p': 0.02, 'mag': 0.75},
    {'level': 3, 'p': 0.04, 'mag': 1.00},
    {'level': 4, 'p': 0.06, 'mag': 1.25},
    {'level': 5, 'p': 0.08, 'mag': 1.50},
]
rng = np.random.default_rng(2026)
for cfg in spike_levels:
    pred_c = _tail_spikes(pred_base, p_spike=cfg['p'], mag_scale=cfg['mag'], region_iqr=region_iqr, rng=rng)
    sc = _tail_scores_quiet(gt_base, pred_c, region_names)
    rows.append({
        'family': 'tail_spikes',
        'level': float(cfg['level']),
        'QNT_tail_D': sc['QNT_tail_D'],
        'QNT_score01': sc['QNT_score01'],
    })

qnt_corr_df = pd.DataFrame(rows)
display(qnt_corr_df.round(4))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

sub_var = qnt_corr_df[qnt_corr_df['family'] == 'variance_scale'].sort_values('level')
sub_sp = qnt_corr_df[qnt_corr_df['family'] == 'tail_spikes'].sort_values('level')

# Score plot
axes[0].plot(sub_var['level'], sub_var['QNT_score01'], marker='o', label='variance_scale (a)')
axes[0].plot(sub_sp['level'], sub_sp['QNT_score01'], marker='o', label='tail_spikes (level)')
axes[0].axhline(baseline['QNT_score01'], linestyle='--', color='black', linewidth=1, label='baseline')
axes[0].set_title('QNT_score01 degradation')
axes[0].set_xlabel('corruption level')
axes[0].set_ylabel('QNT_score01 (higher better)')
axes[0].grid(alpha=0.25)
axes[0].legend(loc='best')

# Raw tail distance plot
axes[1].plot(sub_var['level'], sub_var['QNT_tail_D'], marker='o', label='variance_scale (a)')
axes[1].plot(sub_sp['level'], sub_sp['QNT_tail_D'], marker='o', label='tail_spikes (level)')
axes[1].axhline(baseline['QNT_tail_D'], linestyle='--', color='black', linewidth=1, label='baseline')
axes[1].set_title('Tail distance (raw D)')
axes[1].set_xlabel('corruption level')
axes[1].set_ylabel('QNT_tail_D (lower better)')
axes[1].grid(alpha=0.25)
axes[1].legend(loc='best')

plt.tight_layout()
plt.show()

# This cell uses the archived perfected higher-order moment metric from the legacy benchmark branch. After alignment, it compares variance, skewness, and kurtosis region by region on flattened GT and prediction activity, builds a single distance `|log(var_p / var_g)| + 0.5 * |skew_p - skew_g| + 0.25 * |kurt_p - kurt_g|`, and maps that distance to the bounded score `MOM_score01 = 1 / (1 + distance)`. The code cell below runs that exact legacy perfected implementation and shows the original gain-scaling corruption sweep.

# In[ ]:


# === Higher-order moments realism (simple, strict, benchmark-friendly) ===
import pandas as pd

from nethobench.analysis.direct_neuro_metrics import compute_moment_score01
from nethobench.analysis.refined_neuro_metric_replacements import compute_moment_replacement

mom_simple = compute_moment_score01(gt_array_denorm, data_predicted)

display(pd.DataFrame([{
    "metric": "MOM_score01",
    "value": mom_simple["scores"]["MOM_score01"],
    "candidate_name": mom_simple["candidate_name"],
}]))

# 1) Tail spikes
_, mom_corr_df = compute_moment_replacement(
    gt_array_denorm,
    data_predicted,
    enable_plots=bool(globals().get("ENABLE_PLOTS", True)),
)
display(mom_corr_df.round(4))


# This cell uses the archived perfected graph metric from the legacy benchmark branch. It builds pooled GT and prediction correlation matrices, converts each into a binary strong-edge topology by keeping the top fraction of absolute edges, and compares three graph summaries: edge-set Jaccard overlap, weighted-degree agreement, and local clustering agreement. The final `GRAPH_score01` is the geometric mean of those valid graph-summary terms. The code cell below runs that exact legacy perfected implementation and shows the original region-permutation corruption sweep.

# In[ ]:


# === Corr-connectivity GRAPH realism (simple, strict, benchmark-friendly) ===
import numpy as np
import pandas as pd

from nethobench.analysis.direct_neuro_metrics import compute_graph_score01
from nethobench.analysis.refined_neuro_metric_replacements import compute_graph_replacement

graph_simple = compute_graph_score01(gt_array_denorm, data_predicted)

display(pd.DataFrame([{
    "metric": "GRAPH_score01",
    "value": graph_simple["scores"]["GRAPH_score01"],
    "GRAPH_jacc_mean": graph_simple["scores"].get("GRAPH_jacc_mean", np.nan),
    "GRAPH_deg_mean": graph_simple["scores"].get("GRAPH_deg_mean", np.nan),
    "GRAPH_cluster_mean": graph_simple["scores"].get("GRAPH_cluster_mean", np.nan),
    "candidate_name": graph_simple["candidate_name"],
}]))

# 1) Corruption sweep
_, graph_corr_df = compute_graph_replacement(
    gt_array_denorm,
    data_predicted,
    enable_plots=bool(globals().get("ENABLE_PLOTS", True)),
)
display(graph_corr_df.round(4))


# ## Manifold Realism
# This cell measures low-dimensional geometric realism with a GT-anchored manifold score. The active implementation combines persistent-homology lifetime agreement with local neighborhood geometry agreement, so `MANI_score01` is high only when both the coarse topological organization and the fine local manifold structure of the prediction match the real data. This is the finalized manifold replacement used by the benchmark.

# In[ ]:


# === MANIFOLD realism (simple geometry, strict, benchmark-friendly) ===
import pandas as pd

from nethobench.analysis.direct_neuro_metrics import compute_manifold_score01
from nethobench.analysis.refined_neuro_metric_replacements import compute_manifold_replacement

mani_simple = compute_manifold_score01(gt_array_denorm, data_predicted)

display(pd.DataFrame([{
    "metric": "MANI_score01",
    "value": mani_simple["scores"]["MANI_score01"],
    "candidate_name": mani_simple["candidate_name"],
}]))

# 1) Corruption sweep
_, mani_corr_df = compute_manifold_replacement(
    gt_array_denorm,
    data_predicted,
    enable_plots=bool(globals().get("ENABLE_PLOTS", True)),
)
display(mani_corr_df.round(4))


# ## Trajectory Realism
# This cell scores the distribution of trajectories in a shared GT-defined latent space. The active implementation combines occupancy-and-velocity agreement with a path-feature agreement term, so `TRJDIST_score01` rewards predictions that visit the same latent regions with similar motion statistics and similar path geometry. This is the finalized trajectory replacement used by the benchmark.

# In[ ]:


# === TRAJECTORY DISTRIBUTION realism (FIXED: global GT-PCA basis, pooled across sequences) ===
import pandas as pd

from nethobench.analysis.direct_neuro_metrics import compute_trajectory_score01
from nethobench.analysis.refined_neuro_metric_replacements import compute_trajectory_replacement

trajectory_dist_simple = compute_trajectory_score01(gt_array_denorm, data_predicted)

embedding_gt_seq = None
embedding_pred_seq = None
embedding_ref_seq = None
gt_aligned = gt_array_denorm
pred_aligned = data_predicted

display(pd.DataFrame([{
    "metric": "TRJDIST_score01",
    "value": trajectory_dist_simple["scores"]["TRJDIST_score01"],
    "candidate_name": trajectory_dist_simple["candidate_name"],
}]))

# -------------------------------------------------
# 6) Corruption sweep
_, trjdist_corr_df = compute_trajectory_replacement(
    gt_array_denorm,
    data_predicted,
    enable_plots=bool(globals().get("ENABLE_PLOTS", True)),
)
display(trjdist_corr_df)


# ## Relational, Geometry, and State-Dynamics Terms
# 
# This notebook computes four additional structural terms directly from aligned tensors:
# 
# ### Cross-region mutual-information structure
# 
# For each dataset we estimate a region-by-region mutual-information matrix
# 
# \[
# M_{ij} = \operatorname{MI}(x_i, x_j),
# \]
# 
# then compare the upper-triangular entries of \(M^{gt}\) and \(M^{pred}\) with a correlation-plus-RMSE similarity.
# 
# ### Lagged covariance
# 
# For lags \(\ell \in \{1,2,4\}\), the notebook forms
# 
# \[
# C_{\ell} = \frac{1}{T-\ell-1}(X_{1:T-\ell} - \bar X)^\top (X_{1+\ell:T} - \bar X),
# \]
# 
# and scores agreement between the GT and prediction lagged covariance matrices.
# 
# ### Impulse response
# 
# Each sequence stack is summarized by a ridge-regularized VAR(1) operator
# 
# \[
# x_{t+1} \approx A x_t,
# \]
# 
# and the resulting coefficient matrices are compared as a simple low-order impulse-response proxy.
# 
# ### Subspace angle
# 
# Let \(U\) and \(V\) be GT and prediction principal subspaces. The score is the mean squared cosine of the principal angles:
# 
# \[
# s_{subspace} = \frac{1}{k}\sum_{i=1}^{k} \cos^2 \theta_i.
# \]
# 
# ### State dynamics
# 
# State dynamics are defined in a **GT-derived latent space**. Ground truth is standardized, projected into a GT PCA basis, and clustered into discrete latent states. The notebook then compares:
# 
# - state occupancies for `k=11` and `k=12`
# - 1-step, 2-step, and 3-step transition structure for the `k=11` partition
# 
# This family tests whether the model visits the right metastable states and moves between them with the right short-horizon transition logic.
# 

# In[ ]:


# === ADDITIONAL STRUCTURAL METRICS (cross-region information, lagged dynamics, subspaces, states) ===
import pandas as pd

from nethobench.analysis.additional_neuro_metrics import compute_additional_structural_metrics

additional_structural = compute_additional_structural_metrics(gt_array_denorm, data_predicted)
additional_structural_scores = additional_structural.get('scores', {})

selected_structural_metrics = [
    'CrossRegionMI_score01',
    'SubspaceAngle_score01',
    'LaggedCovariance_score01',
    'ImpulseResponse_score01',
    'LatentStateOccupancyK11_score01',
    'LatentStateOccupancyK12_score01',
    'LatentStateTransitionLag1K11_score01',
    'LatentStateTransitionLag2K11_score01',
    'LatentStateTransitionLag3K11_score01',
]

display(
    pd.DataFrame(
        [{'metric': key, 'value': additional_structural_scores.get(key, float('nan'))} for key in selected_structural_metrics]
    )
    .sort_values('metric')
    .reset_index(drop=True)
)


# ## Final Neuro Score
# 
# This cell assembles the active neuro score from the metric cells above. Only the retained metric families contribute to the package-level neuro score:
# 
# - distribution
# - temporal
# - relational
# - geometry
# - state dynamics
# 
# The fidelity family is intentionally excluded here and lives behind the separate `fidelity-scores` command.
# 

# In[ ]:


# === FINAL NEURO SCORE COMPOSITE ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nethobench.analysis.score_definitions import (
    build_neuro_families_df,
    build_neuro_metrics_df,
    compute_neuro_composite,
)


def _finite_scalar(x):
    return np.isscalar(x) and np.isfinite(x)


def _safe_get(mapping, *keys, default=np.nan):
    cur = mapping
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return float(cur) if _finite_scalar(cur) else default


def _from_df(df_name, value_col, row_col=None, row_value=None, default=np.nan):
    if df_name not in globals():
        return default
    df = globals()[df_name]
    if not isinstance(df, pd.DataFrame) or value_col not in df.columns:
        return default
    sub = df
    if row_col is not None and row_col in df.columns:
        sub = df[df[row_col] == row_value]
    if sub.empty:
        return default
    vals = pd.to_numeric(sub[value_col], errors='coerce').to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals[0]) if vals.size else default


KL_or_JSD_score01 = np.nan
if 'dist_scores' in globals() and isinstance(dist_scores, dict):
    KL_or_JSD_score01 = _safe_get(dist_scores, 'KL_score01_avg', default=np.nan)
if not np.isfinite(KL_or_JSD_score01) and 'kl_sweep_df' in globals():
    KL_or_JSD_score01 = _from_df('kl_sweep_df', 'KL_score01_avg', row_col='corruption', row_value='baseline')

Mean_score01 = _from_df('mean_top10_df', 'Mean_score01', row_col='corruption', row_value='baseline')

QNT_score01 = np.nan
if 'qnt_simple' in globals() and isinstance(qnt_simple, dict):
    QNT_score01 = _safe_get(qnt_simple, 'scores', 'QNT_score01', default=np.nan)
if not np.isfinite(QNT_score01):
    QNT_score01 = _from_df('qnt_corr_df', 'QNT_score01', row_col='family', row_value='baseline')

MOM_score01 = np.nan
if 'mom_simple' in globals() and isinstance(mom_simple, dict):
    MOM_score01 = _safe_get(mom_simple, 'scores', 'MOM_score01', default=np.nan)
if not np.isfinite(MOM_score01):
    MOM_score01 = _from_df('mom_corr_df', 'MOM_score01', row_col='family', row_value='baseline')

GRAPH_score01 = np.nan
if 'graph_simple' in globals() and isinstance(graph_simple, dict):
    GRAPH_score01 = _safe_get(graph_simple, 'scores', 'GRAPH_score01', default=np.nan)
if not np.isfinite(GRAPH_score01):
    GRAPH_score01 = _from_df('graph_corr_df', 'GRAPH_score01', row_col='family', row_value='baseline')

MANI_score01 = np.nan
if 'mani_simple' in globals() and isinstance(mani_simple, dict):
    MANI_score01 = _safe_get(mani_simple, 'scores', 'MANI_score01', default=np.nan)
if not np.isfinite(MANI_score01):
    MANI_score01 = _from_df('mani_corr_df', 'MANI_score01', row_col='family', row_value='baseline')

TRJDIST_score01 = np.nan
if 'trajectory_dist_simple' in globals() and isinstance(trajectory_dist_simple, dict):
    TRJDIST_score01 = _safe_get(trajectory_dist_simple, 'scores', 'TRJDIST_score01', default=np.nan)
if not np.isfinite(TRJDIST_score01):
    TRJDIST_score01 = _from_df('trjdist_corr_df', 'TRJDIST_score01', row_col='family', row_value='baseline')

extra_metric_names = [
    'CrossRegionMI_score01',
    'SubspaceAngle_score01',
    'LaggedCovariance_score01',
    'ImpulseResponse_score01',
    'LatentStateOccupancyK11_score01',
    'LatentStateOccupancyK12_score01',
    'LatentStateTransitionLag1K11_score01',
    'LatentStateTransitionLag2K11_score01',
    'LatentStateTransitionLag3K11_score01',
]

extra_scores = {}
if 'additional_structural' in globals() and isinstance(additional_structural, dict):
    maybe_scores = additional_structural.get('scores', {})
    if isinstance(maybe_scores, dict):
        extra_scores = maybe_scores

SCORES = {
    'KL_or_JSD_score01': KL_or_JSD_score01,
    'QNT_score01': QNT_score01,
    'MOM_score01': MOM_score01,
    'Mean_score01': Mean_score01,
    'TRJDIST_score01': TRJDIST_score01,
    'GRAPH_score01': GRAPH_score01,
    'MANI_score01': MANI_score01,
}
for metric_name in extra_metric_names:
    SCORES[metric_name] = _safe_get({'scores': extra_scores}, 'scores', metric_name, default=np.nan)

metrics_df = build_neuro_metrics_df(SCORES)
families_df = build_neuro_families_df(SCORES)
FINAL_COMPOSITE_SCORE = compute_neuro_composite(SCORES)
FINAL_NEURO_COMPOSITE_SCORE = FINAL_COMPOSITE_SCORE
composite_score = FINAL_COMPOSITE_SCORE

family_colors = {
    'distribution': '#4C78A8',
    'temporal_spectral': '#54A24B',
    'relational': '#E45756',
    'geometry': '#72B7B2',
    'state_dynamics': '#9D755D',
}

plot_df = metrics_df[np.isfinite(metrics_df['value'].to_numpy(dtype=float))].copy()
plot_colors = [family_colors[f] for f in plot_df['family']]

fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))
axes[0].bar(plot_df['metric'], plot_df['value'], color=plot_colors, edgecolor='k', alpha=0.9)
axes[0].set_title('Metric-level neuro score terms')
axes[0].set_ylabel('score (0-1)')
axes[0].tick_params(axis='x', rotation=90)
axes[0].grid(axis='y', alpha=0.25)

family_plot_df = families_df[np.isfinite(families_df['value'].to_numpy(dtype=float))].copy()
family_plot_colors = [family_colors[f] for f in family_plot_df['family']]
axes[1].bar(family_plot_df['family'], family_plot_df['value'], color=family_plot_colors, edgecolor='k', alpha=0.9)
axes[1].set_title(f'Family scores | FINAL={FINAL_COMPOSITE_SCORE:.4f}')
axes[1].set_ylabel('score (0-1)')
axes[1].tick_params(axis='x', rotation=18)
axes[1].grid(axis='y', alpha=0.25)

plt.tight_layout()
plt.show()

print('=== METRICS USED ===')
print(metrics_df.to_string(index=False))
print('\n=== FAMILY SCORES ===')
print(families_df.to_string(index=False))
print(f'\nFINAL_COMPOSITE_SCORE: {float(FINAL_COMPOSITE_SCORE):.6f}')


# ## Corruption Sensitivity Dashboard
# 
# This section stress-tests the retained metrics under structured corruptions. The purpose is not to define the score itself, but to verify that the active neuro score families degrade in interpretable directions under:
# 
# - mean and scale distortions
# - additive noise
# - temporal shuffling and oversmoothing
# - region remapping and mixing
# - latent-space rotations
# 
# The line plots show score trajectories across corruption magnitude. The polar views summarize the largest relative drop per corruption family.
# 

# In[ ]:


# === UNIFIED CORRUPTION SENSITIVITY DASHBOARD (active neuro score metrics) ===
import math
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nethobench.analysis.additional_neuro_metrics import compute_additional_structural_metrics
from nethobench.analysis.direct_neuro_metrics import (
    compute_graph_score01,
    compute_manifold_score01,
    compute_moment_score01,
    compute_trajectory_score01,
)

UCR_EPS = 1e-12


def _ucr_align(gt_arr, pred_arr):
    gt_arr = np.asarray(gt_arr, dtype=np.float64)
    pred_arr = np.asarray(pred_arr, dtype=np.float64)
    if pred_arr.shape[1] != gt_arr.shape[1] and pred_arr.shape[1] % gt_arr.shape[1] == 0:
        factor = pred_arr.shape[1] // gt_arr.shape[1]
        pred_arr = pred_arr.reshape(pred_arr.shape[0], gt_arr.shape[1], factor, pred_arr.shape[2]).mean(axis=2)
    elif pred_arr.shape[1] != gt_arr.shape[1]:
        keep = min(gt_arr.shape[1], pred_arr.shape[1])
        gt_arr = gt_arr[:, :keep, :]
        pred_arr = pred_arr[:, :keep, :]
    return gt_arr, pred_arr


def _ucr_global_iqr(arr):
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    q25, q75 = np.quantile(flat, [0.25, 0.75])
    out = float(q75 - q25)
    if not np.isfinite(out) or out <= UCR_EPS:
        out = float(np.nanstd(flat))
    return out if np.isfinite(out) and out > UCR_EPS else 1.0


def _ucr_temporal_shuffle(pred_arr, level, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    frac = min(max(float(level), 0.0), 1.0)
    n_time = out.shape[1]
    n_swap = max(1, int(round(frac * n_time)))
    for s in range(out.shape[0]):
        idx = rng.choice(n_time, size=n_swap, replace=False)
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        out[s, idx, :] = out[s, shuffled, :]
    return out


def _ucr_region_mix(pred_arr, level, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    perm = rng.permutation(out.shape[2])
    return (1.0 - level) * out + level * out[:, :, perm]


def _ucr_region_permute(pred_arr, level, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    if level < 1.0:
        return _ucr_region_mix(out, level, rng)
    return out[:, :, rng.permutation(out.shape[2])]


def _ucr_additive_noise(pred_arr, level, rng, scale):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    return out + rng.normal(scale=level * 0.35 * scale, size=out.shape)


def _ucr_mean_shift(pred_arr, level, scale):
    return np.asarray(pred_arr, dtype=np.float64) + level * 0.30 * scale


def _ucr_variance_scale(pred_arr, level):
    out = np.asarray(pred_arr, dtype=np.float64)
    mean = np.mean(out, axis=1, keepdims=True)
    factor = 1.0 + 0.80 * level
    return mean + factor * (out - mean)


def _ucr_oversmooth(pred_arr, level):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    alpha = min(max(float(level), 0.0), 0.95)
    smooth = out.copy()
    for t in range(1, out.shape[1]):
        smooth[:, t, :] = alpha * smooth[:, t - 1, :] + (1.0 - alpha) * out[:, t, :]
    return smooth


def _ucr_latent_rotation(pred_arr, level, rng):
    out = np.asarray(pred_arr, dtype=np.float64)
    n_reg = out.shape[2]
    mat = rng.normal(size=(n_reg, n_reg))
    q, _ = np.linalg.qr(mat)
    mixed = (1.0 - level) * np.eye(n_reg) + level * q
    q_mixed, _ = np.linalg.qr(mixed)
    rotated = np.einsum('str,rc->stc', out, q_mixed)
    return rotated


def _compute_dashboard_scores(gt_arr, pred_arr):
    scores = {}
    dist_out = _compute_kl_metrics(gt_arr, pred_arr, bins=60, support_q=(0.001, 0.999))
    scores['KL_or_JSD_score01'] = float(dist_out.get('KL_score01_avg', np.nan))
    scores['Mean_score01'] = float(compute_mean_score01_top10(gt_arr, pred_arr).get('Mean_score01', np.nan))
    scores['QNT_score01'] = float(compute_quantile_score01_simple(gt_arr, pred_arr).get('scores', {}).get('QNT_score01', np.nan))
    scores['MOM_score01'] = float(compute_moment_score01(gt_arr, pred_arr).get('scores', {}).get('MOM_score01', np.nan))
    scores['GRAPH_score01'] = float(compute_graph_score01(gt_arr, pred_arr).get('scores', {}).get('GRAPH_score01', np.nan))
    scores['MANI_score01'] = float(compute_manifold_score01(gt_arr, pred_arr).get('scores', {}).get('MANI_score01', np.nan))
    scores['TRJDIST_score01'] = float(compute_trajectory_score01(gt_arr, pred_arr).get('scores', {}).get('TRJDIST_score01', np.nan))
    extra = compute_additional_structural_metrics(gt_arr, pred_arr).get('scores', {})
    for key in [
        'CrossRegionMI_score01',
        'SubspaceAngle_score01',
        'LaggedCovariance_score01',
        'ImpulseResponse_score01',
        'LatentStateOccupancyK11_score01',
        'LatentStateOccupancyK12_score01',
        'LatentStateTransitionLag1K11_score01',
        'LatentStateTransitionLag2K11_score01',
        'LatentStateTransitionLag3K11_score01',
    ]:
        scores[key] = float(extra.get(key, np.nan))
    return scores


family_specs = [
    dict(name='mean_shift', label='Mean shift', levels=[0.25, 0.50, 0.75, 1.00], apply=lambda x, lv, rng, sc: _ucr_mean_shift(x, lv, sc)),
    dict(name='variance_scale', label='Variance scale', levels=[0.25, 0.50, 0.75, 1.00], apply=lambda x, lv, rng, sc: _ucr_variance_scale(x, lv)),
    dict(name='additive_noise', label='Additive noise', levels=[0.25, 0.50, 0.75, 1.00], apply=lambda x, lv, rng, sc: _ucr_additive_noise(x, lv, rng, sc)),
    dict(name='temporal_shuffle', label='Temporal shuffle', levels=[0.25, 0.50, 0.75, 1.00], apply=lambda x, lv, rng, sc: _ucr_temporal_shuffle(x, lv, rng)),
    dict(name='region_mix', label='Region mix', levels=[0.25, 0.50, 0.75, 1.00], apply=lambda x, lv, rng, sc: _ucr_region_mix(x, lv, rng)),
    dict(name='region_permutation', label='Region permutation', levels=[0.50, 1.00], apply=lambda x, lv, rng, sc: _ucr_region_permute(x, lv, rng)),
    dict(name='latent_rotation', label='Latent rotation', levels=[0.25, 0.50, 0.75, 1.00], apply=lambda x, lv, rng, sc: _ucr_latent_rotation(x, lv, rng)),
    dict(name='oversmoothing', label='Oversmoothing', levels=[0.25, 0.50, 0.75, 0.90], apply=lambda x, lv, rng, sc: _ucr_oversmooth(x, lv)),
]

family_order = [spec['name'] for spec in family_specs]
family_labels = {spec['name']: spec['label'] for spec in family_specs}
family_colors = dict(zip(family_order, plt.cm.tab10(np.linspace(0, 1, len(family_order)))))

gt_base, pred_base = _ucr_align(gt_array_denorm, data_predicted)
base_scale = _ucr_global_iqr(gt_base)
baseline_scores = _compute_dashboard_scores(gt_base, pred_base)
score_order = [score for score in [
    'KL_or_JSD_score01',
    'QNT_score01',
    'MOM_score01',
    'Mean_score01',
    'TRJDIST_score01',
    'GRAPH_score01',
    'CrossRegionMI_score01',
    'LaggedCovariance_score01',
    'ImpulseResponse_score01',
    'MANI_score01',
    'SubspaceAngle_score01',
    'LatentStateOccupancyK11_score01',
    'LatentStateOccupancyK12_score01',
    'LatentStateTransitionLag1K11_score01',
    'LatentStateTransitionLag2K11_score01',
    'LatentStateTransitionLag3K11_score01',
] if np.isfinite(baseline_scores.get(score, np.nan))]

rows = []
for family_name in family_order:
    for score_name in score_order:
        rows.append({
            'score': score_name,
            'family': family_name,
            'level': 0.0,
            'relative_magnitude': 0.0,
            'score_value': baseline_scores[score_name],
            'baseline': baseline_scores[score_name],
            'score_drop_abs': 0.0,
            'score_drop_rel': 0.0,
        })

for fam_idx, spec in enumerate(family_specs):
    n_levels = len(spec['levels'])
    for level_idx, level in enumerate(spec['levels'], start=1):
        rng = np.random.default_rng(20260323 + 100 * fam_idx + level_idx)
        pred_corrupted = spec['apply'](pred_base, level, rng, base_scale)
        score_values = _compute_dashboard_scores(gt_base, pred_corrupted)
        rel_mag = level_idx / float(n_levels)
        for score_name in score_order:
            baseline = baseline_scores.get(score_name, np.nan)
            value = float(score_values.get(score_name, np.nan))
            drop_abs = float(np.clip(baseline - value, 0.0, 1.0)) if np.isfinite(baseline) and np.isfinite(value) else np.nan
            drop_rel = float(np.clip(drop_abs / (baseline + UCR_EPS), 0.0, 1.0)) if np.isfinite(drop_abs) else np.nan
            rows.append({
                'score': score_name,
                'family': spec['name'],
                'level': float(level),
                'relative_magnitude': float(rel_mag),
                'score_value': value,
                'baseline': baseline,
                'score_drop_abs': drop_abs,
                'score_drop_rel': drop_rel,
            })

unified_corruption_master_df = pd.DataFrame(rows)
unified_corruption_family_worst_df = (
    unified_corruption_master_df.groupby(['score', 'family'], as_index=False)['score_drop_rel']
    .max()
    .sort_values(['score', 'family'])
)

n_scores = len(score_order)
ncols = 3
nrows = int(math.ceil(n_scores / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.0 * nrows))
axes = np.atleast_1d(axes).ravel()
for ax, score_name in zip(axes, score_order):
    sub = unified_corruption_master_df[unified_corruption_master_df['score'] == score_name]
    for family_name in family_order:
        fam = sub[sub['family'] == family_name].sort_values('relative_magnitude')
        ax.plot(
            fam['relative_magnitude'].to_numpy(dtype=float),
            fam['score_value'].to_numpy(dtype=float),
            color=family_colors[family_name],
            marker='o',
            linewidth=1.5,
            markersize=3.0,
            alpha=0.9,
        )
    ax.axhline(baseline_scores[score_name], color='black', linestyle='--', linewidth=1.0)
    ax.set_title(score_name)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('relative corruption magnitude')
    ax.set_ylabel('score')
    ax.grid(alpha=0.25)
for ax in axes[n_scores:]:
    ax.axis('off')
handles = [plt.Line2D([0], [0], color=family_colors[name], marker='o', linewidth=1.8, label=family_labels[name]) for name in family_order]
handles.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.0, label='baseline'))
fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()

theta = np.linspace(0.0, 2.0 * np.pi, len(family_order), endpoint=False)
theta_closed = np.concatenate([theta, theta[:1]])
nrows = int(math.ceil(n_scores / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(6.3 * ncols, 5.8 * nrows), subplot_kw={'projection': 'polar'})
axes = np.atleast_1d(axes).ravel()
for ax, score_name in zip(axes, score_order):
    sub = unified_corruption_family_worst_df[unified_corruption_family_worst_df['score'] == score_name]
    radii = []
    for family_name in family_order:
        fam = sub[sub['family'] == family_name]
        radii.append(float(fam.iloc[0]['score_drop_rel']) if not fam.empty else 0.0)
    radii = np.asarray(radii, dtype=float)
    radii_closed = np.concatenate([radii, radii[:1]])
    ax.plot(theta_closed, radii_closed, linewidth=2.0, color='#1f77b4')
    ax.fill(theta_closed, radii_closed, alpha=0.18, color='#1f77b4')
    ax.set_xticks(theta)
    ax.set_xticklabels([family_labels[name] for name in family_order], fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8)
    ax.set_title(score_name, va='bottom')
for ax in axes[n_scores:]:
    ax.axis('off')
fig.suptitle('Worst relative corruption drop by score and corruption family', y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()


# ## Overlay Polar Summary
# This extra cell overlays the per-metric polar sensitivity profiles on a single polar axis. It uses the same strongest-family relative degradation values computed in the unified dashboard, but places every score on one chart to make cross-metric sensitivity patterns easier to compare.

# In[ ]:


# === OVERLAID POLAR SUMMARY (active neuro score metrics) ===
import numpy as np
import matplotlib.pyplot as plt

if 'unified_corruption_family_worst_df' not in globals() or unified_corruption_family_worst_df.empty:
    raise RuntimeError('Run the unified corruption sensitivity dashboard cell first.')

theta = np.linspace(0.0, 2.0 * np.pi, len(family_order), endpoint=False)
theta_closed = np.concatenate([theta, theta[:1]])

cmap = plt.cm.get_cmap('tab20', max(len(score_order), 1))
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

for idx, score_name in enumerate(score_order):
    sub = unified_corruption_family_worst_df[unified_corruption_family_worst_df['score'] == score_name]
    radii = []
    for family_name in family_order:
        fam = sub[sub['family'] == family_name]
        radii.append(float(fam.iloc[0]['score_drop_rel']) if not fam.empty else 0.0)
    radii = np.asarray(radii, dtype=float)
    radii_closed = np.concatenate([radii, radii[:1]])
    color = cmap(idx)
    ax.plot(theta_closed, radii_closed, linewidth=2.0, alpha=0.80, color=color, label=score_name)

ax.set_xticks(theta)
ax.set_xticklabels([family_labels[name] for name in family_order], fontsize=9)
ax.set_ylim(0.0, 1.0)
ax.set_yticks([0.25, 0.50, 0.75, 1.00])
ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8)
ax.set_title('Overlaid polar sensitivity map across active neuro-score metrics', va='bottom')
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.02), frameon=False)
plt.tight_layout()
plt.show()


# ## Separate Fidelity Score
# 
# The old direct-fidelity family is still available, but it is no longer part of the default neuro score.
# 
# Use:
# 
# ```bash
# nethobench fidelity-scores --gt gt_neural.csv --preds pred_neural.csv
# ```
# 
# That command reports:
# 
# - `Error_score01`
# - `MI_score01`
# - `family_fidelity`
# 
# with the same fidelity weighting as the legacy benchmark:
# 
# \[
# S_{fidelity} = 0.65 \, s_{Error} + 0.35 \, s_{MI}.
# \]
# 
