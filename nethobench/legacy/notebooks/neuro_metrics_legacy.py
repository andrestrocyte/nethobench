#!/usr/bin/env python
# coding: utf-8

# ## Setup
# This cell configures the runtime used by the benchmark notebook. It imports the scientific stack, enables inline plotting for interactive sessions, and defines the file paths and options consumed by the downstream metric cells. No score is computed here; the purpose is to make the remaining cells deterministic and reproducible from a single source of truth.

# In[1]:


%matplotlib inline

# plot matplolib in tk for popup window
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure `import nethobench...` works when the notebook is opened directly in Jupyter.
_cwd = Path.cwd().resolve()
_candidate_roots = [_cwd, *_cwd.parents]
_candidate_roots += [Path.home() / "nethobench", Path.home() / "Desktop" / "nethobench"]
_seen = set()
_repo_root = None
for root in _candidate_roots:
    root = root.resolve()
    if root in _seen:
        continue
    _seen.add(root)
    if (root / "nethobench" / "__init__.py").is_file():
        _repo_root = root
        break
if _repo_root is not None and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# autoreload packages and functions
%load_ext autoreload
%autoreload 2

from matplotlib import pyplot as plt

# ## Data Loading and Alignment
# This cell loads ground-truth and prediction CSV files, aligns them by `sequenceId` and `itemPosition`, extracts the neural activity tensor, and prepares the denormalized arrays that every metric reuses. Mathematically, it defines the common sample space
# 
# \[X^{gt}, X^{pred} \in \mathbb{R}^{N_{seq} 	imes T 	imes R},\]
# 
# where `N_seq` is the number of sequences, `T` the aligned time horizon, and `R` the shared neural regions. Every metric below is evaluated on these same aligned tensors.

# In[2]:


import pandas as pd
import numpy as np
import json

preds_fname = "/Users/deviandr/netholabs/evaluate_LLMs/sequifier-netho-hp-search-9-run-2-best-10000-predictions.csv"
gt_fname = "/Users/deviandr/netholabs/evaluate_LLMs/base_sequifier/data-clean.csv"
ddconfig_path = "/Users/deviandr/netholabs/evaluate_LLMs/data-clean-all.json"

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

with open(ddconfig_path, "r") as fh:
    stats = json.load(fh)["selected_columns_statistics"]

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


# ## Sanity-Check Plots
# This cell visualizes the aligned data before any scoring. Its role is diagnostic: confirm that scales, ranges, and sequence structure are sensible before interpreting the benchmark. These plots do not affect any metric values; they are a data-quality gate for the notebook run.

# In[3]:


%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# --- Selección de trials ---
num_trials = 4
pred_trials = np.linspace(0, data_predicted.shape[0] - 1, num_trials, dtype=int)
gt_trials = np.linspace(0, gt_array_denorm.shape[0] - 1, num_trials, dtype=int)

# --- Predicciones ---
fig_pred, axes_pred = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
axes_pred = axes_pred.flatten()
for ax, seq in zip(axes_pred, pred_trials):
    for r in range(data_predicted.shape[2]):
        ax.plot(data_predicted[seq, :, r], label=pred_region_names[r])
    ax.set_title(f"Predicted trial {seq}")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
axes_pred[-1].set_xlabel("Step index")
axes_pred[-2].set_xlabel("Step index")
handles, labels = axes_pred[0].get_legend_handles_labels()
fig_pred.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, 1.03))
fig_pred.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()

# --- Ground truth (denormalizada) ---
fig_gt, axes_gt = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
axes_gt = axes_gt.flatten()
for ax, seq in zip(axes_gt, gt_trials):
    for r in range(gt_array_denorm.shape[2]):
        ax.plot(gt_array_denorm[seq, :, r], label=gt_regions[r])
    ax.set_title(f"GT trial {seq}")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
axes_gt[-1].set_xlabel("Step index")
axes_gt[-2].set_xlabel("Step index")
handles_gt, labels_gt = axes_gt[0].get_legend_handles_labels()
fig_gt.legend(handles_gt, labels_gt, loc="upper center", ncol=min(len(labels_gt), 4), bbox_to_anchor=(0.5, 1.03))
fig_gt.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()


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
    
    assert gt_arr.shape == pred_arr.shape, f"{pred_arr }, {gt_arr.shape = } != {pred_arr.shape = }"
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

# In[17]:


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


# ## Dependence Realism (kNN Mutual Information)
# This cell evaluates whether the prediction preserves nonlinear statistical dependence. It estimates mutual information with the same k-nearest-neighbor procedure on ground-truth and predicted signals, compares the resulting dependence summaries, and converts the mismatch into a bounded `MI_score01`. The cell therefore asks whether the inferred activity reproduces the strength of dependence structure present in the data, beyond simple first- or second-order moments.

# In[18]:


# --- Mutual information realism (STRICT, simple, benchmark-friendly) ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from contextlib import redirect_stdout
import io

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

def _safe_quantile(x, q):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else np.nan

# ----------------------------
# 2) Core MI computation
# ----------------------------
def compute_mi_strict(
    gt_arr,
    pred_arr,
    region_names=None,
    n_neighbors=5,
    min_samples=80,
    max_time=1200,
    standardize=True,
    worst_q_regions=0.10,
    rng_seed=0,
):
    gt_arr, pred_arr = _align_gt_pred(gt_arr, pred_arr)
    n_seq, _, n_reg = gt_arr.shape

    if region_names is None:
        region_names = [f"R{i}" for i in range(n_reg)]
    else:
        region_names = list(region_names)
        if len(region_names) != n_reg:
            region_names = [f"R{i}" for i in range(n_reg)]

    rng = np.random.default_rng(rng_seed)

    # MI per sequence, per region
    mi_all = np.full((n_seq, n_reg), np.nan, dtype=np.float64)

    for seq_idx in range(n_seq):
        for r in range(n_reg):
            x = gt_arr[seq_idx, :, r]
            y = pred_arr[seq_idx, :, r]

            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < min_samples:
                continue

            x = x[mask]
            y = y[mask]

            if x.size > max_time:
                idx = rng.choice(x.size, size=max_time, replace=False)
                x = x[idx]
                y = y[idx]

            if standardize:
                mx, sx = _robust_median_mad(x)
                my, sy = _robust_median_mad(y)
                x = (x - mx) / (sx + EPS)
                y = (y - my) / (sy + EPS)

            try:
                mi_val = mutual_info_regression(
                    x.reshape(-1, 1),
                    y,
                    discrete_features=False,
                    n_neighbors=int(n_neighbors),
                    random_state=int(rng_seed),
                )[0]
                if np.isfinite(mi_val):
                    mi_all[seq_idx, r] = max(0.0, float(mi_val))
            except Exception:
                pass

    # Per-region summaries
    mi_mean = np.nanmean(mi_all, axis=0)
    mi_std = np.nanstd(mi_all, axis=0)
    mi_q10 = np.nanquantile(mi_all, 0.10, axis=0)

    mi_df = pd.DataFrame({
        "region": region_names,
        "mi_mean": mi_mean,
        "mi_std": mi_std,
        "mi_q10": mi_q10,
        "valid_frac": np.mean(np.isfinite(mi_all), axis=0),
    }).set_index("region")

    # Per-sequence summaries
    seq_mean = np.full(n_seq, np.nan, dtype=np.float64)
    seq_worst = np.full(n_seq, np.nan, dtype=np.float64)

    for i in range(n_seq):
        row = mi_all[i]
        row = row[np.isfinite(row)]
        if row.size == 0:
            continue
        seq_mean[i] = float(np.mean(row))
        seq_worst[i] = float(np.quantile(row, worst_q_regions))

    # Final benchmark scalars
    MI_score_mean = float(np.nanmean(seq_mean))
    MI_score_strict = _safe_quantile(seq_worst, 0.10)
    MI_score01 = float(MI_score_strict / (1.0 + MI_score_strict)) if np.isfinite(MI_score_strict) else np.nan

    scores = {
        "MI_score_mean": MI_score_mean,      # smoother
        "MI_score_strict": MI_score_strict,  # strict nested q10 scalar
        "MI_score01": MI_score01,            # final bounded score: strict/(1+strict)
        "MI_coverage_seq": float(np.mean(np.isfinite(seq_mean))),
    }

    print("\n=== Mutual Information realism (STRICT, simple) ===")
    print(f"n_seq={n_seq} | n_reg={n_reg}")
    print(f"n_neighbors={n_neighbors} | min_samples={min_samples} | max_time={max_time} | standardize={standardize}")
    print(f"worst_q_regions={worst_q_regions}")
    print("\nBenchmark scalars:")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}" if np.isfinite(v) else f"  {k}: nan")

    return {
        "mi_all": mi_all,
        "per_region": mi_df.sort_values("mi_mean", ascending=False),
        "per_seq_mean": seq_mean,
        "per_seq_worst": seq_worst,
        "scores": scores,
        "region_names": region_names,
    }

# ----------------------------
# 3) Run
# ----------------------------
_region_names = None
if "pred_region_names" in globals() and globals().get("pred_region_names") is not None:
    _region_names = globals()["pred_region_names"]
elif "gt_regions" in globals() and globals().get("gt_regions") is not None:
    _region_names = globals()["gt_regions"]

results_mi = compute_mi_strict(
    gt_array_denorm,
    data_predicted,
    region_names=_region_names,
    n_neighbors=5,
    min_samples=80,
    max_time=1200,
    standardize=True,
    worst_q_regions=0.10,
    rng_seed=0,
)

display(results_mi["per_region"].round(4))

# ----------------------------
# 4) Plots
# ----------------------------
mi_df = results_mi["per_region"].reindex(results_mi["region_names"])

plt.figure(figsize=(10, 4))
plt.bar(mi_df.index, mi_df["mi_mean"], yerr=mi_df["mi_std"],
        color="tab:purple", alpha=0.8, ecolor="black", capsize=3)
plt.xticks(rotation=90)
plt.ylabel("Mutual information (nats)")
plt.title("GT vs prediction MI per region (per-sequence averaged)")
plt.grid(alpha=0.2, axis="y")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3))
vals = results_mi["per_seq_worst"]
vals = vals[np.isfinite(vals)]
plt.hist(vals, bins=20, color="tab:blue", alpha=0.7, edgecolor="k")
if vals.size:
    plt.axvline(np.quantile(vals, 0.10), linestyle="--", linewidth=2, label="q10 strict")
plt.xlabel("Per-sequence worst-q MI across regions")
plt.ylabel("Count")
plt.title("Distribution of strict per-sequence MI")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ----------------------------
# 5) Corruption Sweep (merged)
# M_strict = q0.10_i(q0.10_r(MI_{i,r})); MI_score01 = M_strict/(1+M_strict)
# ----------------------------
if 'compute_mi_strict' not in globals():
    raise RuntimeError('compute_mi_strict is not available. Run the MI cell above first.')


def _robust_iqr(x):
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


def _mi_scores_quiet(gt_arr, pred_arr, region_names):
    sink = io.StringIO()
    with redirect_stdout(sink):
        out = compute_mi_strict(
            gt_arr,
            pred_arr,
            region_names=region_names,
            n_neighbors=5,
            min_samples=80,
            max_time=1200,
            standardize=True,
            worst_q_regions=0.10,
            rng_seed=0,
        )
    mean_s = float(out['scores'].get('MI_score_mean', np.nan))
    strict_s = float(out['scores'].get('MI_score_strict', np.nan))
    score01_s = float(out['scores'].get('MI_score01', np.nan))
    return mean_s, strict_s, score01_s


def _temporal_shuffle_partial(pred, frac, rng):
    # Shuffle a fraction of timepoints within each sequence (same shuffled indices across regions).
    out = np.array(pred, copy=True)
    n_seq, T, _ = out.shape
    m = int(np.floor(frac * T))
    if m <= 1:
        return out
    for i in range(n_seq):
        idx = rng.choice(T, size=m, replace=False)
        perm = rng.permutation(idx)
        out[i, idx, :] = out[i, perm, :]
    return out


def _additive_noise(pred, sigma_scale, region_iqr, rng):
    out = np.array(pred, copy=True)
    noise = rng.normal(loc=0.0, scale=1.0, size=out.shape)
    out = out + noise * (sigma_scale * region_iqr[None, None, :])
    return out


# Use aligned arrays so all corruptions run on same shape
if '_align_gt_pred' in globals() and callable(globals()['_align_gt_pred']):
    gt_base, pred_base = _align_gt_pred(gt_array_denorm, data_predicted)
else:
    gt_base, pred_base = gt_array_denorm, data_predicted

gt_base = np.asarray(gt_base, dtype=np.float64)
pred_base = np.asarray(pred_base, dtype=np.float64)
region_names = globals().get('pred_region_names', globals().get('gt_regions', None))

region_iqr = np.array([_robust_iqr(gt_base[:, :, r].reshape(-1)) for r in range(gt_base.shape[2])], dtype=np.float64)

# Progressive corruption levels
shuffle_levels = [0.10, 0.25, 0.50, 0.75, 1.00]       # fraction of timepoints shuffled
noise_levels = [0.10, 0.25, 0.50, 0.75, 1.00]         # sigma in GT-IQR units

rng = np.random.default_rng(42)

# Baseline
base_mean, base_strict, base_score01 = _mi_scores_quiet(gt_base, pred_base, region_names)
rows = [{
    'family': 'baseline',
    'level': 0.0,
    'MI_score_mean': base_mean,
    'MI_score_strict': base_strict,
    'MI_score01': base_score01,
}]

# A) Temporal shuffle sweep
for lv in shuffle_levels:
    pred_c = _temporal_shuffle_partial(pred_base, frac=lv, rng=rng)
    m, s, sc = _mi_scores_quiet(gt_base, pred_c, region_names)
    rows.append({'family': 'temporal_shuffle', 'level': float(lv), 'MI_score_mean': m, 'MI_score_strict': s, 'MI_score01': sc})

# B) Additive noise sweep
for lv in noise_levels:
    pred_c = _additive_noise(pred_base, sigma_scale=lv, region_iqr=region_iqr, rng=rng)
    m, s, sc = _mi_scores_quiet(gt_base, pred_c, region_names)
    rows.append({'family': 'additive_noise', 'level': float(lv), 'MI_score_mean': m, 'MI_score_strict': s, 'MI_score01': sc})

mi_corruption_df = pd.DataFrame(rows)
display(mi_corruption_df.round(4))

# Plot score degradation per corruption family
families = ['temporal_shuffle', 'additive_noise']
metrics = ['MI_score_mean', 'MI_score_strict', 'MI_score01']
titles = ['MI_score_mean', 'MI_score_strict', 'MI_score01 = strict/(1+strict)']

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, metric, title in zip(axes, metrics, titles):
    for fam in families:
        sub = mi_corruption_df[mi_corruption_df['family'] == fam].sort_values('level')
        ax.plot(sub['level'], sub[metric], marker='o', label=fam)
    ax.axhline(base_score01 if metric == 'MI_score01' else (base_mean if metric == 'MI_score_mean' else base_strict),
               linestyle='--', color='black', linewidth=1, label='baseline')
    ax.set_title(title)
    ax.set_xlabel('corruption level')
    ax.set_ylabel('score')
    ax.grid(alpha=0.25)

axes[0].legend(loc='best')
plt.tight_layout()
plt.show()

# ## Pointwise Error Realism
# This cell measures direct signal fidelity after alignment. It computes a normalized pointwise error between `X^{gt}` and `X^{pred}` and maps that error to `Error_score01 \in [0, 1]`, so lower normalized error yields higher realism. The corruption sweep is a controlled stress test of exact sample-level fidelity: as corruption grows, the pointwise mismatch should grow and the score should fall.

# In[19]:


# === Error / fidelity metric (simple, strict, benchmark-friendly) ===
# Formula:
#   RMSE(i,r)      = sqrt(mean_t (Pred[i,t,r] - GT[i,t,r])^2)
#   nRMSE(i,r)     = RMSE(i,r) / (IQR_GT(r) + eps)
#   D_i            = mean of top-25% largest nRMSE(i,r) across regions
#   D              = mean_i D_i
#   Error_score01  = 1 / (1 + D)

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

def _compute_rmse_1d(gt_1d, pr_1d, min_samples=50):
    gt_1d = np.asarray(gt_1d, dtype=np.float64)
    pr_1d = np.asarray(pr_1d, dtype=np.float64)
    m = np.isfinite(gt_1d) & np.isfinite(pr_1d)
    if m.sum() < min_samples:
        return np.nan
    d = pr_1d[m] - gt_1d[m]
    return float(np.sqrt(np.mean(d * d)))

# ----------------------------
# 2) Main computation
# ----------------------------
def compute_error_score01_simple(
    gt_arr,
    pred_arr,
    region_names=None,
    min_samples=50,
    top_q_regions=0.25,
):
    gt_arr, pred_arr = _align_gt_pred(gt_arr, pred_arr)
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"Aligned mismatch: {gt_arr.shape} vs {pred_arr.shape}")

    n_seq, _, n_reg = gt_arr.shape

    if region_names is None:
        region_names = [f"R{i}" for i in range(n_reg)]
    else:
        region_names = list(region_names)
        if len(region_names) != n_reg:
            region_names = [f"R{i}" for i in range(n_reg)]

    # GT-only robust scale per region
    iqr_gt = np.array([
        _robust_iqr(gt_arr[:, :, r].reshape(-1))
        for r in range(n_reg)
    ], dtype=np.float64)

    # Per-sequence, per-region nRMSE
    nrmse_sr = np.full((n_seq, n_reg), np.nan, dtype=np.float64)

    for i in range(n_seq):
        for r in range(n_reg):
            rmse = _compute_rmse_1d(gt_arr[i, :, r], pred_arr[i, :, r], min_samples=min_samples)
            if np.isfinite(rmse):
                nrmse_sr[i, r] = rmse / (iqr_gt[r] + EPS)

    # Per-sequence strict distance: mean of worst top-q regions
    D_seq = np.full(n_seq, np.nan, dtype=np.float64)
    for i in range(n_seq):
        D_seq[i] = _topq_mean(nrmse_sr[i], q=top_q_regions)

    # Final scalar
    D = float(np.nanmean(D_seq)) if np.isfinite(D_seq).any() else np.nan
    Error_score01 = float(1.0 / (1.0 + D)) if np.isfinite(D) else np.nan

    # Region summaries
    per_region = pd.DataFrame({
        "region": region_names,
        "IQR_GT": iqr_gt,
        "nRMSE_mean": np.nanmean(nrmse_sr, axis=0),
        "nRMSE_q90": np.nanquantile(nrmse_sr, 0.90, axis=0),
        "valid_seq_frac": np.mean(np.isfinite(nrmse_sr), axis=0),
    }).sort_values("nRMSE_mean", ascending=False)

    per_seq = pd.DataFrame({
        "seq": np.arange(n_seq),
        "D_seq_topq": D_seq,
        "valid_regions": np.sum(np.isfinite(nrmse_sr), axis=1),
    }).sort_values("D_seq_topq", ascending=False)

    scores = {
        "ERR_nRMSE_topq_mean": D,
        "Error_score01": Error_score01,
        "ERR_coverage_seq": float(np.mean(np.isfinite(D_seq))),
        "ERR_coverage_regions": float(np.mean(np.isfinite(iqr_gt))),
    }

    print("\n=== Error / fidelity metric (simple, strict) ===")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}" if np.isfinite(v) else f"  {k}: nan")

    return {
        "nrmse_sr": nrmse_sr,
        "D_seq": D_seq,
        "per_region": per_region,
        "per_seq": per_seq,
        "scores": scores,
        "region_names": region_names,
        "params": {
            "min_samples": min_samples,
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

err_simple = compute_error_score01_simple(
    gt_array_denorm,
    data_predicted,
    region_names=_region_names,
    min_samples=50,
    top_q_regions=0.25,
)

display(err_simple["per_region"].round(4))

# ----------------------------
# 4) Plots
# ----------------------------
per_region = err_simple["per_region"].set_index("region").reindex(err_simple["region_names"])

plt.figure(figsize=(12, 4.5))
plt.bar(per_region.index, per_region["nRMSE_mean"], alpha=0.85, edgecolor="k")
plt.xticks(rotation=90)
plt.ylabel("Mean nRMSE")
plt.title("Per-region normalized RMSE")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3.5))
vals = err_simple["D_seq"]
vals = vals[np.isfinite(vals)]
plt.hist(vals, bins=20, alpha=0.8, edgecolor="k")
if vals.size:
    plt.axvline(np.mean(vals), linestyle="--", linewidth=2, label="mean(D_seq)")
plt.xlabel("Per-sequence top-q region error")
plt.ylabel("Count")
plt.title("Distribution of strict per-sequence error")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# Additive-noise corruption sweep (merged)
# ----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if 'compute_error_score01_simple' not in globals():
    raise RuntimeError('compute_error_score01_simple is not available. Run the error metric cell above first.')


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


def _align_local(gt_arr, pred_arr):
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
    return gt_arr, pred_arr


def _add_noise(pred_arr, sigma, region_iqr, rng):
    noise = rng.normal(loc=0.0, scale=1.0, size=pred_arr.shape)
    return pred_arr + noise * (sigma * region_iqr[None, None, :])


gt_base, pred_base = _align_local(gt_array_denorm, data_predicted)
region_iqr = np.array([_iqr_robust(gt_base[:, :, r].reshape(-1)) for r in range(gt_base.shape[2])], dtype=np.float64)
region_names = globals().get('pred_region_names', globals().get('gt_regions', None))

sigmas = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
rng = np.random.default_rng(123)

rows = []
for sigma in sigmas:
    if sigma == 0.0:
        pred_c = np.array(pred_base, copy=True)
    else:
        pred_c = _add_noise(pred_base, sigma=sigma, region_iqr=region_iqr, rng=rng)

    out = compute_error_score01_simple(
        gt_base,
        pred_c,
        region_names=region_names,
        top_q_regions=0.25,
    )
    rows.append({
        'sigma': float(sigma),
        'D': float(out['scores']['ERR_nRMSE_topq_mean']),
        'Error_score01': float(out['scores']['Error_score01']),
    })

err_noise_df = pd.DataFrame(rows)
display(err_noise_df.round(4))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)

axes[0].plot(err_noise_df['sigma'], err_noise_df['Error_score01'], marker='o')
axes[0].set_title("Error_score01 vs additive noise")
axes[0].set_xlabel('noise sigma (x GT-IQR)')
axes[0].set_ylabel('Error_score01 (higher better)')
axes[0].grid(alpha=0.25)

axes[1].plot(err_noise_df['sigma'], err_noise_df['D'], marker='o')
axes[1].set_title("D (top-q nRMSE) vs additive noise")
axes[1].set_xlabel('noise sigma (x GT-IQR)')
axes[1].set_ylabel('D (lower better)')
axes[1].grid(alpha=0.25)

plt.tight_layout()
plt.show()

# ## Quantile and Tail-Shape Realism
# This cell asks whether predictions reproduce the distributional shape of each region, not just its mean. It compares matched quantiles or tail summaries between ground truth and predictions, aggregates the resulting discrepancies, and maps them into `QNT_score01`. Because the score is driven by quantile structure, it is sensitive to asymmetric tails, broadening, compression, and other changes that may not be visible in a mean-only metric.

# In[20]:


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

# ### Relational Metrics
# The next group measures whether predictions preserve relationships among regions rather than only single-region marginals.

# ## Functional Connectivity Realism
# This cell computes a static functional-connectivity representation for ground truth and predictions, typically through region-by-region correlation structure, and then scores how similar those connectivity summaries are. The resulting `FC_score01` is high only when the inferred activity reproduces the same large-scale correlation geometry that appears in the real data. The associated corruption sweep probes how robustly this relational structure degrades under region mixing or desynchronization.

# In[21]:


# === FC realism (simple, strict, benchmark-friendly) ===
# Formula:
#   1) For each sequence i:
#        - compute GT FC matrix Corr(GT_i)
#        - compute Pred FC matrix Corr(Pred_i)
#        - vectorize upper triangle
#        - compute edge-pattern similarity r_i = corr(vec(FC_GT_i), vec(FC_Pred_i))
#   2) Strict summary across sequences:
#        FC_core = q10_i(r_i)
#   3) Bounded score:
#        FC_score01 = (FC_core + 1) / 2
#
# Interpretation:
#   - FC_core in [-1, 1], higher better
#   - FC_score01 in [0, 1], higher better

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
# 1) Helpers
# ----------------------------
def _safe_corrcoef_matrix(X, min_samples=20):
    X = np.asarray(X, dtype=np.float64)
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    if X.shape[0] < min_samples:
        return None
    C = np.corrcoef(X, rowvar=False)
    if not np.all(np.isfinite(C)):
        return None
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    return C

def _safe_pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a = a[m]
    b = b[m]
    a = a - np.mean(a)
    b = b - np.mean(b)
    da = np.sqrt(np.sum(a * a))
    db = np.sqrt(np.sum(b * b))
    if da < 1e-12 or db < 1e-12:
        return np.nan
    return float(np.sum(a * b) / (da * db + EPS))

def _safe_quantile(x, q):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.quantile(x, q))

# ----------------------------
# 2) Main computation
# ----------------------------
def compute_fc_score01_simple(
    gt_arr,
    pred_arr,
    region_names=None,
    min_samples=20,
):
    gt_arr, pred_arr = _align_gt_pred(gt_arr, pred_arr)
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"Aligned mismatch: {gt_arr.shape} vs {pred_arr.shape}")

    n_seq, _, n_reg = gt_arr.shape
    iu = np.triu_indices(n_reg, k=1)

    if region_names is None:
        region_names = [f"R{i}" for i in range(n_reg)]
    else:
        region_names = list(region_names)
        if len(region_names) != n_reg:
            region_names = [f"R{i}" for i in range(n_reg)]

    r_seq = np.full(n_seq, np.nan, dtype=np.float64)

    for i in range(n_seq):
        Cg = _safe_corrcoef_matrix(gt_arr[i], min_samples=min_samples)
        Cp = _safe_corrcoef_matrix(pred_arr[i], min_samples=min_samples)
        if Cg is None or Cp is None:
            continue

        vg = Cg[iu]
        vp = Cp[iu]
        r_seq[i] = _safe_pearson(vg, vp)

    FC_core = _safe_quantile(r_seq, 0.10)
    FC_score01 = float((FC_core + 1.0) / 2.0) if np.isfinite(FC_core) else np.nan
    FC_mean = float(np.nanmean(r_seq)) if np.isfinite(r_seq).any() else np.nan

    scores = {
        "FC_mean_r": FC_mean,
        "FC_core_q10_r": FC_core,
        "FC_score01": FC_score01,
        "FC_coverage_seq": float(np.mean(np.isfinite(r_seq))),
    }

    per_seq = pd.DataFrame({
        "seq": np.arange(n_seq),
        "FC_r": r_seq,
    }).sort_values("FC_r", ascending=True)

    print("\n=== FC realism (simple, strict) ===")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}" if np.isfinite(v) else f"  {k}: nan")

    return {
        "per_seq": per_seq,
        "scores": scores,
        "region_names": region_names,
    }

# ----------------------------
# 3) Run
# ----------------------------
_region_names = None
if "pred_region_names" in globals() and globals().get("pred_region_names") is not None:
    _region_names = globals()["pred_region_names"]
elif "gt_regions" in globals() and globals().get("gt_regions") is not None:
    _region_names = globals()["gt_regions"]

fc_simple = compute_fc_score01_simple(
    gt_array_denorm,
    data_predicted,
    region_names=_region_names,
    min_samples=20,
)

display(fc_simple["per_seq"].round(4))

# ----------------------------
# 4) Plot
# ----------------------------
vals = fc_simple["per_seq"]["FC_r"].values
vals = vals[np.isfinite(vals)]

plt.figure(figsize=(6, 3.5))
plt.hist(vals, bins=20, alpha=0.8, edgecolor="k")
if vals.size:
    plt.axvline(np.quantile(vals, 0.10), linestyle="--", linewidth=2, label="q10")
plt.xlabel("Per-sequence FC similarity r")
plt.ylabel("Count")
plt.title("Distribution of FC realism across sequences")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# FC corruption sweep (merged)
# ----------------------------
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

if 'compute_fc_score01_simple' not in globals():
    raise RuntimeError('compute_fc_score01_simple is not available. Run the FC realism cell above first.')


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


def _align_local(gt_arr, pred_arr):
    gt_arr = np.asarray(gt_arr, dtype=np.float64)
    pred_arr = np.asarray(pred_arr, dtype=np.float64)
    if '_align_gt_pred' in globals() and callable(globals()['_align_gt_pred']):
        return globals()['_align_gt_pred'](gt_arr, pred_arr)
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


def _fc_scores_quiet(gt_arr, pred_arr, region_names):
    sink = io.StringIO()
    with redirect_stdout(sink):
        out = compute_fc_score01_simple(gt_arr, pred_arr, region_names=region_names)
    return {
        'FC_core': float(out['scores'].get('FC_core', np.nan)),
        'FC_score01': float(out['scores'].get('FC_score01', np.nan)),
    }


def _region_mixing_noise(pred_arr, sigma, region_iqr, rng):
    noise = rng.normal(0.0, 1.0, size=pred_arr.shape)
    return pred_arr + noise * (float(sigma) * region_iqr[None, None, :])


def _permute_subset_regions(pred_arr, frac, rng):
    out = np.array(pred_arr, copy=True)
    n_reg = out.shape[2]
    k = max(2, int(np.ceil(float(frac) * n_reg)))
    k = min(k, n_reg)
    idx = rng.choice(n_reg, size=k, replace=False)
    perm = rng.permutation(idx)
    # avoid identity on chosen subset for stronger corruption
    if np.all(perm == idx) and k > 1:
        perm = np.roll(perm, 1)
    out[:, :, idx] = out[:, :, perm]
    return out


def _temporal_shuffle_per_region(pred_arr, frac, rng):
    out = np.array(pred_arr, copy=True)
    n_seq, T, n_reg = out.shape
    m = int(np.floor(float(frac) * T))
    if m <= 1:
        return out
    for i in range(n_seq):
        for r in range(n_reg):
            idx = rng.choice(T, size=m, replace=False)
            perm = rng.permutation(idx)
            out[i, idx, r] = out[i, perm, r]
    return out


# Baseline arrays
gt_base, pred_base = _align_local(gt_array_denorm, data_predicted)
region_names = globals().get('pred_region_names', globals().get('gt_regions', None))
region_iqr = np.array([_iqr_robust(gt_base[:, :, r].reshape(-1)) for r in range(gt_base.shape[2])], dtype=np.float64)

baseline = _fc_scores_quiet(gt_base, pred_base, region_names)
rows = [{
    'family': 'baseline',
    'level': 0.0,
    'FC_core': baseline['FC_core'],
    'FC_score01': baseline['FC_score01'],
}]

noise_levels = [0.10, 0.25, 0.50, 0.75, 1.00]
perm_levels = [0.20, 0.40, 0.60, 0.80, 1.00]
shuffle_levels = [0.10, 0.25, 0.50, 0.75, 1.00]
rng = np.random.default_rng(2026)

# 1) Region mixing noise
for lv in noise_levels:
    pred_c = _region_mixing_noise(pred_base, sigma=lv, region_iqr=region_iqr, rng=rng)
    sc = _fc_scores_quiet(gt_base, pred_c, region_names)
    rows.append({'family': 'region_noise', 'level': float(lv), 'FC_core': sc['FC_core'], 'FC_score01': sc['FC_score01']})

# 2) Region permutation (fraction of regions permuted)
for lv in perm_levels:
    pred_c = _permute_subset_regions(pred_base, frac=lv, rng=rng)
    sc = _fc_scores_quiet(gt_base, pred_c, region_names)
    rows.append({'family': 'region_permutation', 'level': float(lv), 'FC_core': sc['FC_core'], 'FC_score01': sc['FC_score01']})

# 3) Temporal shuffle within each region
for lv in shuffle_levels:
    pred_c = _temporal_shuffle_per_region(pred_base, frac=lv, rng=rng)
    sc = _fc_scores_quiet(gt_base, pred_c, region_names)
    rows.append({'family': 'temporal_shuffle_per_region', 'level': float(lv), 'FC_core': sc['FC_core'], 'FC_score01': sc['FC_score01']})

fc_corr_df = pd.DataFrame(rows)
display(fc_corr_df.round(4))

# Plot progressive score degradation
families = ['region_noise', 'region_permutation', 'temporal_shuffle_per_region']
plt.figure(figsize=(8, 4.8))
for fam in families:
    sub = fc_corr_df[fc_corr_df['family'] == fam].sort_values('level')
    plt.plot(sub['level'], sub['FC_score01'], marker='o', label=fam)

plt.axhline(baseline['FC_score01'], linestyle='--', color='black', linewidth=1, label='baseline')
plt.title('FC_score01 under progressive FC corruptions')
plt.xlabel('corruption level')
plt.ylabel('FC_score01 (higher better)')
plt.grid(alpha=0.25)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# ## PCA Realism
# This cell measures whether the prediction occupies the same dominant linear subspace as the data. It computes a ground-truth PCA basis, projects prediction activity into that basis, and scores how well the prediction preserves the variance explained and reconstruction behavior induced by the real-data components. The score is therefore a GT-anchored subspace agreement metric rather than a free PCA fit on the prediction alone.

# In[22]:


# === PCA realism (simple, strict, benchmark-friendly) ===
import nethobench.analysis.refined_neuro_metric_replacements
import pandas as pd
import matplotlib.pyplot as plt

from nethobench.analysis.refined_neuro_metric_replacements import compute_pca_replacement


pca_simple, pca_corr_df = compute_pca_replacement(
    gt_array_denorm,
    data_predicted,
    enable_plots=bool(globals().get("ENABLE_PLOTS", True)),
)

display(pd.DataFrame([{
    "metric": "PCA_score01",
    "value": pca_simple["scores"]["PCA_score01"],
    "candidate_name": pca_simple["candidate_name"],
}]))

# 1) Region permutation
display(pca_corr_df)


# ## Autocorrelation Realism
# This cell measures temporal self-similarity within each region. It compares early-lag autocorrelation structure between ground truth and predictions, aggregates the lag-wise mismatch, and converts it to `AUTO_score01`. In mathematical terms, the cell asks whether each signal's dependence on its own recent past is preserved, which makes the metric sensitive to temporal scrambling, smoothing, and over-regularized dynamics.

# In[23]:


# === AUTOCORR realism (improved, more sensitive, still simple/strict) ===
import pandas as pd
import matplotlib.pyplot as plt

from nethobench.analysis.refined_neuro_metric_replacements import compute_autocorr_replacement


auto_sensitive, auto_corr_df = compute_autocorr_replacement(
    gt_array_denorm,
    data_predicted,
    enable_plots=bool(globals().get("ENABLE_PLOTS", True)),
)
auto_simple = auto_sensitive

display(pd.DataFrame([{
    "metric": "AUTO_score01",
    "value": auto_sensitive["scores"]["AUTO_score01"],
    "candidate_name": auto_sensitive["candidate_name"],
}]))

rows = [{
    "family": row["family"],
    "magnitude": row["magnitude"],
    "AUTO_score01": row["AUTO_score01"],
} for row in auto_corr_df.to_dict("records")]
display(auto_corr_df)


# ## Cross-Correlation Realism
# This cell measures whether lagged relationships between regions are preserved. The active implementation combines a lagged cross-correlation matrix comparison with a strong-edge profile comparison, so `CC_score01` is high only when both the broad cross-region lag structure and the strongest couplings agree between ground truth and prediction. This is one of the finalized replacement metrics in the benchmark.

# In[24]:


# === CROSSCORR realism (more sensitive, still simple/strict/benchmark-friendly) ===
import pandas as pd
import matplotlib.pyplot as plt

from nethobench.analysis.refined_neuro_metric_replacements import compute_crosscorr_replacement


cc_sensitive, cc_corr_df = compute_crosscorr_replacement(
    gt_array_denorm,
    data_predicted,
    enable_plots=bool(globals().get("ENABLE_PLOTS", True)),
)
cc_simple = cc_sensitive

display(pd.DataFrame([{
    "metric": "CC_score01",
    "value": cc_sensitive["scores"]["CC_score01"],
    "candidate_name": cc_sensitive["candidate_name"],
}]))

rows = [{
    "family": row["family"],
    "magnitude": row["magnitude"],
    "CC_score01": row["CC_score01"],
} for row in cc_corr_df.to_dict("records")]
display(cc_corr_df)


# This cell uses the archived perfected higher-order moment metric from the legacy benchmark branch. After alignment, it compares variance, skewness, and kurtosis region by region on flattened GT and prediction activity, builds a single distance `|log(var_p / var_g)| + 0.5 * |skew_p - skew_g| + 0.25 * |kurt_p - kurt_g|`, and maps that distance to the bounded score `MOM_score01 = 1 / (1 + distance)`. The code cell below runs that exact legacy perfected implementation and shows the original gain-scaling corruption sweep.

# In[4]:


# === Higher-order moments realism (simple, strict, benchmark-friendly) ===
# Formula:
#   For each region r after alignment, flatten GT and prediction over sequences and time.
#   Compute:
#
#       D_r = |log(var_pr / var_gt)| + 0.50 * |skew_pr - skew_gt| + 0.25 * |kurt_pr - kurt_gt|
#
#   Map distance to score:
#
#       score_r      = 1 / (1 + D_r)
#       MOM_score01  = mean_r(score_r)
#
# Interpretation:
#   - bounded in [0,1]
#   - higher is better

import numpy as np
import pandas as pd

from nethobench.analysis.refined_neuro_metric_replacements import compute_moment_replacement

mom_simple, mom_corr_df = compute_moment_replacement(
    gt_array_denorm,
    data_predicted,
    enable_plots=bool(globals().get("ENABLE_PLOTS", True)),
)

display(pd.DataFrame([{
    "metric": "MOM_score01",
    "value": mom_simple["scores"]["MOM_score01"],
    "candidate_name": mom_simple["candidate_name"],
}]))
display(mom_corr_df.round(4))

# 1) Tail spikes
pass


# This cell uses the archived perfected graph metric from the legacy benchmark branch. It builds pooled GT and prediction correlation matrices, converts each into a binary strong-edge topology by keeping the top fraction of absolute edges, and compares three graph summaries: edge-set Jaccard overlap, weighted-degree agreement, and local clustering agreement. The final `GRAPH_score01` is the geometric mean of those valid graph-summary terms. The code cell below runs that exact legacy perfected implementation and shows the original region-permutation corruption sweep.

# In[5]:


# === Corr-connectivity GRAPH realism (simple, strict, benchmark-friendly) ===
# Formula:
#   1) Pool all valid timepoints across sequences and compute GT / prediction
#      correlation matrices across regions.
#   2) Build binary strong-edge graphs by keeping the top 15% absolute edges.
#   3) Compute three bounded graph-summary terms:
#        J = Jaccard overlap of the strong-edge sets
#        D = 1 / (1 + mean(|deg_pr - deg_gt|) / mean(|deg_gt|))
#        C = 1 - mean(|clustering_pr - clustering_gt|)
#   4) Final bounded score:
#        GRAPH_score01 = geometric_mean(J, D, C) over finite terms
#
# Interpretation:
#   - bounded in [0,1]
#   - higher is better

import numpy as np
import pandas as pd

from nethobench.analysis.refined_neuro_metric_replacements import compute_graph_replacement

graph_simple, graph_corr_df = compute_graph_replacement(
    gt_array_denorm,
    data_predicted,
    enable_plots=bool(globals().get("ENABLE_PLOTS", True)),
)

display(pd.DataFrame([{
    "metric": "GRAPH_score01",
    "value": graph_simple["scores"]["GRAPH_score01"],
    "GRAPH_jacc_mean": graph_simple["scores"].get("GRAPH_jacc_mean", np.nan),
    "GRAPH_deg_mean": graph_simple["scores"].get("GRAPH_deg_mean", np.nan),
    "GRAPH_cluster_mean": graph_simple["scores"].get("GRAPH_cluster_mean", np.nan),
    "candidate_name": graph_simple["candidate_name"],
}]))
display(graph_corr_df.round(4))

perm_levels = []


# ### Geometry Metrics
# The next group evaluates shared low-dimensional structure and geometric organization.

# ## CV-CCA Realism
# This cell quantifies agreement in shared linear subspaces using canonical correlation analysis. It extracts paired latent directions from ground truth and predictions, measures how well those canonical correlations agree, and maps the result into `CCA_score01`. The metric is geometry-aware: it rewards predictions that preserve coordinated population-level directions, not merely individual region statistics.

# In[27]:


# === CCA realism (stricter, nested-q10, benchmark-friendly) ===
# Formula:
#   1) For each sequence i:
#        - jointly mask finite rows of GT_i and Pred_i
#        - run proper cross-validated CCA:
#            * fit on TRAIN folds only
#            * evaluate on TEST folds only
#        - collect held-out canonical correlations r(i,f,c)
#        - average over folds per component:
#            r_comp(i,c) = mean_f r(i,f,c)
#        - strict per-sequence score:
#            CCA_i = q10_c(r_comp(i,c))
#
#   2) Strict summary across sequences:
#        CCA_core = q10_i(CCA_i)
#
#   3) Final score:
#        CCA_score01 = CCA_core
#
# Interpretation:
#   - bounded in [0,1]
#   - higher is better
#   - stricter than mean-over-components CCA because weak canonical modes matter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold

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
# 1) Helpers
# ----------------------------
def _finite_row_mask(X, Y):
    return np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)

def _zscore_train_apply(Xtr, Xte, eps=1e-9):
    mu = np.mean(Xtr, axis=0, keepdims=True)
    sd = np.std(Xtr, axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd

def _corr_safe(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or b.size < 3:
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else np.nan

def _safe_quantile(x, q):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.quantile(x, q))

# ----------------------------
# 2) Main computation
# ----------------------------
def compute_cca_score01_strict(
    gt_arr,
    pred_arr,
    region_names=None,
    min_seg_len=80,
    max_time=601,
    n_comp=5,
    folds=5,
    max_iter=2000,
    rng_seed=0,
):
    gt_arr, pred_arr = _align_gt_pred(gt_arr, pred_arr)
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"Aligned mismatch: {gt_arr.shape} vs {pred_arr.shape}")
    if gt_arr.ndim != 3:
        raise ValueError(f"Expected 3D arrays [n_seq, T, n_reg]; got ndim={gt_arr.ndim}")

    n_seq, T, n_reg = gt_arr.shape
    T_eff = min(T, int(max_time))
    gt_arr = gt_arr[:, :T_eff, :]
    pred_arr = pred_arr[:, :T_eff, :]

    if region_names is None:
        region_names = [f"R{i}" for i in range(n_reg)]
    else:
        region_names = list(region_names)
        if len(region_names) != n_reg:
            region_names = [f"R{i}" for i in range(n_reg)]

    cca_seq = np.full(n_seq, np.nan, dtype=np.float64)
    cca_mean_seq = np.full(n_seq, np.nan, dtype=np.float64)
    cca_rowperm_seq = np.full(n_seq, np.nan, dtype=np.float64)
    valid_rows_seq = np.zeros(n_seq, dtype=int)
    k_used_seq = np.zeros(n_seq, dtype=int)

    comp_mat = np.full((n_seq, int(n_comp)), np.nan, dtype=np.float64)
    comp_mat_rowperm = np.full((n_seq, int(n_comp)), np.nan, dtype=np.float64)

    def _cv_cca_component_scores(X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        m = _finite_row_mask(X, Y)
        if int(m.sum()) < int(min_seg_len):
            return None, 0

        X = X[m]
        Y = Y[m]
        n = X.shape[0]
        if n < max(12, int(folds) * 3):
            return None, n

        k = int(min(n_comp, X.shape[1], Y.shape[1], n - 2))
        if k < 1:
            return None, n

        kf = KFold(n_splits=int(folds), shuffle=True, random_state=int(rng_seed))
        fold_comp = []

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
                fold_comp.append(np.full(k, np.nan, dtype=np.float64))
                continue

            corrs = np.full(k, np.nan, dtype=np.float64)
            for c in range(k):
                corrs[c] = _corr_safe(Xc[:, c], Yc[:, c])

            fold_comp.append(corrs)

        fold_comp = np.asarray(fold_comp, dtype=np.float64)  # [fold, comp]
        if fold_comp.ndim != 2 or fold_comp.shape[1] == 0:
            return None, n

        comp_scores = np.nanmean(fold_comp, axis=0)  # mean across folds, per component
        return comp_scores, n

    rng = np.random.default_rng(int(rng_seed))

    for s in range(n_seq):
        X = gt_arr[s]
        Y = pred_arr[s]

        comp_scores, n_valid = _cv_cca_component_scores(X, Y)
        valid_rows_seq[s] = int(n_valid)

        if comp_scores is not None:
            k = min(len(comp_scores), int(n_comp))
            comp_mat[s, :k] = comp_scores[:k]
            k_used_seq[s] = int(k)
            cca_mean_seq[s] = float(np.nanmean(comp_scores[:k])) if np.isfinite(comp_scores[:k]).any() else np.nan
            cca_seq[s] = _safe_quantile(comp_scores[:k], 0.10)

        # row-permuted diagnostic
        perm = rng.permutation(Y.shape[0])
        comp_scores_perm, _ = _cv_cca_component_scores(X, Y[perm])
        if comp_scores_perm is not None:
            k = min(len(comp_scores_perm), int(n_comp))
            comp_mat_rowperm[s, :k] = comp_scores_perm[:k]
            cca_rowperm_seq[s] = _safe_quantile(comp_scores_perm[:k], 0.10)

    CCA_core = _safe_quantile(cca_seq, 0.10)
    CCA_score01 = CCA_core
    CCA_mean = float(np.nanmean(cca_seq)) if np.isfinite(cca_seq).any() else np.nan

    scores = {
        "CCA_mean": CCA_mean,
        "CCA_core_q10": CCA_core,
        "CCA_score01": CCA_score01,
        "CCA_coverage_seq": float(np.mean(np.isfinite(cca_seq))),
        "CCA_mean_over_comp_mean": float(np.nanmean(cca_mean_seq)) if np.isfinite(cca_mean_seq).any() else np.nan,
        "CCA_rowperm_mean": float(np.nanmean(cca_rowperm_seq)) if np.isfinite(cca_rowperm_seq).any() else np.nan,
        "CCA_n_valid_rows_mean": float(np.nanmean(valid_rows_seq)) if np.isfinite(valid_rows_seq).any() else np.nan,
        "CCA_k_used_mean": float(np.nanmean(k_used_seq)) if np.isfinite(k_used_seq).any() else np.nan,
    }

    per_seq = pd.DataFrame({
        "seq": np.arange(n_seq),
        "n_valid_rows": valid_rows_seq,
        "k_used": k_used_seq,
        "CCA_i_strict": cca_seq,
        "CCA_i_meancomp": cca_mean_seq,
        "CCA_rowperm_strict": cca_rowperm_seq,
    }).sort_values("CCA_i_strict", ascending=True)

    comp_cols = [f"comp_{i+1}" for i in range(int(n_comp))]
    per_comp = pd.DataFrame(comp_mat, columns=comp_cols)
    per_comp.insert(0, "seq", np.arange(n_seq))

    print("\n=== CCA realism (stricter, nested-q10) ===")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}" if np.isfinite(v) else f"  {k}: nan")

    return {
        "per_seq": per_seq,
        "per_comp": per_comp,
        "scores": scores,
        "arrays": {
            "cca_seq": cca_seq,
            "cca_mean_seq": cca_mean_seq,
            "cca_rowperm_seq": cca_rowperm_seq,
            "valid_rows_seq": valid_rows_seq,
            "k_used_seq": k_used_seq,
            "comp_mat": comp_mat,
            "comp_mat_rowperm": comp_mat_rowperm,
        },
        "region_names": region_names,
        "params": {
            "min_seg_len": min_seg_len,
            "max_time": T_eff,
            "n_comp": n_comp,
            "folds": folds,
            "max_iter": max_iter,
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

cca_strict = compute_cca_score01_strict(
    gt_array_denorm,
    data_predicted,
    region_names=_region_names,
    min_seg_len=80,
    max_time=601,
    n_comp=5,
    folds=5,
    max_iter=2000,
    rng_seed=0,
)

display(cca_strict["per_seq"].round(4))
display(cca_strict["per_comp"].round(4))

# ----------------------------
# 4) Plots
# ----------------------------
vals = cca_strict["arrays"]["cca_seq"]
vals = vals[np.isfinite(vals)]

plt.figure(figsize=(6, 3.5))
plt.hist(vals, bins=20, alpha=0.8, edgecolor="k")
if vals.size:
    plt.axvline(np.quantile(vals, 0.10), linestyle="--", linewidth=2, label="q10 across seq")
plt.xlabel("Per-sequence strict held-out CCA")
plt.ylabel("Count")
plt.title("Distribution of strict CCA realism across sequences")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

mean_vals = cca_strict["arrays"]["cca_mean_seq"]
m = np.isfinite(mean_vals) & np.isfinite(cca_strict["arrays"]["cca_seq"])
plt.figure(figsize=(6, 3.5))
plt.scatter(mean_vals[m], cca_strict["arrays"]["cca_seq"][m], alpha=0.6)
plt.xlabel("Mean over components")
plt.ylabel("Strict q10 over components")
plt.title("Mean-vs-strict per-sequence CCA")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

m = np.isfinite(cca_strict["arrays"]["cca_seq"]) & np.isfinite(cca_strict["arrays"]["cca_rowperm_seq"])
plt.figure(figsize=(6, 3.5))
plt.scatter(
    cca_strict["arrays"]["cca_rowperm_seq"][m],
    cca_strict["arrays"]["cca_seq"][m],
    alpha=0.6
)
plt.xlabel("Row-permuted strict held-out CCA")
plt.ylabel("Matched strict held-out CCA")
plt.title("Matched vs row-permuted strict CCA")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# ----------------------------
# CCA corruption sweep (merged)
# ----------------------------
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

if 'compute_cca_score01_strict' not in globals():
    raise RuntimeError('compute_cca_score01_strict is not available. Run the CCA realism cell above first.')


def _align_local(gt_arr, pred_arr):
    gt_arr = np.asarray(gt_arr, dtype=np.float64)
    pred_arr = np.asarray(pred_arr, dtype=np.float64)
    if '_align_gt_pred' in globals() and callable(globals()['_align_gt_pred']):
        return globals()['_align_gt_pred'](gt_arr, pred_arr)
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


def _cca_scores_quiet(gt_arr, pred_arr, region_names):
    sink = io.StringIO()
    with redirect_stdout(sink):
        out = compute_cca_score01_strict(gt_arr, pred_arr, region_names=region_names)
    sc = out['scores']
    return {
        'CCA_score01': float(sc.get('CCA_score01', np.nan)),
        'CCA_core_q10': float(sc.get('CCA_core_q10', np.nan)),
        'CCA_mean': float(sc.get('CCA_mean', np.nan)),
    }


def _temporal_shuffle(pred_arr, frac, rng):
    # Break rowwise alignment between GT and Pred.
    out = np.array(pred_arr, copy=True)
    n_seq, T, _ = out.shape
    m = int(np.floor(float(frac) * T))
    if m <= 1:
        return out
    for i in range(n_seq):
        idx = rng.choice(T, size=m, replace=False)
        perm = rng.permutation(idx)
        out[i, idx, :] = out[i, perm, :]
    return out


def _latent_contam(pred_arr, a, rng):
    # Add a Pred-only latent mode: Y' = Y + a * u(t) * w^T.
    out = np.array(pred_arr, copy=True)
    n_seq, T, R = out.shape
    t = np.linspace(0.0, 2.0 * np.pi, T, endpoint=False)
    w = rng.normal(0.0, 1.0, size=(R,))
    w = w / (np.linalg.norm(w) + 1e-12)
    for i in range(n_seq):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        u = np.sin(t + phase).reshape(T, 1)
        out[i] = out[i] + float(a) * (u @ w.reshape(1, R))
    return out


def _independent_region_noise(pred_arr, sigma, region_iqr, rng):
    noise = rng.normal(0.0, 1.0, size=pred_arr.shape)
    return pred_arr + noise * (float(sigma) * region_iqr[None, None, :])


# Baseline arrays
gt_base, pred_base = _align_local(gt_array_denorm, data_predicted)
region_names = globals().get('pred_region_names', globals().get('gt_regions', None))
region_iqr = np.array([_iqr_robust(gt_base[:, :, r].reshape(-1)) for r in range(gt_base.shape[2])], dtype=np.float64)

baseline = _cca_scores_quiet(gt_base, pred_base, region_names)
rows = [{
    'family': 'baseline',
    'level': 0.0,
    'CCA_score01': baseline['CCA_score01'],
    'CCA_core_q10': baseline['CCA_core_q10'],
    'CCA_mean': baseline['CCA_mean'],
}]

shuffle_levels = [0.10, 0.25, 0.50, 0.75, 1.00]
latent_levels = [0.10, 0.25, 0.50, 0.75, 1.00]
noise_levels = [0.10, 0.25, 0.50, 0.75, 1.00]
rng = np.random.default_rng(2026)

# 1) Temporal shuffle
for lv in shuffle_levels:
    pred_c = _temporal_shuffle(pred_base, frac=lv, rng=rng)
    sc = _cca_scores_quiet(gt_base, pred_c, region_names)
    rows.append({
        'family': 'temporal_shuffle',
        'level': float(lv),
        'CCA_score01': sc['CCA_score01'],
        'CCA_core_q10': sc['CCA_core_q10'],
        'CCA_mean': sc['CCA_mean'],
    })

# 2) Latent contamination
for lv in latent_levels:
    pred_c = _latent_contam(pred_base, a=lv, rng=rng)
    sc = _cca_scores_quiet(gt_base, pred_c, region_names)
    rows.append({
        'family': 'latent_contamination',
        'level': float(lv),
        'CCA_score01': sc['CCA_score01'],
        'CCA_core_q10': sc['CCA_core_q10'],
        'CCA_mean': sc['CCA_mean'],
    })

# 3) Independent region noise
for lv in noise_levels:
    pred_c = _independent_region_noise(pred_base, sigma=lv, region_iqr=region_iqr, rng=rng)
    sc = _cca_scores_quiet(gt_base, pred_c, region_names)
    rows.append({
        'family': 'independent_region_noise',
        'level': float(lv),
        'CCA_score01': sc['CCA_score01'],
        'CCA_core_q10': sc['CCA_core_q10'],
        'CCA_mean': sc['CCA_mean'],
    })

cca_corr_df = pd.DataFrame(rows)
display(cca_corr_df.round(4))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
families = ['temporal_shuffle', 'latent_contamination', 'independent_region_noise']

for fam in families:
    sub = cca_corr_df[cca_corr_df['family'] == fam].sort_values('level')
    axes[0].plot(sub['level'], sub['CCA_score01'], marker='o', label=fam)
axes[0].axhline(baseline['CCA_score01'], linestyle='--', color='black', linewidth=1, label='baseline')
axes[0].set_title('CCA_score01 under latent-structure corruptions')
axes[0].set_xlabel('corruption level')
axes[0].set_ylabel('CCA_score01 (higher better)')
axes[0].grid(alpha=0.25)
axes[0].legend(loc='best')

for fam in families:
    sub = cca_corr_df[cca_corr_df['family'] == fam].sort_values('level')
    axes[1].plot(sub['level'], sub['CCA_mean'], marker='o', label=fam)
axes[1].axhline(baseline['CCA_mean'], linestyle='--', color='black', linewidth=1, label='baseline')
axes[1].set_title('CCA_mean under latent-structure corruptions')
axes[1].set_xlabel('corruption level')
axes[1].set_ylabel('CCA_mean (higher better)')
axes[1].grid(alpha=0.25)
axes[1].legend(loc='best')

plt.tight_layout()
plt.show()

# ## Manifold Realism
# This cell measures low-dimensional geometric realism with a GT-anchored manifold score. The active implementation combines persistent-homology lifetime agreement with local neighborhood geometry agreement, so `MANI_score01` is high only when both the coarse topological organization and the fine local manifold structure of the prediction match the real data. This is the finalized manifold replacement used by the benchmark.

# In[7]:


# === MANIFOLD realism (simple geometry, strict, benchmark-friendly) ===
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

try:
    from ripser import ripser
except Exception:
    ripser = None

EPS = 1e-8


def _mani_align_arrays(gt, pred):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if gt.shape[1] != pred.shape[1]:
        m = min(gt.shape[1], pred.shape[1])
        gt = gt[:, :m, :]
        pred = pred[:, :m, :]
    if gt.shape != pred.shape:
        raise ValueError(f'Aligned mismatch: {gt.shape} vs {pred.shape}')
    return gt, pred


def _mani_finite_rows(X, Y):
    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    return X[mask], Y[mask]


def _mani_standardize_with_gt(Xg, Xp):
    mu = np.mean(Xg, axis=0, keepdims=True)
    sd = np.std(Xg, axis=0, keepdims=True)
    sd = np.where(sd < EPS, 1.0, sd)
    return (Xg - mu) / sd, (Xp - mu) / sd


def _mani_score_from_distance(distance):
    return float(1.0 / (1.0 + distance)) if np.isfinite(distance) else np.nan


def _mani_robust_scale(x, floor=0.05):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return float(floor)
    q25, q75 = np.quantile(x, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < floor:
        scale = float(np.std(x))
    return float(scale if np.isfinite(scale) and scale >= floor else floor)


def _mani_quantile_distance(x, y, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 8 or y.size < 8:
        return np.nan
    qx = np.quantile(x, qs)
    qy = np.quantile(y, qs)
    scale = _mani_robust_scale(x)
    return float(np.mean(np.abs(qx - qy) / (scale + EPS)))


def _mani_choose_k(Xg, var_target=0.9, k_min=2, k_max=8):
    k_fit = int(min(k_max, Xg.shape[1], Xg.shape[0] - 1))
    if k_fit < 2:
        return 1
    pca = PCA(n_components=k_fit, svd_solver='full', random_state=0).fit(Xg)
    csum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(csum, var_target) + 1)
    return int(np.clip(k, k_min, k_fit))


def _mani_fit_pooled_gt_pca(gt, pred, k_max=3, seed=0):
    gt, pred = _mani_align_arrays(gt, pred)
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = _mani_finite_rows(Xg, Xp)
    if Xg.shape[0] < 64:
        return None, None, None
    mu = np.mean(Xg, axis=0, keepdims=True)
    sd = np.std(Xg, axis=0, keepdims=True)
    sd = np.where(sd < EPS, 1.0, sd)
    Xg_z = (Xg - mu) / sd
    k = _mani_choose_k(Xg_z, k_max=min(k_max, Xg_z.shape[1]))
    pca = PCA(n_components=k, svd_solver='full', random_state=seed).fit(Xg_z)
    return mu, sd, pca


def _mani_pooled_latent_clouds(gt, pred, k_max=3, n_points=128, seed=3):
    gt, pred = _mani_align_arrays(gt, pred)
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = _mani_finite_rows(Xg, Xp)
    if Xg.shape[0] < 64:
        return None, None
    Xg_z, Xp_z = _mani_standardize_with_gt(Xg, Xp)
    k = _mani_choose_k(Xg_z, k_max=min(k_max, Xg_z.shape[1]))
    pca = PCA(n_components=k, svd_solver='full', random_state=seed).fit(Xg_z)
    Zg = pca.transform(Xg_z)
    Zp = pca.transform(Xp_z)
    rng = np.random.default_rng(seed)
    if Zg.shape[0] > n_points:
        Zg = Zg[rng.choice(Zg.shape[0], size=n_points, replace=False)]
    if Zp.shape[0] > n_points:
        Zp = Zp[rng.choice(Zp.shape[0], size=n_points, replace=False)]
    return Zg.astype(np.float64), Zp.astype(np.float64)


def _mani_stratified_latent_clouds(gt, pred, k_max=3, points_per_seq=2, max_sequences=48, seed=5):
    gt, pred = _mani_align_arrays(gt, pred)
    mu, sd, pca = _mani_fit_pooled_gt_pca(gt, pred, k_max=k_max, seed=seed)
    if pca is None:
        return None, None
    seq_indices = np.arange(gt.shape[0], dtype=int)
    if seq_indices.size > max_sequences:
        keep_idx = np.linspace(0, seq_indices.size - 1, num=max_sequences, dtype=int)
        seq_indices = seq_indices[keep_idx]
    clouds_g = []
    clouds_p = []
    for seq_idx in seq_indices:
        Xg, Xp = _mani_finite_rows(gt[seq_idx], pred[seq_idx])
        if Xg.shape[0] < max(8, points_per_seq + 1):
            continue
        Zg = pca.transform((Xg - mu) / sd)
        Zp = pca.transform((Xp - mu) / sd)
        take = min(points_per_seq, Zg.shape[0], Zp.shape[0])
        idx = np.linspace(0, min(Zg.shape[0], Zp.shape[0]) - 1, num=take, dtype=int)
        clouds_g.append(Zg[idx])
        clouds_p.append(Zp[idx])
    if not clouds_g or not clouds_p:
        return None, None
    return np.vstack(clouds_g).astype(np.float64), np.vstack(clouds_p).astype(np.float64)


def _mani_ripser_diagrams(X, maxdim=1, n_perm=64):
    if ripser is None or X is None or X.shape[0] < 16:
        return None
    return ripser(X, maxdim=maxdim, n_perm=min(n_perm, X.shape[0]))['dgms']


def _mani_lifetimes(dgm):
    if dgm is None or len(dgm) == 0:
        return np.array([], dtype=np.float64)
    arr = np.asarray(dgm, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.array([], dtype=np.float64)
    life = arr[:, 1] - arr[:, 0]
    mask = np.isfinite(life) & (life > 0)
    return np.sort(life[mask].astype(np.float64))


def _mani_lifetime_similarity(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 and y.size == 0:
        return 1.0
    if x.size == 0 or y.size == 0:
        return 0.0
    dist = _mani_quantile_distance(x, y, qs=(0.25, 0.5, 0.75, 0.9))
    total = abs(np.sum(x) - np.sum(y)) / (np.sum(x) + 0.05)
    return float(np.nanmean([_mani_score_from_distance(dist), _mani_score_from_distance(total)]))


def _mani_pairwise_distances(X):
    X = np.asarray(X, dtype=np.float64)
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _mani_knn_distance_profile(X, k=5):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < max(8, k + 2):
        return np.array([], dtype=np.float64)
    D = _mani_pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    kth = np.partition(D, kth=min(k - 1, D.shape[1] - 1), axis=1)[:, min(k - 1, D.shape[1] - 1)]
    return kth[np.isfinite(kth)]


def _mani_region_permute_blend(data, level, seed=0):
    rng = np.random.default_rng(seed)
    out = np.array(data, copy=True)
    perm = rng.permutation(out.shape[-1])
    permuted = out[:, :, perm]
    return (1.0 - level) * out + level * permuted


def _mani_topology_score(gt, pred):
    Zg, Zp = _mani_stratified_latent_clouds(gt, pred, k_max=3, points_per_seq=2, max_sequences=48, seed=5)
    if Zg is None or Zp is None:
        return np.nan
    dgms_g = _mani_ripser_diagrams(Zg, maxdim=1, n_perm=64)
    dgms_p = _mani_ripser_diagrams(Zp, maxdim=1, n_perm=64)
    if dgms_g is None or dgms_p is None:
        return np.nan
    score_h0 = _mani_lifetime_similarity(_mani_lifetimes(dgms_g[0]), _mani_lifetimes(dgms_p[0]))
    score_h1 = _mani_lifetime_similarity(_mani_lifetimes(dgms_g[1]), _mani_lifetimes(dgms_p[1]))
    return float(np.nanmean([score_h0, score_h1]))


def _mani_local_score(gt, pred):
    Zg, Zp = _mani_pooled_latent_clouds(gt, pred, k_max=3, n_points=128, seed=3)
    if Zg is None or Zp is None:
        return np.nan
    score_life = _mani_topology_score(gt, pred)
    knn_g = _mani_knn_distance_profile(Zg, k=5)
    knn_p = _mani_knn_distance_profile(Zp, k=5)
    score_knn = _mani_score_from_distance(_mani_quantile_distance(knn_g, knn_p, qs=(0.1, 0.25, 0.5, 0.75, 0.9)))
    if not np.isfinite(score_life) or not np.isfinite(score_knn):
        return np.nan
    return float(np.sqrt(score_life * score_knn))


def _mani_final_score(gt, pred):
    topology = _mani_topology_score(gt, pred)
    local = _mani_local_score(gt, pred)
    vals = []
    wts = []
    if np.isfinite(topology):
        vals.append(topology)
        wts.append(0.75)
    if np.isfinite(local):
        vals.append(local)
        wts.append(0.25)
    if not vals:
        return np.nan
    return float(np.average(vals, weights=wts))


def _mani_build_corruption_df(gt, score_fn, levels=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), seed=0):
    rows = []
    for level in levels:
        corrupted = _mani_region_permute_blend(gt, float(level), seed=seed)
        rows.append({
            'family': 'baseline' if level == 0 else 'region_permute_blend',
            'magnitude': float(level),
            'MANI_score01': _mani_final_score(gt, corrupted),
        })
    return pd.DataFrame(rows)


mani_score = _mani_final_score(gt_array_denorm, data_predicted)
mani_simple = {
    'candidate_name': 'final_manifold_score',
    'description': (
        'Persistent-homology lifetime agreement stabilized by local-neighborhood geometry. '
        'It combines a stratified PH topology term with a local k-nearest-neighbor geometry term.'
    ),
    'scores': {
        'MANI_score01': float(mani_score) if np.isfinite(mani_score) else np.nan,
        'MANI_mean': float(mani_score) if np.isfinite(mani_score) else np.nan,
    },
}
mani_corr_df = _mani_build_corruption_df(gt_array_denorm, _mani_final_score)

print('=== Manifold replacement metric ===')
print(mani_simple['description'])
print(f"MANI_score01: {mani_simple['scores']['MANI_score01']:.6f}" if np.isfinite(mani_simple['scores']['MANI_score01']) else 'MANI_score01: NaN')
if ripser is None:
    print('ripser is not available in this kernel, so persistent-homology terms return NaN.')
print()
print('Corruption sweep:')
print(mani_corr_df.to_string(index=False))

display(pd.DataFrame([{
    'metric': 'MANI_score01',
    'value': mani_simple['scores']['MANI_score01'],
    'candidate_name': mani_simple['candidate_name'],
}]))

# 1) Region permutation corruption sweep
display(mani_corr_df)

if bool(globals().get('ENABLE_PLOTS', True)) and not mani_corr_df.empty:
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(mani_corr_df['magnitude'], mani_corr_df['MANI_score01'], marker='o', linewidth=2.2, color='#1f77b4')
    ax.set_xlabel('Corruption magnitude')
    ax.set_ylabel('score (0-1)')
    ax.set_ylim(0.0, 1.0)
    ax.set_title('Manifold replacement degradation')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()


# ## Embedding Diagnostic
# This cell builds a low-dimensional visualization of ground truth, prediction, and a shuffled reference. It is diagnostic rather than part of the scalar benchmark: the goal is to inspect whether prediction trajectories occupy the same neighborhood structure as the real data and whether they remain distinguishable from a deliberately corrupted baseline.

# In[29]:


# --- UMAP baseline embedding: shuffled GT vs GT vs predictions ---
import numpy as np
import matplotlib.pyplot as plt

try:
    from umap import UMAP
    umap_constructor = UMAP
except ImportError:
    from sklearn.manifold import TSNE
    umap_constructor = lambda **kwargs: TSNE(n_components=kwargs.get('n_components', 2), random_state=kwargs.get('random_state', 42))
    print('umap-learn not installed; falling back to t-SNE for embedding.')

if '_flatten_sequences' not in globals():
    def _flatten_sequences(arr):
        return arr.reshape(-1, arr.shape[-1])

def _align_regions_for_embedding(gt_arr, pred_arr, gt_regions, pred_regions):
    labels = gt_regions if pred_regions == gt_regions else [r for r in gt_regions if r in pred_regions]
    if not labels:
        raise ValueError('No overlapping regions between GT and predictions for embedding analysis.')
    gt_idx = [gt_regions.index(r) for r in labels]
    pred_idx = [pred_regions.index(r) for r in labels]
    return labels, gt_arr[..., gt_idx], pred_arr[..., pred_idx]

def _circular_shuffle_per_region(arr, rng):
    # Circularly shift each region independently along the time axis
    arr = np.asarray(arr)
    shuffled = arr.copy()
    n_time = arr.shape[1]
    shifts = rng.integers(0, n_time, size=arr.shape[2])
    for region_idx, shift in enumerate(shifts):
        if shift == 0:
            continue
        shuffled[..., region_idx] = np.roll(shuffled[..., region_idx], shift=shift, axis=1)
    return shuffled

region_labels, gt_aligned, pred_aligned = _align_regions_for_embedding(
    gt_array_denorm, data_predicted, gt_regions, pred_region_names
)

rng = np.random.default_rng(42)
gt_shuffled = _circular_shuffle_per_region(gt_aligned, rng)

# Flatten sequences -> (samples, regions)
gt_flat = _flatten_sequences(gt_aligned)
pred_flat = _flatten_sequences(pred_aligned)
shuffled_flat = _flatten_sequences(gt_shuffled)

reducer = umap_constructor(n_components=2, random_state=42)
embedding_ref = reducer.fit_transform(shuffled_flat)
embedding_gt = reducer.transform(gt_flat) if hasattr(reducer, 'transform') else reducer.fit_transform(gt_flat)
embedding_pred = reducer.transform(pred_flat) if hasattr(reducer, 'transform') else reducer.fit_transform(pred_flat)

# Optional subsampling for visualization clarity
def _maybe_subsample(points, labels, max_points=5000):
    if points.shape[0] <= max_points:
        return points, labels
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx], labels[idx]

plot_data = []
plot_data.append((embedding_ref, np.full(embedding_ref.shape[0], 'Shuffled GT', dtype=object)))
plot_data.append((embedding_gt, np.full(embedding_gt.shape[0], 'Ground Truth', dtype=object)))
plot_data.append((embedding_pred, np.full(embedding_pred.shape[0], 'Predicted', dtype=object)))

fig, ax = plt.subplots(figsize=(8, 6))
colors = {'Shuffled GT': '#bbbbbb', 'Ground Truth': '#1f77b4', 'Predicted': '#ff7f0e'}
for emb, label_arr in plot_data:
    emb, label_arr = _maybe_subsample(emb, label_arr)
    ax.scatter(emb[:, 0], emb[:, 1], s=8, alpha=0.3, label=label_arr[0], c=colors.get(label_arr[0], None))

ax.set_title('Embedding comparison (UMAP trained on circularly shuffled GT)')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.legend()
ax.grid(True, alpha=0.2)
plt.tight_layout()

print('UMAP fitted on circularly shuffled GT; lower separation between GT and predictions suggests closer distributional alignment.')


# ## Trajectory Realism
# This cell scores the distribution of trajectories in a shared GT-defined latent space. The active implementation combines occupancy-and-velocity agreement with a path-feature agreement term, so `TRJDIST_score01` rewards predictions that visit the same latent regions with similar motion statistics and similar path geometry. This is the finalized trajectory replacement used by the benchmark.

# In[32]:


# === TRAJECTORY DISTRIBUTION realism (FIXED: global GT-PCA basis, pooled across sequences) ===
import pandas as pd
import matplotlib.pyplot as plt

from nethobench.analysis.refined_neuro_metric_replacements import compute_trajectory_replacement


trajectory_dist_simple, trjdist_corr_df = compute_trajectory_replacement(
    gt_array_denorm,
    data_predicted,
    enable_plots=bool(globals().get("ENABLE_PLOTS", True)),
)

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
display(trjdist_corr_df)


# ## Optional Animation
# This cell defines an animation helper for interactive inspection of latent trajectories. It does not contribute to any score. Its purpose is to let a reader visually compare one sequence of ground truth, prediction, and shuffled reference in the same low-dimensional embedding.

# In[33]:


# --- Animated trajectory visualizer (optional video export) ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

_required = ['embedding_gt_seq', 'embedding_pred_seq', 'embedding_ref_seq']
_missing = [name for name in _required if name not in globals()]
if _missing:
    raise RuntimeError(f'Required trajectory tensors not found: {_missing}. Run the trajectory cell first.')

def animate_sequence(seq_idx=0, include_shuffled=True, fps=12, tail=25, save_path=None):
    if seq_idx >= embedding_gt_seq.shape[0]:
        raise IndexError(f'seq_idx {seq_idx} out of range (n_seq={embedding_gt_seq.shape[0]}).')
    gt_traj = embedding_gt_seq[seq_idx]
    pred_traj = embedding_pred_seq[seq_idx]
    shuffled_traj = embedding_ref_seq[seq_idx]
    n_frames = gt_traj.shape[0]

    if include_shuffled:
        all_points = np.concatenate([gt_traj, pred_traj, shuffled_traj], axis=0)
    else:
        all_points = np.concatenate([gt_traj, pred_traj], axis=0)
    padding = 0.05 * (all_points.max(axis=0) - all_points.min(axis=0) + 1e-9)
    mins = all_points.min(axis=0) - padding
    maxs = all_points.max(axis=0) + padding

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_title(f'Embedding trajectory animation (sequence {seq_idx})')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.grid(True, alpha=0.2)

    line_gt, = ax.plot([], [], color='#1f77b4', linewidth=2, label='GT path')
    point_gt, = ax.plot([], [], marker='o', color='#1f77b4', markersize=6)
    line_pred, = ax.plot([], [], color='#ff7f0e', linewidth=2, label='Pred path')
    point_pred, = ax.plot([], [], marker='o', color='#ff7f0e', markersize=6)

    if include_shuffled:
        line_shuffled, = ax.plot([], [], color='#bbbbbb', linewidth=1.5, linestyle='--', alpha=0.6, label='Shuffled path')
    else:
        line_shuffled = None

    tail = max(1, tail)

    def _update(frame):
        start = max(0, frame - tail)
        gt_segment = gt_traj[start:frame + 1]
        pred_segment = pred_traj[start:frame + 1]
        line_gt.set_data(gt_segment[:, 0], gt_segment[:, 1])
        line_pred.set_data(pred_segment[:, 0], pred_segment[:, 1])
        point_gt.set_data([gt_traj[frame, 0]], [gt_traj[frame, 1]])
        point_pred.set_data([pred_traj[frame, 0]], [pred_traj[frame, 1]])
        artists = [line_gt, point_gt, line_pred, point_pred]
        if include_shuffled and line_shuffled is not None:
            shuffled_segment = shuffled_traj[start:frame + 1]
            line_shuffled.set_data(shuffled_segment[:, 0], shuffled_segment[:, 1])
            artists.append(line_shuffled)
        return artists

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=1000 / fps, blit=True)

    try:
        html = HTML(anim.to_jshtml())
    finally:
        plt.close(fig)

    if save_path:
        try:
            anim.save(save_path, writer='ffmpeg', dpi=120)
            print(f'Animation saved to {save_path}')
        except Exception as exc:
            print(f'Could not save animation to {save_path}: {exc}')

    return html

_ready = all(name in globals() for name in _required)
if _ready:
    _ready = all(globals()[name] is not None for name in _required)
if _ready:
    _ready = all(hasattr(globals()[name], 'shape') and getattr(globals()[name], 'ndim', 0) >= 3 for name in _required)

if _ready:
    preview_animation = animate_sequence(seq_idx=0, include_shuffled=True, tail=40)
    display(preview_animation)
    print('Use animate_sequence(seq_idx, include_shuffled, fps, tail, save_path) for custom previews or exports.')
else:
    print('Animation preview skipped because trajectory embedding tensors are unavailable in this run.')


# ## Bandpower Realism
# This cell measures spectral realism in predefined frequency bands. It computes bandpower summaries from the ground truth and the predictions, compares their relative band occupancy and regional spectral profile, and maps that discrepancy into `BP_score01`. The associated corruption sweep is designed to detect temporal perturbations that preserve rough amplitudes while distorting frequency content.

# In[34]:


# === BANDPOWER realism (simple, strict, benchmark-friendly) ===
import pandas as pd
import matplotlib.pyplot as plt

from nethobench.analysis.refined_neuro_metric_replacements import compute_bandpower_replacement


bandpower_simple, bp_corr_df = compute_bandpower_replacement(
    gt_array_denorm,
    data_predicted,
    enable_plots=bool(globals().get("ENABLE_PLOTS", True)),
)
bandpower_corr_df = bp_corr_df

display(pd.DataFrame([{
    "metric": "BP_score01",
    "value": bandpower_simple["scores"]["BP_score01"],
    "candidate_name": bandpower_simple["candidate_name"],
}]))

# 1) Temporal shuffle corruption sweep
display(bp_corr_df)


# ## Score Table
# This cell assembles the scalar neuro metrics computed above into a single summary table. It is a reporting cell: no new metric is introduced here. The purpose is to expose the per-metric score values in a compact form before they are grouped into families and collapsed into the final composite.

# In[35]:


#####FINAL COMPOSITE SCORE (SEE BELOW)#####

# ## Final Composite
# This cell groups the scalar metrics into the benchmark families `distribution`, `fidelity`, `temporal_spectral`, `relational`, and `geometry`, then computes the final neuro composite as the weighted arithmetic mean over the families that are available in the current run. The cell therefore defines the single headline neuro score reported by the package while preserving interpretable family-level subscores.

# ## Additional Structural Metrics
# 
# This cell computes supplementary relational and geometry metrics. It compares shrinkage-based conditional interactions, spectral shape, effective dimensionality, covariance and precision eigenspectra, dominant subspaces, lagged covariance structure, and simple lag-1 response kernels. All outputs are bounded in `[0,1]`, with larger values indicating better agreement between GT and prediction.

# In[ ]:


# === ADDITIONAL STRUCTURAL METRICS (partial corr, spectra, subspaces, lagged dynamics) ===
import pandas as pd

from nethobench.analysis.additional_neuro_metrics import compute_additional_structural_metrics

additional_structural = compute_additional_structural_metrics(gt_array_denorm, data_predicted)
additional_structural_scores = additional_structural.get('scores', {})

display(pd.DataFrame([{'metric': key, 'value': value} for key, value in additional_structural_scores.items()]).sort_values('metric').reset_index(drop=True))


# In[36]:


# === FINAL COMPOSITE (family-based, using exact upstream notebook result names) ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This cell reads the score variables from the exact result objects created earlier
# in this notebook. It does not recompute metrics.


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


def _weighted_mean_available(values, weights):
    keys = [k for k, v in values.items() if np.isfinite(v) and weights.get(k, 0.0) > 0]
    if not keys:
        return np.nan
    wsum = float(np.sum([weights[k] for k in keys]))
    if wsum <= 0:
        return np.nan
    return float(np.sum([weights[k] * values[k] for k in keys]) / wsum)


# ----------------------------
# Exact upstream notebook names
# ----------------------------
KL_score01 = np.nan
if 'dist_scores' in globals() and isinstance(dist_scores, dict):
    KL_score01 = _safe_get(dist_scores, 'KL_score01_avg', default=np.nan)
if not np.isfinite(KL_score01) and 'kl_sweep_df' in globals():
    KL_score01 = _from_df('kl_sweep_df', 'KL_score01_avg', row_col='corruption', row_value='baseline')
if not np.isfinite(KL_score01) and 'dist_out' in globals() and isinstance(dist_out, dict):
    kl_mean = _safe_get(dist_out, 'summary', 'KL_geo', 'mean', default=np.nan)
    kl_q10 = _safe_get(dist_out, 'summary', 'KL_geo', 'q10_strict', default=np.nan)
    if np.isfinite(kl_mean) and np.isfinite(kl_q10):
        KL_score01 = 0.5 * (kl_mean + kl_q10)

JSD_score01 = np.nan
Mean_score01 = _from_df('mean_top10_df', 'Mean_score01', row_col='corruption', row_value='baseline')

MI_score01 = np.nan
if 'results_mi' in globals() and isinstance(results_mi, dict):
    MI_score01 = _safe_get(results_mi, 'scores', 'MI_score01', default=np.nan)
if not np.isfinite(MI_score01):
    MI_score01 = _from_df('mi_corruption_df', 'MI_score01', row_col='family', row_value='baseline')

Error_score01 = np.nan
if 'err_simple' in globals() and isinstance(err_simple, dict):
    Error_score01 = _safe_get(err_simple, 'scores', 'Error_score01', default=np.nan)
if not np.isfinite(Error_score01):
    Error_score01 = _from_df('err_noise_df', 'Error_score01', row_col='sigma', row_value=0.0)

QNT_score01 = np.nan
if 'qnt_simple' in globals() and isinstance(qnt_simple, dict):
    QNT_score01 = _safe_get(qnt_simple, 'scores', 'QNT_score01', default=np.nan)
if not np.isfinite(QNT_score01):
    QNT_score01 = _from_df('qnt_corr_df', 'QNT_score01', row_col='family', row_value='baseline')

FC_score01 = np.nan
if 'fc_simple' in globals() and isinstance(fc_simple, dict):
    FC_score01 = _safe_get(fc_simple, 'scores', 'FC_score01', default=np.nan)
if not np.isfinite(FC_score01):
    FC_score01 = _from_df('fc_corr_df', 'FC_score01', row_col='family', row_value='baseline')

PCA_score01 = np.nan
if 'pca_simple' in globals() and isinstance(pca_simple, dict):
    PCA_score01 = _safe_get(pca_simple, 'scores', 'PCA_score01', default=np.nan)
if not np.isfinite(PCA_score01):
    PCA_score01 = _from_df('pca_corr_df', 'PCA_score01', row_col='family', row_value='baseline')

AUTO_score01 = np.nan
if 'auto_sensitive' in globals() and isinstance(auto_sensitive, dict):
    AUTO_score01 = _safe_get(auto_sensitive, 'scores', 'AUTO_score01', default=np.nan)
if not np.isfinite(AUTO_score01) and 'auto_simple' in globals() and isinstance(auto_simple, dict):
    AUTO_score01 = _safe_get(auto_simple, 'scores', 'AUTO_score01', default=np.nan)
if not np.isfinite(AUTO_score01):
    AUTO_score01 = _from_df('auto_corr_df', 'AUTO_score01', row_col='family', row_value='baseline')

CC_score01 = np.nan
if 'cc_sensitive' in globals() and isinstance(cc_sensitive, dict):
    CC_score01 = _safe_get(cc_sensitive, 'scores', 'CC_score01', default=np.nan)
if not np.isfinite(CC_score01) and 'cc_simple' in globals() and isinstance(cc_simple, dict):
    CC_score01 = _safe_get(cc_simple, 'scores', 'CC_score01', default=np.nan)
if not np.isfinite(CC_score01):
    CC_score01 = _from_df('cc_corr_df', 'CC_score01', row_col='family', row_value='baseline')

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

CCA_score01 = np.nan
if 'cca_strict' in globals() and isinstance(cca_strict, dict):
    CCA_score01 = _safe_get(cca_strict, 'scores', 'CCA_score01', default=np.nan)
if not np.isfinite(CCA_score01):
    CCA_score01 = _from_df('cca_corr_df', 'CCA_score01', row_col='family', row_value='baseline')

MANI_score01 = np.nan
if 'mani_simple' in globals() and isinstance(mani_simple, dict):
    MANI_score01 = _safe_get(mani_simple, 'scores', 'MANI_score01', default=np.nan)

BP_score01 = np.nan
if 'bandpower_simple' in globals() and isinstance(bandpower_simple, dict):
    BP_score01 = _safe_get(bandpower_simple, 'scores', 'BP_score01', default=np.nan)

TRJDIST_score01 = np.nan
if 'trajectory_dist_simple' in globals() and isinstance(trajectory_dist_simple, dict):
    TRJDIST_score01 = _safe_get(trajectory_dist_simple, 'scores', 'TRJDIST_score01', default=np.nan)
if not np.isfinite(TRJDIST_score01):
    TRJDIST_score01 = _from_df('trjdist_corr_df', 'TRJDIST_score01', row_col='family', row_value='baseline')

extra_metric_names = [
    'PartialCorr_score01',
    'PSDShape_score01',
    'Dimensionality_score01',
    'CrossRegionMI_score01',
    'PrecisionMatrixSpectrum_score01',
    'SubspaceAngle_score01',
    'EigenspectrumShape_score01',
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
    'KL_score01': KL_score01,
    'JSD_score01': JSD_score01,
    'Mean_score01': Mean_score01,
    'MI_score01': MI_score01,
    'Error_score01': Error_score01,
    'QNT_score01': QNT_score01,
    'FC_score01': FC_score01,
    'PCA_score01': PCA_score01,
    'AUTO_score01': AUTO_score01,
    'CC_score01': CC_score01,
    'MOM_score01': MOM_score01,
    'GRAPH_score01': GRAPH_score01,
    'CCA_score01': CCA_score01,
    'MANI_score01': MANI_score01,
    'BP_score01': BP_score01,
    'TRJDIST_score01': TRJDIST_score01,
}
for metric_name in extra_metric_names:
    SCORES[metric_name] = _safe_get({'scores': extra_scores}, 'scores', metric_name, default=np.nan)

DIST_DIVERGENCE = SCORES['KL_score01'] if np.isfinite(SCORES['KL_score01']) else SCORES['JSD_score01']

DIST_VALUES = {
    'Divergence': DIST_DIVERGENCE,
    'Quantiles': SCORES['QNT_score01'],
    'Moments': SCORES['MOM_score01'],
    'Mean': SCORES['Mean_score01'],
}
DIST_WEIGHTS = {
    'Divergence': 0.35,
    'Quantiles': 0.30,
    'Moments': 0.25,
    'Mean': 0.10,
}
S_dist = _weighted_mean_available(DIST_VALUES, DIST_WEIGHTS)

FID_VALUES = {
    'Error': SCORES['Error_score01'],
    'MI': SCORES['MI_score01'],
}
FID_WEIGHTS = {
    'Error': 0.65,
    'MI': 0.35,
}
S_fid = _weighted_mean_available(FID_VALUES, FID_WEIGHTS)

TEMP_VALUES = {
    'Autocorr': SCORES['AUTO_score01'],
    'Bandpower': SCORES['BP_score01'],
    'Trajectory': SCORES['TRJDIST_score01'],
    'PSDShape': SCORES['PSDShape_score01'],
}
TEMP_WEIGHTS = {
    'Autocorr': 0.28,
    'Bandpower': 0.20,
    'Trajectory': 0.30,
    'PSDShape': 0.22,
}
S_temp = _weighted_mean_available(TEMP_VALUES, TEMP_WEIGHTS)

REL_VALUES = {
    'FC': SCORES['FC_score01'],
    'CrossCorr': SCORES['CC_score01'],
    'Graph': SCORES['GRAPH_score01'],
    'PartialCorr': SCORES['PartialCorr_score01'],
    'CrossRegionMI': SCORES['CrossRegionMI_score01'],
    'PrecisionMatrixSpectrum': SCORES['PrecisionMatrixSpectrum_score01'],
    'LaggedCovariance': SCORES['LaggedCovariance_score01'],
    'ImpulseResponse': SCORES['ImpulseResponse_score01'],
}
REL_WEIGHTS = {
    'FC': 0.15,
    'CrossCorr': 0.13,
    'Graph': 0.12,
    'PartialCorr': 0.15,
    'CrossRegionMI': 0.10,
    'PrecisionMatrixSpectrum': 0.12,
    'LaggedCovariance': 0.13,
    'ImpulseResponse': 0.10,
}
S_rel = _weighted_mean_available(REL_VALUES, REL_WEIGHTS)

GEOM_VALUES = {
    'PCA': SCORES['PCA_score01'],
    'Manifold': SCORES['MANI_score01'],
    'CCA': SCORES['CCA_score01'],
    'Dimensionality': SCORES['Dimensionality_score01'],
    'SubspaceAngle': SCORES['SubspaceAngle_score01'],
    'EigenspectrumShape': SCORES['EigenspectrumShape_score01'],
}
GEOM_WEIGHTS = {
    'PCA': 0.16,
    'Manifold': 0.18,
    'CCA': 0.18,
    'Dimensionality': 0.12,
    'SubspaceAngle': 0.18,
    'EigenspectrumShape': 0.18,
}
S_geom = _weighted_mean_available(GEOM_VALUES, GEOM_WEIGHTS)

STATE_VALUES = {
    'LatentStateOccupancyK11': SCORES['LatentStateOccupancyK11_score01'],
    'LatentStateOccupancyK12': SCORES['LatentStateOccupancyK12_score01'],
    'LatentStateTransitionLag1K11': SCORES['LatentStateTransitionLag1K11_score01'],
    'LatentStateTransitionLag2K11': SCORES['LatentStateTransitionLag2K11_score01'],
    'LatentStateTransitionLag3K11': SCORES['LatentStateTransitionLag3K11_score01'],
}
STATE_WEIGHTS = {
    'LatentStateOccupancyK11': 0.22,
    'LatentStateOccupancyK12': 0.18,
    'LatentStateTransitionLag1K11': 0.20,
    'LatentStateTransitionLag2K11': 0.20,
    'LatentStateTransitionLag3K11': 0.20,
}
S_state = _weighted_mean_available(STATE_VALUES, STATE_WEIGHTS)

FAMILY_VALUES = {
    'distribution': S_dist,
    'fidelity': S_fid,
    'temporal_spectral': S_temp,
    'relational': S_rel,
    'geometry': S_geom,
    'state_dynamics': S_state,
}
FAMILY_WEIGHTS = {
    'distribution': 0.20,
    'fidelity': 0.08,
    'temporal_spectral': 0.16,
    'relational': 0.22,
    'geometry': 0.18,
    'state_dynamics': 0.16,
}
FINAL_COMPOSITE_SCORE = _weighted_mean_available(FAMILY_VALUES, FAMILY_WEIGHTS)

metrics_df = pd.DataFrame([
    {'family': 'distribution', 'metric': 'KL_or_JSD_score01', 'value': DIST_DIVERGENCE},
    {'family': 'distribution', 'metric': 'QNT_score01', 'value': SCORES['QNT_score01']},
    {'family': 'distribution', 'metric': 'MOM_score01', 'value': SCORES['MOM_score01']},
    {'family': 'distribution', 'metric': 'Mean_score01', 'value': SCORES['Mean_score01']},
    {'family': 'fidelity', 'metric': 'Error_score01', 'value': SCORES['Error_score01']},
    {'family': 'fidelity', 'metric': 'MI_score01', 'value': SCORES['MI_score01']},
    {'family': 'temporal_spectral', 'metric': 'AUTO_score01', 'value': SCORES['AUTO_score01']},
    {'family': 'temporal_spectral', 'metric': 'BP_score01', 'value': SCORES['BP_score01']},
    {'family': 'temporal_spectral', 'metric': 'TRJDIST_score01', 'value': SCORES['TRJDIST_score01']},
    {'family': 'temporal_spectral', 'metric': 'PSDShape_score01', 'value': SCORES['PSDShape_score01']},
    {'family': 'relational', 'metric': 'FC_score01', 'value': SCORES['FC_score01']},
    {'family': 'relational', 'metric': 'CC_score01', 'value': SCORES['CC_score01']},
    {'family': 'relational', 'metric': 'GRAPH_score01', 'value': SCORES['GRAPH_score01']},
    {'family': 'relational', 'metric': 'PartialCorr_score01', 'value': SCORES['PartialCorr_score01']},
    {'family': 'relational', 'metric': 'CrossRegionMI_score01', 'value': SCORES['CrossRegionMI_score01']},
    {'family': 'relational', 'metric': 'PrecisionMatrixSpectrum_score01', 'value': SCORES['PrecisionMatrixSpectrum_score01']},
    {'family': 'relational', 'metric': 'LaggedCovariance_score01', 'value': SCORES['LaggedCovariance_score01']},
    {'family': 'relational', 'metric': 'ImpulseResponse_score01', 'value': SCORES['ImpulseResponse_score01']},
    {'family': 'geometry', 'metric': 'PCA_score01', 'value': SCORES['PCA_score01']},
    {'family': 'geometry', 'metric': 'MANI_score01', 'value': SCORES['MANI_score01']},
    {'family': 'geometry', 'metric': 'CCA_score01', 'value': SCORES['CCA_score01']},
    {'family': 'geometry', 'metric': 'Dimensionality_score01', 'value': SCORES['Dimensionality_score01']},
    {'family': 'geometry', 'metric': 'SubspaceAngle_score01', 'value': SCORES['SubspaceAngle_score01']},
    {'family': 'geometry', 'metric': 'EigenspectrumShape_score01', 'value': SCORES['EigenspectrumShape_score01']},
    {'family': 'state_dynamics', 'metric': 'LatentStateOccupancyK11_score01', 'value': SCORES['LatentStateOccupancyK11_score01']},
    {'family': 'state_dynamics', 'metric': 'LatentStateOccupancyK12_score01', 'value': SCORES['LatentStateOccupancyK12_score01']},
    {'family': 'state_dynamics', 'metric': 'LatentStateTransitionLag1K11_score01', 'value': SCORES['LatentStateTransitionLag1K11_score01']},
    {'family': 'state_dynamics', 'metric': 'LatentStateTransitionLag2K11_score01', 'value': SCORES['LatentStateTransitionLag2K11_score01']},
    {'family': 'state_dynamics', 'metric': 'LatentStateTransitionLag3K11_score01', 'value': SCORES['LatentStateTransitionLag3K11_score01']},
])

families_df = pd.DataFrame([
    {'family': 'distribution', 'value': S_dist, 'weight': FAMILY_WEIGHTS['distribution']},
    {'family': 'fidelity', 'value': S_fid, 'weight': FAMILY_WEIGHTS['fidelity']},
    {'family': 'temporal_spectral', 'value': S_temp, 'weight': FAMILY_WEIGHTS['temporal_spectral']},
    {'family': 'relational', 'value': S_rel, 'weight': FAMILY_WEIGHTS['relational']},
    {'family': 'geometry', 'value': S_geom, 'weight': FAMILY_WEIGHTS['geometry']},
    {'family': 'state_dynamics', 'value': S_state, 'weight': FAMILY_WEIGHTS['state_dynamics']},
])

family_colors = {
    'distribution': '#4C78A8',
    'fidelity': '#F58518',
    'temporal_spectral': '#54A24B',
    'relational': '#E45756',
    'geometry': '#72B7B2',
    'state_dynamics': '#9D755D',
}

plot_df = metrics_df[np.isfinite(metrics_df['value'].to_numpy(dtype=float))].copy()
plot_colors = [family_colors[f] for f in plot_df['family']]

fig, axes = plt.subplots(1, 2, figsize=(19, 5.5))
axes[0].bar(plot_df['metric'], plot_df['value'], color=plot_colors, edgecolor='k', alpha=0.9)
axes[0].set_title('Upstream score01 metrics used in composite')
axes[0].set_ylabel('score (0-1)')
axes[0].tick_params(axis='x', rotation=90)
axes[0].grid(axis='y', alpha=0.25)

family_plot_df = families_df[np.isfinite(families_df['value'].to_numpy(dtype=float))].copy()
family_plot_colors = [family_colors[f] for f in family_plot_df['family']]
axes[1].bar(family_plot_df['family'], family_plot_df['value'], color=family_plot_colors, edgecolor='k', alpha=0.9)
axes[1].set_title(f'Family composites | FINAL={FINAL_COMPOSITE_SCORE:.4f}')
axes[1].set_ylabel('score (0-1)')
axes[1].tick_params(axis='x', rotation=22)
axes[1].grid(axis='y', alpha=0.25)

plt.tight_layout()
plt.show()

print('=== METRICS USED ===')
print(metrics_df.to_string(index=False))
print('\n=== FAMILY COMPOSITES ===')
print(families_df.to_string(index=False))
print(f'\nFINAL_COMPOSITE_SCORE: {float(FINAL_COMPOSITE_SCORE):.6f}')


# ## Unified Corruption Sensitivity Dashboard
# This cell concatenates the corruption sweeps produced by the metric cells above, computes family-wise relative degradations, and renders the notebook-wide sensitivity summary. The line plots show how each score changes with corruption magnitude, while the polar plots summarize the strongest relative drop per corruption family. This dashboard is the main visual audit of whether the benchmark is sensitive to realistic failure modes.

# In[37]:



# === UNIFIED CORRUPTION SENSITIVITY DASHBOARD (all notebook corruption families) ===
import math
import io
import json
import contextlib
import warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

UCR_EPS = 1e-12
UCR_RNG_BASE_SEED = 20260312


def _ucr_quiet_call(func, *args, **kwargs):
    sink = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            return func(*args, **kwargs)


def _ucr_float_or_nan(value):
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def _ucr_extract_score(result, *keys):
    if isinstance(result, dict):
        for key in keys:
            if key in result:
                val = _ucr_float_or_nan(result[key])
                if np.isfinite(val):
                    return val
        nested = result.get('scores')
        if isinstance(nested, dict):
            for key in keys:
                if key in nested:
                    val = _ucr_float_or_nan(nested[key])
                    if np.isfinite(val):
                        return val
    return np.nan


def _ucr_align_gt_pred(gt_arr, pred_arr):
    gt_arr = np.asarray(gt_arr, dtype=np.float64)
    pred_arr = np.asarray(pred_arr, dtype=np.float64)

    align_fn = globals().get('_align_gt_pred')
    if callable(align_fn):
        try:
            gt_a, pred_a = align_fn(gt_arr, pred_arr)
            gt_a = np.asarray(gt_a, dtype=np.float64)
            pred_a = np.asarray(pred_a, dtype=np.float64)
            if gt_a.ndim == 3 and pred_a.ndim == 3 and gt_a.shape[0] == pred_a.shape[0] and gt_a.shape[2] == pred_a.shape[2]:
                return gt_a, pred_a
        except Exception:
            pass

    gt_len = gt_arr.shape[1]
    pred_len = pred_arr.shape[1]
    if pred_len != gt_len and pred_len % gt_len == 0:
        factor = pred_len // gt_len
        pred_res = pred_arr.reshape(pred_arr.shape[0], gt_len, factor, pred_arr.shape[2]).mean(axis=2)
        return gt_arr, pred_res

    T = min(gt_len, pred_len)
    return gt_arr[:, :T, :], pred_arr[:, :T, :]


def _ucr_region_iqr(gt_arr):
    vals = []
    for r in range(gt_arr.shape[2]):
        x = np.asarray(gt_arr[:, :, r], dtype=np.float64).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size == 0:
            vals.append(1.0)
            continue
        q25, q75 = np.nanquantile(x, [0.25, 0.75])
        sc = float(q75 - q25)
        if not np.isfinite(sc) or sc <= 1e-12:
            sc = float(np.nanstd(x))
        vals.append(sc if np.isfinite(sc) and sc > 1e-12 else 1.0)
    return np.asarray(vals, dtype=np.float64)


def _ucr_global_iqr(gt_arr):
    x = np.asarray(gt_arr, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    q25, q75 = np.nanquantile(x, [0.25, 0.75])
    sc = float(q75 - q25)
    if not np.isfinite(sc) or sc <= 1e-12:
        sc = float(np.nanstd(x))
    return sc if np.isfinite(sc) and sc > 1e-12 else 1.0


def _ucr_deranged(indices, rng):
    indices = np.asarray(indices, dtype=int)
    if indices.size <= 1:
        return indices.copy()
    for _ in range(64):
        perm = indices[rng.permutation(indices.size)]
        if not np.any(perm == indices):
            return perm
    return np.roll(indices, 1)


def _ucr_with_patched_globals(patches, fn, *args, **kwargs):
    saved = {}
    missing = set()
    for name in patches:
        if name in globals():
            saved[name] = globals()[name]
        else:
            missing.add(name)
    try:
        for name, value in patches.items():
            globals()[name] = value
        return _ucr_quiet_call(fn, *args, **kwargs)
    finally:
        for name in patches:
            if name in missing:
                globals().pop(name, None)
            else:
                globals()[name] = saved[name]


# --- lazy function bootstrap for notebook kernels that did not run all metric cells ---
_UCR_BOOTSTRAP_NOTEBOOK = None
_UCR_BOOTSTRAP_FUNC_CACHE = {}
_UCR_BOOTSTRAP_MARKERS = {
    '_compute_kl_metrics': '# Align once and keep names compatible with previous sections',
    'compute_mean_score01_top10': '# Baseline',
    'compute_mi_strict': '# ----------------------------\n# 3)',
    'compute_error_score01_simple': '# ----------------------------\n# 3)',
    'compute_quantile_score01_simple': '# ----------------------------\n# 3)',
    'compute_fc_score01_simple': '# ----------------------------\n# 3)',
    'compute_pca_score01_simple': '# ----------------------------\n# 3)',
    'compute_autocorr_score01_sensitive': '# ----------------------------\n# 3)',
    'compute_crosscorr_score01_sensitive': '# ----------------------------\n# 3)',
    'compute_moments_score01_simple': '# ----------------------------\n# 3)',
    'compute_graph_score01_simple': '# ----------------------------\n# 3)',
    'compute_cca_score01_strict': '# ----------------------------\n# 3)',
    'compute_manifold_score01_geometry': '# ----------------------------\n# 3)',
    'compute_trajectory_distribution_score01_fixed': '# -------------------------------------------------\n# 3)',
    'compute_bandpower_score01_strict': '# ----------------------------\n# 3)',
}


def _ucr_find_notebook_path():
    cwd = Path.cwd().resolve()
    candidates = []
    for base in [cwd, *cwd.parents]:
        candidates.append(base / 'nethobench' / 'notebooks' / 'neuro_metrics.ipynb')
        candidates.append(base / 'notebooks' / 'neuro_metrics.ipynb')
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _ucr_get_notebook_cells():
    global _UCR_BOOTSTRAP_NOTEBOOK
    if _UCR_BOOTSTRAP_NOTEBOOK is None:
        nb_file = _ucr_find_notebook_path()
        if nb_file is None:
            return None, None
        nb_obj = json.loads(nb_file.read_text())
        _UCR_BOOTSTRAP_NOTEBOOK = (nb_file, nb_obj.get('cells', []))
    return _UCR_BOOTSTRAP_NOTEBOOK


def _ucr_bootstrap_function(func_name):
    if func_name in _UCR_BOOTSTRAP_FUNC_CACHE:
        return _UCR_BOOTSTRAP_FUNC_CACHE[func_name]

    nb_file, cells = _ucr_get_notebook_cells()
    if nb_file is None:
        return None

    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        cell_src = ''.join(cell.get('source', []))
        if f'def {func_name}' not in cell_src:
            continue

        marker = _UCR_BOOTSTRAP_MARKERS.get(func_name)
        if marker and marker in cell_src:
            cell_src = cell_src[:cell_src.index(marker)]

        ns = {'__builtins__': __builtins__}
        exec(compile(cell_src, f'{nb_file}:{func_name}', 'exec'), ns, ns)
        fn = ns.get(func_name)
        if callable(fn):
            _UCR_BOOTSTRAP_FUNC_CACHE[func_name] = fn
            return fn
        return None

    return None


# ---------------------------------------------------------------------
# Corruption helpers (union of corruption logic used across notebook)
# ---------------------------------------------------------------------

# shared noise / scale helpers
SPIKE_LEVEL_CFG = {
    1: (0.01, 0.50),
    2: (0.02, 0.75),
    3: (0.04, 1.00),
    4: (0.06, 1.25),
    5: (0.08, 1.50),
}


def _ucr_mean_shift_region_iqr(pred_arr, level, ctx, rng):
    return np.asarray(pred_arr, dtype=np.float64) + float(level) * ctx['region_iqr'][None, None, :]


def _ucr_global_mean_shift(pred_arr, level, ctx, rng):
    return np.asarray(pred_arr, dtype=np.float64) + float(level) * float(ctx['global_iqr'])


def _ucr_region_mean_shift(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    pattern = np.linspace(-1.0, 1.0, out.shape[2], dtype=np.float64)
    out += float(level) * ctx['region_iqr'][None, None, :] * pattern[None, None, :]
    return out


def _ucr_sequence_mean_shift(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    offsets = rng.normal(loc=0.0, scale=float(level) * float(ctx['global_iqr']), size=(out.shape[0], 1, 1))
    out += offsets
    return out


def _ucr_slow_drift(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    T = out.shape[1]
    trend = np.linspace(0.0, 1.0, T, dtype=np.float64).reshape(1, T, 1)
    out += float(level) * float(ctx['global_iqr']) * trend
    return out


def _ucr_variance_scale_kl(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    center = np.nanmedian(out, axis=1, keepdims=True)
    return center + (1.0 + float(level)) * (out - center)


def _ucr_variance_scale_qnt(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    center = np.nanmedian(out, axis=1, keepdims=True)
    return center + float(level) * (out - center)


def _ucr_tail_spikes_kl(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    m = float(level)
    spike_prob = min(0.02 + 0.08 * m, 0.35)
    mask = rng.random(out.shape) < spike_prob
    spikes = rng.laplace(loc=0.0, scale=m, size=out.shape) * ctx['region_iqr'][None, None, :]
    out += mask * spikes
    return out


def _ucr_tail_spikes_qnt(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    lv = int(round(float(level)))
    p_spike, mag = SPIKE_LEVEL_CFG[lv]
    mask = rng.random(out.shape) < p_spike
    spikes = rng.laplace(loc=0.0, scale=1.0, size=out.shape) * (mag * ctx['region_iqr'][None, None, :])
    out += mask * spikes
    return out


def _ucr_tail_spikes_mom(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    lv = int(round(float(level)))
    p_spike, mag = SPIKE_LEVEL_CFG[lv]
    mask = rng.random(out.shape) < p_spike
    heavy = rng.standard_t(df=3, size=out.shape)
    out += mask * (heavy * (mag * ctx['region_iqr'][None, None, :]))
    return out


def _ucr_one_sided_spikes(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    lv = int(round(float(level)))
    p_spike, mag = SPIKE_LEVEL_CFG[lv]
    mask = rng.random(out.shape) < p_spike
    amp = np.abs(rng.standard_t(df=3, size=out.shape))
    out += mask * (amp * (mag * ctx['region_iqr'][None, None, :]))
    return out


def _ucr_additive_noise_iqr(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    noise = rng.normal(0.0, 1.0, size=out.shape)
    out += noise * (float(level) * ctx['region_iqr'][None, None, :])
    return out


def _ucr_temporal_shuffle(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    n_seq, T, _ = out.shape
    m = int(np.floor(float(level) * T))
    if m <= 1:
        return out
    for i in range(n_seq):
        idx = rng.choice(T, size=m, replace=False)
        perm = rng.permutation(idx)
        out[i, idx, :] = out[i, perm, :]
    return out


def _ucr_temporal_shuffle_per_region(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    n_seq, T, n_reg = out.shape
    m = int(np.floor(float(level) * T))
    if m <= 1:
        return out
    for i in range(n_seq):
        for r in range(n_reg):
            idx = rng.choice(T, size=m, replace=False)
            perm = rng.permutation(idx)
            out[i, idx, r] = out[i, perm, r]
    return out


def _ucr_region_noise(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    noise = rng.normal(0.0, 1.0, size=out.shape)
    out += noise * (float(level) * ctx['region_iqr'][None, None, :])
    return out


def _ucr_region_permutation(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    n_reg = out.shape[2]
    k = max(2, int(np.ceil(float(level) * n_reg)))
    k = min(k, n_reg)
    idx = rng.choice(n_reg, size=k, replace=False)
    perm = rng.permutation(idx)
    if np.all(perm == idx) and k > 1:
        perm = np.roll(perm, 1)
    out[:, :, idx] = out[:, :, perm]
    return out


def _ucr_region_permute_frac(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    n_reg = out.shape[2]
    k = max(1, int(np.round(float(level) * n_reg)))
    k = min(k, n_reg)
    sel = rng.choice(n_reg, size=k, replace=False)
    perm = sel[rng.permutation(k)]
    out[:, :, sel] = out[:, :, perm]
    return out


def _ucr_region_mixing(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    partner = np.roll(out, shift=1, axis=2)
    lam = float(level)
    return (1.0 - lam) * out + lam * partner


def _ucr_region_mix_lambda(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    n_reg = out.shape[2]
    perm = rng.permutation(n_reg)
    lam = float(level)
    return (1.0 - lam) * out + lam * out[:, :, perm]


def _ucr_ar_oversmoothing(pred_arr, level, ctx, rng):
    lam = float(level)
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    sm = out.copy()
    for i in range(sm.shape[0]):
        for r in range(sm.shape[2]):
            for t in range(1, sm.shape[1]):
                sm[i, t, r] = (1.0 - lam) * out[i, t, r] + lam * sm[i, t - 1, r]
    return sm


def _ucr_oversmoothing_lambda(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    lam = float(level)
    for t in range(1, out.shape[1]):
        out[:, t, :] = (1.0 - lam) * out[:, t, :] + lam * out[:, t - 1, :]
    return out


def _ucr_region_specific_lags(pred_arr, level, ctx, rng):
    out = np.full_like(np.asarray(pred_arr, dtype=np.float64), np.nan)
    n_seq, T, n_reg = out.shape
    lag_max = int(level)
    lags = rng.integers(-lag_max, lag_max + 1, size=n_reg)
    for r in range(n_reg):
        lag = int(lags[r])
        if lag == 0:
            out[:, :, r] = pred_arr[:, :, r]
        elif lag > 0:
            if lag < T:
                out[:, lag:, r] = pred_arr[:, :T - lag, r]
        else:
            k = -lag
            if k < T:
                out[:, :T - k, r] = pred_arr[:, k:, r]
    return out


def _ucr_fixed_lag_jitter_desync(pred_arr, level, ctx, rng):
    out = np.full_like(np.asarray(pred_arr, dtype=np.float64), np.nan)
    n_seq, T, n_reg = out.shape
    jmax = int(level)
    for i in range(n_seq):
        jit = rng.integers(-jmax, jmax + 1, size=n_reg)
        for r in range(n_reg):
            lag = int(jit[r])
            if lag == 0:
                out[i, :, r] = pred_arr[i, :, r]
            elif lag > 0:
                if lag < T:
                    out[i, lag:, r] = pred_arr[i, :T - lag, r]
            else:
                k = -lag
                if k < T:
                    out[i, :T - k, r] = pred_arr[i, k:, r]
    return out


def _ucr_anti_sync_mixing(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    partner = np.roll(out, shift=1, axis=2)
    lam = float(level)
    return (1.0 - lam) * out - lam * partner


def _ucr_phase_scramble(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    n_seq, T, n_reg = out.shape
    k = max(1, int(np.ceil(float(level) * n_reg)))
    k = min(k, n_reg)
    for i in range(n_seq):
        chosen = rng.choice(n_reg, size=k, replace=False)
        for r in chosen:
            x = out[i, :, r]
            if not np.isfinite(x).all():
                continue
            X = np.fft.rfft(x)
            mag = np.abs(X)
            ph = np.angle(X)
            rand_ph = rng.uniform(-np.pi, np.pi, size=ph.shape)
            rand_ph[0] = ph[0]
            if T % 2 == 0 and rand_ph.size > 1:
                rand_ph[-1] = ph[-1]
            X_new = mag * np.exp(1j * rand_ph)
            out[i, :, r] = np.real(np.fft.irfft(X_new, n=T))
    return out


def _ucr_temporal_circular_lag(pred_arr, level, ctx, rng):
    return np.roll(np.asarray(pred_arr, dtype=np.float64), shift=int(level), axis=1)


def _ucr_temporal_nanpad_lag(pred_arr, level, ctx, rng):
    lag = int(level)
    out = np.full_like(np.asarray(pred_arr, dtype=np.float64), np.nan)
    if lag == 0:
        return np.asarray(pred_arr, dtype=np.float64).copy()
    if lag > 0:
        if lag < out.shape[1]:
            out[:, lag:, :] = pred_arr[:, :-lag, :]
    else:
        k = -lag
        if k < out.shape[1]:
            out[:, :-k, :] = pred_arr[:, k:, :]
    return out


def _ucr_temporal_block_shuffle(pred_arr, level, ctx, rng):
    out = np.empty_like(np.asarray(pred_arr, dtype=np.float64))
    n_seq, t_len, _ = out.shape
    b = int(level)
    if b <= 1:
        return np.asarray(pred_arr, dtype=np.float64).copy()
    for s in range(n_seq):
        starts = np.arange(0, t_len, b)
        blocks = [pred_arr[s, st:min(st + b, t_len), :].copy() for st in starts]
        perm = rng.permutation(len(blocks))
        pos = 0
        for idx in perm:
            blk = blocks[idx]
            n = blk.shape[0]
            out[s, pos:pos + n, :] = blk
            pos += n
    return out


def _ucr_local_time_jitter(pred_arr, level, ctx, rng):
    out = np.empty_like(np.asarray(pred_arr, dtype=np.float64))
    n_seq, t_len, _ = out.shape
    L = int(level)
    if L <= 0:
        return np.asarray(pred_arr, dtype=np.float64).copy()
    for s in range(n_seq):
        jitter = rng.integers(-L, L + 1, size=t_len)
        idx = np.clip(np.arange(t_len) + jitter, 0, t_len - 1)
        out[s] = pred_arr[s, idx, :]
    return out


def _ucr_latent_contamination_pca(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    n_seq, T, R = out.shape
    t = np.linspace(0.0, 2.0 * np.pi, T, endpoint=False)
    w = rng.normal(0.0, 1.0, size=(R,))
    w /= (np.linalg.norm(w) + UCR_EPS)
    for i in range(n_seq):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        u = np.sin(t + phase).reshape(T, 1)
        out[i] += float(level) * (u @ w.reshape(1, R))
    return out


def _ucr_latent_contamination_cca(pred_arr, level, ctx, rng):
    return _ucr_latent_contamination_pca(pred_arr, level, ctx, rng)


def _ucr_latent_contamination_amp(pred_arr, level, ctx, rng):
    out = np.asarray(pred_arr, dtype=np.float64).copy()
    n_seq, t_len, n_reg = out.shape
    t = np.linspace(0.0, 1.0, t_len, endpoint=True)
    w = rng.standard_normal(n_reg)
    w /= (np.linalg.norm(w) + UCR_EPS)
    w = w * ctx['region_iqr']
    for s in range(n_seq):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        freq = rng.uniform(1.0, 3.0)
        drift = 2.0 * (t - 0.5)
        u = np.sin(2.0 * np.pi * freq * t + phase) + 0.5 * drift
        u /= (np.std(u) + UCR_EPS)
        out[s] += float(level) * np.outer(u, w)
    return out


def _ucr_pair_valid_rows_mani(Xg, Xp):
    Xg = np.asarray(Xg, dtype=np.float64)
    Xp = np.asarray(Xp, dtype=np.float64)
    m = np.isfinite(Xg).all(axis=1) & np.isfinite(Xp).all(axis=1)
    return Xg[m], Xp[m]


def _ucr_pair_valid_rows_trj(Xg, Xp):
    Xg = np.asarray(Xg, dtype=np.float64)
    Xp = np.asarray(Xp, dtype=np.float64)
    m = np.isfinite(Xg).all(axis=1) & np.isfinite(Xp).all(axis=1)
    return Xg[m], Xp[m], m


# ---------------------------
# Corruption specification
# ---------------------------
family_specs = [
    # Mean/bias
    dict(name='mean_shift', label='mean_shift', group='mean_bias', levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=_ucr_mean_shift_region_iqr),
    dict(name='global_mean_shift', label='global_mean_shift', group='mean_bias', levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=_ucr_global_mean_shift),
    dict(name='region_mean_shift', label='region_mean_shift', group='mean_bias', levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=_ucr_region_mean_shift),
    dict(name='sequence_mean_shift', label='sequence_mean_shift', group='mean_bias', levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=_ucr_sequence_mean_shift),
    dict(name='slow_drift', label='slow_drift', group='mean_bias', levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=_ucr_slow_drift),

    # Scale/tails
    dict(name='variance_scale_kl', label='variance_scale (KL)', group='scale_tail', levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=_ucr_variance_scale_kl),
    dict(name='variance_scale_qnt', label='variance_scale (QNT)', group='scale_tail', levels=[1.10, 1.25, 1.50, 1.75, 2.00], apply=_ucr_variance_scale_qnt),
    dict(name='tail_spikes_kl', label='tail_spikes (KL)', group='scale_tail', levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=_ucr_tail_spikes_kl),
    dict(name='tail_spikes_qnt', label='tail_spikes (QNT)', group='scale_tail', levels=[1, 2, 3, 4, 5], apply=_ucr_tail_spikes_qnt),
    dict(name='tail_spikes_mom', label='tail_spikes (MOM)', group='scale_tail', levels=[1, 2, 3, 4, 5], apply=_ucr_tail_spikes_mom),
    dict(name='one_sided_spikes', label='one_sided_spikes', group='scale_tail', levels=[1, 2, 3, 4, 5], apply=_ucr_one_sided_spikes),

    # Noise
    dict(name='additive_noise', label='additive_noise', group='noise', levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=_ucr_additive_noise_iqr),
    dict(name='independent_region_noise', label='independent_region_noise', group='noise', levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=_ucr_additive_noise_iqr),
    dict(name='region_noise', label='region_noise', group='noise', levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=_ucr_region_noise),
    dict(name='additive_noise_iqr', label='additive_noise_iqr', group='noise', levels=[0.25, 0.75, 1.00], apply=_ucr_additive_noise_iqr),

    # Temporal alignment
    dict(name='temporal_shuffle', label='temporal_shuffle', group='temporal', levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=_ucr_temporal_shuffle),
    dict(name='temporal_shuffle_per_region', label='temporal_shuffle_per_region', group='temporal', levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=_ucr_temporal_shuffle_per_region),
    dict(name='temporal_circular_lag', label='temporal_circular_lag', group='temporal', levels=[1, 4, 16], apply=_ucr_temporal_circular_lag),
    dict(name='temporal_nanpad_lag', label='temporal_nanpad_lag', group='temporal', levels=[1, 4, 16], apply=_ucr_temporal_nanpad_lag),
    dict(name='temporal_block_shuffle', label='temporal_block_shuffle', group='temporal', levels=[5, 20, 40], apply=_ucr_temporal_block_shuffle),
    dict(name='local_time_jitter', label='local_time_jitter', group='temporal', levels=[1, 4, 16], apply=_ucr_local_time_jitter),
    dict(name='region_specific_lags', label='region_specific_lags', group='temporal', levels=[1, 2, 4, 8, 12], apply=_ucr_region_specific_lags),
    dict(name='fixed_lag_jitter_desync', label='fixed_lag_jitter_desync', group='temporal', levels=[1, 2, 4, 8, 12], apply=_ucr_fixed_lag_jitter_desync),

    # Region mapping/mixing
    dict(name='region_permutation', label='region_permutation', group='region_mapping', levels=[0.20, 0.40, 0.60, 0.80, 1.00], apply=_ucr_region_permutation),
    dict(name='region_permute_frac', label='region_permute_frac', group='region_mapping', levels=[0.25, 0.50, 1.00], apply=_ucr_region_permute_frac),
    dict(name='region_mixing', label='region_mixing', group='region_mapping', levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=_ucr_region_mixing),
    dict(name='region_mix_lambda', label='region_mix_lambda', group='region_mapping', levels=[0.25, 0.50, 1.00], apply=_ucr_region_mix_lambda),
    dict(name='anti_sync_mixing', label='anti_sync_mixing', group='region_mapping', levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=_ucr_anti_sync_mixing),

    # Spectral/phase/latent
    dict(name='phase_scramble', label='phase_scramble', group='spectral_latent', levels=[0.20, 0.40, 0.60, 0.80, 1.00], apply=_ucr_phase_scramble),
    dict(name='latent_contamination_pca', label='latent_contamination (PCA)', group='spectral_latent', levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=_ucr_latent_contamination_pca),
    dict(name='latent_contamination_cca', label='latent_contamination (CCA)', group='spectral_latent', levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=_ucr_latent_contamination_cca),
    dict(name='latent_contamination_amp', label='latent_contamination_amp', group='spectral_latent', levels=[0.25, 0.75, 1.50], apply=_ucr_latent_contamination_amp),

    # Smoothing
    dict(name='ar_oversmoothing', label='ar_oversmoothing', group='smoothing', levels=[0.10, 0.25, 0.50, 0.75, 0.90], apply=_ucr_ar_oversmoothing),
    dict(name='oversmoothing_lambda', label='oversmoothing_lambda', group='smoothing', levels=[0.25, 0.75, 0.90], apply=_ucr_oversmoothing_lambda),
]

logic_group_order = ['mean_bias', 'scale_tail', 'noise', 'temporal', 'region_mapping', 'spectral_latent', 'smoothing']
logic_group_label = {
    'mean_bias': 'Mean/Bias',
    'scale_tail': 'Scale/Tails',
    'noise': 'Noise',
    'temporal': 'Temporal',
    'region_mapping': 'Region Mapping',
    'spectral_latent': 'Spectral/Latent',
    'smoothing': 'Smoothing',
}

family_specs_by_name = OrderedDict((spec['name'], spec) for spec in family_specs)
family_order = [spec['name'] for spec in family_specs]

# colors for many families
cmaps = [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c]
color_list = []
for cmap in cmaps:
    color_list.extend(cmap(np.linspace(0, 1, 20)))
family_colors = {name: color_list[i % len(color_list)] for i, name in enumerate(family_order)}


if 'gt_array_denorm' not in globals() or 'data_predicted' not in globals():
    raise RuntimeError('Expected gt_array_denorm and data_predicted in notebook globals.')

gt_base, pred_base = _ucr_align_gt_pred(gt_array_denorm, data_predicted)
gt_base = np.asarray(gt_base, dtype=np.float64)
pred_base = np.asarray(pred_base, dtype=np.float64)
region_iqr = _ucr_region_iqr(gt_base)
region_names = globals().get('pred_region_names', globals().get('gt_regions', None))
if region_names is None or len(region_names) != gt_base.shape[2]:
    region_names = [f'region_{r}' for r in range(gt_base.shape[2])]

ctx = {
    'region_iqr': region_iqr,
    'global_iqr': _ucr_global_iqr(gt_base),
    'region_names': region_names,
}

trj_params = {
    'min_samples': 80,
    'var_target': 0.90,
    'k_min': 3,
    'k_max': 10,
    'w_pos': 0.20,
    'w_vel': 0.40,
    'w_speed': 0.40,
    'rng_seed': 0,
    'max_global_fit_rows': 200000,
}
if isinstance(globals().get('trajectory_dist_simple'), dict) and isinstance(globals()['trajectory_dist_simple'].get('params'), dict):
    trj_params.update(globals()['trajectory_dist_simple']['params'])

score_registry = OrderedDict()
skipped_metrics = []


def _register_score(score_name, func_name, scorer):
    fn = globals().get(func_name)
    if not callable(fn):
        fn = _ucr_bootstrap_function(func_name)
        if callable(fn):
            globals()[func_name] = fn
    if callable(fn):
        score_registry[score_name] = scorer
    else:
        skipped_metrics.append({'score': score_name, 'reason': f'missing function: {func_name}'})


_register_score(
    'KL_score01',
    '_compute_kl_metrics',
    lambda gt_arr, pred_arr: _ucr_extract_score(
        _ucr_quiet_call(_compute_kl_metrics, gt_arr, pred_arr, bins=60, support_q=(0.001, 0.999)),
        'KL_score01_avg', 'KL_score01'
    ),
)
_register_score(
    'Mean_score01',
    'compute_mean_score01_top10',
    lambda gt_arr, pred_arr: _ucr_extract_score(
        _ucr_quiet_call(compute_mean_score01_top10, gt_arr, pred_arr, eps=1e-12),
        'Mean_score01'
    ),
)
_register_score(
    'MI_score01',
    'compute_mi_strict',
    lambda gt_arr, pred_arr: _ucr_extract_score(
        _ucr_quiet_call(
            compute_mi_strict,
            gt_arr,
            pred_arr,
            region_names=region_names,
            n_neighbors=5,
            min_samples=80,
            max_time=1200,
            standardize=True,
            worst_q_regions=0.10,
            rng_seed=0,
        ),
        'MI_score01'
    ),
)
_register_score(
    'Error_score01',
    'compute_error_score01_simple',
    lambda gt_arr, pred_arr: _ucr_extract_score(
        _ucr_quiet_call(
            compute_error_score01_simple,
            gt_arr,
            pred_arr,
            region_names=region_names,
            min_samples=50,
            top_q_regions=0.25,
        ),
        'Error_score01'
    ),
)
_register_score(
    'QNT_score01',
    'compute_quantile_score01_simple',
    lambda gt_arr, pred_arr: _ucr_extract_score(
        _ucr_quiet_call(
            compute_quantile_score01_simple,
            gt_arr,
            pred_arr,
            region_names=region_names,
            q_lo=0.01,
            q_hi=0.99,
            n_q=99,
            tail_lo=0.10,
            tail_hi=0.90,
            min_samples=80,
            max_time=1200,
            top_q_regions=0.25,
            rng_seed=0,
        ),
        'QNT_score01'
    ),
)
_register_score(
    'FC_score01',
    'compute_fc_score01_simple',
    lambda gt_arr, pred_arr: _ucr_extract_score(
        _ucr_quiet_call(compute_fc_score01_simple, gt_arr, pred_arr, region_names=region_names, min_samples=20),
        'FC_score01'
    ),
)
_register_score(
    'PCA_score01',
    'compute_pca_score01_simple',
    lambda gt_arr, pred_arr: _ucr_extract_score(
        _ucr_quiet_call(
            compute_pca_score01_simple,
            gt_arr,
            pred_arr,
            region_names=region_names,
            min_samples=80,
            max_time=1200,
            var_target=0.90,
            k_min=5,
            k_max=20,
            rng_seed=0,
        ),
        'PCA_score01'
    ),
)
_register_score(
    'AUTO_score01',
    'compute_autocorr_score01_sensitive',
    lambda gt_arr, pred_arr: _ucr_extract_score(
        _ucr_quiet_call(
            compute_autocorr_score01_sensitive,
            gt_arr,
            pred_arr,
            region_names=region_names,
            min_seg_len=80,
            max_time=601,
            max_lag=120,
            short_lag_end=20,
            tau_lag_end=60,
            top_q_regions=0.25,
            w_curve=0.35,
            w_lag1=0.30,
            w_tau=0.20,
            w_auc=0.15,
            floor_curve=0.05,
            floor_feat=0.05,
        ),
        'AUTO_score01'
    ),
)
_register_score(
    'CC_score01',
    'compute_crosscorr_score01_sensitive',
    lambda gt_arr, pred_arr: _ucr_extract_score(
        _ucr_quiet_call(
            compute_crosscorr_score01_sensitive,
            gt_arr,
            pred_arr,
            region_names=region_names,
            max_lag=24,
            min_seg_len=80,
            min_abs_lag=1,
            max_time=601,
            top_q_pairs=0.10,
            rms_floor=0.03,
            peak_floor=0.05,
            w_curve=0.35,
            w_lag=0.30,
            w_peak=0.25,
            w_sign=0.10,
        ),
        'CC_score01'
    ),
)
_register_score(
    'MOM_score01',
    'compute_moments_score01_simple',
    lambda gt_arr, pred_arr: _ucr_extract_score(
        _ucr_quiet_call(
            compute_moments_score01_simple,
            gt_arr,
            pred_arr,
            region_names=region_names,
            fisher=True,
            min_seg_len=80,
            alpha_moment=0.50,
            rms_floor_skew=0.05,
            rms_floor_kurt=0.10,
        ),
        'MOM_score01'
    ),
)
_register_score(
    'GRAPH_score01',
    'compute_graph_score01_simple',
    lambda gt_arr, pred_arr: _ucr_extract_score(
        _ucr_quiet_call(
            compute_graph_score01_simple,
            gt_arr,
            pred_arr,
            region_names=region_names,
            min_seg_len=80,
            max_time=1200,
            topk_mode='k',
            topk_k=30,
            topk_frac=0.25,
            min_abs_corr=0.0,
            w_jacc=0.40,
            w_wgt=0.35,
            w_deg=0.25,
        ),
        'GRAPH_score01'
    ),
)
_register_score(
    'CCA_score01',
    'compute_cca_score01_strict',
    lambda gt_arr, pred_arr: _ucr_extract_score(
        _ucr_quiet_call(
            compute_cca_score01_strict,
            gt_arr,
            pred_arr,
            region_names=region_names,
            min_seg_len=80,
            max_time=601,
            n_comp=5,
            folds=5,
            max_iter=2000,
            rng_seed=0,
        ),
        'CCA_score01'
    ),
)
_register_score(
    'MANI_score01',
    'compute_manifold_score01_geometry',
    lambda gt_arr, pred_arr: _ucr_extract_score(
        _ucr_with_patched_globals(
            {'_pair_valid_rows': _ucr_pair_valid_rows_mani},
            compute_manifold_score01_geometry,
            gt_arr,
            pred_arr,
            region_names=region_names,
            min_samples=80,
            max_time=1200,
            var_target=0.90,
            k_min=3,
            k_max=10,
            w_proc=0.45,
            w_rad=0.25,
            w_spec=0.30,
            rng_seed=0,
        ),
        'MANI_score01'
    ),
)
_register_score(
    'BP_score01',
    'compute_bandpower_score01_strict',
    lambda gt_arr, pred_arr: _ucr_extract_score(
        _ucr_quiet_call(
            compute_bandpower_score01_strict,
            gt_arr,
            pred_arr,
            region_names=region_names,
            fs=30.0,
            nperseg=256,
            noverlap=192,
            fine_bins=None,
            min_t_needed=None,
            w_shape=0.55,
            w_mag=0.45,
            shape_gamma=2.0,
            mag_gamma=1.5,
        ),
        'BP_score01'
    ),
)
_register_score(
    'TRJDIST_score01',
    'compute_trajectory_distribution_score01_fixed',
    lambda gt_arr, pred_arr: _ucr_extract_score(
        _ucr_with_patched_globals(
            {
                '_align_gt_pred': _ucr_align_gt_pred,
                '_pair_valid_rows': _ucr_pair_valid_rows_trj,
            },
            compute_trajectory_distribution_score01_fixed,
            gt_arr,
            pred_arr,
            params=trj_params,
            return_embeddings=False,
        ),
        'TRJDIST_score01'
    ),
)

baseline_scores = OrderedDict()
skipped_runtime = []
for score_name, scorer in score_registry.items():
    try:
        score_val = scorer(gt_base, pred_base)
    except Exception as exc:
        score_val = np.nan
        skipped_runtime.append({'score': score_name, 'reason': f'baseline error: {exc}'})
    if np.isfinite(score_val):
        baseline_scores[score_name] = float(np.clip(score_val, 0.0, 1.0))
    elif score_name not in {item['score'] for item in skipped_runtime}:
        skipped_runtime.append({'score': score_name, 'reason': 'baseline returned NaN'})

active_score_registry = OrderedDict((k, score_registry[k]) for k in baseline_scores)
if not active_score_registry:
    raise RuntimeError('No score functions were available for the unified corruption dashboard.')

n_conditions = sum(len(spec['levels']) for spec in family_specs)
print(
    f'Unified corruption sweep: {len(active_score_registry)} scores x {n_conditions} corrupted settings '
    f'using aligned GT={gt_base.shape} and Pred={pred_base.shape}.'
)

master_rows = []
evaluation_failures = []

# baseline row per score per family for consistent plotting
for spec in family_specs:
    family_name = spec['name']
    for score_name, baseline in baseline_scores.items():
        master_rows.append({
            'score': score_name,
            'family': family_name,
            'logic_group': spec['group'],
            'level': 0.0,
            'relative_magnitude': 0.0,
            'score_value': baseline,
            'baseline': baseline,
            'score_drop_abs': 0.0,
            'score_drop_rel': 0.0,
        })

for fam_idx, spec in enumerate(family_specs):
    family_name = spec['name']
    levels = list(spec['levels'])
    n_levels = len(levels)
    for level_idx, level in enumerate(levels, start=1):
        rel_mag = level_idx / float(n_levels)
        rng = np.random.default_rng(UCR_RNG_BASE_SEED + 1000 * fam_idx + level_idx)
        pred_corrupted = spec['apply'](pred_base, level, ctx, rng)
        pred_corrupted = np.asarray(pred_corrupted, dtype=np.float64)

        for score_name, scorer in active_score_registry.items():
            try:
                score_val = scorer(gt_base, pred_corrupted)
            except Exception as exc:
                score_val = np.nan
                evaluation_failures.append({
                    'score': score_name,
                    'family': family_name,
                    'level': level,
                    'reason': str(exc),
                })

            score_val = _ucr_float_or_nan(score_val)
            if np.isfinite(score_val):
                score_val = float(np.clip(score_val, 0.0, 1.0))

            baseline = baseline_scores[score_name]
            if np.isfinite(score_val):
                score_drop_abs = float(np.clip(baseline - score_val, 0.0, 1.0))
                score_drop_rel = float(np.clip(score_drop_abs / (baseline + UCR_EPS), 0.0, 1.0))
            else:
                score_drop_abs = np.nan
                score_drop_rel = np.nan

            master_rows.append({
                'score': score_name,
                'family': family_name,
                'logic_group': spec['group'],
                'level': float(level),
                'relative_magnitude': float(rel_mag),
                'score_value': score_val,
                'baseline': float(baseline),
                'score_drop_abs': score_drop_abs,
                'score_drop_rel': score_drop_rel,
            })

unified_corruption_master_df = pd.DataFrame(master_rows)

score_order = [score for score in [
    'KL_score01', 'Mean_score01', 'MI_score01', 'Error_score01', 'QNT_score01',
    'FC_score01', 'PCA_score01', 'AUTO_score01', 'CC_score01', 'MOM_score01',
    'GRAPH_score01', 'CCA_score01', 'MANI_score01', 'BP_score01', 'TRJDIST_score01',
] if score in baseline_scores]

# line dashboard
line_legend_handles = []
for family in family_order:
    line_legend_handles.append(
        Line2D([0], [0], color=family_colors[family], linewidth=2.0, marker='o', markersize=3.5, label=family_specs_by_name[family]['label'])
    )
line_legend_handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=1.2, label='baseline'))

n_scores = len(score_order)
line_ncols = 3
line_nrows = int(math.ceil(n_scores / line_ncols))
fig, axes = plt.subplots(line_nrows, line_ncols, figsize=(6.4 * line_ncols, 4.0 * line_nrows), sharex=False, sharey=False)
axes = np.atleast_1d(axes).ravel()

for ax, score_name in zip(axes, score_order):
    sub = unified_corruption_master_df[unified_corruption_master_df['score'] == score_name].copy()
    baseline = baseline_scores[score_name]

    for family in family_order:
        fam = sub[(sub['family'] == family) & np.isfinite(sub['score_value'])].sort_values('relative_magnitude')
        if fam.empty:
            continue
        ax.plot(
            fam['relative_magnitude'].to_numpy(dtype=float),
            fam['score_value'].to_numpy(dtype=float),
            color=family_colors[family],
            marker='o',
            linewidth=1.25,
            markersize=2.8,
            alpha=0.92,
        )

    ax.axhline(baseline, color='black', linestyle='--', linewidth=1.0, alpha=0.75)
    ax.set_title(score_name)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel('relative corruption magnitude')
    ax.set_ylabel('score01')
    ax.grid(alpha=0.20)

for ax in axes[n_scores:]:
    ax.axis('off')

fig.suptitle('Unified corruption sensitivity dashboard (all notebook corruption families)', fontsize=15)
fig.legend(handles=line_legend_handles, loc='upper center', ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=8)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

# strongest family-wise drops for polar and summary
strongest_rows = []
valid_rows = unified_corruption_master_df[
    (unified_corruption_master_df['relative_magnitude'] > 0.0) & np.isfinite(unified_corruption_master_df['score_drop_rel'])
]
for (score_name, family_name), sub in valid_rows.groupby(['score', 'family']):
    strongest = sub.sort_values(['score_drop_rel', 'score_drop_abs', 'relative_magnitude'], ascending=[False, False, False]).iloc[0]
    strongest_rows.append({
        'score': score_name,
        'family': family_name,
        'logic_group': strongest['logic_group'],
        'level': float(strongest['level']),
        'relative_magnitude': float(strongest['relative_magnitude']),
        'score_value': float(strongest['score_value']),
        'baseline': float(strongest['baseline']),
        'score_drop_abs': float(strongest['score_drop_abs']),
        'score_drop_rel': float(strongest['score_drop_rel']),
    })

unified_corruption_family_worst_df = pd.DataFrame(strongest_rows)

# polar dashboard (bigger) with logic-group ordering
polar_ncols = 2
polar_nrows = int(math.ceil(n_scores / polar_ncols))
fig, axes = plt.subplots(
    polar_nrows,
    polar_ncols,
    figsize=(8.8 * polar_ncols, 6.8 * polar_nrows),
    subplot_kw={'projection': 'polar'},
)
axes = np.atleast_1d(axes).ravel()

theta = np.linspace(0.0, 2.0 * np.pi, len(family_order), endpoint=False)
theta_closed = np.concatenate([theta, theta[:1]])

for ax, score_name in zip(axes, score_order):
    sub = unified_corruption_family_worst_df[unified_corruption_family_worst_df['score'] == score_name]
    radii = []
    for family_name in family_order:
        fam = sub[sub['family'] == family_name]
        radii.append(float(fam.iloc[0]['score_drop_rel']) if not fam.empty else 0.0)
    radii = np.asarray(radii, dtype=float)
    radii_closed = np.concatenate([radii, radii[:1]])

    ax.plot(theta_closed, radii_closed, color='#2f2f2f', linewidth=1.0)
    ax.fill(theta_closed, radii_closed, color='#9e9e9e', alpha=0.08)

    for ang, radius, family_name in zip(theta, radii, family_order):
        ax.plot([ang, ang], [0.0, radius], color=family_colors[family_name], linewidth=1.8, alpha=0.85)
        ax.scatter([ang], [radius], color=family_colors[family_name], s=36, zorder=3)

    ax.set_xticks(theta)
    ax.set_xticklabels([family_specs_by_name[name]['label'] for name in family_order], fontsize=6)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=7)
    ax.set_title(score_name, va='bottom')
    ax.grid(alpha=0.25)

for ax in axes[n_scores:]:
    ax.axis('off')

fig.suptitle('Polar sensitivity map (ordered by corruption logic family)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# top-5 table per score
top_rows = []
if not unified_corruption_family_worst_df.empty:
    for score_name, sub in unified_corruption_family_worst_df.groupby('score'):
        top_rows.append(sub.sort_values(['score_drop_rel', 'score_drop_abs'], ascending=False).head(5))

unified_corruption_top5_df = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()

print('=== Unified corruption sensitivity: strongest families per score ===')
if not unified_corruption_top5_df.empty:
    print(unified_corruption_top5_df.round(4).to_string(index=False))
else:
    print('No valid corruption sensitivity rows were produced.')

if skipped_metrics or skipped_runtime:
    skip_df = pd.DataFrame(skipped_metrics + skipped_runtime)
    print('\n=== Skipped metrics ===')
    print(skip_df.to_string(index=False))

if evaluation_failures:
    fail_df = pd.DataFrame(evaluation_failures)
    fail_summary = fail_df.groupby(['score', 'family']).size().reset_index(name='n_fail')
    fail_summary = fail_summary.sort_values(['score', 'n_fail', 'family'], ascending=[True, False, True])
    print('\n=== Metric/corruption evaluation failures (summary) ===')
    print(fail_summary.to_string(index=False))

# show corruption logic ordering used in polar axes
logic_order_table = []
for fam in family_order:
    logic_order_table.append({
        'family': fam,
        'label': family_specs_by_name[fam]['label'],
        'logic_group': logic_group_label[family_specs_by_name[fam]['group']],
    })

unified_corruption_family_order_df = pd.DataFrame(logic_order_table)
print('\n=== Polar family order (logic grouped) ===')
print(unified_corruption_family_order_df.to_string(index=False))


# ## Overlay Polar Summary
# This extra cell overlays the per-metric polar sensitivity profiles on a single polar axis. It uses the same strongest-family relative degradation values computed in the unified dashboard, but places every score on one chart to make cross-metric sensitivity patterns easier to compare.

# In[38]:


# === OVERLAID POLAR SUMMARY (all scores on one axis) ===
import numpy as np
import matplotlib.pyplot as plt

if 'unified_corruption_family_worst_df' not in globals() or unified_corruption_family_worst_df.empty:
    raise RuntimeError('Run the unified corruption sensitivity dashboard cell first.')

overlay_scores = score_order if 'score_order' in globals() else sorted(unified_corruption_family_worst_df['score'].unique())
theta = np.linspace(0.0, 2.0 * np.pi, len(family_order), endpoint=False)
theta_closed = np.concatenate([theta, theta[:1]])

cmap = plt.cm.get_cmap('tab20', max(len(overlay_scores), 1))
fig, ax = plt.subplots(figsize=(9.5, 9.5), subplot_kw={'projection': 'polar'})

for idx, score_name in enumerate(overlay_scores):
    sub = unified_corruption_family_worst_df[unified_corruption_family_worst_df['score'] == score_name]
    radii = []
    for family_name in family_order:
        fam = sub[sub['family'] == family_name]
        radii.append(float(fam.iloc[0]['score_drop_rel']) if not fam.empty else 0.0)
    radii = np.asarray(radii, dtype=float)
    radii_closed = np.concatenate([radii, radii[:1]])
    color = cmap(idx)
    ax.plot(theta_closed, radii_closed, linewidth=2.0, alpha=0.78, color=color, label=score_name)

ax.set_xticks(theta)
ax.set_xticklabels([family_specs_by_name[name]['label'] for name in family_order], fontsize=8)
ax.set_ylim(0.0, 1.0)
ax.set_yticks([0.25, 0.50, 0.75, 1.00])
ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8)
ax.set_title('Overlaid polar sensitivity map across all scores', va='bottom')
ax.grid(alpha=0.25)
ax.legend(loc='upper left', bbox_to_anchor=(1.12, 1.10), frameon=False, fontsize=8)
plt.tight_layout()
plt.show()
