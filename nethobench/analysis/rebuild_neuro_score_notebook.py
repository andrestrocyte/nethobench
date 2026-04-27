from __future__ import annotations

import copy
from pathlib import Path

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_NOTEBOOK = REPO_ROOT / "nethobench" / "notebooks" / "neuro_metrics.ipynb"


TITLE_MD = """# Neuro Score

This notebook defines the active **neural realism score** used by NethoBench. The score is intentionally focused on the metric families that remained most discriminative and scientifically robust across the project:

- **distribution**
- **temporal**
- **relational**
- **geometry**
- **state dynamics**

The direct pointwise fidelity family is no longer folded into the default neural benchmark. It is now exposed separately through the `fidelity-scores` command.

## Composite definition

The active neuro score is

\\[
S_{neuro}
= 0.22 D
+ 0.18 T
+ 0.24 R
+ 0.18 G
+ 0.18 S_d,
\\]

where

\\[
D = \\frac{1}{4}\\left(
s_{KL/JSD}
+ s_{QNT}
+ s_{MOM}
+ s_{Mean}
\\right),
\\]

\\[
T = s_{TRJDIST},
\\]

\\[
R = \\frac{1}{4}\\left(
s_{GRAPH}
+ s_{CrossRegionMI}
+ s_{LaggedCovariance}
+ s_{ImpulseResponse}
\\right),
\\]

\\[
G = \\frac{1}{2}\\left(
s_{MANI}
+ s_{SubspaceAngle}
\\right),
\\]

and

\\[
S_d = \\frac{1}{5}\\left(
s_{Occ11}
+ s_{Occ12}
+ s_{Trans1}
+ s_{Trans2}
+ s_{Trans3}
\\right).
\\]

Every scalar metric is bounded in \\([0,1]\\), with larger values meaning better agreement between ground truth and prediction.
"""


SETUP_MD = """## Setup

This notebook is runnable both interactively and through the package CLI. The setup cell imports the scientific stack, inserts the repository root into `sys.path` when needed, and defines the default file-path variables patched by the CLI runner.
"""


DATA_MD = """## Data Loading and Alignment

This cell loads the prediction and ground-truth CSVs, aligns them by shared neural regions and common horizon, and creates the aligned tensors

\\[
X^{gt}, X^{pred} \\in \\mathbb{R}^{N_{seq} \\times T \\times R}.
\\]

All downstream metrics operate on these same aligned arrays.
"""


DATA_CODE = """%matplotlib inline

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
"""


STRUCTURAL_MD = """## Relational, Geometry, and State-Dynamics Terms

This notebook computes four additional structural terms directly from aligned tensors:

### Cross-region mutual-information structure

For each dataset we estimate a region-by-region mutual-information matrix

\\[
M_{ij} = \\operatorname{MI}(x_i, x_j),
\\]

then compare the upper-triangular entries of \\(M^{gt}\\) and \\(M^{pred}\\) with a correlation-plus-RMSE similarity.

### Lagged covariance

For lags \\(\\ell \\in \\{1,2,4\\}\\), the notebook forms

\\[
C_{\\ell} = \\frac{1}{T-\\ell-1}(X_{1:T-\\ell} - \\bar X)^\\top (X_{1+\\ell:T} - \\bar X),
\\]

and scores agreement between the GT and prediction lagged covariance matrices.

### Impulse response

Each sequence stack is summarized by a ridge-regularized VAR(1) operator

\\[
x_{t+1} \\approx A x_t,
\\]

and the resulting coefficient matrices are compared as a simple low-order impulse-response proxy.

### Subspace angle

Let \\(U\\) and \\(V\\) be GT and prediction principal subspaces. The score is the mean squared cosine of the principal angles:

\\[
s_{subspace} = \\frac{1}{k}\\sum_{i=1}^{k} \\cos^2 \\theta_i.
\\]

### State dynamics

State dynamics are defined in a **GT-derived latent space**. Ground truth is standardized, projected into a GT PCA basis, and clustered into discrete latent states. The notebook then compares:

- state occupancies for `k=11` and `k=12`
- 1-step, 2-step, and 3-step transition structure for the `k=11` partition

This family tests whether the model visits the right metastable states and moves between them with the right short-horizon transition logic.
"""


FINAL_MD = """## Final Neuro Score

This cell assembles the active neuro score from the metric cells above. Only the retained metric families contribute to the package-level neuro score:

- distribution
- temporal
- relational
- geometry
- state dynamics

The fidelity family is intentionally excluded here and lives behind the separate `fidelity-scores` command.
"""


FIDELITY_NOTE_MD = """## Separate Fidelity Score

The old direct-fidelity family is still available, but it is no longer part of the default neuro score.

Use:

```bash
nethobench fidelity-scores --gt gt_neural.csv --preds pred_neural.csv
```

That command reports:

- `Error_score01`
- `MI_score01`
- `family_fidelity`

with the same fidelity weighting as the legacy benchmark:

\\[
S_{fidelity} = 0.65 \\, s_{Error} + 0.35 \\, s_{MI}.
\\]
"""


MOMENT_CODE = """# === Higher-order moments realism (simple, strict, benchmark-friendly) ===
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
"""


GRAPH_CODE = """# === Corr-connectivity GRAPH realism (simple, strict, benchmark-friendly) ===
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
"""


MANI_CODE = """# === MANIFOLD realism (simple geometry, strict, benchmark-friendly) ===
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
"""


TRJ_CODE = """# === TRAJECTORY DISTRIBUTION realism (FIXED: global GT-PCA basis, pooled across sequences) ===
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
"""


ADDITIONAL_CODE = """# === ADDITIONAL STRUCTURAL METRICS (cross-region information, lagged dynamics, subspaces, states) ===
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
"""


FINAL_CODE = """# === FINAL NEURO SCORE COMPOSITE ===
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
print('\\n=== FAMILY SCORES ===')
print(families_df.to_string(index=False))
print(f'\\nFINAL_COMPOSITE_SCORE: {float(FINAL_COMPOSITE_SCORE):.6f}')
"""


DASHBOARD_MD = """## Corruption Sensitivity Dashboard

This section stress-tests the retained metrics under structured corruptions. The purpose is not to define the score itself, but to verify that the active neuro score families degrade in interpretable directions under:

- mean and scale distortions
- additive noise
- temporal shuffling and oversmoothing
- region remapping and mixing
- latent-space rotations

The line plots show score trajectories across corruption magnitude. The polar views summarize the largest relative drop per corruption family.
"""


DASHBOARD_CODE = """# === UNIFIED CORRUPTION SENSITIVITY DASHBOARD (active neuro score metrics) ===
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
"""


OVERLAY_CODE = """# === OVERLAID POLAR SUMMARY (active neuro score metrics) ===
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
"""



def _find_cell_index(cells, marker: str) -> int:
    for idx, cell in enumerate(cells):
        source = "".join(cell.get("source", []))
        if source.lstrip().startswith(marker):
            return idx
    raise ValueError(f"Could not find cell starting with marker: {marker}")


def _copy_cell(cell):
    new_cell = copy.deepcopy(cell)
    if new_cell.get("cell_type") == "code":
        new_cell["outputs"] = []
        new_cell["execution_count"] = None
    return new_cell


def _copy_markdown_plus_code(cells, marker: str):
    idx = _find_cell_index(cells, marker)
    out = []
    if idx > 0 and cells[idx - 1].get("cell_type") == "markdown":
        out.append(_copy_cell(cells[idx - 1]))
    out.append(_copy_cell(cells[idx]))
    return out


def _new_markdown(source: str):
    return nbformat.v4.new_markdown_cell(source)


def _new_code(source: str):
    cell = nbformat.v4.new_code_cell(source)
    cell["execution_count"] = None
    cell["outputs"] = []
    return cell


def build_notebook() -> nbformat.NotebookNode:
    legacy_nb = _load_legacy_notebook()
    cells = legacy_nb.cells

    new_cells = [
        _new_markdown(TITLE_MD),
        _new_markdown(SETUP_MD),
        _new_code(DATA_CODE),
        _new_markdown(DATA_MD),
    ]

    new_cells.extend(_copy_markdown_plus_code(cells, "# === Distribution realism (KL only): baseline + corruption degradation ==="))
    new_cells.extend(_copy_markdown_plus_code(cells, "# === Mean difference metric (Top-10% regions) ==="))
    new_cells.extend(_copy_markdown_plus_code(cells, "# === Quantile / tail realism (simple, strict, benchmark-friendly) ==="))

    if (_find_cell_index(cells, "# === Higher-order moments realism (simple, strict, benchmark-friendly) ===") - 1) >= 0:
        idx = _find_cell_index(cells, "# === Higher-order moments realism (simple, strict, benchmark-friendly) ===")
        if cells[idx - 1].get("cell_type") == "markdown":
            new_cells.append(_copy_cell(cells[idx - 1]))
    new_cells.append(_new_code(MOMENT_CODE))

    if (_find_cell_index(cells, "# === Corr-connectivity GRAPH realism (simple, strict, benchmark-friendly) ===") - 1) >= 0:
        idx = _find_cell_index(cells, "# === Corr-connectivity GRAPH realism (simple, strict, benchmark-friendly) ===")
        if cells[idx - 1].get("cell_type") == "markdown":
            new_cells.append(_copy_cell(cells[idx - 1]))
    new_cells.append(_new_code(GRAPH_CODE))

    if (_find_cell_index(cells, "# === MANIFOLD realism (simple geometry, strict, benchmark-friendly) ===") - 1) >= 0:
        idx = _find_cell_index(cells, "# === MANIFOLD realism (simple geometry, strict, benchmark-friendly) ===")
        if cells[idx - 1].get("cell_type") == "markdown":
            new_cells.append(_copy_cell(cells[idx - 1]))
    new_cells.append(_new_code(MANI_CODE))

    if (_find_cell_index(cells, "# === TRAJECTORY DISTRIBUTION realism (FIXED: global GT-PCA basis, pooled across sequences) ===") - 1) >= 0:
        idx = _find_cell_index(cells, "# === TRAJECTORY DISTRIBUTION realism (FIXED: global GT-PCA basis, pooled across sequences) ===")
        if cells[idx - 1].get("cell_type") == "markdown":
            new_cells.append(_copy_cell(cells[idx - 1]))
    new_cells.append(_new_code(TRJ_CODE))

    new_cells.append(_new_markdown(STRUCTURAL_MD))
    new_cells.append(_new_code(ADDITIONAL_CODE))
    new_cells.append(_new_markdown(FINAL_MD))
    new_cells.append(_new_code(FINAL_CODE))
    new_cells.append(_new_markdown(DASHBOARD_MD))
    new_cells.append(_new_code(DASHBOARD_CODE))

    overlay_idx = _find_cell_index(cells, "# === OVERLAID POLAR SUMMARY (all scores on one axis) ===")
    if overlay_idx > 0 and cells[overlay_idx - 1].get("cell_type") == "markdown":
        new_cells.append(_copy_cell(cells[overlay_idx - 1]))
    new_cells.append(_new_code(OVERLAY_CODE))
    new_cells.append(_new_markdown(FIDELITY_NOTE_MD))

    nb = nbformat.v4.new_notebook(cells=new_cells, metadata=copy.deepcopy(legacy_nb.metadata))
    return nb


def main() -> None:
    nb = build_notebook()
    TARGET_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    with TARGET_NOTEBOOK.open("w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)
    print(f"Wrote {TARGET_NOTEBOOK}")


if __name__ == "__main__":
    main()
