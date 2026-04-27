#!/usr/bin/env python
# coding: utf-8

# ### What this does
# Loads ground-truth and inference parquet/CSV files, enforces matching headers, concatenates them, and previews shapes/headers.
# 

# In[15]:



from pathlib import Path
import re
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

# Set your ground-truth and inference directories here
# (can contain .parquet and/or .csv)
gt_dir = Path('/home/andres/b2data/test_gt')
inf_dir = Path('/home/andres/b2data/test_inference')


def collect_files(directory: Path):
    files = list(directory.glob('*.parquet')) + list(directory.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f'No parquet or csv files found in {directory}')

    def sort_key(path: Path):
        m = re.search(r'_([0-9]+)\.(parquet|csv)$', path.name)
        return int(m.group(1)) if m else path.name

    return sorted(files, key=sort_key)


def load_file(path: Path) -> pd.DataFrame:
    if path.suffix == '.csv':
        return pd.read_csv(path)
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        table = pq.read_table(path, use_pandas_metadata=False)
        return table.to_pandas()

# Load GT and INF sets
gt_files = collect_files(gt_dir)
inf_files = collect_files(inf_dir)
print(f'Found {len(gt_files)} GT files and {len(inf_files)} INF files.')

# Align headers based on the first GT file reference
gt_first = load_file(gt_files[0])
columns = list(gt_first.columns)

aligned_gt = []
aligned_inf = []

for path in gt_files:
    df = load_file(path)
    if set(df.columns) != set(columns):
        missing = set(columns) - set(df.columns)
        extra = set(df.columns) - set(columns)
        raise ValueError(f'GT header mismatch in {path}: missing {missing}, extra {extra}')
    aligned_gt.append(df[columns])

for path in inf_files:
    df = load_file(path)
    if set(df.columns) != set(columns):
        missing = set(columns) - set(df.columns)
        extra = set(df.columns) - set(columns)
        raise ValueError(f'INF header mismatch in {path}: missing {missing}, extra {extra}')
    aligned_inf.append(df[columns])

# Concatenate and convert to NumPy
gt_df_all = pd.concat(aligned_gt, ignore_index=True)
inf_df_all = pd.concat(aligned_inf, ignore_index=True)
headers = columns
gt_array = gt_df_all.to_numpy()
inf_array = inf_df_all.to_numpy()

print(f'GT array shape:  {gt_array.shape}')
print(f'INF array shape: {inf_array.shape}')
print('Headers:', headers)

# Preview first 20 rows for the first 10 files in each set
for label, file_list, dfs in [
    ('GT', gt_files, aligned_gt),
    ('INF', inf_files, aligned_inf),
]:
    print(f"==== {label} preview (up to first 10 files) ====")
    for idx, path in enumerate(file_list[:10]):
        print(f"{label} File {idx+1}: {path.name} (rows: {len(dfs[idx])})")
        display(dfs[idx].head(20))

# Small sanity preview of the arrays
print('GT array first 3 rows:')
print(gt_array[:3])
print('INF array first 3 rows:')
print(inf_array[:3])
# Similarity score for data readiness (1 means schemas aligned and loaded)
load_score = 1.0
print(f"Similarity score (data readiness): {load_score:.3f}")
# Plot: loaded rows and unique sequences
import matplotlib.pyplot as plt
plt.figure(figsize=(6,3))
plt.bar(['GT rows', 'INF rows'], [len(gt_df_all), len(inf_df_all)], color=['tab:blue', 'tab:orange'])
plt.title('Loaded rows per domain')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,3))
plt.bar(['GT seqs', 'INF seqs'], [gt_df_all.sequenceId.nunique(), inf_df_all.sequenceId.nunique()], color=['tab:blue', 'tab:orange'])
plt.title('Unique sequenceIds loaded')
plt.tight_layout()
plt.show()


# ### What this does
# Detects body-part coordinate pairs, merges GT and inference on sequenceId/itemPosition, and builds per-body-part arrays for later metrics.
# 

# In[16]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Ensure GT/INF data is loaded
if 'gt_df_all' not in globals() or 'inf_df_all' not in globals():
    raise RuntimeError('Run the GT/INF loader cell first to define gt_df_all and inf_df_all.')

# Identify body parts from headers ending with _X/_Y
coord_pairs = []
for col in headers:
    if col.endswith('_X'):
        base = col[:-2]
        y_col = f'{base}_Y'
        if y_col in headers:
            coord_pairs.append((base, col, y_col))
body_parts = [b for b, _, _ in coord_pairs]

print(f'Body parts detected: {body_parts}')

# Merge GT/INF on sequenceId + itemPosition (inner join to align time)
merge_keys = ['sequenceId', 'itemPosition']
value_headers = [h for h in headers if h not in merge_keys]
paired_df = pd.merge(
    gt_df_all[merge_keys + value_headers],
    inf_df_all[merge_keys + value_headers],
    on=merge_keys,
    suffixes=('_gt', '_inf'),
    how='inner'
)
print(f'Matched rows after merge: {len(paired_df)} (GT rows: {len(gt_df_all)}, INF rows: {len(inf_df_all)})')
print(f'Sequences present: {paired_df.sequenceId.nunique()}')

# Helper: per-sequence grouping sorted by itemPosition
paired_groups = paired_df.sort_values('itemPosition').groupby('sequenceId')

def get_part_arrays(df: pd.DataFrame, suffix: str):
    arrays = {}
    for base, x_col, y_col in coord_pairs:
        arrays[base] = df[[f'{x_col}{suffix}', f'{y_col}{suffix}']].to_numpy()
    return arrays

# Keep separate dictionaries of body-part arrays for GT and inference
# (makes it clearer which version is being inspected later)
gt_part_arrays = get_part_arrays(paired_df, '_gt')
inf_part_arrays = get_part_arrays(paired_df, '_inf')


# Similarity score for merge coverage (aligned rows vs available min rows)
merge_score = 0.0
if len(paired_df):
    denom = max(1, min(len(gt_df_all), len(inf_df_all)))
    merge_score = min(1.0, len(paired_df) / denom)
print(f"Similarity score (merge coverage): {merge_score:.3f}")
# Plot: matched rows per sequence (top 20)
import matplotlib.pyplot as plt
if not paired_df.empty:
    counts = paired_df.groupby('sequenceId').size().sort_values(ascending=False)
    plt.figure(figsize=(8,3))
    counts.head(20).plot(kind='bar', color='tab:blue')
    plt.ylabel('Matched rows')
    plt.title('Top sequences by matched frames')
    plt.tight_layout()
    plt.show()
else:
    print('No matched rows to plot.')


# ### What this does
# Computes per-body-part RMSE/MAE/correlations between GT and inference and summarizes per-sequence RMSE, with NaN-safe handling.
# 

# In[17]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

# Per-body-part error metrics (nan-safe)
rows = []
for base, x_col, y_col in coord_pairs:
    # Handle empty merges gracefully
    if paired_df.empty:
        rows.append({'body_part': base, 'rmse_pos': np.nan, 'mae_x': np.nan, 'mae_y': np.nan, 'corr_x': np.nan, 'corr_y': np.nan})
        continue

    gt_xy = paired_df[[f'{x_col}_gt', f'{y_col}_gt']].to_numpy()
    inf_xy = paired_df[[f'{x_col}_inf', f'{y_col}_inf']].to_numpy()

    # Drop rows with NaNs before metrics
    valid_mask = ~np.isnan(gt_xy).any(axis=1) & ~np.isnan(inf_xy).any(axis=1)
    gt_xy = gt_xy[valid_mask]
    inf_xy = inf_xy[valid_mask]

    if len(gt_xy) == 0:
        rows.append({'body_part': base, 'rmse_pos': np.nan, 'mae_x': np.nan, 'mae_y': np.nan, 'corr_x': np.nan, 'corr_y': np.nan})
        continue

    diff = gt_xy - inf_xy
    dist = np.linalg.norm(diff, axis=1)
    rmse = np.sqrt(np.nanmean(dist ** 2)) if dist.size else np.nan
    mae_vec = np.nanmean(np.abs(diff), axis=0) if diff.size else np.array([np.nan, np.nan])

    def safe_corr(a, b):
        # Need at least 2 non-constant points for Pearson
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return np.corrcoef(a, b)[0, 1]

    corr_x = safe_corr(gt_xy[:, 0], inf_xy[:, 0])
    corr_y = safe_corr(gt_xy[:, 1], inf_xy[:, 1])

    rows.append({
        'body_part': base,
        'rmse_pos': rmse,
        'mae_x': mae_vec[0],
        'mae_y': mae_vec[1],
        'corr_x': corr_x,
        'corr_y': corr_y,
    })

error_df = pd.DataFrame(rows)
if not error_df.empty:
    error_df = error_df.sort_values('rmse_pos')
print('Per-body-part error summary (sorted by RMSE):')
display(error_df)

# Sequence-aware RMSE (per sequence) and summary
seq_rows = []
if not paired_df.empty:
    for seq_id, seq_df in paired_groups:
        seq_rmse = {}
        for base, x_col, y_col in coord_pairs:
            gt_xy = seq_df[[f'{x_col}_gt', f'{y_col}_gt']].to_numpy()
            inf_xy = seq_df[[f'{x_col}_inf', f'{y_col}_inf']].to_numpy()
            valid_mask = ~np.isnan(gt_xy).any(axis=1) & ~np.isnan(inf_xy).any(axis=1)
            gt_xy = gt_xy[valid_mask]
            inf_xy = inf_xy[valid_mask]
            if len(gt_xy) == 0:
                continue
            dist = np.linalg.norm(gt_xy - inf_xy, axis=1)
            seq_rmse[base] = np.sqrt(np.nanmean(dist ** 2)) if dist.size else np.nan
        if seq_rmse:
            seq_rows.append({'sequenceId': seq_id, 'rmse_mean': np.nanmean(list(seq_rmse.values()))})

if seq_rows:
    seq_df = pd.DataFrame(seq_rows).sort_values('rmse_mean')
else:
    seq_df = pd.DataFrame(columns=['sequenceId', 'rmse_mean'])
print('Per-sequence mean RMSE:')
display(seq_df.head(20))
# Similarity score for per-part errors (lower RMSE => higher score)
if error_df.empty:
    metrics_score = np.nan
else:
    mean_rmse = error_df['rmse_pos'].mean()
    metrics_score = 1.0 / (1.0 + mean_rmse)
print(f"Similarity score (per-part errors): {metrics_score:.3f}")
# Plot: per-body-part RMSE and per-sequence RMSE distribution
import matplotlib.pyplot as plt
if not error_df.empty:
    plt.figure(figsize=(8,3))
    plt.bar(error_df['body_part'], error_df['rmse_pos'], color='tab:blue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('RMSE')
    plt.title('Per-body-part RMSE')
    plt.tight_layout()
    plt.show()
if 'seq_df' in locals() and not seq_df.empty:
    plt.figure(figsize=(6,3))
    plt.hist(seq_df['rmse_mean'], bins=20, color='tab:orange', alpha=0.8)
    plt.xlabel('Per-sequence mean RMSE')
    plt.ylabel('Count')
    plt.title('Distribution of sequence RMSE')
    plt.tight_layout()
    plt.show()


# ### What this does
# Uses CENTER tracks to derive speed/acceleration stats, stationary fractions, and quadrant occupancy for GT vs inference (robust to missing data).
# 

# In[18]:


# Velocity and acceleration using CENTER as proxy for body centroid
center_gt = paired_df[['CENTER_X_gt', 'CENTER_Y_gt', 'sequenceId', 'itemPosition']]
center_inf = paired_df[['CENTER_X_inf', 'CENTER_Y_inf', 'sequenceId', 'itemPosition']]

if paired_df.empty or center_gt.empty or center_inf.empty:
    print('No paired data available; velocity/acc/occupancy metrics set to NaN/empty.')
    vel_df = pd.DataFrame(columns=['sequenceId', 'speed', 'label'])
    acc_df = pd.DataFrame(columns=['sequenceId', 'acc', 'label'])
    speed_thresh = np.nan
    stationary = pd.Series({'gt': np.nan, 'inf': np.nan})
else:
    vel_records = []
    acc_records = []
    for label, df in [('gt', center_gt), ('inf', center_inf)]:
        for seq, seq_df in df.sort_values('itemPosition').groupby('sequenceId'):
            coords = seq_df[['CENTER_X_' + label, 'CENTER_Y_' + label]].to_numpy()
            if len(coords) < 2:
                continue
            vel = np.diff(coords, axis=0)
            speed = np.linalg.norm(vel, axis=1)
            acc = np.diff(vel, axis=0)
            acc_mag = np.linalg.norm(acc, axis=1)
            vel_records.append(pd.DataFrame({'sequenceId': seq, 'speed': speed, 'label': label}))
            acc_records.append(pd.DataFrame({'sequenceId': seq, 'acc': acc_mag, 'label': label}))

    vel_df = pd.concat(vel_records, ignore_index=True) if vel_records else pd.DataFrame(columns=['sequenceId', 'speed', 'label'])
    acc_df = pd.concat(acc_records, ignore_index=True) if acc_records else pd.DataFrame(columns=['sequenceId', 'acc', 'label'])

    if vel_df.empty or vel_df.loc[vel_df.label == 'gt', 'speed'].empty:
        speed_thresh = np.nan
        stationary = pd.Series({'gt': np.nan, 'inf': np.nan})
    else:
        speed_thresh = np.percentile(vel_df.loc[vel_df.label == 'gt', 'speed'], 10)
        stationary = vel_df.groupby('label').apply(lambda g: (g['speed'] <= speed_thresh).mean()).rename('stationary_frac')

print(f"Stationary threshold (10th percentile GT speed): {speed_thresh:.3f}")
print(f"Stationary fraction GT: {stationary.get('gt', np.nan):.3f} INF: {stationary.get('inf', np.nan):.3f}")

# Velocity/acceleration distribution summary
if vel_df.empty:
    print('Velocity distribution stats: empty')
    vel_stats = pd.DataFrame(columns=['mean', 'std', '50%', '90%'])
else:
    vel_stats = vel_df.groupby('label')['speed'].agg(
        mean='mean',
        std='std',
        **{'50%': lambda s: s.quantile(0.5), '90%': lambda s: s.quantile(0.9)}
    )
print('Velocity distribution stats (mean/std/median/p90):')
display(vel_stats)

if acc_df.empty:
    print('Acceleration distribution stats: empty')
    acc_stats = pd.DataFrame(columns=['mean', 'std', '50%', '90%'])
else:
    acc_stats = acc_df.groupby('label')['acc'].agg(
        mean='mean',
        std='std',
        **{'50%': lambda s: s.quantile(0.5), '90%': lambda s: s.quantile(0.9)}
    )
print('Acceleration distribution stats (mean/std/median/p90):')
display(acc_stats)

# Quadrant occupancy using CENTER positions
if paired_df.empty:
    print('Quadrant occupancy: no paired data')
    quad_df = pd.DataFrame(columns=['label', 'quadrant', 'fraction'])
else:
    all_center = pd.DataFrame({
        'x_gt': center_gt['CENTER_X_gt'],
        'y_gt': center_gt['CENTER_Y_gt'],
        'x_inf': center_inf['CENTER_X_inf'],
        'y_inf': center_inf['CENTER_Y_inf'],
    }).dropna()

    if all_center.empty:
        quad_df = pd.DataFrame(columns=['label', 'quadrant', 'fraction'])
    else:
        min_x = all_center[['x_gt', 'x_inf']].min().min()
        max_x = all_center[['x_gt', 'x_inf']].max().max()
        min_y = all_center[['y_gt', 'y_inf']].min().min()
        max_y = all_center[['y_gt', 'y_inf']].max().max()
        mid_x = 0.5 * (min_x + max_x)
        mid_y = 0.5 * (min_y + max_y)

        quad_records = []
        for label, x_col, y_col in [('gt', 'CENTER_X_gt', 'CENTER_Y_gt'), ('inf', 'CENTER_X_inf', 'CENTER_Y_inf')]:
            coords = paired_df[[x_col, y_col]].dropna().to_numpy()
            if len(coords) == 0:
                continue
            quadrant_counts = defaultdict(int)
            for x, y in coords:
                qx = 0 if x <= mid_x else 1
                qy = 0 if y <= mid_y else 1
                quadrant_counts[(qx, qy)] += 1
            total = sum(quadrant_counts.values())
            for q, cnt in quadrant_counts.items():
                quad_records.append({'label': label, 'quadrant': str(q), 'fraction': cnt / total if total else np.nan})

        quad_df = pd.DataFrame(quad_records) if quad_records else pd.DataFrame(columns=['label', 'quadrant', 'fraction'])

print('Quadrant occupancy fractions:')
display(quad_df)
# Similarity score for kinematics (stationary fraction agreement)
if 'stationary' in locals() and not stationary.isnull().any():
    diff = abs(stationary.get('gt', np.nan) - stationary.get('inf', np.nan))
    kin_score = max(0.0, 1.0 - min(1.0, diff))
else:
    kin_score = np.nan
print(f"Similarity score (kinematics): {kin_score:.3f}")
# Plot: speed distributions GT vs INF
import matplotlib.pyplot as plt
if not vel_df.empty:
    plt.figure(figsize=(8,3))
    for lbl, color in [('gt', 'tab:blue'), ('inf', 'tab:orange')]:
        subset = vel_df.loc[vel_df.label == lbl, 'speed']
        if not subset.empty:
            plt.hist(subset, bins=50, alpha=0.5, label=lbl, color=color)
    plt.xlabel('Speed')
    plt.ylabel('Count')
    plt.title('Speed distribution GT vs INF')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ### What this does
# Measures inter-limb distances (nose–tail, ears, nose–center) and compares movement direction to body axis via cosine similarity for GT vs inference.
# 

# In[19]:



# Inter-limb distances and movement direction vs body axis

def pair_distance(df: pd.DataFrame, part_a: str, part_b: str, suffix: str):
    ax, ay = f'{part_a}_X{suffix}', f'{part_a}_Y{suffix}'
    bx, by = f'{part_b}_X{suffix}', f'{part_b}_Y{suffix}'
    diff = df[[ax, ay]].to_numpy() - df[[bx, by]].to_numpy()
    return np.linalg.norm(diff, axis=1)

pairs_to_check = [
    ('NOSE', 'TAIL_BASE'),
    ('LEFT_EAR', 'RIGHT_EAR'),
    ('NOSE', 'CENTER'),
]

rows = []
for label, suffix in [('gt', '_gt'), ('inf', '_inf')]:
    for p0, p1 in pairs_to_check:
        dist = pair_distance(paired_df, p0, p1, suffix)
        rows.append({'label': label, 'pair': f'{p0}-{p1}', 'mean': dist.mean(), 'std': dist.std(), 'median': np.median(dist)})

dist_df = pd.DataFrame(rows)
print('Inter-limb distance stats:')
display(dist_df)

# Directionality: compare velocity vector vs body axis (nose -> tail_base)
rows = []
for label, x_prefix, y_prefix in [('gt', 'CENTER_X_gt', 'CENTER_Y_gt'), ('inf', 'CENTER_X_inf', 'CENTER_Y_inf')]:
    for seq, seq_df in paired_df.sort_values('itemPosition').groupby('sequenceId'):
        coords = seq_df[[x_prefix, y_prefix]].to_numpy()
        vel = np.diff(coords, axis=0)
        nose = seq_df[['NOSE_X_' + label, 'NOSE_Y_' + label]].to_numpy()[1:]
        tail = seq_df[['TAIL_BASE_X_' + label, 'TAIL_BASE_Y_' + label]].to_numpy()[1:]
        axis_vec = nose - tail
        # Normalize
        vel_norm = np.linalg.norm(vel, axis=1) + 1e-8
        axis_norm = np.linalg.norm(axis_vec, axis=1) + 1e-8
        cos_sim = np.sum(vel * axis_vec, axis=1) / (vel_norm * axis_norm)
        rows.append({'label': label, 'sequenceId': seq, 'cos_mean': np.mean(cos_sim), 'cos_median': np.median(cos_sim)})

direction_df = pd.DataFrame(rows)
print('Direction vs body-axis cosine similarity (1=forward):')
display(direction_df.head(20))
# Similarity score for geometry/direction (distance + cosine alignment)
import numpy as np
geom_score = np.nan
if 'dist_df' in locals() and not dist_df.empty:
    pivot = dist_df.pivot(index='pair', columns='label', values='mean')
    if {'gt', 'inf'}.issubset(pivot.columns):
        dist_gap = (pivot['gt'] - pivot['inf']).abs().mean()
        geom_score = 1.0 / (1.0 + dist_gap)
if 'direction_df' in locals() and not direction_df.empty:
    dir_pivot = direction_df.pivot(index='sequenceId', columns='label', values='cos_mean')
    if {'gt', 'inf'}.issubset(dir_pivot.columns):
        dir_gap = (dir_pivot['gt'] - dir_pivot['inf']).abs().mean()
        dir_score = 1.0 - min(1.0, dir_gap / 2.0)
        if np.isnan(geom_score):
            geom_score = dir_score
        else:
            geom_score = 0.5 * geom_score + 0.5 * dir_score
print(f"Similarity score (geometry/direction): {geom_score:.3f}")
# Plot: inter-limb distance means and directionality
import matplotlib.pyplot as plt
if 'dist_df' in locals() and not dist_df.empty:
    pivot = dist_df.pivot(index='pair', columns='label', values='mean')
    plt.figure(figsize=(8,3))
    pivot.plot(kind='bar', ax=plt.gca(), color={'gt': 'tab:blue', 'inf': 'tab:orange'})
    plt.ylabel('Mean distance')
    plt.title('Inter-limb distances (GT vs INF)')
    plt.tight_layout()
    plt.show()
if 'direction_df' in locals() and not direction_df.empty:
    plt.figure(figsize=(6,3))
    for lbl, color in [('gt', 'tab:blue'), ('inf', 'tab:orange')]:
        subset = direction_df.loc[direction_df.label == lbl, 'cos_mean']
        if not subset.empty:
            plt.hist(subset, bins=30, alpha=0.5, label=lbl, color=color, range=(-1, 1))
    plt.xlabel('Cosine similarity (velocity vs body axis)')
    plt.ylabel('Count')
    plt.title('Direction alignment')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ### What this does
# Clusters movement “syllables” via k-means on velocity/posture features and compares GT vs inference cluster counts/proportions and symmetric KL.
# 

# In[20]:



# Movement syllables via k-means clustering on velocity + posture features
try:
    from sklearn.cluster import KMeans
except ImportError:
    raise ImportError('sklearn not available for syllable clustering; install scikit-learn to run this cell.')

# Features: center velocity (dx, dy), nose-tail vector, ear spread
feat_rows = []
labels = []
for label, x_prefix, y_prefix in [('gt', 'CENTER_X_gt', 'CENTER_Y_gt'), ('inf', 'CENTER_X_inf', 'CENTER_Y_inf')]:
    for seq, seq_df in paired_df.sort_values('itemPosition').groupby('sequenceId'):
        coords = seq_df[[x_prefix, y_prefix]].to_numpy()
        vel = np.diff(coords, axis=0)
        vel_pad = np.vstack([vel[:1], vel])  # align lengths with frames
        nose = seq_df[['NOSE_X_' + label, 'NOSE_Y_' + label]].to_numpy()
        tail = seq_df[['TAIL_BASE_X_' + label, 'TAIL_BASE_Y_' + label]].to_numpy()
        axis_vec = nose - tail
        ears = seq_df[['LEFT_EAR_X_' + label, 'LEFT_EAR_Y_' + label, 'RIGHT_EAR_X_' + label, 'RIGHT_EAR_Y_' + label]].to_numpy()
        ear_span = np.linalg.norm(ears[:, :2] - ears[:, 2:], axis=1)
        feats = np.column_stack([vel_pad, axis_vec, ear_span])
        feat_rows.append(feats)
        labels.extend([label] * len(feats))

features = np.vstack(feat_rows)
labels = np.array(labels)

k = 8
kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
clusters = kmeans.fit_predict(features)

cluster_df = pd.DataFrame({'label': labels, 'cluster': clusters})
counts = cluster_df.groupby(['label', 'cluster']).size().unstack(fill_value=0)
probs = counts.div(counts.sum(axis=1), axis=0)
print(f'Syllable cluster counts (k={k}):')
display(counts)
print('Syllable cluster proportions:')
display(probs)

# Symmetric KL divergence between GT and INF syllable distributions
p = probs.loc['gt'].values + 1e-8
q = probs.loc['inf'].values + 1e-8
p /= p.sum()
q /= q.sum()
kl_gt_inf = np.sum(p * np.log(p / q))
kl_inf_gt = np.sum(q * np.log(q / p))
sym_kl = 0.5 * (kl_gt_inf + kl_inf_gt)
print(f'Symmetric KL between GT and INF syllable distributions: {sym_kl:.4f}')
# Similarity score for syllable distributions (sym KL -> [0,1])
similarity_score = 1.0 / (1.0 + sym_kl)
print(f"Similarity score (syllable KL): {similarity_score:.3f}")
# Plot: syllable distribution (cluster probabilities)
import matplotlib.pyplot as plt
if 'probs' in locals() and not probs.empty:
    ax = probs.T.plot(kind='bar', figsize=(8,3), color=['tab:blue', 'tab:orange'])
    ax.set_ylabel('Probability')
    ax.set_title('Syllable distribution (GT vs INF)')
    plt.tight_layout()
    plt.show()


# In[21]:


# Trajectory similarity via DTW on CENTER track
# Uses dynamic time warping to compare GT vs INF center trajectories per sequence

def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray, window: int | None = None) -> float:
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return np.nan
    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))
    inf = np.inf
    dtw = np.full((n + 1, m + 1), inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            cost = np.linalg.norm(seq_a[i - 1] - seq_b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    # Normalize by path length upper bound to keep distances comparable
    return dtw[n, m] / (n + m)

traj_rows = []
for seq, seq_df in paired_df.sort_values('itemPosition').groupby('sequenceId'):
    gt_seq = seq_df[['CENTER_X_gt', 'CENTER_Y_gt']].to_numpy()
    inf_seq = seq_df[['CENTER_X_inf', 'CENTER_Y_inf']].to_numpy()
    if len(gt_seq) == 0 or len(inf_seq) == 0:
        continue
    # Sakoe-Chiba band ~10% of longer length to limit cost
    w = int(max(len(gt_seq), len(inf_seq)) * 0.1)
    w = max(w, abs(len(gt_seq) - len(inf_seq)))
    dist = dtw_distance(gt_seq, inf_seq, window=w)
    sim = np.nan if np.isnan(dist) else 1.0 / (1.0 + dist)
    traj_rows.append({'sequenceId': seq, 'dtw_dist': dist, 'similarity_0_1': sim})

traj_df = pd.DataFrame(traj_rows).sort_values('similarity_0_1', ascending=False) if traj_rows else pd.DataFrame(columns=['sequenceId', 'dtw_dist', 'similarity_0_1'])
print('Trajectory DTW similarity per sequence (center track):')
display(traj_df.head(20))
if not traj_df.empty:
    overall_sim = traj_df['similarity_0_1'].mean()
    print(f'Overall DTW-based trajectory similarity (0-1): {overall_sim:.3f}')
else:
    print('No trajectory similarity computed (no paired sequences).')
# Plot: DTW trajectory similarity distribution
import matplotlib.pyplot as plt
if not traj_df.empty:
    plt.figure(figsize=(6,3))
    plt.hist(traj_df['similarity_0_1'], bins=20, color='tab:green', alpha=0.8)
    plt.xlabel('DTW similarity (0-1)')
    plt.ylabel('Count')
    plt.title('Trajectory similarity across sequences')
    plt.tight_layout()
    plt.show()


# In[22]:


# Movement manifold alignment metrics (PCA + Procrustes + MMD)
try:
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
    from scipy.spatial import procrustes
except ImportError:
    raise ImportError('PCA/Procrustes/MMD dependencies missing (install scikit-learn and scipy).')

if paired_df.empty:
    print('No paired data; skipping manifold metrics.')
else:
    def build_features(df: pd.DataFrame, label: str) -> np.ndarray:
        center = df[[f'CENTER_X_{label}', f'CENTER_Y_{label}']].to_numpy()
        if len(center) == 0:
            return np.empty((0, 5))
        vel = np.diff(center, axis=0)
        if len(vel) == 0:
            vel_pad = np.zeros_like(center)
        else:
            vel_pad = np.vstack([vel[:1], vel])
        nose = df[[f'NOSE_X_{label}', f'NOSE_Y_{label}']].to_numpy()
        tail = df[[f'TAIL_BASE_X_{label}', f'TAIL_BASE_Y_{label}']].to_numpy()
        axis_vec = nose - tail
        ears = df[[f'LEFT_EAR_X_{label}', f'LEFT_EAR_Y_{label}', f'RIGHT_EAR_X_{label}', f'RIGHT_EAR_Y_{label}']].to_numpy()
        ear_span = np.linalg.norm(ears[:, :2] - ears[:, 2:], axis=1, keepdims=True)
        feats = np.hstack([vel_pad, axis_vec, ear_span])
        return feats

    gt_feats_all = []
    inf_feats_all = []
    for seq, seq_df in paired_df.sort_values('itemPosition').groupby('sequenceId'):
        gt_feats = build_features(seq_df, 'gt')
        inf_feats = build_features(seq_df, 'inf')
        if len(gt_feats) == 0 or len(inf_feats) == 0:
            continue
        mask = ~np.isnan(gt_feats).any(axis=1) & ~np.isnan(inf_feats).any(axis=1)
        gt_feats = gt_feats[mask]
        inf_feats = inf_feats[mask]
        if len(gt_feats) == 0:
            continue
        gt_feats_all.append(gt_feats)
        inf_feats_all.append(inf_feats)

    if not gt_feats_all:
        print('No valid trajectory features for manifold metrics.')
    else:
        gt_stack = np.vstack(gt_feats_all)
        inf_stack = np.vstack(inf_feats_all)
        n_components = min(3, gt_stack.shape[1])
        pca = PCA(n_components=n_components)
        gt_emb = pca.fit_transform(gt_stack)
        inf_emb = pca.transform(inf_stack)
        proc_a, proc_b, proc_dist = procrustes(gt_emb, inf_emb)
        proc_sim = 1.0 / (1.0 + proc_dist)

        # RBF MMD with median heuristic for gamma
        all_emb = np.vstack([gt_emb, inf_emb])
        if len(all_emb) > 1:
            dists = pairwise_distances(all_emb)
            nonzero = dists[np.triu_indices_from(dists, k=1)]
            med = np.median(nonzero) if len(nonzero) else 1.0
            gamma = 1.0 / (2.0 * (med ** 2 + 1e-8))
        else:
            gamma = 1.0
        Kxx = rbf_kernel(gt_emb, gt_emb, gamma=gamma)
        Kyy = rbf_kernel(inf_emb, inf_emb, gamma=gamma)
        Kxy = rbf_kernel(gt_emb, inf_emb, gamma=gamma)
        mmd2 = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
        mmd_sim = 1.0 / (1.0 + max(0.0, mmd2))

        print('Manifold metrics (GT PCA space):')
        print(f'  PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}')
        print(f'  Procrustes distance: {proc_dist:.4f} -> similarity: {proc_sim:.3f}')
        print(f'  RBF MMD^2: {mmd2:.4f} -> similarity: {mmd_sim:.3f}')


# In[23]:


# UMAP embedding of 5-frame movement chunks (GT, INF, GT shuffled)
try:
    import umap  # umap-learn
    UMAP = umap.UMAP
except ImportError:
    UMAP = None

from sklearn.decomposition import PCA

chunk_size = 5

if paired_df.empty:
    print('No paired data; skipping UMAP chunk embedding.')
else:
    def build_features(df: pd.DataFrame, label: str) -> np.ndarray:
        center = df[[f'CENTER_X_{label}', f'CENTER_Y_{label}']].to_numpy()
        if len(center) == 0:
            return np.empty((0, 5))
        vel = np.diff(center, axis=0)
        if len(vel) == 0:
            vel_pad = np.zeros_like(center)
        else:
            vel_pad = np.vstack([vel[:1], vel])
        nose = df[[f'NOSE_X_{label}', f'NOSE_Y_{label}']].to_numpy()
        tail = df[[f'TAIL_BASE_X_{label}', f'TAIL_BASE_Y_{label}']].to_numpy()
        axis_vec = nose - tail
        ears = df[[f'LEFT_EAR_X_{label}', f'LEFT_EAR_Y_{label}', f'RIGHT_EAR_X_{label}', f'RIGHT_EAR_Y_{label}']].to_numpy()
        ear_span = np.linalg.norm(ears[:, :2] - ears[:, 2:], axis=1, keepdims=True)
        feats = np.hstack([vel_pad, axis_vec, ear_span])
        return feats

    chunks = []
    labels = []
    seq_ids = []

    for seq, seq_df in paired_df.sort_values('itemPosition').groupby('sequenceId'):
        gt_feats = build_features(seq_df, 'gt')
        inf_feats = build_features(seq_df, 'inf')
        if len(gt_feats) < chunk_size or len(inf_feats) < chunk_size:
            continue
        # Drop NaNs
        mask_gt = ~np.isnan(gt_feats).any(axis=1)
        mask_inf = ~np.isnan(inf_feats).any(axis=1)
        gt_feats = gt_feats[mask_gt]
        inf_feats = inf_feats[mask_inf]
        if len(gt_feats) < chunk_size or len(inf_feats) < chunk_size:
            continue
        # contiguous chunks
        for arr, lbl in [(gt_feats, 'gt'), (inf_feats, 'inf')]:
            for start in range(0, len(arr) - chunk_size + 1, chunk_size):
                window = arr[start:start + chunk_size]
                chunks.append(window.flatten())
                labels.append(lbl)
                seq_ids.append(seq)
        # shuffled GT baseline
        shuffled = gt_feats.copy()
        rng = np.random.default_rng(0)
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled) - chunk_size + 1, chunk_size):
            window = shuffled[start:start + chunk_size]
            chunks.append(window.flatten())
            labels.append('gt_shuffled')
            seq_ids.append(seq)

    if not chunks:
        print('No chunks available for UMAP embedding.')
    else:
        import numpy as np
        X = np.vstack(chunks)
        lbl_arr = np.array(labels)
        n_neighbors = max(2, min(15, len(X) - 1))
        if UMAP is not None and len(X) > 5:
            reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=0)
            emb = reducer.fit_transform(X)
            emb_method = 'UMAP'
        else:
            pca = PCA(n_components=2)
            emb = pca.fit_transform(X)
            emb_method = 'PCA (fallback)'
        print(f'Embedding method: {emb_method}, points: {len(X)}')

        # Plot embedding colored by label
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,4))
        for lbl, color in [('gt', 'tab:blue'), ('inf', 'tab:orange'), ('gt_shuffled', 'tab:gray')]:
            mask = lbl_arr == lbl
            if mask.any():
                plt.scatter(emb[mask,0], emb[mask,1], s=8, alpha=0.6, label=lbl, color=color)
        plt.title('5-frame movement chunks embedding')
        plt.xlabel('dim 1'); plt.ylabel('dim 2')
        plt.legend(markerscale=2)
        plt.tight_layout()
        plt.show()

