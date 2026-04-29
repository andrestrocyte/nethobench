from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from nethobench.helpers import load_gt_and_preds


def _clip01(x: float, eps: float = 1e-6) -> float:
    return float(np.clip(x, eps, 1.0 - eps))


def _geometric_mean_scores(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    arr = np.asarray([_clip01(v) for v in arr], dtype=np.float64)
    return float(np.exp(np.mean(np.log(arr))))


def load_file(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_parquet(path)
    except Exception:
        import pyarrow.parquet as pq
        table = pq.read_table(path, use_pandas_metadata=False)
        return table.to_pandas()


def merge_paired(gt_df: pd.DataFrame, inf_df: pd.DataFrame, headers: List[str]) -> pd.DataFrame:
    merge_keys = ["sequenceId", "itemPosition"]
    value_headers = [h for h in headers if h not in merge_keys]
    return pd.merge(
        gt_df[merge_keys + value_headers],
        inf_df[merge_keys + value_headers],
        on=merge_keys,
        how="inner",
        suffixes=("_gt", "_inf"),
    )


def _calc_sym_kl(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    return 0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))


# ---------------------------------------------------------
# Core Behavioral Metrics
# ---------------------------------------------------------

def stationary_score(paired_df: pd.DataFrame) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    if paired_df.empty:
        return np.nan, {}, {}

    all_speeds_gt = []
    seq_speeds = {"gt": {}, "inf": {}}
    for label in ["gt", "inf"]:
        x_col, y_col = f"CENTER_X_{label}", f"CENTER_Y_{label}"
        for seq, seq_df in paired_df.sort_values("itemPosition").groupby("sequenceId"):
            coords = seq_df[[x_col, y_col]].to_numpy()
            vel = np.diff(coords, axis=0)
            speed = np.linalg.norm(vel, axis=1)
            seq_speeds[label][seq] = speed
            if label == "gt":
                all_speeds_gt.extend(speed)

    if not all_speeds_gt:
        return np.nan, {}, {}

    thresh = np.percentile(all_speeds_gt, 10)
    seq_scores = {}
    seq_weights = {}
    gaps = []
    valid_seqs = set(seq_speeds["gt"].keys()) & set(seq_speeds["inf"].keys())
    for seq in valid_seqs:
        s_gt = seq_speeds["gt"][seq]
        s_inf = seq_speeds["inf"][seq]
        if len(s_gt) == 0 or len(s_inf) == 0:
            continue
        stat_gt = float(np.mean(s_gt <= thresh))
        stat_inf = float(np.mean(s_inf <= thresh))
        diff = abs(stat_gt - stat_inf)
        gaps.append(diff)
        seq_scores[str(seq)] = 1.0 - min(1.0, diff)
        seq_weights[str(seq)] = len(s_gt)

    if not gaps:
        return np.nan, {}, {}
    global_gap = float(np.mean(gaps))
    global_score = 1.0 - min(1.0, global_gap)
    return global_score, seq_scores, seq_weights


def velocity_distribution_score(paired_df: pd.DataFrame) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    if paired_df.empty:
        return np.nan, {}, {}

    speeds = {"gt": {}, "inf": {}}
    all_vals = []

    for label in ("gt", "inf"):
        x_col, y_col = f"CENTER_X_{label}", f"CENTER_Y_{label}"
        for seq, seq_df in paired_df.sort_values("itemPosition").groupby("sequenceId"):
            coords = seq_df[[x_col, y_col]].to_numpy()
            vel = np.diff(coords, axis=0)
            speed = np.linalg.norm(vel, axis=1)
            if len(speed) > 0:
                speeds[label][seq] = speed
                all_vals.extend(speed)

    if not all_vals:
        return np.nan, {}, {}

    lo, hi = min(all_vals), max(all_vals)
    if lo == hi:
        return np.nan, {}, {}

    bins = np.linspace(lo, hi, 60)
    seq_scores = {}
    seq_weights = {}

    global_gt = np.concatenate(list(speeds["gt"].values())) if speeds["gt"] else np.array([])
    global_inf = np.concatenate(list(speeds["inf"].values())) if speeds["inf"] else np.array([])

    if global_gt.size < 10 or global_inf.size < 10:
        return np.nan, {}, {}

    Pg, _ = np.histogram(global_gt, bins=bins, density=True)
    Pi, _ = np.histogram(global_inf, bins=bins, density=True)
    global_kl = _calc_sym_kl(Pg, Pi)
    global_score = 1.0 / (1.0 + global_kl)

    valid_seqs = set(speeds["gt"].keys()) & set(speeds["inf"].keys())
    for seq in valid_seqs:
        pg_seq, _ = np.histogram(speeds["gt"][seq], bins=bins, density=True)
        pi_seq, _ = np.histogram(speeds["inf"][seq], bins=bins, density=True)
        kl = _calc_sym_kl(pg_seq, pi_seq)
        seq_scores[str(seq)] = 1.0 / (1.0 + kl)
        seq_weights[str(seq)] = len(speeds["gt"][seq])

    return global_score, seq_scores, seq_weights


def acceleration_distribution_score(paired_df: pd.DataFrame) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    if paired_df.empty:
        return np.nan, {}, {}

    accs = {"gt": {}, "inf": {}}
    all_vals = []

    for label in ("gt", "inf"):
        x_col, y_col = f"CENTER_X_{label}", f"CENTER_Y_{label}"
        for seq, seq_df in paired_df.sort_values("itemPosition").groupby("sequenceId"):
            coords = seq_df[[x_col, y_col]].to_numpy()
            vel = np.diff(coords, axis=0)
            acc = np.diff(vel, axis=0)
            mag = np.linalg.norm(acc, axis=1)
            if len(mag) > 0:
                accs[label][seq] = mag
                all_vals.extend(mag)

    if not all_vals:
        return np.nan, {}, {}

    lo, hi = min(all_vals), max(all_vals)
    if lo == hi:
        return np.nan, {}, {}

    bins = np.linspace(lo, hi, 60)
    seq_scores = {}
    seq_weights = {}

    global_gt = np.concatenate(list(accs["gt"].values())) if accs["gt"] else np.array([])
    global_inf = np.concatenate(list(accs["inf"].values())) if accs["inf"] else np.array([])

    if global_gt.size < 10 or global_inf.size < 10:
        return np.nan, {}, {}

    Pg, _ = np.histogram(global_gt, bins=bins, density=True)
    Pi, _ = np.histogram(global_inf, bins=bins, density=True)
    global_kl = _calc_sym_kl(Pg, Pi)
    global_score = 1.0 / (1.0 + global_kl)

    valid_seqs = set(accs["gt"].keys()) & set(accs["inf"].keys())
    for seq in valid_seqs:
        pg_seq, _ = np.histogram(accs["gt"][seq], bins=bins, density=True)
        pi_seq, _ = np.histogram(accs["inf"][seq], bins=bins, density=True)
        kl = _calc_sym_kl(pg_seq, pi_seq)
        seq_scores[str(seq)] = 1.0 / (1.0 + kl)
        seq_weights[str(seq)] = len(accs["gt"][seq])

    return global_score, seq_scores, seq_weights


def direction_score(paired_df: pd.DataFrame, nose_label: str = "NOSE", tail_label: str = "TAIL_BASE") -> Tuple[float, Dict[str, float], Dict[str, int]]:
    if paired_df.empty:
        return np.nan, {}, {}

    gaps = []
    seq_scores = {}
    seq_weights = {}

    grouped = paired_df.sort_values("itemPosition").groupby("sequenceId")
    for seq, seq_df in grouped:
        cos_vals = {}
        for label in ("gt", "inf"):
            coords = seq_df[[f"CENTER_X_{label}", f"CENTER_Y_{label}"]].to_numpy()
            vel = np.diff(coords, axis=0)
            if len(vel) == 0:
                continue
            nose = seq_df[[f"{nose_label}_X_{label}", f"{nose_label}_Y_{label}"]].to_numpy()[1:]
            tail = seq_df[[f"{tail_label}_X_{label}", f"{tail_label}_Y_{label}"]].to_numpy()[1:]
            axis_vec = nose - tail
            vel_norm = np.linalg.norm(vel, axis=1) + 1e-8
            axis_norm = np.linalg.norm(axis_vec, axis=1) + 1e-8
            cos_sim = np.sum(vel * axis_vec, axis=1) / (vel_norm * axis_norm)
            cos_vals[label] = cos_sim

        if {"gt", "inf"} <= cos_vals.keys():
            n = min(len(cos_vals["gt"]), len(cos_vals["inf"]))
            if n == 0:
                continue
            diff = np.abs(cos_vals["gt"][:n] - cos_vals["inf"][:n])
            gap = float(np.mean(diff))
            gaps.append(gap)
            seq_scores[str(seq)] = 1.0 - min(1.0, gap / 2.0)
            seq_weights[str(seq)] = len(cos_vals["gt"])

    if not gaps:
        return np.nan, {}, {}

    global_gap = np.mean(gaps)
    global_score = 1.0 - min(1.0, global_gap / 2.0)
    return global_score, seq_scores, seq_weights


def quadrant_score(paired_df: pd.DataFrame) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    if paired_df.empty:
        return np.nan, {}, {}

    centers = paired_df[["CENTER_X_gt", "CENTER_Y_gt", "CENTER_X_inf", "CENTER_Y_inf", "sequenceId"]].dropna()
    if centers.empty:
        return np.nan, {}, {}

    min_x = centers[["CENTER_X_gt", "CENTER_X_inf"]].min().min()
    max_x = centers[["CENTER_X_gt", "CENTER_X_inf"]].max().max()
    min_y = centers[["CENTER_Y_gt", "CENTER_Y_inf"]].min().min()
    max_y = centers[["CENTER_Y_gt", "CENTER_Y_inf"]].max().max()
    mid_x = 0.5 * (min_x + max_x)
    mid_y = 0.5 * (min_y + max_y)

    def get_quad_counts(df_slice, x_col, y_col):
        coords = df_slice[[x_col, y_col]].to_numpy()
        qx = (coords[:, 0] > mid_x).astype(int)
        qy = (coords[:, 1] > mid_y).astype(int)
        idx = qx * 2 + qy
        counts = np.bincount(idx, minlength=4).astype(float)
        return counts

    p_glob = get_quad_counts(centers, "CENTER_X_gt", "CENTER_Y_gt")
    q_glob = get_quad_counts(centers, "CENTER_X_inf", "CENTER_Y_inf")

    if p_glob.sum() == 0 or q_glob.sum() == 0:
        return np.nan, {}, {}

    global_kl = _calc_sym_kl(p_glob / p_glob.sum(), q_glob / q_glob.sum())
    global_score = 1.0 / (1.0 + global_kl)

    seq_scores = {}
    seq_weights = {}
    for seq, seq_df in centers.groupby("sequenceId"):
        p_seq = get_quad_counts(seq_df, "CENTER_X_gt", "CENTER_Y_gt")
        q_seq = get_quad_counts(seq_df, "CENTER_X_inf", "CENTER_Y_inf")
        if p_seq.sum() > 0 and q_seq.sum() > 0:
            kl = _calc_sym_kl(p_seq / p_seq.sum(), q_seq / q_seq.sum())
            seq_scores[str(seq)] = 1.0 / (1.0 + kl)
            seq_weights[str(seq)] = int(p_seq.sum())

    return global_score, seq_scores, seq_weights


def position_kl_score(paired_df: pd.DataFrame) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    if paired_df.empty:
        return np.nan, {}, {}

    centers = paired_df[["CENTER_X_gt", "CENTER_Y_gt", "CENTER_X_inf", "CENTER_Y_inf", "sequenceId"]].dropna()
    if centers.empty:
        return np.nan, {}, {}

    combined = np.vstack([
        centers[["CENTER_X_gt", "CENTER_Y_gt"]].to_numpy(),
        centers[["CENTER_X_inf", "CENTER_Y_inf"]].to_numpy()
    ])
    x_min, x_max = combined[:, 0].min(), combined[:, 0].max()
    y_min, y_max = combined[:, 1].min(), combined[:, 1].max()

    if x_min == x_max: x_min -= 0.5; x_max += 0.5
    if y_min == y_max: y_min -= 0.5; y_max += 0.5

    bins = 60
    x_edges = np.linspace(x_min, x_max, bins + 1)
    y_edges = np.linspace(y_min, y_max, bins + 1)

    def get_hist(df_slice, x_col, y_col):
        H, _, _ = np.histogram2d(df_slice[x_col], df_slice[y_col], bins=[x_edges, y_edges], density=True)
        return H.flatten()

    Pg = get_hist(centers, "CENTER_X_gt", "CENTER_Y_gt")
    Pi = get_hist(centers, "CENTER_X_inf", "CENTER_Y_inf")
    global_kl = _calc_sym_kl(Pg, Pi)
    global_score = 1.0 / (1.0 + global_kl)

    seq_scores = {}
    seq_weights = {}
    for seq, seq_df in centers.groupby("sequenceId"):
        p_seq = get_hist(seq_df, "CENTER_X_gt", "CENTER_Y_gt")
        q_seq = get_hist(seq_df, "CENTER_X_inf", "CENTER_Y_inf")
        kl = _calc_sym_kl(p_seq, q_seq)
        seq_scores[str(seq)] = 1.0 / (1.0 + kl)
        seq_weights[str(seq)] = len(seq_df)

    return global_score, seq_scores, seq_weights


def trajectory_shape_score(paired_df: pd.DataFrame, k: int = 6) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    if paired_df.empty:
        return np.nan, {}, {}

    rows = []
    seq_lens = {}
    for seq, seq_df in paired_df.sort_values("itemPosition").groupby("sequenceId"):
        seq_feats = {}
        for label in ("gt", "inf"):
            pts = seq_df[[f"CENTER_X_{label}", f"CENTER_Y_{label}"]].to_numpy()
            if len(pts) < 3:
                continue
            if label == "gt":
                seq_lens[str(seq)] = len(pts)
            disp = pts[-1] - pts[0]
            path_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
            net_disp = float(np.linalg.norm(disp))
            straightness = net_disp / (path_len + 1e-8)
            mean_speed = path_len / max(1, len(pts) - 1)
            step_vecs = np.diff(pts, axis=0)
            norms = np.linalg.norm(step_vecs, axis=1) + 1e-8
            dirs = step_vecs / norms[:, None]
            dots = np.sum(dirs[:-1] * dirs[1:], axis=1) if len(dirs) > 1 else np.array([])
            dots = np.clip(dots, -1.0, 1.0)
            mean_turn = float(np.arccos(dots).mean()) if len(dots) else 0.0
            seq_feats[label] = [path_len, net_disp, straightness, mean_speed, mean_turn]
        if "gt" in seq_feats and "inf" in seq_feats:
            rows.append({"sequenceId": seq, "gt": seq_feats["gt"], "inf": seq_feats["inf"]})

    if not rows:
        return np.nan, {}, {}

    df = pd.DataFrame(rows)
    X = np.vstack(df["gt"].tolist() + df["inf"].tolist())
    if len(X) < k * 2:
        return np.nan, {}, {}

    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    clusters = km.fit_predict(X)
    labels = ["gt"] * len(df) + ["inf"] * len(df)
    seq_map = df["sequenceId"].tolist() + df["sequenceId"].tolist()
    cluster_df = pd.DataFrame({"label": labels, "cluster": clusters, "sequenceId": seq_map})

    counts = cluster_df.groupby(["label", "cluster"]).size().unstack(fill_value=0)
    if not {"gt", "inf"}.issubset(counts.index):
        return np.nan, {}, {}

    probs = counts.div(counts.sum(axis=1), axis=0)
    p = probs.loc["gt"].to_numpy()
    q = probs.loc["inf"].to_numpy()
    global_score = 1.0 / (1.0 + _calc_sym_kl(p, q))

    seq_scores = {}
    seq_weights = {}
    for seq, seq_df in cluster_df.groupby("sequenceId"):
        sc = seq_df.groupby(["label", "cluster"]).size().unstack(fill_value=0)
        sc = sc.reindex(columns=range(k), fill_value=0)
        if "gt" in sc.index and "inf" in sc.index:
            sp = sc.loc["gt"].to_numpy()
            sq = sc.loc["inf"].to_numpy()
            kl = _calc_sym_kl(sp, sq)
            seq_scores[str(seq)] = 1.0 / (1.0 + kl)
            seq_weights[str(seq)] = int(sc.loc["gt"].sum())

    return global_score, seq_scores, seq_weights


def syllable_score(paired_df: pd.DataFrame, k: int = 8) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    if paired_df.empty:
        return np.nan, {}, {}

    feat_rows = []
    labels = []
    seq_map = []

    def build_feats(df, label):
        coords = df[[f"CENTER_X_{label}", f"CENTER_Y_{label}"]].to_numpy()
        vel = np.diff(coords, axis=0)
        acc = np.diff(vel, axis=0)
        speed = np.linalg.norm(vel, axis=1)
        acc_mag = np.linalg.norm(acc, axis=1) if len(acc) else np.array([])
        feats = []
        for i in range(min(len(speed), len(acc_mag))):
            feats.append([speed[i], acc_mag[i]])
        return np.array(feats)

    for seq, seq_df in paired_df.sort_values("itemPosition").groupby("sequenceId"):
        for label in ("gt", "inf"):
            feats = build_feats(seq_df, label)
            feats = feats[~np.isnan(feats).any(axis=1)]
            if len(feats) > 0:
                feat_rows.append(feats)
                labels.extend([label] * len(feats))
                seq_map.extend([seq] * len(feats))

    if not feat_rows:
        return np.nan, {}, {}

    X = np.vstack(feat_rows)
    if len(X) < k * 2:
        return np.nan, {}, {}

    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    clusters = km.fit_predict(X)
    df = pd.DataFrame({"label": labels, "cluster": clusters, "sequenceId": seq_map})

    counts = df.groupby(["label", "cluster"]).size().unstack(fill_value=0)
    if not {"gt", "inf"}.issubset(counts.index):
        return np.nan, {}, {}

    probs = counts.div(counts.sum(axis=1), axis=0)
    p = probs.loc["gt"].to_numpy()
    q = probs.loc["inf"].to_numpy()
    global_score = 1.0 / (1.0 + _calc_sym_kl(p, q))

    seq_scores = {}
    seq_weights = {}
    for seq, seq_df in df.groupby("sequenceId"):
        sc = seq_df.groupby(["label", "cluster"]).size().unstack(fill_value=0)
        sc = sc.reindex(columns=range(k), fill_value=0)
        if "gt" in sc.index and "inf" in sc.index:
            sp = sc.loc["gt"].to_numpy()
            sq = sc.loc["inf"].to_numpy()
            kl = _calc_sym_kl(sp, sq)
            seq_scores[str(seq)] = 1.0 / (1.0 + kl)
            seq_weights[str(seq)] = int(sc.loc["gt"].sum())

    return global_score, seq_scores, seq_weights


# ---------------------------------------------------------
# Extended Metrics (Previously in Notebook)
# ---------------------------------------------------------

def body_part_errors(paired_df: pd.DataFrame, coord_pairs: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """Computes RMSE, MAE and Pearson correlation per body part."""
    rows = []
    for base, x_col, y_col in coord_pairs:
        if paired_df.empty:
            continue
        gt_xy = paired_df[[f'{x_col}_gt', f'{y_col}_gt']].to_numpy()
        inf_xy = paired_df[[f'{x_col}_inf', f'{y_col}_inf']].to_numpy()
        
        valid = ~np.isnan(gt_xy).any(axis=1) & ~np.isnan(inf_xy).any(axis=1)
        if not valid.any():
            continue
            
        diff = gt_xy[valid] - inf_xy[valid]
        dist = np.linalg.norm(diff, axis=1)
        rmse = float(np.sqrt(np.mean(dist**2)))
        mae_x, mae_y = np.mean(np.abs(diff), axis=0)
        
        a_x, b_x = gt_xy[valid, 0], inf_xy[valid, 0]
        corr_x = float(np.corrcoef(a_x, b_x)[0, 1]) if len(a_x) > 1 and np.std(a_x)>0 and np.std(b_x)>0 else np.nan
        
        a_y, b_y = gt_xy[valid, 1], inf_xy[valid, 1]
        corr_y = float(np.corrcoef(a_y, b_y)[0, 1]) if len(a_y) > 1 and np.std(a_y)>0 and np.std(b_y)>0 else np.nan
        
        rows.append({
            'body_part': base,
            'rmse_pos': rmse,
            'mae_x': float(mae_x),
            'mae_y': float(mae_y),
            'corr_x': corr_x,
            'corr_y': corr_y
        })
    return pd.DataFrame(rows)


def inter_limb_distances(paired_df: pd.DataFrame, pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """Computes mean distances between specific body parts."""
    rows = []
    for label, suffix in [('gt', '_gt'), ('inf', '_inf')]:
        for p0, p1 in pairs:
            try:
                ax, ay = f'{p0}_X{suffix}', f'{p0}_Y{suffix}'
                bx, by = f'{p1}_X{suffix}', f'{p1}_Y{suffix}'
                diff = paired_df[[ax, ay]].to_numpy() - paired_df[[bx, by]].to_numpy()
                dist = np.linalg.norm(diff, axis=1)
                rows.append({
                    'label': label,
                    'pair': f'{p0}-{p1}',
                    'mean': float(np.nanmean(dist)),
                    'std': float(np.nanstd(dist))
                })
            except KeyError:
                pass
    return pd.DataFrame(rows)
def dtw_trajectory_similarity(paired_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """Calculates Dynamic Time Warping (DTW) distance on CENTER tracking in O(M) memory."""
    if paired_df.empty or 'CENTER_X_gt' not in paired_df.columns:
        return np.nan, {}
        
    def dtw_dist(seq_a, seq_b, window):
        n, m = len(seq_a), len(seq_b)
        if n == 0 or m == 0:
            return np.nan
        window = max(window, abs(n - m))
        
        # O(M) memory optimization: only keep the previous and current row
        prev_row = np.full(m + 1, np.inf)
        curr_row = np.full(m + 1, np.inf)
        prev_row[0] = 0.0
        
        for i in range(1, n + 1):
            curr_row[:] = np.inf
            j_start = max(1, i - window)
            j_end = min(m, i + window)
            for j in range(j_start, j_end + 1):
                cost = np.linalg.norm(seq_a[i - 1] - seq_b[j - 1])
                curr_row[j] = cost + min(prev_row[j], curr_row[j - 1], prev_row[j - 1])
            prev_row[:] = curr_row
            
        return curr_row[m] / (n + m)

    seq_scores = {}
    for seq, seq_df in paired_df.sort_values('itemPosition').groupby('sequenceId'):
        gt_seq = seq_df[['CENTER_X_gt', 'CENTER_Y_gt']].dropna().to_numpy()
        inf_seq = seq_df[['CENTER_X_inf', 'CENTER_Y_inf']].dropna().to_numpy()
        if not len(gt_seq) or not len(inf_seq):
            continue
        w = max(int(max(len(gt_seq), len(inf_seq)) * 0.1), abs(len(gt_seq) - len(inf_seq)))
        d = dtw_dist(gt_seq, inf_seq, w)
        seq_scores[str(seq)] = 1.0 / (1.0 + d) if not np.isnan(d) else np.nan
        
    vals = [v for v in seq_scores.values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan, seq_scores

def manifold_alignment_metrics(paired_df: pd.DataFrame) -> Dict[str, float]:
    """Calculates Procrustes and MMD metrics for movement manifolds (memory-safe)."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
        from scipy.spatial import procrustes
    except ImportError:
        return {'procrustes_sim': np.nan, 'mmd_sim': np.nan}
    
    def build_features(df, label):
        try:
            center = df[[f'CENTER_X_{label}', f'CENTER_Y_{label}']].to_numpy()
            vel = np.diff(center, axis=0)
            vel_pad = np.vstack([vel[:1], vel]) if len(vel) else np.zeros_like(center)
            nose = df[[f'NOSE_X_{label}', f'NOSE_Y_{label}']].to_numpy()
            tail = df[[f'TAIL_BASE_X_{label}', f'TAIL_BASE_Y_{label}']].to_numpy()
            axis_vec = nose - tail
            ears = df[[f'LEFT_EAR_X_{label}', f'LEFT_EAR_Y_{label}', f'RIGHT_EAR_X_{label}', f'RIGHT_EAR_Y_{label}']].to_numpy()
            ear_span = np.linalg.norm(ears[:, :2] - ears[:, 2:], axis=1, keepdims=True)
            return np.hstack([vel_pad, axis_vec, ear_span])
        except KeyError:
            return np.empty((len(df), 5))

    gt_feats_all, inf_feats_all = [], []
    for seq, seq_df in paired_df.sort_values('itemPosition').groupby('sequenceId'):
        gt_f = build_features(seq_df, 'gt')
        inf_f = build_features(seq_df, 'inf')
        mask = ~np.isnan(gt_f).any(axis=1) & ~np.isnan(inf_f).any(axis=1)
        if mask.any():
            gt_feats_all.append(gt_f[mask])
            inf_feats_all.append(inf_f[mask])
    
    if not gt_feats_all:
        return {'procrustes_sim': np.nan, 'mmd_sim': np.nan}

    gt_stack = np.vstack(gt_feats_all)
    inf_stack = np.vstack(inf_feats_all)
    pca = PCA(n_components=min(3, gt_stack.shape[1]))
    gt_emb = pca.fit_transform(gt_stack)
    inf_emb = pca.transform(inf_stack)
    
    try:
        _, _, proc_dist = procrustes(gt_emb, inf_emb)
        proc_sim = float(1.0 / (1.0 + proc_dist))
    except ValueError:
        proc_sim = np.nan
    
    # Subsample to prevent O(N^2) memory explosion in distance matrix
    MAX_MMD_POINTS = 2000
    rng = np.random.default_rng(42)
    if len(gt_emb) > MAX_MMD_POINTS:
        gt_emb_sub = gt_emb[rng.choice(len(gt_emb), MAX_MMD_POINTS, replace=False)]
    else:
        gt_emb_sub = gt_emb
        
    if len(inf_emb) > MAX_MMD_POINTS:
        inf_emb_sub = inf_emb[rng.choice(len(inf_emb), MAX_MMD_POINTS, replace=False)]
    else:
        inf_emb_sub = inf_emb

    all_emb_sub = np.vstack([gt_emb_sub, inf_emb_sub])
    if len(all_emb_sub) > 1:
        dists = pairwise_distances(all_emb_sub)
        nonzero = dists[np.triu_indices_from(dists, k=1)]
        med = np.median(nonzero) if len(nonzero) else 1.0
        gamma = 1.0 / (2.0 * (med ** 2 + 1e-8))
    else:
        gamma = 1.0
        
    Kxx = rbf_kernel(gt_emb_sub, gt_emb_sub, gamma=gamma)
    Kyy = rbf_kernel(inf_emb_sub, inf_emb_sub, gamma=gamma)
    Kxy = rbf_kernel(gt_emb_sub, inf_emb_sub, gamma=gamma)
    mmd2 = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    mmd_sim = float(1.0 / (1.0 + max(0.0, mmd2)))
    
    return {'procrustes_sim': proc_sim, 'mmd_sim': mmd_sim}


def get_chunked_embeddings(paired_df: pd.DataFrame, chunk_size: int = 5) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generates 2D embeddings of short movement chunks for diagnostics."""
    try:
        try:
            import umap
            reducer = umap.UMAP(n_components=2, min_dist=0.1, random_state=0)
        except ImportError:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            
        def build_features(df, label):
            try:
                center = df[[f'CENTER_X_{label}', f'CENTER_Y_{label}']].to_numpy()
                vel = np.diff(center, axis=0)
                vel_pad = np.vstack([vel[:1], vel]) if len(vel) else np.zeros_like(center)
                nose = df[[f'NOSE_X_{label}', f'NOSE_Y_{label}']].to_numpy()
                tail = df[[f'TAIL_BASE_X_{label}', f'TAIL_BASE_Y_{label}']].to_numpy()
                axis_vec = nose - tail
                ears = df[[f'LEFT_EAR_X_{label}', f'LEFT_EAR_Y_{label}', f'RIGHT_EAR_X_{label}', f'RIGHT_EAR_Y_{label}']].to_numpy()
                ear_span = np.linalg.norm(ears[:, :2] - ears[:, 2:], axis=1, keepdims=True)
                return np.hstack([vel_pad, axis_vec, ear_span])
            except KeyError:
                return np.empty((0, 5))

        chunks, labels, seq_ids = [], [], []
        for seq, seq_df in paired_df.sort_values('itemPosition').groupby('sequenceId'):
            gt_feats = build_features(seq_df, 'gt')
            inf_feats = build_features(seq_df, 'inf')
            if len(gt_feats) < chunk_size or len(inf_feats) < chunk_size:
                continue
            mask_gt = ~np.isnan(gt_feats).any(axis=1)
            mask_inf = ~np.isnan(inf_feats).any(axis=1)
            gt_feats, inf_feats = gt_feats[mask_gt], inf_feats[mask_inf]
            
            if len(gt_feats) < chunk_size or len(inf_feats) < chunk_size:
                continue
            
            for arr, lbl in [(gt_feats, 'gt'), (inf_feats, 'inf')]:
                for start in range(0, len(arr) - chunk_size + 1, chunk_size):
                    chunks.append(arr[start:start + chunk_size].flatten())
                    labels.append(lbl)
                    seq_ids.append(seq)
            
            shuffled = gt_feats.copy()
            rng = np.random.default_rng(0)
            rng.shuffle(shuffled)
            for start in range(0, len(shuffled) - chunk_size + 1, chunk_size):
                chunks.append(shuffled[start:start + chunk_size].flatten())
                labels.append('gt_shuffled')
                seq_ids.append(seq)

        if not chunks:
            return None
            
        X = np.vstack(chunks)
        emb = reducer.fit_transform(X) if len(X) > 5 else np.zeros((len(X), 2))
        return emb, np.array(labels), np.array(seq_ids)
    except Exception:
        return None

# ---------------------------------------------------------
# Master Metric Aggregator
# ---------------------------------------------------------
def compute_etho_scores(
    gt_dir: Optional[Path] = None, 
    inf_dir: Optional[Path] = None, 
    *, 
    paired_df: Optional[pd.DataFrame] = None
) -> Tuple[Dict[str, float], Dict[str, List[float]], Dict[str, float], Dict[str, float]]:
    
    # Avoid redundant loading if already passed from run_etho_full_analysis
    if paired_df is None:
        if gt_dir is None or inf_dir is None:
            raise ValueError("Must provide either paired_df, or both gt_dir and inf_dir.")
        gt_df, inf_df = load_gt_and_preds(gt_dir, inf_dir)
        paired_df = merge_paired(gt_df, inf_df, list(gt_df.columns))

    pos_res = position_kl_score(paired_df)
    stat_res = stationary_score(paired_df)
    vel_res = velocity_distribution_score(paired_df)
    acc_res = acceleration_distribution_score(paired_df)
    dir_res = direction_score(paired_df)
    quad_res = quadrant_score(paired_df)
    syll_res = syllable_score(paired_df)
    traj_res = trajectory_shape_score(paired_df)

    # Calculate memory-optimized DTW similarity
    dtw_score, dtw_seq_scores = dtw_trajectory_similarity(paired_df)
    
    # Calculate memory-optimized Manifold similarities
    manifold_res = manifold_alignment_metrics(paired_df)

    scores = {
        "position_kl_score": float(pos_res[0]) if np.isfinite(pos_res[0]) else np.nan,
        "stationary_score": float(stat_res[0]) if np.isfinite(stat_res[0]) else np.nan,
        "velocity_score": float(vel_res[0]) if np.isfinite(vel_res[0]) else np.nan,
        "acceleration_score": float(acc_res[0]) if np.isfinite(acc_res[0]) else np.nan,
        "direction_score": float(dir_res[0]) if np.isfinite(dir_res[0]) else np.nan,
        "quadrant_score": float(quad_res[0]) if np.isfinite(quad_res[0]) else np.nan,
        "syllable_score": float(syll_res[0]) if np.isfinite(syll_res[0]) else np.nan,
        "trajectory_shape_score": float(traj_res[0]) if np.isfinite(traj_res[0]) else np.nan,
        "dtw_similarity_score": dtw_score,
        "procrustes_similarity": manifold_res['procrustes_sim'],
        "mmd_similarity": manifold_res['mmd_sim']
    }

    # Only core structural metrics go into composite logic
    core_scores = {k: v for k, v in scores.items() if k not in ["procrustes_similarity", "mmd_similarity", "dtw_similarity_score"]}
    scores["composite_score"] = _geometric_mean_scores(list(core_scores.values()))

    all_seq_dicts = {
        "position_kl_score": pos_res[1],
        "stationary_score": stat_res[1],
        "velocity_score": vel_res[1],
        "acceleration_score": acc_res[1],
        "direction_score": dir_res[1],
        "quadrant_score": quad_res[1],
        "syllable_score": syll_res[1],
        "trajectory_shape_score": traj_res[1],
        "dtw_similarity_score": dtw_seq_scores,
    }
    
    # Generate weights dictionary (for DTW, default to 1 since it's sequence-level)
    all_weight_dicts = {
        "position_kl_score": pos_res[2],
        "stationary_score": stat_res[2],
        "velocity_score": vel_res[2],
        "acceleration_score": acc_res[2],
        "direction_score": dir_res[2],
        "quadrant_score": quad_res[2],
        "syllable_score": syll_res[2],
        "trajectory_shape_score": traj_res[2],
        "dtw_similarity_score": {seq: 1 for seq in dtw_seq_scores}, 
    }

    all_keys = set()
    for d in all_seq_dicts.values():
        all_keys.update(d.keys())
    sorted_seq_ids = sorted(list(all_keys))

    sequence_level_scores = {"sequence_length": [int(v) for v in paired_df.groupby("sequenceId").size().reindex(sorted_seq_ids, fill_value=0)]}
    sequence_means = {}
    sequence_stds = {}

    for metric_name in all_seq_dicts.keys():
        d_scores = all_seq_dicts[metric_name]
        d_weights = all_weight_dicts[metric_name]
        sequence_level_scores[metric_name] = [
            float(d_scores[k]) if k in d_scores and np.isfinite(d_scores[k]) else np.nan
            for k in sorted_seq_ids
        ]
        vals = []
        wts = []
        for k in d_scores.keys():
            if k in d_weights and np.isfinite(d_scores[k]):
                vals.append(d_scores[k])
                wts.append(d_weights[k])
        if vals and sum(wts) > 0:
            weighted_mean = np.average(vals, weights=wts)
            sequence_means[metric_name] = float(weighted_mean)
            variance = np.average((np.array(vals) - weighted_mean) ** 2, weights=wts)
            sequence_stds[metric_name] = float(np.sqrt(variance))
        else:
            sequence_means[metric_name] = np.nan
            sequence_stds[metric_name] = np.nan

    return scores, sequence_level_scores, sequence_means, sequence_stds


def run_etho_full_analysis(gt_dir: Path, inf_dir: Path, *, output_root: Path | None = None) -> Path:
    """
    Executes the full behavioral analysis pipeline headlessly, saves metrics,
    and generates visualization figures directly into the specified output directory.
    """
    import json
    try:
        from nethobench.analysis.etho_reporting import generate_full_etho_report
    except ImportError:
        def generate_full_etho_report(*args, **kwargs):
            pass # Failsafe if reporting module has not been fully implemented yet

    outdir = _timestamped_outdir(output_root, prefix="etho-analysis")
    
    # 1. Load Data
    gt_df, inf_df = load_gt_and_preds(gt_dir, inf_dir)
    paired_df = merge_paired(gt_df, inf_df, list(gt_df.columns))

    # 2. Extract Additional Features for the Report
    coord_pairs = []
    for col in gt_df.columns:
        if col.endswith('_X'):
            base = col[:-2]
            if f'{base}_Y' in gt_df.columns:
                coord_pairs.append((base, col, f'{base}_Y'))
                
    errors_df = body_part_errors(paired_df, coord_pairs)
    pairs_to_check = [('NOSE', 'TAIL_BASE'), ('LEFT_EAR', 'RIGHT_EAR'), ('NOSE', 'CENTER')]
    distances_df = inter_limb_distances(paired_df, pairs_to_check)
    chunk_data = get_chunked_embeddings(paired_df, chunk_size=5)

    # 3. Compute Structural Scores (Pass paired_df to avoid double-loading!)
    scores, seq_scores, seq_means, seq_stds = compute_etho_scores(paired_df=paired_df)

    # 4. Save JSON
    payload = {
        "global_scores": scores,
        "per_sequence": seq_scores,
        "per_sequence_mean": seq_means,
        "per_sequence_std": seq_stds,
    }
    with (outdir / "scores.json").open("w") as f:
        json.dump(payload, f, indent=2)

    # 5. Generate Matplotlib Figures
    generate_full_etho_report(gt_df, inf_df, paired_df, errors_df, distances_df, chunk_data, outdir)
    
    return outdir

def _timestamped_outdir(base: Path | None = None, prefix: str = "ethobench") -> Path:
    base = Path(base) if base is not None else Path.cwd() / "outputs"
    outdir = base / f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

