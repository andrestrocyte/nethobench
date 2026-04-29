from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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


def compute_etho_scores(gt_dir: Path, inf_dir: Path) -> Tuple[Dict[str, float], Dict[str, List[float]], Dict[str, float], Dict[str, float]]:


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

    scores = {
        "position_kl_score": float(pos_res[0]) if np.isfinite(pos_res[0]) else np.nan,
        "stationary_score": float(stat_res[0]) if np.isfinite(stat_res[0]) else np.nan,
        "velocity_score": float(vel_res[0]) if np.isfinite(vel_res[0]) else np.nan,
        "acceleration_score": float(acc_res[0]) if np.isfinite(acc_res[0]) else np.nan,
        "direction_score": float(dir_res[0]) if np.isfinite(dir_res[0]) else np.nan,
        "quadrant_score": float(quad_res[0]) if np.isfinite(quad_res[0]) else np.nan,
        "syllable_score": float(syll_res[0]) if np.isfinite(syll_res[0]) else np.nan,
        "trajectory_shape_score": float(traj_res[0]) if np.isfinite(traj_res[0]) else np.nan,
    }

    scores["composite_score"] = _geometric_mean_scores(list(scores.values()))

    all_seq_dicts = {
        "position_kl_score": pos_res[1],
        "stationary_score": stat_res[1],
        "velocity_score": vel_res[1],
        "acceleration_score": acc_res[1],
        "direction_score": dir_res[1],
        "quadrant_score": quad_res[1],
        "syllable_score": syll_res[1],
        "trajectory_shape_score": traj_res[1],
    }
    all_weight_dicts = {
        "position_kl_score": pos_res[2],
        "stationary_score": stat_res[2],
        "velocity_score": vel_res[2],
        "acceleration_score": acc_res[2],
        "direction_score": dir_res[2],
        "quadrant_score": quad_res[2],
        "syllable_score": syll_res[2],
        "trajectory_shape_score": traj_res[2],
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


def _timestamped_outdir(base: Path | None = None, prefix: str = "ethobench") -> Path:
    base = Path(base) if base is not None else Path.cwd() / "outputs"
    outdir = base / f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def run_ethobench_notebook(gt_dir: Path, inf_dir: Path, *, output_root: Path | None = None) -> Path:
    """
    Execute the bundled ethobench notebook headlessly, saving figures to disk and
    returning the output directory.
    """
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    nb_path = Path(__file__).parent / "notebooks" / "behavior_metrics.ipynb"
    if not nb_path.is_file():
        raise FileNotFoundError(
            f"Bundled ethobench notebook not found at {nb_path}. "
            "Copy your notebook there or drop --run-notebook."
        )
    outdir = _timestamped_outdir(output_root, prefix="ethobench-analysis")

    nb = nbformat.read(nb_path, as_version=4)
    patch = f"""
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
outdir = Path(r"{outdir}")
outdir.mkdir(parents=True, exist_ok=True)
plot_counter = {{'n': 0}}
orig_show = plt.show

def saving_show(*args, **kwargs):
    figs = [plt.figure(num) for num in plt.get_fignums()]
    for fig in figs:
        plot_counter['n'] += 1
        fig.savefig(outdir / f"figure_{{plot_counter['n']:03d}}.png", dpi=200, bbox_inches='tight')
    plt.close('all')

plt.show = saving_show
gt_dir = Path(r"{gt_dir}")
inf_dir = Path(r"{inf_dir}")
"""
    nb.cells.insert(0, nbformat.v4.new_code_cell(patch))

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {'metadata': {'path': nb_path.parent}})
    executed_path = outdir / "executed_notebook.ipynb"
    with executed_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return outdir
