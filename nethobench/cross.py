from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nbformat
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .neuro import _compute_scores_from_arrays
from .etho import (
    position_kl_score,
    quadrant_score,
    stationary_score,
    velocity_distribution_score,
    acceleration_distribution_score,
    direction_score,
    syllable_score,
    trajectory_shape_score,
)


def _clip01(x: float, eps: float = 1e-6) -> float:
    return float(np.clip(x, eps, 1.0 - eps))


def _geometric_mean_scores(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    arr = np.asarray([_clip01(v) for v in arr], dtype=np.float64)
    return float(np.exp(np.mean(np.log(arr))))


def _load_config(config: Union[Path, dict, None], sample_df: pd.DataFrame | None = None) -> dict:
    if isinstance(config, dict):
        cfg = dict(config)
    elif config is None:
        if sample_df is None:
            raise ValueError("Cannot infer config without a DataFrame.")
        cfg = _infer_config(sample_df)
    else:
        cfg = json.loads(Path(config).read_text())
    cfg.setdefault("sequence_key", "sequenceId")
    cfg.setdefault("time_key", "itemPosition")
    cfg.setdefault("behavior_parts", ["CENTER", "NOSE", "TAIL_BASE"])
    cfg.setdefault("body_axis", ["NOSE", "TAIL_BASE"])
    return cfg


def _infer_config(df: pd.DataFrame) -> dict:
    seq_key = "sequenceId"
    time_key = "itemPosition"
    behavior_parts = []
    for col in df.columns:
        if col.endswith("_X"):
            base = col[:-2]
            if f"{base}_Y" in df.columns:
                behavior_parts.append(base)
    behavior_parts = sorted(set(behavior_parts))
    neuro_cols = [
        c for c in df.columns
        if c not in {seq_key, time_key}
        and not (c.endswith("_X") or c.endswith("_Y"))
    ]
    return {
        "sequence_key": seq_key,
        "time_key": time_key,
        "behavior_parts": behavior_parts,
        "neuro_cols": neuro_cols,
        "body_axis": ["NOSE", "TAIL_BASE"] if "NOSE_X" in df and "TAIL_BASE_X" in df else behavior_parts[:2],
    }


def _merge_aligned(gt_path: Path, pred_path: Path, cfg: dict) -> pd.DataFrame:
    seq_key = cfg.get("sequence_key", "sequenceId")
    time_key = cfg.get("time_key", "itemPosition")
    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)
    for col in [seq_key, time_key]:
        if col not in gt.columns or col not in pred.columns:
            raise ValueError(f"Missing alignment column {col} in GT or predictions.")
    merged = pd.merge(
        gt.sort_values([seq_key, time_key]),
        pred.sort_values([seq_key, time_key]),
        on=[seq_key, time_key],
        suffixes=("_gt", "_inf"),
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping sequence/time rows after merge.")
    return merged


def _arrays_from_aligned(aligned: pd.DataFrame, cols: list[str], suffix: str, cfg: dict) -> np.ndarray:
    seq_key = cfg.get("sequence_key", "sequenceId")
    time_key = cfg.get("time_key", "itemPosition")
    seq_arrays = []
    for _, sdf in aligned.sort_values([seq_key, time_key]).groupby(seq_key):
        cols_suff = [f"{c}_{suffix}" for c in cols]
        if not set(cols_suff).issubset(sdf.columns):
            missing = set(cols_suff) - set(sdf.columns)
            raise ValueError(f"Missing columns for modality {suffix}: {missing}")
        seq_arrays.append(sdf[cols_suff].to_numpy())
    return np.stack(seq_arrays, axis=0)


def _behavior_feature_matrix(aligned: pd.DataFrame, cfg: dict, suffix: str) -> np.ndarray:
    seq_key = cfg.get("sequence_key", "sequenceId")
    time_key = cfg.get("time_key", "itemPosition")
    center = cfg.get("center_part", "CENTER")
    axis = cfg.get("body_axis", ["NOSE", "TAIL_BASE"])
    nose, tail = axis if len(axis) == 2 else ("NOSE", "TAIL_BASE")

    feats = []
    for _, sdf in aligned.sort_values([seq_key, time_key]).groupby(seq_key):
        cx = sdf[f"{center}_X_{suffix}"].to_numpy()
        cy = sdf[f"{center}_Y_{suffix}"].to_numpy()
        coords = np.stack([cx, cy], axis=1)
        speed = np.concatenate([[0.0], np.linalg.norm(np.diff(coords, axis=0), axis=1)])
        heading_cos = np.zeros_like(speed)
        heading_sin = np.zeros_like(speed)
        nose_x_col = f"{nose}_X_{suffix}"
        tail_x_col = f"{tail}_X_{suffix}"
        if nose_x_col in sdf.columns and tail_x_col in sdf.columns:
            nose_xy = sdf[[f"{nose}_X_{suffix}", f"{nose}_Y_{suffix}"]].to_numpy()
            tail_xy = sdf[[f"{tail}_X_{suffix}", f"{tail}_Y_{suffix}"]].to_numpy()
            axis_vec = nose_xy - tail_xy
            norm = np.linalg.norm(axis_vec, axis=1) + 1e-8
            heading_cos = axis_vec[:, 0] / norm
            heading_sin = axis_vec[:, 1] / norm
        seq_feats = np.stack([cx, cy, speed, heading_cos, heading_sin], axis=1)
        feats.append(seq_feats)
    return np.vstack(feats)


def _behavior_feature_sequences(aligned: pd.DataFrame, cfg: dict, suffix: str) -> list[np.ndarray]:
    seq_key = cfg.get("sequence_key", "sequenceId")
    time_key = cfg.get("time_key", "itemPosition")
    center = cfg.get("center_part", "CENTER")
    axis = cfg.get("body_axis", ["NOSE", "TAIL_BASE"])
    nose, tail = axis if len(axis) == 2 else ("NOSE", "TAIL_BASE")

    out: list[np.ndarray] = []
    for _, sdf in aligned.sort_values([seq_key, time_key]).groupby(seq_key):
        cx = sdf[f"{center}_X_{suffix}"].to_numpy()
        cy = sdf[f"{center}_Y_{suffix}"].to_numpy()
        coords = np.stack([cx, cy], axis=1)
        speed = np.concatenate([[0.0], np.linalg.norm(np.diff(coords, axis=0), axis=1)])
        heading_cos = np.zeros_like(speed)
        heading_sin = np.zeros_like(speed)
        nose_x_col = f"{nose}_X_{suffix}"
        tail_x_col = f"{tail}_X_{suffix}"
        if nose_x_col in sdf.columns and tail_x_col in sdf.columns:
            nose_xy = sdf[[f"{nose}_X_{suffix}", f"{nose}_Y_{suffix}"]].to_numpy()
            tail_xy = sdf[[f"{tail}_X_{suffix}", f"{tail}_Y_{suffix}"]].to_numpy()
            axis_vec = nose_xy - tail_xy
            norm = np.linalg.norm(axis_vec, axis=1) + 1e-8
            heading_cos = axis_vec[:, 0] / norm
            heading_sin = axis_vec[:, 1] / norm
        out.append(np.stack([cx, cy, speed, heading_cos, heading_sin], axis=1))
    return out


def _neural_feature_matrix(aligned: pd.DataFrame, neuro_cols: list[str], suffix: str, cfg: dict) -> np.ndarray:
    arr = _arrays_from_aligned(aligned, neuro_cols, suffix, cfg)
    return arr.reshape(-1, arr.shape[-1])


def _neural_feature_sequences(aligned: pd.DataFrame, neuro_cols: list[str], suffix: str, cfg: dict) -> list[np.ndarray]:
    arr = _arrays_from_aligned(aligned, neuro_cols, suffix, cfg)
    return [arr[i] for i in range(arr.shape[0])]


def _cca_mean(X: np.ndarray, Y: np.ndarray, n_components: int = 5) -> float:
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    Xs = scaler_x.fit_transform(X)
    Ys = scaler_y.fit_transform(Y)
    n_comp = max(1, min(n_components, Xs.shape[1], Ys.shape[1]))
    cca = CCA(n_components=n_comp, max_iter=1000)
    X_c, Y_c = cca.fit_transform(Xs, Ys)
    corrs = []
    for i in range(X_c.shape[1]):
        xi = X_c[:, i]
        yi = Y_c[:, i]
        if np.allclose(np.std(xi), 0) or np.allclose(np.std(yi), 0):
            continue
        corrs.append(np.corrcoef(xi, yi)[0, 1])
    return float(np.mean(corrs)) if corrs else np.nan


def _stack_sequences(sequences: list[np.ndarray]) -> np.ndarray:
    if not sequences:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(sequences)


def _predictive_r2(X_sequences: list[np.ndarray], Y_sequences: list[np.ndarray], test_frac: float = 0.2, seed: int = 0) -> float:
    n = min(len(X_sequences), len(Y_sequences))
    if n < 2:
        return np.nan
    pairs = []
    for X, Y in zip(X_sequences[:n], Y_sequences[:n]):
        if X.shape[0] == 0 or Y.shape[0] == 0:
            continue
        m = min(X.shape[0], Y.shape[0])
        pairs.append((X[:m], Y[:m]))
    if len(pairs) < 2:
        return np.nan

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    split = int(round(len(pairs) * (1 - test_frac)))
    split = int(np.clip(split, 1, len(pairs) - 1))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train = _stack_sequences([pairs[i][0] for i in train_idx])
    Y_train = _stack_sequences([pairs[i][1] for i in train_idx])
    X_test = _stack_sequences([pairs[i][0] for i in test_idx])
    Y_test = _stack_sequences([pairs[i][1] for i in test_idx])
    if X_train.size == 0 or Y_train.size == 0 or X_test.size == 0 or Y_test.size == 0:
        return np.nan

    reg = LinearRegression()
    reg.fit(X_train, Y_train)
    y_pred = reg.predict(X_test)
    ss_res = np.sum((Y_test - y_pred) ** 2)
    ss_tot = np.sum((Y_test - Y_test.mean(axis=0)) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return float(r2)


def _lead_lag_peak(neural_sequences: list[np.ndarray], behavior_sequences: list[np.ndarray], max_lag: int = 30) -> int:
    lags = np.arange(-max_lag, max_lag + 1)
    seq_cors = []
    for neural_arr, behavior_arr in zip(neural_sequences, behavior_sequences):
        if neural_arr.shape[0] != behavior_arr.shape[0] or neural_arr.shape[0] < max(5, max_lag + 3):
            continue
        X = neural_arr - neural_arr.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        pc1 = (X @ Vt.T)[:, 0] if Vt.shape[0] > 0 else X.mean(axis=1)
        pc1 = (pc1 - pc1.mean()) / (pc1.std() + 1e-9)
        b = (behavior_arr - behavior_arr.mean()) / (behavior_arr.std() + 1e-9)
        cors = []
        for lag in lags:
            if lag < 0:
                a = pc1[:lag]
                bb = b[-lag:]
            elif lag > 0:
                a = pc1[lag:]
                bb = b[:-lag]
            else:
                a = pc1
                bb = b
            if a.size < 5 or bb.size < 5:
                cors.append(np.nan)
            else:
                cors.append(np.corrcoef(a, bb)[0, 1])
        seq_cors.append(np.asarray(cors, dtype=np.float64))
    if not seq_cors:
        return 0
    cors = np.nanmean(np.vstack(seq_cors), axis=0)
    if not np.isfinite(cors).any():
        return 0
    idx = int(np.nanargmax(np.abs(cors)))
    return int(lags[idx])


def _speed_from_behavior(aligned: pd.DataFrame, cfg: dict, suffix: str) -> Tuple[list[np.ndarray], list[int]]:
    seq_key = cfg.get("sequence_key", "sequenceId")
    time_key = cfg.get("time_key", "itemPosition")
    center = cfg.get("center_part", "CENTER")
    speeds: list[np.ndarray] = []
    lengths = []
    for _, sdf in aligned.sort_values([seq_key, time_key]).groupby(seq_key):
        coords = sdf[[f"{center}_X_{suffix}", f"{center}_Y_{suffix}"]].to_numpy()
        sp = np.concatenate([[0.0], np.linalg.norm(np.diff(coords, axis=0), axis=1)])
        speeds.append(sp)
        lengths.append(len(sp))
    return speeds, lengths


def compute_cross_scores(predictions_csv: Path, ground_truth_csv: Path, config: Union[Path, dict, None]) -> Dict[str, object]:
    # Read GT once for config inference if needed
    sample_df = pd.read_csv(ground_truth_csv)
    cfg = _load_config(config, sample_df=sample_df)
    aligned = _merge_aligned(ground_truth_csv, predictions_csv, cfg)
    neuro_cols = cfg.get("neuro_cols")
    if not neuro_cols:
        raise ValueError("Config must include neuro_cols for cross-scores.")

    # --- Neuro axis ---
    gt_neuro = _arrays_from_aligned(aligned, neuro_cols, "gt", cfg)
    pr_neuro = _arrays_from_aligned(aligned, neuro_cols, "inf", cfg)
    neuro_scores = _compute_scores_from_arrays(gt_neuro, pr_neuro, region_names=neuro_cols)

    # --- Behavior axis (reuse ethobench metrics on merged df) ---
    beh_scores = {}
    beh_scores["position_kl_score"] = position_kl_score(aligned)[0]
    beh_scores["quadrant_score"] = quadrant_score(aligned)[0]
    beh_scores["stationary_score"] = stationary_score(aligned)[0]
    beh_scores["velocity_score"] = velocity_distribution_score(aligned)[0]
    beh_scores["acceleration_score"] = acceleration_distribution_score(aligned)[0]
    beh_scores["direction_score"] = direction_score(aligned, nose_label=cfg.get("body_axis", ["NOSE", "TAIL_BASE"])[0], tail_label=cfg.get("body_axis", ["NOSE", "TAIL_BASE"])[-1])[0]
    beh_scores["syllable_score"] = syllable_score(aligned)[0]
    beh_scores["trajectory_shape_score"] = trajectory_shape_score(aligned)[0]
    beh_scores["composite_score"] = _geometric_mean_scores(list(beh_scores.values()))

    # --- Cross-modal axis ---
    neuro_gt_flat = _neural_feature_matrix(aligned, neuro_cols, "gt", cfg)
    neuro_pr_flat = _neural_feature_matrix(aligned, neuro_cols, "inf", cfg)
    beh_gt_flat = _behavior_feature_matrix(aligned, cfg, "gt")
    beh_pr_flat = _behavior_feature_matrix(aligned, cfg, "inf")
    neuro_gt_seq = _neural_feature_sequences(aligned, neuro_cols, "gt", cfg)
    neuro_pr_seq = _neural_feature_sequences(aligned, neuro_cols, "inf", cfg)
    beh_gt_seq = _behavior_feature_sequences(aligned, cfg, "gt")
    beh_pr_seq = _behavior_feature_sequences(aligned, cfg, "inf")

    cca_gt = _cca_mean(neuro_gt_flat, beh_gt_flat)
    cca_pr = _cca_mean(neuro_pr_flat, beh_pr_flat)
    cca_alignment_score = float(1.0 - min(1.0, abs(cca_gt - cca_pr))) if np.isfinite(cca_gt) and np.isfinite(cca_pr) else np.nan

    r2_n2b_gt = _predictive_r2(neuro_gt_seq, beh_gt_seq)
    r2_n2b_pr = _predictive_r2(neuro_pr_seq, beh_pr_seq)
    n2b_similarity = float(1.0 / (1.0 + abs(r2_n2b_gt - r2_n2b_pr))) if np.isfinite(r2_n2b_gt) and np.isfinite(r2_n2b_pr) else np.nan

    r2_b2n_gt = _predictive_r2(beh_gt_seq, neuro_gt_seq)
    r2_b2n_pr = _predictive_r2(beh_pr_seq, neuro_pr_seq)
    b2n_similarity = float(1.0 / (1.0 + abs(r2_b2n_gt - r2_b2n_pr))) if np.isfinite(r2_b2n_gt) and np.isfinite(r2_b2n_pr) else np.nan

    speed_gt, _ = _speed_from_behavior(aligned, cfg, "gt")
    speed_pr, _ = _speed_from_behavior(aligned, cfg, "inf")
    lag_gt = _lead_lag_peak(neuro_gt_seq, speed_gt)
    lag_pr = _lead_lag_peak(neuro_pr_seq, speed_pr)
    lead_lag_score = 1.0 - min(1.0, abs(lag_gt - lag_pr) / 30.0) if (lag_gt == lag_gt) and (lag_pr == lag_pr) else np.nan

    cross_scores = {
        "cca_gt": cca_gt,
        "cca_pred": cca_pr,
        "cca_alignment_score": cca_alignment_score,
        "r2_neural_to_behavior_gt": r2_n2b_gt,
        "r2_neural_to_behavior_pred": r2_n2b_pr,
        "neural_to_behavior_similarity": n2b_similarity,
        "r2_behavior_to_neural_gt": r2_b2n_gt,
        "r2_behavior_to_neural_pred": r2_b2n_pr,
        "behavior_to_neural_similarity": b2n_similarity,
        "lead_lag_gt": lag_gt,
        "lead_lag_pred": lag_pr,
        "lead_lag_score": lead_lag_score,
    }

    cross_vals = [v for k, v in cross_scores.items() if k.endswith("score") or k.endswith("similarity")]
    cross_vals = [v for v in cross_vals if np.isfinite(v)]
    cross_composite = float(np.prod(cross_vals) ** (1 / len(cross_vals))) if cross_vals else np.nan
    cross_scores["cross_composite"] = cross_composite

    neuro_composite = neuro_scores.get("composite_score", np.nan)
    etho_composite = beh_scores.get("composite_score", np.nan)
    comps = [v for v in [neuro_composite, etho_composite, cross_composite] if np.isfinite(v)]
    final_comp = float(np.mean(comps)) if comps else np.nan

    return {
        "neuro_scores": neuro_scores,
        "behavior_scores": beh_scores,
        "cross_scores": cross_scores,
        "neuro_composite": neuro_composite,
        "etho_composite": etho_composite,
        "cross_composite": cross_composite,
        "composite": final_comp,
    }


def run_cross_full_analysis(
    predictions_csv: Path,
    ground_truth_csv: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> Path:
    """
    Execute the bundled cross-modal notebook headlessly, saving figures + executed notebook.
    """
    from nbconvert.preprocessors import ExecutePreprocessor

    nb_path = Path(__file__).parent / "notebooks" / "cross_modal_metrics.ipynb"
    if not nb_path.is_file():
        raise FileNotFoundError(f"Cross-modal notebook missing at {nb_path}")

    outdir_root = Path(output_root) if output_root else Path.cwd() / "outputs"
    outdir = outdir_root / f"cross-analysis-{Path(predictions_csv).stem}"
    outdir.mkdir(parents=True, exist_ok=True)

    nb = nbformat.read(nb_path, as_version=4)
    patch = f"""
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
gt_path = Path(r"{ground_truth_csv}")
pred_path = Path(r"{predictions_csv}")
config_path = Path(r"{config_path}")
outdir = Path(r"{outdir}")
plot_counter = {{'n': 0}}
orig_show = plt.show
def saving_show(*args, **kwargs):
    figs = [plt.figure(num) for num in plt.get_fignums()]
    for fig in figs:
        plot_counter['n'] += 1
        fig.savefig(outdir / f"figure_{{plot_counter['n']:03d}}.png", dpi=200, bbox_inches='tight')
    plt.close('all')
plt.show = saving_show
"""
    nb.cells.insert(0, nbformat.v4.new_code_cell(patch))

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {'metadata': {'path': nb_path.parent}})
    executed_path = outdir / "executed_cross_modal.ipynb"
    with executed_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return outdir
