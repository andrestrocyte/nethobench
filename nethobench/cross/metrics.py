from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from nethobench.utils.helpers import _clip01, _geometric_mean_scores
from nethobench.utils.calculation import _merge_aligned
from nethobench.neuro.pipeline import load_and_run_neuro_full_analysis
from nethobench.etho.metrics import (
    position_kl_score,
    quadrant_score,
    stationary_score,
    velocity_distribution_score,
    acceleration_distribution_score,
    direction_score,
    syllable_score,
    trajectory_shape_score,
)



def _load_config(
    config: Union[Path, dict, None], sample_df: pd.DataFrame | None = None
) -> dict:
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
        c
        for c in df.columns
        if c not in {seq_key, time_key} and not (c.endswith("_X") or c.endswith("_Y"))
    ]
    return {
        "sequence_key": seq_key,
        "time_key": time_key,
        "behavior_parts": behavior_parts,
        "neuro_cols": neuro_cols,
        "body_axis": (
            ["NOSE", "TAIL_BASE"]
            if "NOSE_X" in df and "TAIL_BASE_X" in df
            else behavior_parts[:2]
        ),
    }




def _arrays_from_aligned(
    aligned: pd.DataFrame, cols: list[str], suffix: str, cfg: dict
) -> np.ndarray:
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


def _behavior_feature_matrix(
    aligned: pd.DataFrame, cfg: dict, suffix: str
) -> np.ndarray:
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


def _behavior_feature_sequences(
    aligned: pd.DataFrame, cfg: dict, suffix: str
) -> list[np.ndarray]:
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


def _neural_feature_matrix(
    aligned: pd.DataFrame, neuro_cols: list[str], suffix: str, cfg: dict
) -> np.ndarray:
    arr = _arrays_from_aligned(aligned, neuro_cols, suffix, cfg)
    return arr.reshape(-1, arr.shape[-1])


def _neural_feature_sequences(
    aligned: pd.DataFrame, neuro_cols: list[str], suffix: str, cfg: dict
) -> list[np.ndarray]:
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


def _predictive_r2(
    X_sequences: list[np.ndarray],
    Y_sequences: list[np.ndarray],
    test_frac: float = 0.2,
    seed: int = 0,
) -> float:
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


def _lead_lag_peak(
    neural_sequences: list[np.ndarray],
    behavior_sequences: list[np.ndarray],
    max_lag: int = 30,
) -> int:
    lags = np.arange(-max_lag, max_lag + 1)
    seq_cors = []
    for neural_arr, behavior_arr in zip(neural_sequences, behavior_sequences):
        if neural_arr.shape[0] != behavior_arr.shape[0] or neural_arr.shape[0] < max(
            5, max_lag + 3
        ):
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


def _speed_from_behavior(
    aligned: pd.DataFrame, cfg: dict, suffix: str
) -> Tuple[list[np.ndarray], list[int]]:
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
