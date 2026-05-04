import pandas as pd
from pathlib import Path
import numpy as np
import contextlib
import io
import tempfile
from typing import Mapping

from nethobench.utils.validation import (
    validate_dataframe_schema,
    validate_alignment_overlap,
)
from nethobench.utils.evaluation_constants import (
    MIN_ITEMS_IQR_ROBUST_FALLBACK,
    QUANTILE_IQR_LO,
    QUANTILE_IQR_HI,
)


EPS = 1e-8


def merge_aligned(gt, pred, cfg: dict) -> pd.DataFrame:
    seq_key = cfg.get("sequence_key", "sequenceId")
    time_key = cfg.get("time_key", "itemPosition")
    gt_df = gt if isinstance(gt, pd.DataFrame) else pd.read_csv(gt)
    pred_df = pred if isinstance(pred, pd.DataFrame) else pd.read_csv(pred)

    # --- Validation: Schema & Index Integrity ---
    validate_dataframe_schema(gt_df, seq_key=seq_key, time_key=time_key)
    validate_dataframe_schema(pred_df, seq_key=seq_key, time_key=time_key)

    for col in [seq_key, time_key]:
        if col not in gt_df.columns or col not in pred_df.columns:
            raise ValueError(f"Missing alignment column {col} in GT or predictions.")
    merged = pd.merge(
        gt_df.sort_values([seq_key, time_key]),
        pred_df.sort_values([seq_key, time_key]),
        on=[seq_key, time_key],
        suffixes=("_gt", "_inf"),
        how="inner",
    )

    # --- Validation: Alignment & Overlap Guards ---
    validate_alignment_overlap(
        gt_df, pred_df, merged, seq_key=seq_key, time_key=time_key
    )

    return merged


def weighted_mean_available(
    values: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    keys = [
        k for k, v in values.items() if np.isfinite(v) and weights.get(k, 0.0) > 0.0
    ]
    if not keys:
        return np.nan
    denom = float(np.sum([weights[k] for k in keys]))
    if denom <= 0.0:
        return np.nan
    return float(np.sum([weights[k] * float(values[k]) for k in keys]) / denom)




def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    q25, q75 = np.quantile(values, [QUANTILE_IQR_LO, QUANTILE_IQR_HI])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(np.nanmean(np.abs(values)))
    if not np.isfinite(scale) or scale < 1e-9:
        return 1.0
    return scale + EPS


def correlation_score(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 3:
        return np.nan
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) < EPS or np.std(bb) < EPS:
        return np.nan
    return float(np.clip((np.corrcoef(aa, bb)[0, 1] + 1.0) / 2.0, 0.0, 1.0))


def rmse_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    err = float(np.sqrt(np.mean((x[mask] - y[mask]) ** 2)))
    scale = float(np.nanstd(x[mask]))
    if not np.isfinite(scale) or scale < EPS:
        scale = float(np.nanmean(np.abs(x[mask])))
    if not np.isfinite(scale) or scale < EPS:
        scale = 1.0
    return float(1.0 / (1.0 + err / scale))


def align_arrays(
    gt_arr: np.ndarray, pred_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    gt = np.asarray(gt_arr, dtype=np.float64)
    pred = np.asarray(pred_arr, dtype=np.float64)
    
    if gt.ndim != 3 or pred.ndim != 3:
        raise ValueError(f"Expected [n_seq, T, n_reg] arrays, got {gt.shape} and {pred.shape}")
    
    # We no longer need the downsampling/truncation math here, 
    # just an assertion that the inner join worked correctly.
    if gt.shape != pred.shape:
        raise ValueError(f"GT/pred arrays must have identical shapes post-merge: {gt.shape} vs {pred.shape}")
        
    return gt, pred






def dataset_to_sequence_frame(arr: np.ndarray, region_names: list[str]) -> pd.DataFrame:
    """Converts a 3D [seq, time, region] array to a flat Nethobench-aligned DataFrame."""
    arr = np.asarray(arr, dtype=np.float64)
    n_seq, n_time, n_regions = arr.shape
    seq_ids = np.repeat(np.arange(n_seq), n_time)
    item_pos = np.tile(np.arange(n_time), n_seq)

    df = pd.DataFrame(arr.reshape(-1, n_regions), columns=region_names)
    df.insert(0, "itemPosition", item_pos)
    df.insert(0, "sequenceId", seq_ids)
    return df


def iqr_robust(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < MIN_ITEMS_IQR_ROBUST_FALLBACK:
        s = np.nanstd(x)
        return float(s if np.isfinite(s) and s > 0 else 1.0)
    q25, q75 = np.nanquantile(x, [QUANTILE_IQR_LO, QUANTILE_IQR_HI])
    s = float(q75 - q25)
    if not np.isfinite(s) or s <= 0:
        s = float(np.nanstd(x))
    return float(s if np.isfinite(s) and s > 0 else 1.0)
