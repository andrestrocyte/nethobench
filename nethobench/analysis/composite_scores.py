from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from nethobench.analysis.neuro_scoring import calculate_neuro_composites
from nethobench.analysis.score_definitions import (
    compute_neuro_composite,
    compute_fidelity_composite,
)
from nethobench.analysis.neuro_reporting import generate_full_neuro_report
from nethobench.helpers import _load_and_align

# --- Sequence Loading & Alignment Helpers ---
  
# --- Legacy Fidelity Implementations ---

def _robust_iqr(x: np.ndarray) -> float:
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

def _robust_median_mad(x: np.ndarray) -> tuple[float, float]:
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    s = 1.4826 * mad
    if not np.isfinite(s) or s <= 1e-12: s = float(np.nanstd(x))
    if not np.isfinite(s) or s <= 1e-12: s = 1.0
    return med, s

def compute_error_score(gt_arr: np.ndarray, pred_arr: np.ndarray) -> float:
    n_seq, _, n_reg = gt_arr.shape
    iqr_gt = np.array([_robust_iqr(gt_arr[:, :, r].reshape(-1)) for r in range(n_reg)])
    nrmse_sr = np.full((n_seq, n_reg), np.nan)
    
    for i in range(n_seq):
        for r in range(n_reg):
            x = gt_arr[i, :, r]
            y = pred_arr[i, :, r]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 50: 
                continue
            rmse = np.sqrt(np.mean((y[m] - x[m])**2))
            nrmse_sr[i, r] = rmse / (iqr_gt[r] + 1e-12)
            
    D_seq = np.full(n_seq, np.nan)
    for i in range(n_seq):
        row = nrmse_sr[i]
        row = row[np.isfinite(row)]
        if row.size == 0: 
            continue
        k = max(1, int(np.ceil(0.25 * row.size)))
        D_seq[i] = np.mean(np.sort(row)[-k:])
        
    D = np.nanmean(D_seq)
    return float(1.0 / (1.0 + D)) if np.isfinite(D) else np.nan

def compute_mi_score(gt_arr: np.ndarray, pred_arr: np.ndarray) -> float:
    n_seq, _, n_reg = gt_arr.shape
    rng = np.random.default_rng(0)
    mi_all = np.full((n_seq, n_reg), np.nan)
    
    for i in range(n_seq):
        for r in range(n_reg):
            x = gt_arr[i, :, r]
            y = pred_arr[i, :, r]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 80: 
                continue
            x, y = x[m], y[m]
            
            if x.size > 1200:
                idx = rng.choice(x.size, size=1200, replace=False)
                x, y = x[idx], y[idx]
            
            mx, sx = _robust_median_mad(x)
            my, sy = _robust_median_mad(y)
            
            x = (x - mx) / (sx + 1e-12)
            y = (y - my) / (sy + 1e-12)
            
            try:
                mi_val = mutual_info_regression(x.reshape(-1, 1), y, discrete_features=False, n_neighbors=5, random_state=0)[0]
                if np.isfinite(mi_val):
                    mi_all[i, r] = max(0.0, float(mi_val))
            except Exception:
                pass
                
    seq_worst = np.full(n_seq, np.nan)
    for i in range(n_seq):
        row = mi_all[i]
        row = row[np.isfinite(row)]
        if row.size == 0: 
            continue
        seq_worst[i] = float(np.quantile(row, 0.10))
        
    strict = float(np.nanquantile(seq_worst, 0.10)) if np.isfinite(seq_worst).any() else np.nan
    return float(strict / (1.0 + strict)) if np.isfinite(strict) else np.nan


# --- Core Scoring Functions ---

def _compute_scores_from_arrays(gt_arr: np.ndarray, pred_arr: np.ndarray) -> Dict[str, float]:
    """Computes legacy composites directly from in-memory arrays."""
    
    if gt_arr.ndim != 3 or pred_arr.ndim != 3:
        raise ValueError("Expected gt_arr/pred_arr arrays with shape [n_seq, n_time, n_reg].")
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"Shape mismatch: {gt_arr.shape} vs {pred_arr.shape}")

    flat_scores = calculate_neuro_composites(gt_arr, pred_arr)
    
    # Calculate fidelity implementations locally
    flat_scores["Error_score01"] = compute_error_score(gt_arr, pred_arr)
    flat_scores["MI_score01"] = compute_mi_score(gt_arr, pred_arr)

    # Compute master composites
    fid_composite = compute_fidelity_composite(flat_scores)
    final_composite = compute_neuro_composite(flat_scores)

    flat_scores["family_fidelity"] = fid_composite
    flat_scores["FINAL_COMPOSITE_SCORE"] = final_composite
    flat_scores["FINAL_NEURO_COMPOSITE_SCORE"] = final_composite
    flat_scores["composite_score"] = final_composite

    return flat_scores


def compute_composite_scores(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    per_sequence_stats: bool = False,
    neuro_cols: Optional[list[str]] = None,
    ddconfig_path: Optional[Path] = None, # Parameter kept for signature compatibility
) -> Dict[str, float]:
    
    if per_sequence_stats:
        raise ValueError("per_sequence_stats is not supported for legacy notebook-based neuro scores.")

    gt, pred, _ = _load_and_align(
        Path(predictions_csv),
        Path(ground_truth_csv),
        neuro_cols=neuro_cols,
    )
    
    return _compute_scores_from_arrays(gt, pred)


