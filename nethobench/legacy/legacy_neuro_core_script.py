from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from nethobench.legacy.legacy_metrics import _load_and_align
from nethobench.analysis.neuro_scoring import calculate_neuro_composites, _align_gt_pred
from nethobench.analysis.score_definitions import (
    build_neuro_metrics_df,
    build_fidelity_metrics_df,
    build_neuro_families_df,
    compute_neuro_composite,
    compute_fidelity_composite,
)

# --- Legacy Notebook Fidelity Implementations ---
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
    # Crucial: align and resample blocks (e.g. 1140 to 30) before scoring
    gt_arr, pred_arr = _align_gt_pred(gt_arr, pred_arr)
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
    # Crucial: align and resample blocks (e.g. 1140 to 30) before scoring
    gt_arr, pred_arr = _align_gt_pred(gt_arr, pred_arr)
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

# --- Main Execution ---
preds_fname = globals().get("preds_fname")
gt_fname = globals().get("gt_fname")
SAVE_PLOTS_DIR = globals().get("SAVE_PLOTS_DIR")
WRAPPED_NOTEBOOK_PATH = globals().get("WRAPPED_NOTEBOOK_PATH")
ENABLE_PLOTS = bool(globals().get("ENABLE_PLOTS", False))

if preds_fname is None or gt_fname is None:
    raise ValueError("preds_fname and gt_fname must be provided")

gt_arr, pred_arr, overlap = _load_and_align(Path(preds_fname), Path(gt_fname))

# 1. Compute core structure and inject the legacy fidelity scores
flat_scores = calculate_neuro_composites(gt_arr, pred_arr)
flat_scores["Error_score01"] = compute_error_score(gt_arr, pred_arr)
flat_scores["MI_score01"] = compute_mi_score(gt_arr, pred_arr)

# 2. Expose expected DataFrames for legacy `_flatten_scores` via globals
# By explicitly merging the Fidelity family into these tables, the CLI parser finds them.
metrics_df = pd.concat([
    build_neuro_metrics_df(flat_scores),
    build_fidelity_metrics_df(flat_scores)
], ignore_index=True)

families_df = build_neuro_families_df(flat_scores)
fid_composite = compute_fidelity_composite(flat_scores)

fidelity_family_row = pd.DataFrame([{
    "family": "fidelity", 
    "value": fid_composite, 
    "weight": 0.08
}])
families_df = pd.concat([families_df, fidelity_family_row], ignore_index=True)

FINAL_COMPOSITE_SCORE = compute_neuro_composite(flat_scores)
flat_scores["FINAL_COMPOSITE_SCORE"] = FINAL_COMPOSITE_SCORE
flat_scores["FINAL_NEURO_COMPOSITE_SCORE"] = FINAL_COMPOSITE_SCORE

# 3. Handle plotting if requested, delegating to the modern reporting module
if ENABLE_PLOTS and SAVE_PLOTS_DIR:
    from nethobench.analysis.neuro_reporting import generate_full_neuro_report
    generate_full_neuro_report(
        gt_arr, 
        pred_arr, 
        overlap, 
        flat_scores, 
        Path(SAVE_PLOTS_DIR)
    )

# 4. Provide a stub wrapped notebook to satisfy legacy consumers
if WRAPPED_NOTEBOOK_PATH:
    wrapped_path = Path(WRAPPED_NOTEBOOK_PATH)
    wrapped_path.parent.mkdir(parents=True, exist_ok=True)
    empty_nb = {
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# Notebook Bypassed"]}], 
        "metadata": {}, 
        "nbformat": 4, 
        "nbformat_minor": 5
    }
    wrapped_path.write_text(json.dumps(empty_nb, indent=2), encoding="utf-8")