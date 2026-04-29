from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from nethobench.analysis.neuro_scoring import calculate_neuro_composites, _align_gt_pred
from nethobench.analysis.score_definitions import (
    compute_neuro_composite,
    compute_fidelity_composite,
)
from nethobench.analysis.neuro_reporting import generate_full_neuro_report


# --- Sequence Loading & Alignment Helpers ---

def _load_sequences(
    csv_path: Path,
    sequence_key: str = "sequenceId",
    time_key: str = "itemPosition",
) -> tuple[np.ndarray, list[str]]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if {sequence_key, time_key}.issubset(df.columns):
        df = df.sort_values([sequence_key, time_key]).reset_index(drop=True)
        region_cols = [column for column in df.columns if column not in {sequence_key, time_key}]
        if not region_cols:
            raise ValueError(f"No region columns found in {csv_path}")
        seq_lengths = df.groupby(sequence_key).size()
        if seq_lengths.nunique() != 1:
            raise ValueError(
                "Sequences must all have identical length. "
                f"Distribution:\n{seq_lengths.describe()}"
            )
        n_seq = int(seq_lengths.size)
        n_time = int(seq_lengths.iloc[0])
        arr = df[region_cols].to_numpy(dtype=np.float64).reshape(n_seq, n_time, len(region_cols))
        return arr, region_cols

    df = pd.read_csv(csv_path, index_col=0)
    if df.index.dtype.kind not in {"i", "u"}:
        raise ValueError(f"Prediction CSV {csv_path} must have integral sequence ids in the index.")
    counts = df.index.value_counts()
    if counts.nunique() != 1:
        raise ValueError("Prediction sequences must all have the same length.")
    n_seq = int(counts.shape[0])
    n_time = int(counts.iloc[0])
    region_cols = df.columns.tolist()
    arr = df.to_numpy(dtype=np.float64).reshape(n_seq, n_time, len(region_cols))
    return arr, region_cols


def _load_and_align(
    predictions_csv: Path,
    ground_truth_csv: Path,
    neuro_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pred_arr, pred_regions = _load_sequences(predictions_csv)
    gt_arr, gt_regions = _load_sequences(ground_truth_csv)

    if neuro_cols:
        overlap = [region for region in neuro_cols if region in gt_regions and region in pred_regions]
    else:
        overlap = [region for region in gt_regions if region in pred_regions]
    
    if not overlap:
        raise ValueError("No overlapping neural regions between GT and predictions.")

    gt_idx = [gt_regions.index(region) for region in overlap]
    pred_idx = [pred_regions.index(region) for region in overlap]
    gt_arr = gt_arr[..., gt_idx]
    pred_arr = pred_arr[..., pred_idx]

    max_len = min(gt_arr.shape[1], pred_arr.shape[1])
    gt_arr = gt_arr[:, :max_len, :]
    pred_arr = pred_arr[:, :max_len, :]
    return gt_arr, pred_arr, overlap


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


# --- Full Analysis Run ---

def _timestamped_outdir(base: Optional[Path] = None, stem: Optional[str] = None) -> Path:
    base = Path(base) if base is not None else Path.cwd() / "outputs"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S") # Fixed undefined 'ts' bug from original code
    outdir = base / (f"{stem}-legacy-analysis-{ts}" if stem else f"legacy-neuro-analysis-{ts}")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def run_legacy_neuro_full_analysis(
    predictions_csv: Path,
    ground_truth_csv: Path,
    ddconfig_path: Path, # Parameter kept for signature compatibility
    *,
    output_root: Optional[Path] = None,
) -> Dict[str, object]:
    
    preds_path = Path(predictions_csv)
    gt_path = Path(ground_truth_csv)
    outdir = _timestamped_outdir(output_root, stem=preds_path.stem)

    gt_arr, pred_arr, overlap = _load_and_align(preds_path, gt_path)
    
    # 1. Compute in-memory
    scores = _compute_scores_from_arrays(gt_arr, pred_arr)

    # 2. Render Modern Matplotlib Figures natively
    generate_full_neuro_report(gt_arr, pred_arr, overlap, scores, outdir)

    # 3. Create dummy notebook for legacy consumers checking for an artifact
    wrapped_notebook = outdir / "wrapped_legacy_neuro_metrics.ipynb"
    empty_nb = {
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# Notebook Bypassed"]}],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    wrapped_notebook.write_text(json.dumps(empty_nb, indent=2), encoding="utf-8")

    # 4. Dump Scores JSON
    scores_path = outdir / "scores.json"
    scores_path.write_text(json.dumps(scores, indent=2))

    return {
        "output_dir": outdir,
        "scores": scores,
        "scores_path": scores_path,
        "wrapped_notebook": wrapped_notebook,
    }