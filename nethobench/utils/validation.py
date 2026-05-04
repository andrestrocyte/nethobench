"""
Centralized data-validation layer for NethoBench.

Validation is designed to fail *early* — right after data loading and alignment —
so that downstream metric code can assume mathematically sound inputs.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from nethobench.utils.evaluation_constants import MIN_SAMPLES_CORR_WINDOW


class DataValidationError(ValueError):
    """Raised when input data fails a structural or mathematical integrity check."""


# ---------------------------------------------------------------------------
# 1. Schema & Index Integrity (Post-Load)
# ---------------------------------------------------------------------------

def validate_dataframe_schema(
    df: pd.DataFrame,
    *,
    seq_key: str = "sequenceId",
    time_key: str = "itemPosition",
    value_cols: Optional[list[str]] = None,
    strict_time_numeric: bool = True,
) -> None:
    """
    Verify that a loaded DataFrame satisfies the structural requirements
    for NethoBench alignment.

    Checks
    ------
    * ``seq_key`` and ``time_key`` columns are present.
    * ``time_key`` is numeric (so it sorts chronologically, not lexicographically).
    * The composite key ``(seq_key, time_key)`` is strictly unique.
    * If ``value_cols`` is given, every column exists in the frame.
    """
    df = pd.DataFrame(df)  # shallow copy to avoid mutating caller
    missing = []
    for col in (seq_key, time_key):
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise DataValidationError(
            f"Missing required alignment columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    if strict_time_numeric:
        time_dtype = df[time_key].dtype
        if not pd.api.types.is_numeric_dtype(time_dtype):
            raise DataValidationError(
                f"Column '{time_key}' must be numeric so it sorts chronologically, "
                f"but found dtype {time_dtype}."
            )

    # Primary-key uniqueness: duplicate (sequenceId, itemPosition) pairs can
    # silently create Cartesian cross-products in inner joins.
    n_rows = len(df)
    n_unique = df[[seq_key, time_key]].drop_duplicates().shape[0]
    if n_unique != n_rows:
        dupes = n_rows - n_unique
        raise DataValidationError(
            f"Duplicate composite keys detected: ({seq_key}, {time_key}) is not "
            f"unique ({dupes} duplicate row(s)). Cartesian explosions during "
            f"alignment will distort time-series metrics."
        )

    if value_cols is not None:
        missing_vals = [c for c in value_cols if c not in df.columns]
        if missing_vals:
            raise DataValidationError(
                f"Missing expected value columns: {missing_vals}"
            )


# ---------------------------------------------------------------------------
# 2. Alignment & Overlap Guards (The Join Phase)
# ---------------------------------------------------------------------------

def validate_alignment_overlap(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    *,
    seq_key: str = "sequenceId",
    time_key: str = "itemPosition",
    min_overlap_ratio: float = 0.50,
    warn_truncation_ratio: float = 0.95,
) -> None:
    """
    Guard against silent data loss during the inner-join alignment step.

    Checks
    ------
    * The merged DataFrame is non-empty.
    * The merged DataFrame retains at least ``min_overlap_ratio`` of the
      expected rows (computed from the *smaller* of GT and Pred row counts).
    * If sequence lengths become highly non-uniform after truncation,
      a warning is emitted so the user knows frames were discarded.
    """
    if merged_df.empty:
        raise DataValidationError(
            "Alignment produced an empty DataFrame — no overlapping "
            f"({seq_key}, {time_key}) pairs between GT and predictions."
        )

    expected_rows = min(len(gt_df), len(pred_df))
    merged_rows = len(merged_df)
    overlap_ratio = merged_rows / expected_rows if expected_rows > 0 else 0.0

    if overlap_ratio < min_overlap_ratio:
        raise DataValidationError(
            f"Severe data loss after alignment: only {merged_rows:,} rows "
            f"retained out of {expected_rows:,} expected "
            f"(overlap ratio {overlap_ratio:.2%}, threshold {min_overlap_ratio:.2%}). "
            f"Check that GT and predictions share the same sequences and time grid."
        )

    # Warn if overlap is below the higher warn threshold
    if overlap_ratio < warn_truncation_ratio:
        warnings.warn(
            f"Data truncation detected: alignment retained {merged_rows:,} rows "
            f"({overlap_ratio:.2%} of expected {expected_rows:,}). "
            f"Downstream metrics may be noisier than expected.",
            UserWarning,
            stacklevel=3,
        )

    # Sequence-length uniformity check after join
    seq_lengths = merged_df.groupby(seq_key).size()
    if seq_lengths.nunique() != 1:
        min_len = int(seq_lengths.min())
        max_len = int(seq_lengths.max())
        if max_len > 0 and min_len / max_len < warn_truncation_ratio:
            warnings.warn(
                f"Sequence lengths are highly non-uniform after alignment: "
                f"min={min_len}, max={max_len}. "
                f"Any 3-D tensor stacking will truncate all sequences to {min_len}, "
                f"discarding {(max_len - min_len) / max_len:.1%} of frames from "
                f"the longest sequence.",
                UserWarning,
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# 3. Data Quality & Mathematical Validity (Pre-Metric)
# ---------------------------------------------------------------------------

def validate_array_quality(
    arr: np.ndarray,
    *,
    name: str = "array",
    min_n_seq: Optional[int] = None,
    min_seq_len: Optional[int] = None,
    min_n_reg: Optional[int] = None,
    max_nan_ratio: float = 0.10,
    check_zero_variance: bool = True,
    zero_variance_tol: float = 1e-6,
) -> None:
    """
    Validate a 3-D array ``[n_seq, T, n_reg]`` before it enters metric code.

    Checks
    ------
    * Array is 3-D and meets any requested minimum dimensions.
    * No single column (region) contains more than ``max_nan_ratio`` NaNs.
    * No region has zero variance (flat-lined signal) if
      ``check_zero_variance`` is True.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3:
        raise DataValidationError(
            f"{name} must be 3-D [n_seq, T, n_reg], got shape {arr.shape}"
        )

    n_seq, n_time, n_reg = arr.shape

    if min_n_seq is not None and n_seq < min_n_seq:
        raise DataValidationError(
            f"{name} has {n_seq} sequence(s), but minimum required is {min_n_seq}."
        )
    if min_seq_len is not None and n_time < min_seq_len:
        raise DataValidationError(
            f"{name} has sequence length {n_time}, but minimum required is {min_seq_len}. "
            f"Several metrics (e.g. windowed features with window=60) need longer sequences."
        )
    if min_n_reg is not None and n_reg < min_n_reg:
        raise DataValidationError(
            f"{name} has {n_reg} region(s), but minimum required is {min_n_reg}."
        )

    # NaN threshold — per-column so a single dead sensor does not silently
    # poison a pooled metric.
    total_per_region = n_seq * n_time
    nan_counts = np.isnan(arr).sum(axis=(0, 1))
    if total_per_region > 0:
        worst_ratio = float(nan_counts.max() / total_per_region)
        worst_region = int(nan_counts.argmax())
        if worst_ratio > max_nan_ratio:
            raise DataValidationError(
                f"{name}: region {worst_region} is {worst_ratio:.1%} NaN "
                f"(threshold {max_nan_ratio:.1%}). Statistically meaningless "
                f"metrics are likely."
            )

    # Zero-variance check
    if check_zero_variance and n_time > 1:
        # std across time, pooled over sequences
        stds = np.nanstd(arr, axis=(0, 1))
        flat_regions = np.where(stds < zero_variance_tol)[0]
        if flat_regions.size:
            raise DataValidationError(
                f"{name}: {flat_regions.size} region(s) have near-zero variance "
                f"(std < {zero_variance_tol}): indices {flat_regions.tolist()}. "
                f"Flat-lined signals break correlation, PCA, CCA, and Procrustes."
            )


def validate_scale_consistency(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    *,
    name_gt: str = "GT",
    name_pred: str = "Pred",
    max_log_ratio: float = 3.0,
) -> None:
    """
    Ensure GT and predictions live on roughly comparable scales.

    A large mismatch (e.g. z-scored GT vs. raw fluorescence predictions)
    makes distance-based metrics meaningless.
    """
    gt_arr = np.asarray(gt_arr, dtype=np.float64)
    pred_arr = np.asarray(pred_arr, dtype=np.float64)

    gt_std = float(np.nanstd(gt_arr))
    pr_std = float(np.nanstd(pred_arr))

    if not (np.isfinite(gt_std) and np.isfinite(pr_std) and gt_std > 0 and pr_std > 0):
        warnings.warn(
            f"Cannot compare scales: {name_gt} std={gt_std}, {name_pred} std={pr_std}.",
            UserWarning,
            stacklevel=3,
        )
        return

    log_ratio = abs(np.log10(gt_std / pr_std))
    if log_ratio > max_log_ratio:
        warnings.warn(
            f"Scale mismatch between {name_gt} (std={gt_std:.3g}) and "
            f"{name_pred} (std={pr_std:.3g}) — log10 ratio={log_ratio:.2f}, "
            f"threshold={max_log_ratio:.2f}. Distance metrics may be dominated "
            f"by the larger-magnitude array.",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# 4. Unified Multimodal Validator
# ---------------------------------------------------------------------------

def validate_multimodal_data(
    df: pd.DataFrame,
    cfg: dict,
) -> None:
    """
    Run the full validation suite on an *already-merged* multimodal DataFrame.

    This is intended for cross-modal pipelines that work with the suffixed
    ``_gt`` / ``_inf`` columns produced by ``merge_aligned``.
    """
    seq_key = cfg.get("sequence_key", "sequenceId")
    time_key = cfg.get("time_key", "itemPosition")

    validate_dataframe_schema(df, seq_key=seq_key, time_key=time_key)

    # Check that expected modality columns exist on both sides of the merge
    neuro_cols = cfg.get("neuro_cols", [])
    behavior_parts = cfg.get("behavior_parts", [])
    missing = []
    for side in ("gt", "inf"):
        for c in neuro_cols:
            col = f"{c}_{side}"
            if col not in df.columns:
                missing.append(col)
        for part in behavior_parts:
            for coord in ("_X", "_Y"):
                col = f"{part}{coord}_{side}"
                if col not in df.columns:
                    missing.append(col)
    if missing:
        raise DataValidationError(
            f"Merged DataFrame missing expected suffixed columns: {missing}"
        )

    # Sequence-length sanity for cross-modal metrics
    seq_lengths = df.groupby(seq_key).size()
    min_len = int(seq_lengths.min())
    max_len = int(seq_lengths.max())
    if min_len < MIN_SAMPLES_CORR_WINDOW:
        warnings.warn(
            f"Shortest sequence after alignment has length {min_len}. "
            f"Cross-modal lead-lag metrics require at least {MIN_SAMPLES_CORR_WINDOW} time steps.",
            UserWarning,
            stacklevel=3,
        )

    # Warn if sequences were heavily truncated
    if max_len > 0 and min_len / max_len < 0.95:
        warnings.warn(
            f"Sequence lengths after alignment are non-uniform "
            f"(min={min_len}, max={max_len}). "
            f"Truncation to the shortest sequence will discard "
            f"{(max_len - min_len) / max_len:.1%} of frames.",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# 5. Convenience helpers for the most common pipeline entry points
# ---------------------------------------------------------------------------

def validate_loaded_neuro_arrays(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    region_names: list[str],
    *,
    min_seq_len: int = 60,
    min_n_seq: int = 2,
    min_n_reg: int = 2,
    max_nan_ratio: float = 0.10,
) -> None:
    """
    Validate 3-D neuro arrays right after ``load_and_align``.
    """
    for arr, label in ((gt_arr, "GT"), (pred_arr, "Pred")):
        validate_array_quality(
            arr,
            name=label,
            min_n_seq=min_n_seq,
            min_seq_len=min_seq_len,
            min_n_reg=min_n_reg,
            max_nan_ratio=max_nan_ratio,
            check_zero_variance=True,
        )

    if gt_arr.shape != pred_arr.shape:
        raise DataValidationError(
            f"Shape mismatch after alignment: GT {gt_arr.shape} vs Pred {pred_arr.shape}"
        )

    validate_scale_consistency(gt_arr, pred_arr)

    # Ensure region-name count matches array dimensions
    if len(region_names) != gt_arr.shape[-1]:
        raise DataValidationError(
            f"Region name count ({len(region_names)}) does not match array "
            f"trailing dimension ({gt_arr.shape[-1]})."
        )
