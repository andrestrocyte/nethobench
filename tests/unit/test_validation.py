"""Unit tests for nethobench/utils/validation.py."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nethobench.utils.validation import (
    DataValidationError,
    validate_dataframe_schema,
    validate_alignment_overlap,
    validate_array_quality,
    validate_scale_consistency,
    validate_multimodal_data,
    validate_loaded_neuro_arrays,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_df() -> pd.DataFrame:
    return pd.DataFrame({
        "sequenceId": [0, 0, 1, 1],
        "itemPosition": [0, 1, 0, 1],
        "region_a": [1.0, 2.0, 3.0, 4.0],
    })


@pytest.fixture
def gt_df() -> pd.DataFrame:
    return pd.DataFrame({
        "sequenceId": [0, 0, 1, 1],
        "itemPosition": [0, 1, 0, 1],
        "region_a": [1.0, 2.0, 3.0, 4.0],
    })


@pytest.fixture
def pred_df() -> pd.DataFrame:
    return pd.DataFrame({
        "sequenceId": [0, 0, 1, 1],
        "itemPosition": [0, 1, 0, 1],
        "region_a": [1.1, 2.1, 3.1, 4.1],
    })


# ---------------------------------------------------------------------------
# Schema & Index Integrity
# ---------------------------------------------------------------------------

class TestValidateDataframeSchema:
    def test_valid_df_passes(self, valid_df: pd.DataFrame):
        validate_dataframe_schema(valid_df)

    def test_missing_sequence_key(self, valid_df: pd.DataFrame):
        df = valid_df.drop(columns=["sequenceId"])
        with pytest.raises(DataValidationError, match="Missing required alignment columns"):
            validate_dataframe_schema(df)

    def test_missing_time_key(self, valid_df: pd.DataFrame):
        df = valid_df.drop(columns=["itemPosition"])
        with pytest.raises(DataValidationError, match="Missing required alignment columns"):
            validate_dataframe_schema(df)

    def test_non_numeric_time_key(self, valid_df: pd.DataFrame):
        df = valid_df.copy()
        df["itemPosition"] = ["a", "b", "c", "d"]
        with pytest.raises(DataValidationError, match="must be numeric"):
            validate_dataframe_schema(df)

    def test_duplicate_composite_key(self, valid_df: pd.DataFrame):
        df = pd.concat([valid_df, valid_df.iloc[[0]]], ignore_index=True)
        with pytest.raises(DataValidationError, match="Duplicate composite keys"):
            validate_dataframe_schema(df)

    def test_missing_value_cols(self, valid_df: pd.DataFrame):
        with pytest.raises(DataValidationError, match="Missing expected value columns"):
            validate_dataframe_schema(valid_df, value_cols=["region_b"])

    def test_custom_keys(self, valid_df: pd.DataFrame):
        df = valid_df.rename(columns={"sequenceId": "seq", "itemPosition": "t"})
        validate_dataframe_schema(df, seq_key="seq", time_key="t")


# ---------------------------------------------------------------------------
# Alignment & Overlap Guards
# ---------------------------------------------------------------------------

class TestValidateAlignmentOverlap:
    def test_perfect_overlap(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame):
        merged = pd.merge(gt_df, pred_df, on=["sequenceId", "itemPosition"], suffixes=("_gt", "_inf"))
        validate_alignment_overlap(gt_df, pred_df, merged)

    def test_empty_merge_raises(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame):
        pred_mismatch = pred_df.copy()
        pred_mismatch["sequenceId"] = pred_mismatch["sequenceId"] + 100
        merged = pd.merge(gt_df, pred_mismatch, on=["sequenceId", "itemPosition"], suffixes=("_gt", "_inf"), how="inner")
        with pytest.raises(DataValidationError, match="empty DataFrame"):
            validate_alignment_overlap(gt_df, pred_mismatch, merged)

    def test_severe_data_loss_raises(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame):
        # GT and Pred both have 4 rows, but only 1 row overlaps
        pred_mismatch = pred_df.copy()
        pred_mismatch["itemPosition"] = [0, 100, 100, 100]
        merged = pd.merge(gt_df, pred_mismatch, on=["sequenceId", "itemPosition"], suffixes=("_gt", "_inf"), how="inner")
        with pytest.raises(DataValidationError, match="Severe data loss"):
            validate_alignment_overlap(gt_df, pred_mismatch, merged, min_overlap_ratio=0.5)

    def test_warning_on_moderate_truncation(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame):
        # GT and Pred both have 4 rows, but only 2 rows overlap
        pred_mismatch = pred_df.copy()
        pred_mismatch["sequenceId"] = [0, 0, 10, 10]
        merged = pd.merge(gt_df, pred_mismatch, on=["sequenceId", "itemPosition"], suffixes=("_gt", "_inf"), how="inner")
        with pytest.warns(UserWarning, match="Data truncation detected"):
            validate_alignment_overlap(gt_df, pred_mismatch, merged, warn_truncation_ratio=0.95)

    def test_warning_on_nonuniform_sequences(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame):
        # GT has seq 0 with 2 rows, seq 1 with 2 rows
        # Pred has seq 0 with 2 rows, seq 1 with 1 row
        pred_partial = pred_df.iloc[[0, 1, 2]].copy()
        merged = pd.merge(gt_df, pred_partial, on=["sequenceId", "itemPosition"], suffixes=("_gt", "_inf"), how="inner")
        with pytest.warns(UserWarning, match="non-uniform"):
            validate_alignment_overlap(gt_df, pred_partial, merged, warn_truncation_ratio=0.95)


# ---------------------------------------------------------------------------
# Array Quality
# ---------------------------------------------------------------------------

class TestValidateArrayQuality:
    def test_valid_3d_array_passes(self):
        arr = np.random.randn(4, 100, 8)
        validate_array_quality(arr)

    def test_not_3d_raises(self):
        arr = np.random.randn(10, 10)
        with pytest.raises(DataValidationError, match="must be 3-D"):
            validate_array_quality(arr)

    def test_min_n_seq_violation(self):
        arr = np.random.randn(1, 100, 8)
        with pytest.raises(DataValidationError, match="1 sequence"):
            validate_array_quality(arr, min_n_seq=2)

    def test_min_seq_len_violation(self):
        arr = np.random.randn(4, 10, 8)
        with pytest.raises(DataValidationError, match="sequence length 10"):
            validate_array_quality(arr, min_seq_len=60)

    def test_min_n_reg_violation(self):
        arr = np.random.randn(4, 100, 1)
        with pytest.raises(DataValidationError, match="1 region"):
            validate_array_quality(arr, min_n_reg=2)

    def test_excessive_nans_raises(self):
        arr = np.random.randn(4, 100, 8)
        arr[:, :, 0] = np.nan
        with pytest.raises(DataValidationError, match="NaN"):
            validate_array_quality(arr, max_nan_ratio=0.05)

    def test_zero_variance_raises(self):
        arr = np.random.randn(4, 100, 8)
        arr[:, :, 3] = 5.0
        with pytest.raises(DataValidationError, match="near-zero variance"):
            validate_array_quality(arr, zero_variance_tol=1e-6)

    def test_zero_variance_disabled(self):
        arr = np.random.randn(4, 100, 8)
        arr[:, :, 3] = 5.0
        validate_array_quality(arr, check_zero_variance=False)


# ---------------------------------------------------------------------------
# Scale Consistency
# ---------------------------------------------------------------------------

class TestValidateScaleConsistency:
    def test_comparable_scales_passes(self):
        gt = np.random.randn(100)
        pred = np.random.randn(100) * 2
        validate_scale_consistency(gt, pred)

    def test_mismatched_scales_warns(self):
        gt = np.random.randn(100)
        pred = np.random.randn(100) * 1e4
        with pytest.warns(UserWarning, match="Scale mismatch"):
            validate_scale_consistency(gt, pred, max_log_ratio=2.0)

    def test_zero_std_warns(self):
        gt = np.ones(100)
        pred = np.random.randn(100)
        with pytest.warns(UserWarning, match="Cannot compare scales"):
            validate_scale_consistency(gt, pred)


# ---------------------------------------------------------------------------
# Multimodal Data
# ---------------------------------------------------------------------------

class TestValidateMultimodalData:
    def test_valid_multimodal_passes(self):
        df = pd.DataFrame({
            "sequenceId": [0, 0, 1, 1],
            "itemPosition": [0, 1, 0, 1],
            "region_a_gt": [1.0, 2.0, 3.0, 4.0],
            "region_a_inf": [1.1, 2.1, 3.1, 4.1],
            "CENTER_X_gt": [0, 1, 2, 3],
            "CENTER_Y_gt": [0, 1, 2, 3],
            "CENTER_X_inf": [0, 1, 2, 3],
            "CENTER_Y_inf": [0, 1, 2, 3],
        })
        cfg = {
            "neuro_cols": ["region_a"],
            "behavior_parts": ["CENTER"],
        }
        validate_multimodal_data(df, cfg)

    def test_missing_suffixed_columns_raises(self):
        df = pd.DataFrame({
            "sequenceId": [0, 0, 1, 1],
            "itemPosition": [0, 1, 0, 1],
            "region_a_gt": [1.0, 2.0, 3.0, 4.0],
        })
        cfg = {"neuro_cols": ["region_a"], "behavior_parts": []}
        with pytest.raises(DataValidationError, match="missing expected suffixed columns"):
            validate_multimodal_data(df, cfg)

    def test_short_sequence_warns(self):
        df = pd.DataFrame({
            "sequenceId": [0, 0, 1, 1],
            "itemPosition": [0, 1, 0, 1],
            "region_a_gt": [1.0, 2.0, 3.0, 4.0],
            "region_a_inf": [1.1, 2.1, 3.1, 4.1],
            "CENTER_X_gt": [0, 1, 2, 3],
            "CENTER_Y_gt": [0, 1, 2, 3],
            "CENTER_X_inf": [0, 1, 2, 3],
            "CENTER_Y_inf": [0, 1, 2, 3],
        })
        cfg = {"neuro_cols": ["region_a"], "behavior_parts": ["CENTER"]}
        with pytest.warns(UserWarning, match="Shortest sequence"):
            validate_multimodal_data(df, cfg)


# ---------------------------------------------------------------------------
# Loaded Neuro Arrays
# ---------------------------------------------------------------------------

class TestValidateLoadedNeuroArrays:
    def test_valid_arrays_pass(self):
        gt = np.random.randn(3, 100, 4)
        pred = np.random.randn(3, 100, 4)
        validate_loaded_neuro_arrays(gt, pred, ["r1", "r2", "r3", "r4"])

    def test_mismatched_shapes_raises(self):
        gt = np.random.randn(3, 100, 4)
        pred = np.random.randn(3, 100, 5)
        with pytest.raises(DataValidationError, match="Shape mismatch"):
            validate_loaded_neuro_arrays(gt, pred, ["r1", "r2", "r3", "r4"])

    def test_region_name_count_mismatch_raises(self):
        gt = np.random.randn(3, 100, 4)
        pred = np.random.randn(3, 100, 4)
        with pytest.raises(DataValidationError, match="Region name count"):
            validate_loaded_neuro_arrays(gt, pred, ["r1", "r2", "r3"])
