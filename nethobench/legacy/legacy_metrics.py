from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import runpy
import tempfile
from typing import Dict, Optional

import numpy as np
import pandas as pd


ACTIVE_LEGACY_NOTEBOOK = Path(__file__).parent / "notebooks" / "neuro_metrics_legacy.ipynb"
ACTIVE_LEGACY_CORE_SCRIPT = Path(__file__).parent / "legacy_neuro_core_script.py"


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


def _ensure_ddconfig(ddconfig_path: Optional[Path]) -> tuple[Path, Optional[Path]]:
    if ddconfig_path is not None:
        return Path(ddconfig_path), None
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({"selected_columns_statistics": {}}, tmp)
    tmp.flush()
    tmp.close()
    return Path(tmp.name), Path(tmp.name)


def _flatten_scores(env: dict) -> Dict[str, float]:
    scores: Dict[str, float] = {}

    metrics_df = env.get("metrics_df")
    if isinstance(metrics_df, pd.DataFrame) and {"metric", "value"}.issubset(metrics_df.columns):
        for _, row in metrics_df.iterrows():
            key = str(row["metric"])
            value = float(row["value"]) if np.isfinite(row["value"]) else np.nan
            scores[key] = value

    families_df = env.get("families_df")
    if isinstance(families_df, pd.DataFrame) and {"family", "value"}.issubset(families_df.columns):
        for _, row in families_df.iterrows():
            key = f"family_{row['family']}"
            value = float(row["value"]) if np.isfinite(row["value"]) else np.nan
            scores[key] = value

    composite = env.get("FINAL_COMPOSITE_SCORE", np.nan)
    composite = float(composite) if np.isfinite(composite) else np.nan
    scores["composite_score"] = composite
    scores["FINAL_COMPOSITE_SCORE"] = composite
    scores["FINAL_NEURO_COMPOSITE_SCORE"] = composite
    return scores


def _run_legacy_notebook(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    ddconfig_path: Optional[Path] = None,
    save_plots_dir: Optional[Path] = None,
    save_wrapped_notebook: Optional[Path] = None,
) -> tuple[dict, Dict[str, float]]:
    dd_path, cleanup_path = _ensure_ddconfig(ddconfig_path)
    try:
        init_globals = {
            "preds_fname": str(Path(predictions_csv)),
            "gt_fname": str(Path(ground_truth_csv)),
            "ddconfig_path": str(dd_path),
            "SAVE_PLOTS_DIR": str(Path(save_plots_dir)) if save_plots_dir is not None else None,
            "WRAPPED_NOTEBOOK_PATH": str(Path(save_wrapped_notebook)) if save_wrapped_notebook is not None else None,
            "ENABLE_PLOTS": bool(save_plots_dir is not None),
            "__name__": "__main__",
            "ACTIVE_NEURO_NOTEBOOK": str(ACTIVE_LEGACY_NOTEBOOK),
        }
        env = runpy.run_path(str(ACTIVE_LEGACY_CORE_SCRIPT), init_globals=init_globals)
        env["ACTIVE_NEURO_NOTEBOOK"] = str(ACTIVE_LEGACY_NOTEBOOK)
        env["ACTIVE_NEURO_CORE_SCRIPT"] = str(ACTIVE_LEGACY_CORE_SCRIPT)
        return env, _flatten_scores(env)
    finally:
        if cleanup_path and cleanup_path.exists():
            cleanup_path.unlink(missing_ok=True)


def _compute_scores_from_arrays(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    region_names: Optional[list[str]] = None,
    ddconfig_path: Optional[Path] = None,
) -> Dict[str, float]:
    if gt.ndim != 3 or pred.ndim != 3:
        raise ValueError("Expected gt/pred arrays with shape [n_seq, n_time, n_reg].")
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} vs {pred.shape}")

    n_seq, n_time, n_reg = gt.shape
    region_names = region_names or [f"R{i}" for i in range(n_reg)]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        gt_path = tmpdir_path / "gt.csv"
        pred_path = tmpdir_path / "pred.csv"

        seq_ids = np.repeat(np.arange(n_seq), n_time)
        item_pos = np.tile(np.arange(n_time), n_seq)

        gt_df = pd.DataFrame(gt.reshape(-1, n_reg), columns=region_names)
        gt_df.insert(0, "itemPosition", item_pos)
        gt_df.insert(0, "sequenceId", seq_ids)
        gt_df.to_csv(gt_path, index=False)

        pred_df = pd.DataFrame(pred.reshape(-1, n_reg), columns=region_names)
        pred_df.index = seq_ids
        pred_df.to_csv(pred_path)

        _, scores = _run_legacy_notebook(pred_path, gt_path, ddconfig_path=ddconfig_path)
        return scores


def compute_legacy_neuro_scores(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    per_sequence_stats: bool = False,
    neuro_cols: Optional[list[str]] = None,
    ddconfig_path: Optional[Path] = None,
) -> Dict[str, float]:
    if per_sequence_stats:
        raise ValueError("per_sequence_stats is not supported for notebook-based neuro scores.")

    if neuro_cols:
        gt, pred, overlap = _load_and_align(
            Path(predictions_csv),
            Path(ground_truth_csv),
            neuro_cols=neuro_cols,
        )
        return _compute_scores_from_arrays(gt, pred, region_names=overlap, ddconfig_path=ddconfig_path)

    _, scores = _run_legacy_notebook(
        Path(predictions_csv),
        Path(ground_truth_csv),
        ddconfig_path=ddconfig_path,
    )
    return scores


def _timestamped_outdir(base: Optional[Path] = None, stem: Optional[str] = None) -> Path:
    base = Path(base) if base is not None else Path.cwd() / "outputs"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = base / (f"{stem}-legacy-analysis-{ts}" if stem else f"legacy-neuro-analysis-{ts}")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def run_legacy_neuro_full_analysis(
    predictions_csv: Path,
    ground_truth_csv: Path,
    ddconfig_path: Path,
    *,
    output_root: Optional[Path] = None,
) -> Dict[str, object]:
    preds_path = Path(predictions_csv)
    outdir = _timestamped_outdir(output_root, stem=preds_path.stem)
    wrapped_notebook = outdir / "wrapped_legacy_neuro_metrics.ipynb"
    _, scores = _run_legacy_notebook(
        preds_path,
        Path(ground_truth_csv),
        ddconfig_path=Path(ddconfig_path),
        save_plots_dir=outdir,
        save_wrapped_notebook=wrapped_notebook,
    )
    scores_path = outdir / "scores.json"
    scores_path.write_text(json.dumps(scores, indent=2))
    return {
        "output_dir": outdir,
        "scores": scores,
        "scores_path": scores_path,
        "wrapped_notebook": wrapped_notebook,
    }
