from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import runpy
import tempfile
from typing import Dict, Optional

import numpy as np
import pandas as pd
from nethobench.analysis.neuro_scoring import calculate_neuro_composites
from nethobench.helpers import load_gt_and_preds
from nethobench.analysis.neuro_reporting import generate_full_neuro_report

ACTIVE_NEURO_NOTEBOOK = Path(__file__).parent / "notebooks" / "neuro_metrics.ipynb"
ACTIVE_NEURO_CORE_SCRIPT = Path(__file__).parent / "analysis" / "neuro_metrics_core_script.py"


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


def _flatten_active_notebook_scores(env: dict) -> Dict[str, float]:
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


def _patch_runtime_lines(
    source: str,
    *,
    predictions_csv: Path,
    ground_truth_csv: Path,
    ddconfig_path: Path,
) -> str:
    lines: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        if stripped.startswith("preds_fname ="):
            line = f'preds_fname = r"{Path(predictions_csv)}"'
        elif stripped.startswith("gt_fname ="):
            line = f'gt_fname = r"{Path(ground_truth_csv)}"'
        elif stripped.startswith("ddconfig_path ="):
            line = f'ddconfig_path = r"{Path(ddconfig_path)}"'
        lines.append(line)
    return "\n".join(lines)


def _execute_full_neuro_notebook(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    ddconfig_path: Path,
    save_plots_dir: Path,
    save_wrapped_notebook: Path,
) -> None:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    if not ACTIVE_NEURO_NOTEBOOK.is_file():
        raise FileNotFoundError(f"Active neuro notebook missing at {ACTIVE_NEURO_NOTEBOOK}")

    plot_dir = Path(save_plots_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    nb = nbformat.read(ACTIVE_NEURO_NOTEBOOK, as_version=4)
    runtime_patch = f"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
repo_root = Path(r"{ACTIVE_NEURO_NOTEBOOK.parent.parent.parent}")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
ENABLE_PLOTS = True
outdir = Path(r"{plot_dir}")
outdir.mkdir(parents=True, exist_ok=True)
plot_counter = {{'n': 0}}
def saving_show(*args, **kwargs):
    figs = [plt.figure(num) for num in plt.get_fignums()]
    for fig in figs:
        plot_counter['n'] += 1
        fig.savefig(outdir / f"figure_{{plot_counter['n']:03d}}.png", dpi=200, bbox_inches='tight')
    plt.close('all')
plt.show = saving_show
"""
    nb.cells.insert(0, nbformat.v4.new_code_cell(runtime_patch))

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if "preds_fname =" in source and "gt_fname =" in source and "ddconfig_path =" in source:
            cell["source"] = _patch_runtime_lines(
                source,
                predictions_csv=Path(predictions_csv),
                ground_truth_csv=Path(ground_truth_csv),
                ddconfig_path=Path(ddconfig_path),
            )
            continue
        if source.lstrip().startswith("# --- Animated trajectory visualizer (optional video export) ---"):
            cell["source"] = source.replace(
                "preview_animation = animate_sequence(seq_idx=0, include_shuffled=True, tail=40)\n"
                "display(preview_animation)\n"
                "print('Use animate_sequence(seq_idx, include_shuffled, fps, tail, save_path) for custom previews or exports.')\n",
                "print('Animation helper loaded. Call animate_sequence(...) manually in an interactive notebook if needed.')\n",
            )

    ep = ExecutePreprocessor(timeout=3600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(ACTIVE_NEURO_NOTEBOOK.parent.parent.parent)}})

    save_wrapped_notebook = Path(save_wrapped_notebook)
    save_wrapped_notebook.parent.mkdir(parents=True, exist_ok=True)
    with save_wrapped_notebook.open("w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)


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

        return run_neuro_full_analysis(pred_path, gt_path)

def compute_neuro_scores(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    per_sequence_stats: bool = False,
    neuro_cols: Optional[list[str]] = None,
) -> Dict[str, float]:
    
    if per_sequence_stats:
        raise ValueError("per_sequence_stats is not supported for notebook-based neuro scores.")

    # 1. Use the existing helper to load CSVs and reshape them into 3D tensors
    # shape: [n_sequences, n_timesteps, n_regions]
    gt_arr, pred_arr, overlap = _load_and_align(
        Path(predictions_csv),
        Path(ground_truth_csv),
        neuro_cols=neuro_cols,
    )
    
    return calculate_neuro_composites(gt_arr, pred_arr)


def _timestamped_outdir(base: Optional[Path] = None, stem: Optional[str] = None) -> Path:
    base = Path(base) if base is not None else Path.cwd() / "outputs"
    outdir = base / (f"{stem}-analysis" if stem else f"neuro-analysis")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def run_neuro_full_analysis(
    predictions_csv: Path,
    ground_truth_csv: Path,
    output_root: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Execute the active neuro notebook headlessly, save figures, and export notebook-derived scores.
    """
    preds_path = Path(predictions_csv)
    outdir = _timestamped_outdir(output_root, stem=preds_path.stem)
    gt_arr, pred_arr, region_names = _load_and_align(preds_path, Path(ground_truth_csv))
    
    scores = calculate_neuro_composites(gt_arr, pred_arr)

    generate_full_neuro_report(gt_arr, pred_arr, region_names, scores, outdir)

    scores_path = outdir / "scores.json"
    scores_path.write_text(json.dumps(scores, indent=2))
    return scores