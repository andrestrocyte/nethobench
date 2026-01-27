from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import tempfile
import nbformat
from scipy import signal, stats

# --------------------------------------------------------------------------- #
# Loading helpers                                                             #
# --------------------------------------------------------------------------- #


def _load_sequences(csv_path: Path, sequence_key: str = "sequenceId", time_key: str = "itemPosition") -> tuple[np.ndarray, list[str]]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if {sequence_key, time_key}.issubset(df.columns):
        df = df.sort_values([sequence_key, time_key]).reset_index(drop=True)
        region_cols = [c for c in df.columns if c not in {sequence_key, time_key}]
        if not region_cols:
            raise ValueError(f"No region columns found in {csv_path}")
        seq_lengths = df.groupby(sequence_key).size()
        if seq_lengths.nunique() != 1:
            raise ValueError(
                "Sequences must all have identical length. "
                f"Distribution:\n{seq_lengths.describe()}"
            )
        n_seq = seq_lengths.size
        n_time = int(seq_lengths.iloc[0])
        arr = (
            df[region_cols]
            .to_numpy(dtype=np.float64)
            .reshape(n_seq, n_time, len(region_cols))
        )
        return arr, region_cols

    df = pd.read_csv(csv_path, index_col=0)
    if df.index.dtype.kind not in {"i", "u"}:
        raise ValueError(
            f"Prediction CSV {csv_path} must have integral sequence ids in the index."
        )
    counts = df.index.value_counts()
    if counts.nunique() != 1:
        raise ValueError(
            "Prediction sequences must all have the same length. Counts:\n"
            f"{counts}"
        )
    n_seq = counts.shape[0]
    n_time = counts.iloc[0]
    region_cols = df.columns.tolist()
    arr = df.to_numpy(dtype=np.float64).reshape(n_seq, n_time, len(region_cols))
    return arr, region_cols


def _load_and_align(predictions_csv: Path, ground_truth_csv: Path, neuro_cols: Optional[list[str]] = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pred_arr, pred_regions = _load_sequences(predictions_csv)
    gt_arr, gt_regions = _load_sequences(ground_truth_csv)

    if neuro_cols:
        overlap = [r for r in neuro_cols if r in gt_regions and r in pred_regions]
    else:
        overlap = [r for r in gt_regions if r in pred_regions]
    if not overlap:
        raise ValueError("No overlapping neural regions between GT and predictions.")

    gt_idx = [gt_regions.index(r) for r in overlap]
    pred_idx = [pred_regions.index(r) for r in overlap]
    gt_arr = gt_arr[..., gt_idx]
    pred_arr = pred_arr[..., pred_idx]

    max_len = min(gt_arr.shape[1], pred_arr.shape[1])
    gt_arr = gt_arr[:, :max_len, :]
    pred_arr = pred_arr[:, :max_len, :]
    return gt_arr, pred_arr, overlap


# --------------------------------------------------------------------------- #
# Core scoring logic                                                          #
# --------------------------------------------------------------------------- #


def _exec_notebook_metrics(
    notebook_path: Path,
    predictions_csv: Path,
    ground_truth_csv: Path,
    ddconfig_path: Path,
    *,
    skip_heavy_diagnostics: bool = True,
) -> dict:
    nb = nbformat.read(notebook_path, as_version=4)
    env: dict = {"__name__": "__main__"}

    env["display"] = lambda *args, **kwargs: None
    exec(
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "plt.show = lambda *args, **kwargs: None\n",
        env,
    )

    pred_str = str(predictions_csv)
    gt_str = str(ground_truth_csv)
    dd_str = str(ddconfig_path)

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        if skip_heavy_diagnostics:
            src = cell.source
            # These cells are expensive and do not contribute to METRICS/BUCKETS.
            if (
                "UMAP baseline embedding" in src
                or "Trajectory similarity in embedding space" in src
                or "Advanced dynamical similarity metrics" in src
                or "Animated trajectory visualizer" in src
                or "Trajectory similarity permutation test" in src
            ):
                continue
        src_lines = []
        for line in cell.source.splitlines():
            stripped = line.strip()
            if stripped.startswith("%") or stripped.startswith("!"):
                continue
            if stripped.startswith("preds_fname ="):
                line = f'preds_fname = r\"{pred_str}\"'
            elif stripped.startswith("gt_fname ="):
                line = f'gt_fname = r\"{gt_str}\"'
            elif stripped.startswith("ddconfig_path ="):
                line = f'ddconfig_path = r\"{dd_str}\"'
            src_lines.append(line)
        if not src_lines:
            continue
        exec("\n".join(src_lines), env)

    return env


def _ensure_ddconfig(ddconfig_path: Optional[Path]) -> tuple[Path, Optional[Path]]:
    if ddconfig_path is not None:
        return Path(ddconfig_path), None
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({"selected_columns_statistics": {}}, tmp)
    tmp.flush()
    tmp.close()
    return Path(tmp.name), Path(tmp.name)


def _flatten_scores(metrics: dict, buckets: dict, composite: float) -> Dict[str, float]:
    scores: Dict[str, float] = {k: float(v) for k, v in metrics.items()}
    for k, v in buckets.items():
        scores[f"bucket_{k}"] = float(v)
    scores["composite_score"] = float(composite)
    scores["FINAL_COMPOSITE_SCORE"] = float(composite)
    scores["FINAL_NEURO_COMPOSITE_SCORE"] = float(composite)
    return scores


def _compute_neuro_scores_from_notebook(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    ddconfig_path: Optional[Path] = None,
) -> Dict[str, float]:
    nb_path = Path(__file__).parent / "notebooks" / "final_implementation_benchmark.ipynb"
    if not nb_path.is_file():
        raise FileNotFoundError(f"Neuro notebook missing at {nb_path}")
    dd_path, cleanup_path = _ensure_ddconfig(ddconfig_path)

    env = _exec_notebook_metrics(
        nb_path,
        Path(predictions_csv),
        Path(ground_truth_csv),
        dd_path,
        skip_heavy_diagnostics=True,
    )
    metrics = env.get("METRICS")
    buckets = env.get("BUCKETS")
    composite = env.get("FINAL_COMPOSITE_SCORE", env.get("FINAL_NEURO_COMPOSITE_SCORE"))
    if metrics is None or buckets is None or composite is None:
        raise RuntimeError("Notebook execution did not produce METRICS/BUCKETS/FINAL_COMPOSITE_SCORE.")
    if cleanup_path and cleanup_path.exists():
        cleanup_path.unlink(missing_ok=True)
    return _flatten_scores(metrics, buckets, composite)


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
        pr_path = tmpdir_path / "pred.csv"

        seq_ids = np.repeat(np.arange(n_seq), n_time)
        item_pos = np.tile(np.arange(n_time), n_seq)
        gt_df = pd.DataFrame(gt.reshape(-1, n_reg), columns=region_names)
        gt_df.insert(0, "itemPosition", item_pos)
        gt_df.insert(0, "sequenceId", seq_ids)
        gt_df.to_csv(gt_path, index=False)

        pr_df = pd.DataFrame(pred.reshape(-1, n_reg), columns=region_names)
        pr_df.index = seq_ids
        pr_df.to_csv(pr_path)

        return _compute_neuro_scores_from_notebook(pr_path, gt_path, ddconfig_path=ddconfig_path)


def compute_neuro_scores(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    per_sequence_stats: bool = False,
    neuro_cols: Optional[list[str]] = None,
    ddconfig_path: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Compute neural plausibility scores using the bundled benchmark notebook logic.
    """
    if per_sequence_stats:
        raise ValueError("per_sequence_stats is not supported for notebook-based core scores.")

    if neuro_cols:
        gt, pred, overlap = _load_and_align(
            Path(predictions_csv),
            Path(ground_truth_csv),
            neuro_cols=neuro_cols,
        )
        return _compute_scores_from_arrays(gt, pred, region_names=overlap, ddconfig_path=ddconfig_path)

    return _compute_neuro_scores_from_notebook(
        Path(predictions_csv),
        Path(ground_truth_csv),
        ddconfig_path=ddconfig_path,
    )


def _timestamped_outdir(base: Optional[Path] = None, stem: Optional[str] = None) -> Path:
    base = Path(base) if base is not None else Path.cwd() / "outputs"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if stem:
        outdir = base / f"{stem}-analysis-{ts}"
    else:
        outdir = base / f"neuro-analysis-{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def run_neuro_full_analysis(
    predictions_csv: Path,
    ground_truth_csv: Path,
    ddconfig_path: Path,
    *,
    output_root: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Execute the bundled NeuroBench full analysis (notebook-derived script) and save figures.
    """
    from nbconvert.preprocessors import ExecutePreprocessor
    import nbformat

    nb_path = Path(__file__).parent / "notebooks" / "final_implementation_benchmark.ipynb"
    if not nb_path.is_file():
        raise FileNotFoundError(f"Neuro notebook missing at {nb_path}")

    preds_path = Path(predictions_csv)
    outdir = _timestamped_outdir(output_root, stem=preds_path.stem)

    nb = nbformat.read(nb_path, as_version=4)

    # Patch notebook cell(s) that hardcode file paths.
    pred_str = str(predictions_csv)
    gt_str = str(ground_truth_csv)
    dd_str = str(ddconfig_path)
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        lines = cell.source.splitlines()
        changed = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("preds_fname ="):
                lines[i] = f'preds_fname = r"{pred_str}"'
                changed = True
            elif stripped.startswith("gt_fname ="):
                lines[i] = f'gt_fname = r"{gt_str}"'
                changed = True
            elif stripped.startswith("ddconfig_path ="):
                lines[i] = f'ddconfig_path = r"{dd_str}"'
                changed = True
        if changed:
            cell.source = "\n".join(lines)

    patch = f"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
outdir = Path(r\"{outdir}\")
plot_counter = {{'n': 0}}
orig_show = plt.show
def saving_show(*args, **kwargs):
    figs = [plt.figure(num) for num in plt.get_fignums()]
    for fig in figs:
        plot_counter['n'] += 1
        fig.savefig(outdir / f\"figure_{{plot_counter['n']:03d}}.png\", dpi=200, bbox_inches=\"tight\")
    plt.close('all')
plt.show = saving_show
"""
    nb.cells.insert(0, nbformat.v4.new_code_cell(patch))

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": nb_path.parent}})
    executed_path = outdir / "executed_neurobench.ipynb"
    with executed_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    return {"output_dir": outdir}
