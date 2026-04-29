import pandas as pd
from pathlib import Path
import numpy as np
import contextlib
import io
import tempfile



def _align_arrays(
    gt_arr: np.ndarray, pred_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    gt = np.asarray(gt_arr, dtype=np.float64)
    pred = np.asarray(pred_arr, dtype=np.float64)
    if gt.ndim != 3 or pred.ndim != 3:
        raise ValueError(
            f"Expected [n_seq, T, n_reg] arrays, got {gt.shape} and {pred.shape}"
        )
    if gt.shape[0] != pred.shape[0] or gt.shape[2] != pred.shape[2]:
        raise ValueError(
            f"GT/pred sequence-region mismatch: {gt.shape} vs {pred.shape}"
        )
    if pred.shape[1] != gt.shape[1] and pred.shape[1] % gt.shape[1] == 0:
        factor = pred.shape[1] // gt.shape[1]
        pred = pred.reshape(pred.shape[0], gt.shape[1], factor, pred.shape[2]).mean(
            axis=2
        )
    elif pred.shape[1] != gt.shape[1]:
        keep = min(gt.shape[1], pred.shape[1])
        gt = gt[:, :keep, :]
        pred = pred[:, :keep, :]
    return gt, pred




def quiet_fidelity_from_arrays(
    gt_arr: np.ndarray, pred_arr: np.ndarray, *, region_names: list[str]
) -> dict[str, float]:
    """Computes fidelity scores headlessly by suppressing stdout/stderr via temp files."""
    sink = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="nethobench-synth-fidelity-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        gt_path = tmpdir_path / "gt.csv"
        pred_path = tmpdir_path / "pred.csv"

        dataset_to_sequence_frame(gt_arr, region_names).to_csv(gt_path, index=False)
        dataset_to_sequence_frame(pred_arr, region_names).to_csv(pred_path, index=False)

        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            return compute_fidelity_scores(pred_path, gt_path, neuro_cols=region_names)



def quiet_scores_from_arrays(
    gt_arr: np.ndarray, pred_arr: np.ndarray, *, region_names: list[str]
) -> dict[str, float]:
    """Computes neuro scores headlessly by suppressing stdout/stderr."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return load_and_run_neuro_full_analysis(gt_arr, pred_arr, region_names=region_names)



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


def _iqr_robust(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 10:
        s = np.nanstd(x)
        return float(s if np.isfinite(s) and s > 0 else 1.0)
    q25, q75 = np.nanquantile(x, [0.25, 0.75])
    s = float(q75 - q25)
    if not np.isfinite(s) or s <= 0:
        s = float(np.nanstd(x))
    return float(s if np.isfinite(s) and s > 0 else 1.0)