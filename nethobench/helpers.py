import pandas as pd
from pathlib import Path
import numpy as np
import contextlib
import io

from typing import Optional


def _clip01(x: float, eps: float = 1e-6) -> float:
    return float(np.clip(x, eps, 1.0 - eps))


def _geometric_mean_scores(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    arr = np.asarray([_clip01(v) for v in arr], dtype=np.float64)
    return float(np.exp(np.mean(np.log(arr))))


def load_df(path: Path):
    if str(path).endswith(".csv"):
        return pd.read_csv(path, index_col=None, header=0, sep=",")
    elif str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise Exception(f"Only parquet and csv files allowed")


def load_gt_and_preds(gt_dir: Path, inf_dir: Path):

    gt_df = load_df(gt_dir)
    inf_df = load_df(inf_dir)
    diff1 = set(list(gt_df.columns)).difference(set(list(inf_df.columns)))
    diff2 = set(list(inf_df.columns)).difference(set(list(gt_df.columns)))

    assert (
        len(diff1) == 0 and len(diff2) == 0
    ), f"Unequal cols found: in gt but not preds: {diff1}, in preds but not gt: {diff2}"

    return gt_df, inf_df


def _load_sequences(
    csv_path: Path,
    sequence_key: str = "sequenceId",
    time_key: str = "itemPosition",
) -> tuple[np.ndarray, list[str]]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if {sequence_key, time_key}.issubset(df.columns):
        df = df.sort_values([sequence_key, time_key]).reset_index(drop=True)
        region_cols = [
            column for column in df.columns if column not in {sequence_key, time_key}
        ]
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
        overlap = [
            region
            for region in neuro_cols
            if region in gt_regions and region in pred_regions
        ]
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


def _quiet_call(func, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        return func(*args, **kwargs)


def _timestamped_outdir(base: Path | None = None, prefix: str = "ethobench") -> Path:
    base = Path(base) if base is not None else Path.cwd() / "outputs"
    outdir = base / f"{prefix}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def get_region_names(n_regions: int, prefix: str = "synthetic-region") -> list[str]:
    """Generates standardized region column names."""
    return [f"{prefix}-{idx:02d}" for idx in range(1, n_regions + 1)]


def generate_orthogonal_matrix(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generates a random orthogonal matrix."""
    mat = rng.normal(size=(dim, dim))
    q, _ = np.linalg.qr(mat)
    return q


def get_module_assignments(n_regions: int, latent_dim: int) -> np.ndarray:
    """Calculates bin assignments for regions across latent dimensions."""
    bins = np.linspace(0, latent_dim, n_regions, endpoint=False)
    return np.floor(bins).astype(int)


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


def quiet_scores_from_arrays(
    gt_arr: np.ndarray, pred_arr: np.ndarray, *, region_names: list[str]
) -> dict[str, float]:
    """Computes neuro scores headlessly by suppressing stdout/stderr."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return _compute_scores_from_arrays(gt_arr, pred_arr, region_names=region_names)


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
