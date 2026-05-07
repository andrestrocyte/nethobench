import pandas as pd
from pathlib import Path
import pandas as pd
import numpy as np
import contextlib
import io
from typing import Optional

from nethobench.utils.validation import (
    validate_dataframe_schema,
    validate_alignment_overlap,
    validate_loaded_neuro_arrays,
)


def clip_fn(x: float, eps: float = 1e-6) -> float:
    return float(np.clip(x, eps, 1.0 - eps))


def geometric_mean_scores(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    arr = np.asarray([clip_fn(v) for v in arr], dtype=np.float64)
    return float(np.exp(np.mean(np.log(arr))))


def load_df(path: Path):
    path = Path(path).expanduser()
    if str(path).endswith(".csv"):
        return pd.read_csv(path, index_col=None, header=0, sep=",")
    elif str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise Exception(f"Only parquet and csv files allowed")


def collect_data_files(path: Path) -> list[Path]:
    path = Path(path).expanduser()
    if path.is_file():
        if path.suffix in {".csv", ".parquet"}:
            return [path]
        raise Exception(f"Only parquet and csv files allowed")
    if not path.is_dir():
        raise FileNotFoundError(f"{path} does not exist")
    files = sorted(
        [p for p in path.glob("*.parquet") if p.is_file()]
        + [p for p in path.glob("*.csv") if p.is_file()],
        key=lambda p: p.name,
    )
    if not files:
        raise FileNotFoundError(f"No parquet/csv files in {path}")
    return files


def load_dataframes(path: Path) -> pd.DataFrame:
    frames = []
    columns = None
    for file_path in collect_data_files(path):
        df = load_df(file_path)
        if columns is None:
            columns = list(df.columns)
        elif set(df.columns) != set(columns):
            missing = set(columns) - set(df.columns)
            extra = set(df.columns) - set(columns)
            raise ValueError(
                f"Header mismatch in {file_path}: missing {missing}, extra {extra}"
            )
        frames.append(df[columns])
    return pd.concat(frames, ignore_index=True)


def _assert_same_columns(gt_df: pd.DataFrame, inf_df: pd.DataFrame):
    diff1 = set(list(gt_df.columns)).difference(set(list(inf_df.columns)))
    diff2 = set(list(inf_df.columns)).difference(set(list(gt_df.columns)))

    assert (
        len(diff1) == 0 and len(diff2) == 0
    ), f"Unequal cols found: in gt but not preds: {diff1}, in preds but not gt: {diff2}"


def _tag_sequence_ids(df: pd.DataFrame, file_index: int, sequence_key: str):
    if sequence_key not in df.columns:
        return df
    out = df.copy()
    out[sequence_key] = out[sequence_key].map(lambda value: f"{file_index}:{value}")
    return out


def load_gt_and_preds(
    gt_dir: Path, inf_dir: Path, sequence_key: str = "sequenceId"
):
    gt_files = collect_data_files(gt_dir)
    inf_files = collect_data_files(inf_dir)

    if len(gt_files) == len(inf_files):
        pairs = list(zip(gt_files, inf_files))
    elif len(gt_files) == 1:
        pairs = [(gt_files[0], inf_file) for inf_file in inf_files]
    elif len(inf_files) == 1:
        pairs = [(gt_file, inf_files[0]) for gt_file in gt_files]
    else:
        raise ValueError(
            "Cannot infer GT/prediction pairing from directories with different "
            f"file counts: {len(gt_files)} GT files and {len(inf_files)} prediction files."
        )

    gt_frames = []
    inf_frames = []
    columns = None
    multi_file = len(pairs) > 1
    for idx, (gt_file, inf_file) in enumerate(pairs):
        gt_df = load_df(gt_file)
        inf_df = load_df(inf_file)
        _assert_same_columns(gt_df, inf_df)
        if columns is None:
            columns = list(gt_df.columns)
        elif set(gt_df.columns) != set(columns):
            missing = set(columns) - set(gt_df.columns)
            extra = set(gt_df.columns) - set(columns)
            raise ValueError(
                f"Header mismatch in {gt_file}: missing {missing}, extra {extra}"
            )
        gt_df = gt_df[columns]
        inf_df = inf_df[columns]
        if multi_file:
            gt_df = _tag_sequence_ids(gt_df, idx, sequence_key)
            inf_df = _tag_sequence_ids(inf_df, idx, sequence_key)
        gt_frames.append(gt_df)
        inf_frames.append(inf_df)

    gt_df = pd.concat(gt_frames, ignore_index=True)
    inf_df = pd.concat(inf_frames, ignore_index=True)
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

def load_and_align(
    predictions_csv: Path,
    ground_truth_csv: Path,
    neuro_cols: Optional[list[str]] = None,
    seq_key: str = "sequenceId",
    time_key: str = "itemPosition",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    
    # 1. Load tabular data
    gt_df = pd.read_csv(ground_truth_csv)
    pred_df = pd.read_csv(predictions_csv)

    # Safely format prediction dataframe (matching existing logic)
    if time_key not in pred_df.columns and pred_df.index.name != seq_key:
        pred_df = pd.read_csv(predictions_csv, index_col=0)
        pred_df.index.name = seq_key
        pred_df = pred_df.reset_index()
        # Create an itemPosition counter per sequence
        pred_df[time_key] = pred_df.groupby(seq_key).cumcount()

    # --- Validation: Schema & Index Integrity ---
    validate_dataframe_schema(gt_df, seq_key=seq_key, time_key=time_key)
    validate_dataframe_schema(pred_df, seq_key=seq_key, time_key=time_key)

    # 2. Find overlapping regions
    gt_regions = [c for c in gt_df.columns if c not in {seq_key, time_key}]
    pred_regions = [c for c in pred_df.columns if c not in {seq_key, time_key}]

    if neuro_cols:
        overlap = [r for r in neuro_cols if r in gt_regions and r in pred_regions]
    else:
        overlap = [r for r in gt_regions if r in pred_regions]

    if not overlap:
        raise ValueError("No overlapping neural regions between GT and predictions.")

    # 3. Apply the Inner Join logic
    merged = pd.merge(
        gt_df.sort_values([seq_key, time_key]),
        pred_df.sort_values([seq_key, time_key]),
        on=[seq_key, time_key],
        suffixes=("_gt", "_inf"),
        how="inner",
    )

    # --- Validation: Alignment & Overlap Guards ---
    validate_alignment_overlap(
        gt_df, pred_df, merged, seq_key=seq_key, time_key=time_key
    )

    # 4. Reconstruct the 3D Arrays
    gt_seqs, pred_seqs = [], []
    grouped = merged.groupby(seq_key)

    # CRITICAL: Find minimum sequence length to guarantee uniform 3D stacking
    min_seq_len = grouped.size().min()

    for _, seq_df in grouped:
        # Extract suffixed columns and truncate to uniform length
        gt_cols = [f"{c}_gt" for c in overlap]
        inf_cols = [f"{c}_inf" for c in overlap]

        gt_seqs.append(seq_df[gt_cols].iloc[:min_seq_len].to_numpy(dtype=np.float64))
        pred_seqs.append(seq_df[inf_cols].iloc[:min_seq_len].to_numpy(dtype=np.float64))

    gt_arr = np.stack(gt_seqs, axis=0)
    pred_arr = np.stack(pred_seqs, axis=0)

    # --- Validation: Data Quality & Mathematical Validity ---
    validate_loaded_neuro_arrays(gt_arr, pred_arr, overlap)

    return gt_arr, pred_arr, overlap


def quiet_call(func, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        return func(*args, **kwargs)


def timestamped_outdir(base: Path | None = None, prefix: str = "ethobench") -> Path:
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
