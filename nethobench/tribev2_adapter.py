from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys
import tempfile
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from .neuro.fidelity import compute_fidelity_scores
from .neuro.metrics.composites import (
    calculate_neuro_composites,
    compute_error_score,
    compute_mi_score,
)
from .neuro.metrics.definitions import (
    NEURO_FAMILY_WEIGHTS,
    compute_fidelity_composite,
)


SEQUENCE_KEY = "sequenceId"
TIME_KEY = "itemPosition"
DEFAULT_RESAMPLE_FREQUENCY_HZ = 1.0
DEFAULT_HEMODYNAMIC_LAG_SECONDS = 5.0
DEFAULT_PARCEL_SPACE = "hcp_mmp1_360_fsaverage5"
FSAVERAGE5_VERTICES_PER_HEMI = 10242


@dataclass(frozen=True)
class StimulusSpec:
    sequence_id: str
    events_csv: str | None = None
    video_path: str | None = None
    audio_path: str | None = None
    text_path: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "StimulusSpec":
        return cls(
            sequence_id=str(raw["sequence_id"]),
            events_csv=_optional_str(raw.get("events_csv")),
            video_path=_optional_str(raw.get("video_path")),
            audio_path=_optional_str(raw.get("audio_path")),
            text_path=_optional_str(raw.get("text_path")),
        )

    def validate(self) -> None:
        choices = [
            self.events_csv is not None,
            self.video_path is not None,
            self.audio_path is not None,
            self.text_path is not None,
        ]
        if sum(choices) != 1:
            raise ValueError(
                f"Stimulus {self.sequence_id!r} must specify exactly one of "
                "events_csv, video_path, audio_path, or text_path."
            )


@dataclass(frozen=True)
class SubjectGroundTruthSpec:
    sequence_id: str
    subject_id: str
    parcel_csv: str | None = None
    surface_left_path: str | None = None
    surface_right_path: str | None = None
    volume_path: str | None = None
    frequency_hz: float | None = None
    tr_seconds: float | None = None
    apply_preprocessing: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "SubjectGroundTruthSpec":
        return cls(
            sequence_id=str(raw["sequence_id"]),
            subject_id=str(raw["subject_id"]),
            parcel_csv=_optional_str(raw.get("parcel_csv")),
            surface_left_path=_optional_str(raw.get("surface_left_path")),
            surface_right_path=_optional_str(raw.get("surface_right_path")),
            volume_path=_optional_str(raw.get("volume_path")),
            frequency_hz=_optional_float(raw.get("frequency_hz")),
            tr_seconds=_optional_float(raw.get("tr_seconds")),
            apply_preprocessing=bool(raw.get("apply_preprocessing", True)),
        )

    def validate(self) -> None:
        choices = [
            self.parcel_csv is not None,
            self.surface_left_path is not None or self.surface_right_path is not None,
            self.volume_path is not None,
        ]
        if sum(bool(choice) for choice in choices) != 1:
            raise ValueError(
                f"Ground-truth item ({self.sequence_id!r}, {self.subject_id!r}) "
                "must specify exactly one source: parcel_csv, a surface pair, or volume_path."
            )
        if (self.surface_left_path is None) != (self.surface_right_path is None):
            raise ValueError(
                f"Ground-truth item ({self.sequence_id!r}, {self.subject_id!r}) must "
                "specify both surface_left_path and surface_right_path."
            )
        if self.parcel_csv is None and self.frequency_hz is None and self.tr_seconds is None:
            raise ValueError(
                f"Ground-truth item ({self.sequence_id!r}, {self.subject_id!r}) must specify "
                "frequency_hz or tr_seconds for raw surface/volume inputs."
            )


@dataclass(frozen=True)
class BenchmarkManifest:
    name: str
    dataset: str
    track: str
    tribev2_root: str | None = None
    tribev2_checkpoint: str = "facebook/tribev2"
    tribev2_cache_folder: str | None = None
    tribev2_cluster: str | None = None
    tribev2_device: str = "auto"
    output_root: str | None = None
    resample_frequency_hz: float = DEFAULT_RESAMPLE_FREQUENCY_HZ
    hemodynamic_lag_seconds: float = DEFAULT_HEMODYNAMIC_LAG_SECONDS
    apply_hemodynamic_shift: bool = True
    detrend: bool = True
    zscore: bool = True
    parcel_space: str = DEFAULT_PARCEL_SPACE
    random_seed: int = 7
    stimuli: list[StimulusSpec] = field(default_factory=list)
    subjects: list[SubjectGroundTruthSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "BenchmarkManifest":
        stimuli = [StimulusSpec.from_dict(item) for item in raw.get("stimuli", [])]
        subjects = [
            SubjectGroundTruthSpec.from_dict(item)
            for item in raw.get("subjects", [])
        ]
        manifest = cls(
            name=str(raw["name"]),
            dataset=str(raw["dataset"]),
            track=str(raw["track"]),
            tribev2_root=_optional_str(raw.get("tribev2_root")),
            tribev2_checkpoint=str(raw.get("tribev2_checkpoint", "facebook/tribev2")),
            tribev2_cache_folder=_optional_str(raw.get("tribev2_cache_folder")),
            tribev2_cluster=_optional_str(raw.get("tribev2_cluster")),
            tribev2_device=str(raw.get("tribev2_device", "auto")),
            output_root=_optional_str(raw.get("output_root")),
            resample_frequency_hz=float(
                raw.get("resample_frequency_hz", DEFAULT_RESAMPLE_FREQUENCY_HZ)
            ),
            hemodynamic_lag_seconds=float(
                raw.get("hemodynamic_lag_seconds", DEFAULT_HEMODYNAMIC_LAG_SECONDS)
            ),
            apply_hemodynamic_shift=bool(raw.get("apply_hemodynamic_shift", True)),
            detrend=bool(raw.get("detrend", True)),
            zscore=bool(raw.get("zscore", True)),
            parcel_space=str(raw.get("parcel_space", DEFAULT_PARCEL_SPACE)),
            random_seed=int(raw.get("random_seed", 7)),
            stimuli=stimuli,
            subjects=subjects,
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if not self.stimuli:
            raise ValueError("Manifest must contain at least one stimulus item.")
        if not self.subjects:
            raise ValueError("Manifest must contain at least one ground-truth subject item.")
        for stimulus in self.stimuli:
            stimulus.validate()
        for subject in self.subjects:
            subject.validate()

        stimulus_ids = {item.sequence_id for item in self.stimuli}
        if len(stimulus_ids) != len(self.stimuli):
            raise ValueError("Stimulus sequence_id values must be unique.")
        missing = sorted({item.sequence_id for item in self.subjects} - stimulus_ids)
        if missing:
            raise ValueError(
                f"Ground-truth items reference sequence_id values not present in stimuli: {missing}"
            )
        if self.parcel_space != DEFAULT_PARCEL_SPACE:
            raise ValueError(
                f"Unsupported parcel_space {self.parcel_space!r}. "
                f"Only {DEFAULT_PARCEL_SPACE!r} is supported."
            )


@dataclass(frozen=True)
class Parcelizer:
    parcel_columns: list[str]
    parcel_indices: list[np.ndarray]

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected [n_time, n_vertices], got shape {arr.shape}")
        if arr.shape[1] != 20484:
            raise ValueError(
                f"Expected fsaverage5 cortical data with 20,484 vertices, got {arr.shape[1]}"
            )
        out = np.empty((arr.shape[0], len(self.parcel_indices)), dtype=np.float64)
        for idx, parcel_idx in enumerate(self.parcel_indices):
            out[:, idx] = np.nanmean(arr[:, parcel_idx], axis=1)
        return out


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def load_manifest(path: Path) -> BenchmarkManifest:
    raw = json.loads(Path(path).read_text())
    return BenchmarkManifest.from_dict(raw)


def write_manifest_template(dataset: str, output_path: Path, *, name: str | None = None, tribev2_root: str | None = None) -> Path:
    dataset_key = dataset.lower()
    if dataset_key not in MANIFEST_TEMPLATES:
        raise ValueError(f"Unknown dataset template {dataset!r}.")
    payload = json.loads(json.dumps(MANIFEST_TEMPLATES[dataset_key]))
    if name:
        payload["name"] = name
    if tribev2_root:
        payload["tribev2_root"] = tribev2_root
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return out


def _resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _load_tribe_model(manifest: BenchmarkManifest):
    if manifest.tribev2_root:
        repo_root = str(_resolve_path(manifest.tribev2_root))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
    try:
        from tribev2 import TribeModel
    except Exception as exc:  # pragma: no cover - exercised only with optional runtime deps
        raise RuntimeError(
            "Could not import tribev2. Set tribev2_root in the manifest or install the "
            "package with its inference dependencies."
        ) from exc
    resolved_device = _resolve_tribev2_device(manifest.tribev2_device)
    extractor_device = "cuda" if resolved_device == "cuda" else "cpu"
    config_update = {
        "data.text_feature.device": extractor_device,
        "data.audio_feature.device": extractor_device,
        "data.video_feature.image.device": extractor_device,
    }
    model = TribeModel.from_pretrained(
        manifest.tribev2_checkpoint,
        cache_folder=manifest.tribev2_cache_folder,
        cluster=manifest.tribev2_cluster,
        device=resolved_device,
        config_update=config_update,
    )
    _apply_tribe_device_overrides(model, extractor_device)
    return model


def _resolve_tribev2_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _apply_tribe_device_overrides(model, device: str) -> None:
    data = getattr(model, "data", None)
    if data is None:
        return
    for feature_name in ("text_feature", "audio_feature"):
        feature = getattr(data, feature_name, None)
        if feature is not None and hasattr(feature, "device"):
            setattr(feature, "device", device)
    video_feature = getattr(data, "video_feature", None)
    if video_feature is not None:
        if hasattr(video_feature, "device"):
            setattr(video_feature, "device", device)
        image_feature = getattr(video_feature, "image", None)
        if image_feature is not None and hasattr(image_feature, "device"):
            setattr(image_feature, "device", device)


def _load_hcp_parcelizer(manifest: BenchmarkManifest) -> Parcelizer:
    try:
        import mne
    except Exception as exc:  # pragma: no cover - exercised only with optional runtime deps
        raise RuntimeError(
            "Could not import MNE. Install mne in the active NethoBench environment "
            "to build the HCP fsaverage5 parcelizer."
        ) from exc

    fsaverage_dir = Path(mne.datasets.fetch_fsaverage(verbose=True))
    subjects_dir = fsaverage_dir.parent
    mne.datasets.fetch_hcp_mmp_parcellation(
        subjects_dir=subjects_dir,
        accept=True,
        combine=False,
        verbose=True,
    )
    with mne.utils.use_log_level("error"):
        labels = mne.read_labels_from_annot(
            "fsaverage",
            "HCPMMP1",
            hemi="both",
            subjects_dir=subjects_dir,
        )
    left = _labels_to_fsaverage5_vertices(labels, hemi="lh")
    right = _labels_to_fsaverage5_vertices(labels, hemi="rh")
    left_names = sorted(left.keys())
    right_names = sorted(right.keys())
    parcel_columns = [f"LH_{name}" for name in left_names] + [f"RH_{name}" for name in right_names]
    parcel_indices = [np.asarray(left[name], dtype=np.int64) for name in left_names]
    parcel_indices += [np.asarray(right[name], dtype=np.int64) for name in right_names]
    if len(parcel_columns) != 360:
        raise RuntimeError(
            f"Expected 360 HCP parcels after hemisphere split, got {len(parcel_columns)}."
        )
    return Parcelizer(parcel_columns=parcel_columns, parcel_indices=parcel_indices)


def _labels_to_fsaverage5_vertices(labels: Iterable[object], *, hemi: str) -> dict[str, np.ndarray]:
    if hemi not in {"lh", "rh"}:
        raise ValueError(f"Unsupported hemisphere {hemi!r}.")
    index_offset = 0 if hemi == "lh" else FSAVERAGE5_VERTICES_PER_HEMI
    suffix = f"-{hemi}"
    label_to_vertices: dict[str, np.ndarray] = {}
    for label in labels:
        label_name = getattr(label, "name", "")
        if not isinstance(label_name, str) or not label_name.endswith(suffix):
            continue
        if label_name.startswith("???"):
            continue
        vertices = np.asarray(getattr(label, "vertices", ()), dtype=np.int64)
        vertices = vertices[vertices < FSAVERAGE5_VERTICES_PER_HEMI] + index_offset
        cleaned = label_name[2:].replace("_ROI", "").replace(suffix, "")
        label_to_vertices[cleaned] = vertices
    if len(label_to_vertices) != 180:
        raise RuntimeError(
            f"Expected 180 HCP cortical parcels in hemisphere {hemi}, got {len(label_to_vertices)}."
        )
    return label_to_vertices


def _standardize_events_frame(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if "timeline" not in df.columns:
        df["timeline"] = "default"
    if "subject" not in df.columns:
        df["subject"] = "default"
    return df


def _predict_stimulus_frame(model, parcelizer: Parcelizer, stimulus: StimulusSpec) -> pd.DataFrame:
    if stimulus.events_csv is not None:
        events = _standardize_events_frame(pd.read_csv(stimulus.events_csv))
    elif stimulus.video_path is not None:
        events = model.get_events_dataframe(video_path=stimulus.video_path)
    elif stimulus.audio_path is not None:
        events = model.get_events_dataframe(audio_path=stimulus.audio_path)
    elif stimulus.text_path is not None:
        events = model.get_events_dataframe(text_path=stimulus.text_path)
    else:  # pragma: no cover - validate() prevents this
        raise ValueError(f"No stimulus source provided for {stimulus.sequence_id!r}")

    preds, segments = model.predict(events=events)
    parcel_preds = parcelizer.transform(np.asarray(preds, dtype=np.float64))
    if parcel_preds.shape[0] != len(segments):
        raise RuntimeError("Mismatch between TRIBE predictions and returned segment metadata.")

    starts = []
    for idx, segment in enumerate(segments):
        start = getattr(segment, "start", idx)
        try:
            starts.append(int(round(float(start))))
        except Exception:
            starts.append(int(idx))

    out = pd.DataFrame(parcel_preds, columns=parcelizer.parcel_columns)
    out.insert(0, TIME_KEY, starts)
    out.insert(0, SEQUENCE_KEY, stimulus.sequence_id)
    out = (
        out.groupby([SEQUENCE_KEY, TIME_KEY], as_index=False)
        .mean(numeric_only=True)
        .sort_values([SEQUENCE_KEY, TIME_KEY], ignore_index=True)
    )
    return out


def _require_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only with optional runtime deps
        raise RuntimeError(
            "nibabel is required to load surface or volumetric fMRI inputs. "
            "Install it in the active NethoBench environment."
        ) from exc
    return nib


def _load_gifti_timeseries(path: Path) -> np.ndarray:
    nib = _require_nibabel()
    img = nib.load(str(path))
    if not hasattr(img, "darrays"):
        raise ValueError(f"Expected a GIFTI image at {path}")
    if len(img.darrays) == 0:
        raise ValueError(f"No data arrays found in {path}")
    arrays = [np.asarray(darray.data, dtype=np.float64) for darray in img.darrays]
    if len(arrays) == 1:
        arr = arrays[0]
        if arr.ndim != 2:
            raise ValueError(f"Unexpected single-array GIFTI shape {arr.shape} at {path}")
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        return arr
    return np.stack(arrays, axis=0)


def _downsample_surface_to_fsaverage5(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("Expected [n_time, n_vertices] per hemisphere.")
    if left.shape[0] != right.shape[0]:
        raise ValueError("Left/right hemisphere time dimensions must match.")

    n_left = left.shape[1]
    n_right = right.shape[1]
    expected = {163842, 40962, 10242, 2562, 642}
    if n_left != n_right or n_left not in expected:
        raise ValueError(
            "Unsupported surface vertex counts for fsaverage-style downsampling: "
            f"left={n_left}, right={n_right}"
        )
    if n_left < 10242:
        raise ValueError(
            f"Cannot upsample surfaces with only {n_left} vertices per hemisphere to fsaverage5."
        )
    left5 = left[:, :10242]
    right5 = right[:, :10242]
    return np.concatenate([left5, right5], axis=1)


def _load_volume_timeseries_to_fsaverage5(path: Path, manifest: BenchmarkManifest) -> np.ndarray:
    nib = _require_nibabel()
    if manifest.tribev2_root:
        repo_root = str(_resolve_path(manifest.tribev2_root))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
    try:
        from tribev2.utils_fmri import TribeSurfaceProjector
    except Exception as exc:  # pragma: no cover - exercised only with optional runtime deps
        raise RuntimeError(
            "Volume-to-surface projection requires tribev2.utils_fmri and its nilearn "
            "dependency to be importable."
        ) from exc
    projector = TribeSurfaceProjector(mesh="fsaverage5", kind="ball", radius=3.0, center_depth=0.5)
    img = nib.load(str(path))
    projected = projector.apply(img)
    arr = np.asarray(projected, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected projected volume shape {arr.shape}")
    if arr.shape[0] == 20484:
        arr = arr.T
    elif arr.shape[1] != 20484:
        raise ValueError(f"Projected data does not match fsaverage5 size: {arr.shape}")
    return arr


def _normalize_raw_timeseries(
    matrix: np.ndarray,
    *,
    source_frequency_hz: float,
    target_frequency_hz: float,
    zscore: bool,
    detrend: bool,
    apply_hemodynamic_shift: bool,
    hemodynamic_lag_seconds: float,
) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected [n_time, n_features], got shape {arr.shape}")

    if zscore:
        means = np.nanmean(arr, axis=0, keepdims=True)
        stds = np.nanstd(arr, axis=0, keepdims=True)
        stds = np.where(stds > 1e-12, stds, 1.0)
        arr = (arr - means) / stds
    if detrend:
        arr = signal.detrend(arr, axis=0, type="linear")
    if not np.isclose(source_frequency_hz, target_frequency_hz):
        arr = _resample_linear(arr, source_frequency_hz, target_frequency_hz)
    if apply_hemodynamic_shift and hemodynamic_lag_seconds > 0:
        lag_steps = int(round(hemodynamic_lag_seconds * target_frequency_hz))
        if lag_steps >= arr.shape[0]:
            raise ValueError(
                f"Hemodynamic lag of {lag_steps} samples exceeds sequence length {arr.shape[0]}."
            )
        arr = arr[lag_steps:]
    return arr


def _resample_linear(matrix: np.ndarray, source_frequency_hz: float, target_frequency_hz: float) -> np.ndarray:
    n_time = matrix.shape[0]
    if n_time <= 1:
        return matrix.copy()
    source_times = np.arange(n_time, dtype=np.float64) / float(source_frequency_hz)
    duration = source_times[-1]
    target_steps = max(int(np.floor(duration * target_frequency_hz)) + 1, 1)
    target_times = np.arange(target_steps, dtype=np.float64) / float(target_frequency_hz)
    out = np.empty((target_steps, matrix.shape[1]), dtype=np.float64)
    for col_idx in range(matrix.shape[1]):
        out[:, col_idx] = np.interp(target_times, source_times, matrix[:, col_idx])
    return out


def _subject_frequency_hz(spec: SubjectGroundTruthSpec) -> float:
    if spec.frequency_hz is not None:
        return float(spec.frequency_hz)
    if spec.tr_seconds is not None:
        return 1.0 / float(spec.tr_seconds)
    raise ValueError(
        f"Ground-truth item ({spec.sequence_id!r}, {spec.subject_id!r}) is missing frequency_hz/tr_seconds."
    )


def _load_parcel_frame(path: Path, parcel_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if {SEQUENCE_KEY, TIME_KEY}.issubset(frame.columns):
        seq_col = SEQUENCE_KEY
        time_col = TIME_KEY
    else:
        seq_col = None
        time_col = TIME_KEY if TIME_KEY in frame.columns else None

    numeric_cols = [
        col for col in frame.columns
        if col not in {SEQUENCE_KEY, TIME_KEY}
    ]
    if set(parcel_columns).issubset(frame.columns):
        out = frame[[col for col in [seq_col, time_col] if col] + parcel_columns].copy()
    elif len(numeric_cols) == len(parcel_columns):
        out = frame[[col for col in [seq_col, time_col] if col] + numeric_cols].copy()
        rename = {old: new for old, new in zip(numeric_cols, parcel_columns)}
        out = out.rename(columns=rename)
    else:
        raise ValueError(
            f"Parcel CSV {path} does not expose the expected 360 parcel columns."
        )

    if seq_col is None:
        out.insert(0, SEQUENCE_KEY, "sequence-0")
    if time_col is None:
        out.insert(1, TIME_KEY, np.arange(out.shape[0], dtype=int))
    out = out[[SEQUENCE_KEY, TIME_KEY, *parcel_columns]].copy()
    out[TIME_KEY] = out[TIME_KEY].astype(int)
    return out.sort_values([SEQUENCE_KEY, TIME_KEY], ignore_index=True)


def _load_ground_truth_frame(
    manifest: BenchmarkManifest,
    parcelizer: Parcelizer,
    spec: SubjectGroundTruthSpec,
) -> pd.DataFrame:
    if spec.parcel_csv is not None:
        frame = _load_parcel_frame(Path(spec.parcel_csv), parcelizer.parcel_columns)
        frame[SEQUENCE_KEY] = spec.sequence_id
        if not spec.apply_preprocessing:
            frame["subjectId"] = spec.subject_id
            return frame
        raw = frame[parcelizer.parcel_columns].to_numpy(dtype=np.float64)
        norm = _normalize_raw_timeseries(
            raw,
            source_frequency_hz=_subject_frequency_hz(spec),
            target_frequency_hz=manifest.resample_frequency_hz,
            zscore=manifest.zscore,
            detrend=manifest.detrend,
            apply_hemodynamic_shift=manifest.apply_hemodynamic_shift,
            hemodynamic_lag_seconds=manifest.hemodynamic_lag_seconds,
        )
        frame = _frame_from_array(spec.sequence_id, norm, parcelizer.parcel_columns)
        frame["subjectId"] = spec.subject_id
        return frame

    if spec.surface_left_path is not None and spec.surface_right_path is not None:
        left = _load_gifti_timeseries(Path(spec.surface_left_path))
        right = _load_gifti_timeseries(Path(spec.surface_right_path))
        raw_surface = _downsample_surface_to_fsaverage5(left, right)
    elif spec.volume_path is not None:
        raw_surface = _load_volume_timeseries_to_fsaverage5(Path(spec.volume_path), manifest)
    else:  # pragma: no cover - validate() prevents this
        raise ValueError("Unsupported ground-truth source.")

    normalized = _normalize_raw_timeseries(
        raw_surface,
        source_frequency_hz=_subject_frequency_hz(spec),
        target_frequency_hz=manifest.resample_frequency_hz,
        zscore=manifest.zscore,
        detrend=manifest.detrend,
        apply_hemodynamic_shift=manifest.apply_hemodynamic_shift,
        hemodynamic_lag_seconds=manifest.hemodynamic_lag_seconds,
    )
    parcel_arr = parcelizer.transform(normalized)
    frame = _frame_from_array(spec.sequence_id, parcel_arr, parcelizer.parcel_columns)
    frame["subjectId"] = spec.subject_id
    return frame


def _frame_from_array(sequence_id: str, arr: np.ndarray, parcel_columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(np.asarray(arr, dtype=np.float64), columns=parcel_columns)
    frame.insert(0, TIME_KEY, np.arange(frame.shape[0], dtype=int))
    frame.insert(0, SEQUENCE_KEY, sequence_id)
    return frame


def _weighted_average_score_dicts(score_dicts: list[dict[str, float]], weights: list[float]) -> dict[str, float]:
    keys = sorted({key for score_dict in score_dicts for key in score_dict.keys()})
    out: dict[str, float] = {}
    for key in keys:
        numer = 0.0
        denom = 0.0
        for score_dict, weight in zip(score_dicts, weights):
            value = float(score_dict.get(key, np.nan))
            if np.isfinite(value):
                numer += weight * value
                denom += weight
        out[key] = float(numer / denom) if denom > 0 else np.nan
    return out


def _compute_fidelity_from_arrays(gt_arr: np.ndarray, pred_arr: np.ndarray, parcel_columns: list[str]) -> dict[str, float]:
    del parcel_columns
    scores = {
        "Error_score": float(compute_error_score(gt_arr, pred_arr)),
        "MI_score": float(compute_mi_score(gt_arr, pred_arr)),
    }
    scores["Error_score01"] = scores["Error_score"]
    scores["MI_score01"] = scores["MI_score"]
    fidelity = float(compute_fidelity_composite(scores))
    scores["family_fidelity"] = fidelity
    scores["FIDELITY_SCORE"] = fidelity
    return scores


def _compute_score_bundle(reference_frame: pd.DataFrame, candidate_frame: pd.DataFrame, parcel_columns: list[str]) -> dict[str, object]:
    per_sequence: dict[str, object] = {}
    neuro_scores_per_seq: list[dict[str, float]] = []
    fidelity_scores_per_seq: list[dict[str, float]] = []
    weights: list[float] = []

    ref_sequences = {seq: frame for seq, frame in reference_frame.groupby(SEQUENCE_KEY)}
    cand_sequences = {seq: frame for seq, frame in candidate_frame.groupby(SEQUENCE_KEY)}
    for sequence_id in sorted(set(ref_sequences) & set(cand_sequences)):
        ref_seq = ref_sequences[sequence_id]
        cand_seq = cand_sequences[sequence_id]
        merged = pd.merge(
            ref_seq,
            cand_seq,
            on=[SEQUENCE_KEY, TIME_KEY],
            suffixes=("_gt", "_pred"),
            how="inner",
        ).sort_values([SEQUENCE_KEY, TIME_KEY], ignore_index=True)
        if merged.empty:
            continue
        gt = merged[[f"{col}_gt" for col in parcel_columns]].to_numpy(dtype=np.float64)[None, :, :]
        pred = merged[[f"{col}_pred" for col in parcel_columns]].to_numpy(dtype=np.float64)[None, :, :]
        neuro = calculate_neuro_composites(gt, pred)
        fidelity = _compute_fidelity_from_arrays(gt, pred, parcel_columns)
        weight = float(merged.shape[0])
        per_sequence[sequence_id] = {
            "n_timepoints": int(merged.shape[0]),
            "neuro_scores": neuro,
            "fidelity_scores": fidelity,
        }
        neuro_scores_per_seq.append(neuro)
        fidelity_scores_per_seq.append(fidelity)
        weights.append(weight)

    if not weights:
        raise ValueError("No overlapping aligned sequence/time rows were found for score computation.")

    return {
        "neuro_scores": _weighted_average_score_dicts(neuro_scores_per_seq, weights),
        "fidelity_scores": _weighted_average_score_dicts(fidelity_scores_per_seq, weights),
        "per_sequence": per_sequence,
    }


def _group_mean_frame(subject_frames: list[pd.DataFrame], parcel_columns: list[str]) -> pd.DataFrame:
    stacked = pd.concat(subject_frames, ignore_index=True)
    return (
        stacked.groupby([SEQUENCE_KEY, TIME_KEY], as_index=False)[parcel_columns]
        .mean()
        .sort_values([SEQUENCE_KEY, TIME_KEY], ignore_index=True)
    )


def _leave_one_out_group_frame(subject_frames: list[pd.DataFrame], parcel_columns: list[str], held_out_subject_id: str) -> pd.DataFrame:
    keep = [
        frame for frame in subject_frames
        if str(frame["subjectId"].iloc[0]) != str(held_out_subject_id)
    ]
    if not keep:
        raise ValueError(f"Cannot compute leave-one-out group for subject {held_out_subject_id!r}")
    return _group_mean_frame(keep, parcel_columns)


def _subject_baseline_report(subject_frames: list[pd.DataFrame], parcel_columns: list[str]) -> dict[str, object]:
    reports: dict[str, object] = {}
    neuro_dicts: list[dict[str, float]] = []
    fidelity_dicts: list[dict[str, float]] = []
    for frame in subject_frames:
        subject_id = str(frame["subjectId"].iloc[0])
        loo_group = _leave_one_out_group_frame(subject_frames, parcel_columns, subject_id)
        score_bundle = _compute_score_bundle(loo_group, frame.drop(columns=["subjectId"]), parcel_columns)
        reports[subject_id] = score_bundle
        neuro_dicts.append(score_bundle["neuro_scores"])
        fidelity_dicts.append(score_bundle["fidelity_scores"])
    return {
        "subjects": reports,
        "mean_neuro_scores": _mean_of_score_dicts(neuro_dicts),
        "median_neuro_scores": _median_of_score_dicts(neuro_dicts),
        "mean_fidelity_scores": _mean_of_score_dicts(fidelity_dicts),
        "median_fidelity_scores": _median_of_score_dicts(fidelity_dicts),
    }


def _mean_of_score_dicts(score_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not score_dicts:
        return {}
    keys = sorted({key for score_dict in score_dicts for key in score_dict.keys()})
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(score_dict.get(key, np.nan)) for score_dict in score_dicts], dtype=np.float64)
        values = values[np.isfinite(values)]
        out[key] = float(np.mean(values)) if values.size else np.nan
    return out


def _median_of_score_dicts(score_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not score_dicts:
        return {}
    keys = sorted({key for score_dict in score_dicts for key in score_dict.keys()})
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(score_dict.get(key, np.nan)) for score_dict in score_dicts], dtype=np.float64)
        values = values[np.isfinite(values)]
        out[key] = float(np.median(values)) if values.size else np.nan
    return out


def _split_half_report(subject_frames: list[pd.DataFrame], parcel_columns: list[str], seed: int) -> dict[str, object]:
    subject_ids = np.asarray(
        sorted({str(frame["subjectId"].iloc[0]) for frame in subject_frames}),
        dtype=object,
    )
    if subject_ids.size < 2:
        raise ValueError("Split-half ceiling requires at least two subjects.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(subject_ids.size)
    subject_ids = subject_ids[perm]
    split_idx = max(1, subject_ids.size // 2)
    left_ids = set(subject_ids[:split_idx].tolist())
    right_ids = set(subject_ids[split_idx:].tolist())
    if not right_ids:
        right_ids = {str(subject_ids[-1])}
        left_ids = set(subject_ids[:-1].tolist())
    left_frames = [
        frame.drop(columns=["subjectId"])
        for frame in subject_frames
        if str(frame["subjectId"].iloc[0]) in left_ids
    ]
    right_frames = [
        frame.drop(columns=["subjectId"])
        for frame in subject_frames
        if str(frame["subjectId"].iloc[0]) in right_ids
    ]
    left_group = _group_mean_frame(left_frames, parcel_columns)
    right_group = _group_mean_frame(right_frames, parcel_columns)
    bundle = _compute_score_bundle(left_group, right_group, parcel_columns)
    bundle["left_subject_ids"] = sorted(left_ids)
    bundle["right_subject_ids"] = sorted(right_ids)
    bundle["seed"] = int(seed)
    return bundle


def _normalize_scores_against_ceiling(score_dict: dict[str, float], ceiling_dict: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in score_dict.items():
        ceiling = float(ceiling_dict.get(key, np.nan))
        raw = float(value)
        if not np.isfinite(raw) or not np.isfinite(ceiling) or abs(ceiling) < 1e-12:
            out[key] = np.nan
        else:
            out[key] = float(raw / ceiling)
    return out


def _apply_corruption(frame: pd.DataFrame, parcel_columns: list[str], corruption_name: str, level: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = frame.copy()
    values = out[parcel_columns].to_numpy(dtype=np.float64)
    seq_ids = out[SEQUENCE_KEY].to_numpy()

    if corruption_name == "time_shuffle":
        corrupted = values.copy()
        for sequence_id in np.unique(seq_ids):
            sel = np.where(seq_ids == sequence_id)[0]
            perm = rng.permutation(sel.size)
            corrupted[sel] = corrupted[sel][perm]
    elif corruption_name == "region_permute":
        perm = rng.permutation(values.shape[1])
        blend = float(level)
        corrupted = ((1.0 - blend) * values) + (blend * values[:, perm])
    elif corruption_name == "lag_desync":
        corrupted = values.copy()
        for sequence_id in np.unique(seq_ids):
            sel = np.where(seq_ids == sequence_id)[0]
            max_shift = max(1, int(round(level * max(1, sel.size // 12))))
            for col_idx in range(values.shape[1]):
                shift = int(rng.integers(-max_shift, max_shift + 1))
                corrupted[sel, col_idx] = np.roll(corrupted[sel, col_idx], shift)
    elif corruption_name == "gain_scaling":
        gains = rng.lognormal(mean=0.0, sigma=0.35 * level, size=values.shape[1])
        biases = rng.normal(scale=0.15 * level, size=values.shape[1])
        corrupted = (values * gains[None, :]) + biases[None, :]
    elif corruption_name == "state_scrambling":
        corrupted = values.copy()
        for seq_idx, sequence_id in enumerate(np.unique(seq_ids)):
            sel = np.where(seq_ids == sequence_id)[0]
            corrupted[sel] = _state_scramble(values[sel], level=level, seed=seed + seq_idx)
    else:
        raise ValueError(f"Unknown corruption {corruption_name!r}")

    out.loc[:, parcel_columns] = corrupted
    return out


def _state_scramble(values: np.ndarray, *, level: float, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if values.shape[0] < 8:
        return values.copy()
    rng = np.random.default_rng(seed)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)
    n_components = max(2, min(8, scaled.shape[0] - 1, scaled.shape[1]))
    pca = PCA(n_components=n_components, random_state=seed)
    latent = pca.fit_transform(scaled)
    n_states = max(2, min(8, latent.shape[0] // 8))
    kmeans = KMeans(n_clusters=n_states, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(latent)
    centers = kmeans.cluster_centers_.copy()
    perm = rng.permutation(n_states)
    mapped = centers[perm][labels]
    mixed = ((1.0 - level) * latent) + (level * mapped)
    reconstructed = pca.inverse_transform(mixed)
    return scaler.inverse_transform(reconstructed)


def _corruption_ladder_report(
    pred_frame: pd.DataFrame,
    group_frame: pd.DataFrame,
    parcel_columns: list[str],
    seed: int,
) -> dict[str, object]:
    ladders: dict[str, object] = {}
    corruption_names = [
        "time_shuffle",
        "region_permute",
        "lag_desync",
        "gain_scaling",
        "state_scrambling",
    ]
    levels = [0.25, 0.50, 0.75, 1.0]
    for corr_idx, corruption_name in enumerate(corruption_names):
        runs = []
        for level_idx, level in enumerate(levels):
            corrupted = _apply_corruption(
                pred_frame,
                parcel_columns,
                corruption_name=corruption_name,
                level=float(level),
                seed=seed + (corr_idx * 100) + level_idx,
            )
            bundle = _compute_score_bundle(group_frame, corrupted, parcel_columns)
            runs.append(
                {
                    "level": float(level),
                    "neuro_scores": bundle["neuro_scores"],
                    "fidelity_scores": bundle["fidelity_scores"],
                }
            )
        ladders[corruption_name] = runs
    return ladders


def _plot_family_comparison(
    output_path: Path,
    *,
    tribe_scores: dict[str, float],
    human_scores: dict[str, float],
    ceiling_scores: dict[str, float],
) -> None:
    family_keys = [f"family_{name}" for name in NEURO_FAMILY_WEIGHTS] + ["FINAL_COMPOSITE_SCORE"]
    labels = [
        key.replace("family_", "").replace("_", " ").title().replace("Final Composite Score", "Composite")
        for key in family_keys
    ]
    x = np.arange(len(family_keys))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.bar(x - width, [tribe_scores.get(key, np.nan) for key in family_keys], width=width, label="TRIBE vs group", color="#355070")
    ax.bar(x, [human_scores.get(key, np.nan) for key in family_keys], width=width, label="Human subject median", color="#6d597a")
    ax.bar(x + width, [ceiling_scores.get(key, np.nan) for key in family_keys], width=width, label="Split-half ceiling", color="#b56576")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("TRIBE v2 vs human subjects vs split-half ceiling")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_corruption_ladder(output_path: Path, ladders: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    for corruption_name, runs in ladders.items():
        levels = [float(run["level"]) for run in runs]
        scores = [
            float(run["neuro_scores"].get("FINAL_COMPOSITE_SCORE", np.nan))
            for run in runs
        ]
        ax.plot(levels, scores, marker="o", linewidth=2.0, label=corruption_name.replace("_", " "))
    ax.set_xlabel("Corruption severity")
    ax.set_ylabel("Neuro composite")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("TRIBE corruption sensitivity ladder")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def export_tribev2_benchmark(manifest_path: Path, *, output_root: Path | None = None) -> dict[str, object]:
    manifest = load_manifest(Path(manifest_path))
    out_root = Path(output_root or manifest.output_root or (Path.cwd() / "outputs" / manifest.name))
    out_root.mkdir(parents=True, exist_ok=True)

    parcelizer = _load_hcp_parcelizer(manifest)
    model = _load_tribe_model(manifest)

    pred_frames = [
        _predict_stimulus_frame(model, parcelizer, stimulus)
        for stimulus in manifest.stimuli
    ]
    pred_frame = pd.concat(pred_frames, ignore_index=True)
    pred_frame = pred_frame.sort_values([SEQUENCE_KEY, TIME_KEY], ignore_index=True)

    subject_frames = [
        _load_ground_truth_frame(manifest, parcelizer, subject)
        for subject in manifest.subjects
    ]
    group_frame = _group_mean_frame(
        [frame.drop(columns=["subjectId"]) for frame in subject_frames],
        parcelizer.parcel_columns,
    )

    tribe_vs_group = _compute_score_bundle(group_frame, pred_frame, parcelizer.parcel_columns)
    human_vs_group = _subject_baseline_report(subject_frames, parcelizer.parcel_columns)
    split_half = _split_half_report(subject_frames, parcelizer.parcel_columns, seed=manifest.random_seed)
    corruption_ladder = _corruption_ladder_report(
        pred_frame,
        group_frame,
        parcelizer.parcel_columns,
        seed=manifest.random_seed,
    )

    pred_csv = out_root / "pred_parcels.csv"
    gt_group_csv = out_root / "gt_group_parcels.csv"
    parcel_config_json = out_root / "tribev2_parcel_config.json"
    human_scores_json = out_root / "human_subject_vs_group_scores.json"
    split_half_json = out_root / "split_half_ceiling_scores.json"
    corruption_json = out_root / "corruption_ladder_scores.json"
    report_json = out_root / "tribev2_benchmark_report.json"
    family_plot = out_root / "tribe_vs_human_vs_ceiling.png"
    ladder_plot = out_root / "corruption_sensitivity_ladder.png"

    pred_frame.to_csv(pred_csv, index=False)
    gt_group_csv_frame = group_frame[[SEQUENCE_KEY, TIME_KEY, *parcelizer.parcel_columns]]
    gt_group_csv_frame.to_csv(gt_group_csv, index=False)

    parcel_config = {
        "sequence_key": SEQUENCE_KEY,
        "time_key": TIME_KEY,
        "parcel_space": manifest.parcel_space,
        "parcel_columns": parcelizer.parcel_columns,
        "tribev2_checkpoint": manifest.tribev2_checkpoint,
        "dataset": manifest.dataset,
        "track": manifest.track,
        "resample_frequency_hz": manifest.resample_frequency_hz,
        "hemodynamic_lag_seconds": manifest.hemodynamic_lag_seconds,
        "apply_hemodynamic_shift": manifest.apply_hemodynamic_shift,
    }
    parcel_config_json.write_text(json.dumps(_json_ready(parcel_config), indent=2))
    human_scores_json.write_text(json.dumps(_json_ready(human_vs_group), indent=2))
    split_half_json.write_text(json.dumps(_json_ready(split_half), indent=2))
    corruption_json.write_text(json.dumps(_json_ready(corruption_ladder), indent=2))

    normalized = {
        "tribe_neuro_vs_ceiling": _normalize_scores_against_ceiling(
            tribe_vs_group["neuro_scores"],
            split_half["neuro_scores"],
        ),
        "tribe_fidelity_vs_ceiling": _normalize_scores_against_ceiling(
            tribe_vs_group["fidelity_scores"],
            split_half["fidelity_scores"],
        ),
        "human_median_neuro_vs_ceiling": _normalize_scores_against_ceiling(
            human_vs_group["median_neuro_scores"],
            split_half["neuro_scores"],
        ),
        "human_median_fidelity_vs_ceiling": _normalize_scores_against_ceiling(
            human_vs_group["median_fidelity_scores"],
            split_half["fidelity_scores"],
        ),
    }

    report = {
        "manifest": asdict(manifest),
        "outputs": {
            "pred_parcels_csv": pred_csv,
            "gt_group_parcels_csv": gt_group_csv,
            "tribev2_parcel_config_json": parcel_config_json,
            "human_subject_vs_group_scores_json": human_scores_json,
            "split_half_ceiling_scores_json": split_half_json,
            "corruption_ladder_scores_json": corruption_json,
            "family_comparison_plot": family_plot,
            "corruption_ladder_plot": ladder_plot,
        },
        "comparisons": {
            "tribe_vs_group": tribe_vs_group,
            "human_subject_vs_group": human_vs_group,
            "split_half_ceiling": split_half,
            "corruption_ladder": corruption_ladder,
        },
        "normalized": normalized,
    }

    _plot_family_comparison(
        family_plot,
        tribe_scores=tribe_vs_group["neuro_scores"],
        human_scores=human_vs_group["median_neuro_scores"],
        ceiling_scores=split_half["neuro_scores"],
    )
    _plot_corruption_ladder(ladder_plot, corruption_ladder)

    report_json.write_text(json.dumps(_json_ready(report), indent=2))
    return _json_ready(report)


def cli_export_main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tribev2_export_nethobench",
        description="Run the TRIBE v2 to NethoBench cortical realism export pipeline.",
    )
    parser.add_argument("--manifest", required=True, help="Path to the benchmark manifest JSON.")
    parser.add_argument("--output-root", help="Optional output directory override.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = export_tribev2_benchmark(
        Path(args.manifest),
        output_root=Path(args.output_root) if args.output_root else None,
    )
    outputs = report["outputs"]
    print("TRIBE v2 export complete.")
    print(f"Predictions      : {outputs['pred_parcels_csv']}")
    print(f"GT group         : {outputs['gt_group_parcels_csv']}")
    print(f"Parcel config    : {outputs['tribev2_parcel_config_json']}")
    print(f"Human baselines  : {outputs['human_subject_vs_group_scores_json']}")
    print(f"Split-half ceil. : {outputs['split_half_ceiling_scores_json']}")
    print(f"Benchmark report : {Path(outputs['pred_parcels_csv']).parent / 'tribev2_benchmark_report.json'}")


MANIFEST_TEMPLATES: dict[str, dict[str, object]] = {
    "lahner2024bold": {
        "name": "tribev2-lahner-shakedown",
        "dataset": "Lahner2024Bold",
        "track": "A",
        "tribev2_root": "/ABSOLUTE/PATH/TO/tribev2",
        "tribev2_checkpoint": "facebook/tribev2",
        "tribev2_cache_folder": "./cache/tribev2",
        "tribev2_device": "auto",
        "resample_frequency_hz": 1.0,
        "hemodynamic_lag_seconds": 5.0,
        "apply_hemodynamic_shift": True,
        "zscore": True,
        "detrend": True,
        "parcel_space": "hcp_mmp1_360_fsaverage5",
        "random_seed": 7,
        "stimuli": [
            {
                "sequence_id": "lahner_test_run01",
                "events_csv": "/ABSOLUTE/PATH/TO/lahner_test_run01_events.csv",
            }
        ],
        "subjects": [
            {
                "sequence_id": "lahner_test_run01",
                "subject_id": "sub-01",
                "surface_left_path": "/ABSOLUTE/PATH/TO/sub-01_hemi-L_space-fsaverage_bold.func.gii",
                "surface_right_path": "/ABSOLUTE/PATH/TO/sub-01_hemi-R_space-fsaverage_bold.func.gii",
                "tr_seconds": 1.75,
            },
            {
                "sequence_id": "lahner_test_run01",
                "subject_id": "sub-02",
                "surface_left_path": "/ABSOLUTE/PATH/TO/sub-02_hemi-L_space-fsaverage_bold.func.gii",
                "surface_right_path": "/ABSOLUTE/PATH/TO/sub-02_hemi-R_space-fsaverage_bold.func.gii",
                "tr_seconds": 1.75,
            },
        ],
    },
    "hcp": {
        "name": "tribev2-hcp-zero-shot",
        "dataset": "HCP",
        "track": "B",
        "tribev2_root": "/ABSOLUTE/PATH/TO/tribev2",
        "tribev2_checkpoint": "facebook/tribev2",
        "tribev2_cache_folder": "./cache/tribev2",
        "tribev2_device": "auto",
        "resample_frequency_hz": 1.0,
        "hemodynamic_lag_seconds": 5.0,
        "apply_hemodynamic_shift": True,
        "zscore": True,
        "detrend": True,
        "parcel_space": "hcp_mmp1_360_fsaverage5",
        "random_seed": 7,
        "stimuli": [
            {
                "sequence_id": "hcp_movie",
                "video_path": "/ABSOLUTE/PATH/TO/hcp_movie.mp4",
            }
        ],
        "subjects": [
            {
                "sequence_id": "hcp_movie",
                "subject_id": "100307",
                "volume_path": "/ABSOLUTE/PATH/TO/subject-100307_movie_bold.nii.gz",
                "tr_seconds": 1.0,
            },
            {
                "sequence_id": "hcp_movie",
                "subject_id": "100408",
                "volume_path": "/ABSOLUTE/PATH/TO/subject-100408_movie_bold.nii.gz",
                "tr_seconds": 1.0,
            },
        ],
    },
    "narratives": {
        "name": "tribev2-narratives-zero-shot",
        "dataset": "Narratives",
        "track": "B",
        "tribev2_root": "/ABSOLUTE/PATH/TO/tribev2",
        "tribev2_checkpoint": "facebook/tribev2",
        "tribev2_cache_folder": "./cache/tribev2",
        "tribev2_device": "auto",
        "resample_frequency_hz": 1.0,
        "hemodynamic_lag_seconds": 5.0,
        "apply_hemodynamic_shift": True,
        "zscore": True,
        "detrend": True,
        "parcel_space": "hcp_mmp1_360_fsaverage5",
        "random_seed": 7,
        "stimuli": [
            {
                "sequence_id": "narratives_story",
                "audio_path": "/ABSOLUTE/PATH/TO/narratives_story.wav",
            }
        ],
        "subjects": [
            {
                "sequence_id": "narratives_story",
                "subject_id": "sub-001",
                "volume_path": "/ABSOLUTE/PATH/TO/sub-001_story_bold.nii.gz",
                "tr_seconds": 1.5,
            },
            {
                "sequence_id": "narratives_story",
                "subject_id": "sub-002",
                "volume_path": "/ABSOLUTE/PATH/TO/sub-002_story_bold.nii.gz",
                "tr_seconds": 1.5,
            },
        ],
    },
}
