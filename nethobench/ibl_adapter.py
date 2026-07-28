from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
import json
import shutil
from pathlib import Path
import tempfile
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .neuro.fidelity import compute_fidelity_scores
from .neuro.metrics.composites import (
    calculate_neuro_composites,
    compute_error_score,
    compute_mi_score,
)
from .neuro.metrics.definitions import compute_fidelity_composite


SEQUENCE_KEY = "sequenceId"
TIME_KEY = "itemPosition"


DEFAULT_IBL_BASE_URL = "https://openalyx.internationalbrainlab.org"
DEFAULT_IBL_RELEASE_TAG = "RepeatedSite"
DEFAULT_IBL_SPIKE_SORTING_REVISION = "2024-05-06"
DEFAULT_ALIGNMENT_EVENT = "stimOn_times"
DEFAULT_WINDOW_SECONDS = (-0.5, 1.5)
DEFAULT_BIN_SIZE_SECONDS = 0.02


@dataclass(frozen=True)
class IBLManifest:
    name: str = "ibl-repeated-site-neuropixels"
    base_url: str = DEFAULT_IBL_BASE_URL
    username: str | None = None
    password: str | None = "international"
    release_tag: str = DEFAULT_IBL_RELEASE_TAG
    project: str | None = None
    spike_sorting_revision: str | None = DEFAULT_IBL_SPIKE_SORTING_REVISION
    alignment_event: str = DEFAULT_ALIGNMENT_EVENT
    window_seconds: tuple[float, float] = DEFAULT_WINDOW_SECONDS
    bin_size_seconds: float = DEFAULT_BIN_SIZE_SECONDS
    context_bins: int = 25
    target_bins: int = 75
    min_fr_hz: float = 0.2
    max_fr_hz: float = 100.0
    cluster_label_keep: list[int] = field(default_factory=lambda: [1])
    target_n_regions: int = 12
    min_n_regions: int = 8
    min_units_per_region: int = 5
    max_sessions: int = 24
    max_trials: int | None = 128
    max_unit_sessions: int = 5
    unit_level_top_n: int = 128
    disk_policy: str = "clean_raw_after_binning"
    min_free_disk_gb: float = 8.0
    feature_cache_dir: str | None = None
    split_policy: str = "within_session,cross_session,cross_lab"
    model_suite: list[str] = field(default_factory=lambda: ["psth", "glm_poisson", "var", "lds_var", "ssm_vae", "gru", "transformer", "transformer_nb_reg"])
    neural_seeds: list[int] = field(default_factory=lambda: [7, 11, 19])
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 64
    ssm_vae_latent_dim: int = 8
    ssm_vae_hidden_dim: int = 64
    ssm_vae_beta: float = 0.2
    ssm_vae_free_bits: float = 0.02
    ssm_vae_samples: int = 10
    ssm_vae_score_samples: int = 6
    bootstrap_samples: int = 500
    random_seed: int = 7
    output_root: str | None = None
    session_eids: list[str] = field(default_factory=list)
    predictions_csv: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "IBLManifest":
        window = raw.get("window_seconds", DEFAULT_WINDOW_SECONDS)
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            raise ValueError("window_seconds must contain exactly two values.")
        manifest = cls(
            name=str(raw.get("name", "ibl-repeated-site-neuropixels")),
            base_url=str(raw.get("base_url", DEFAULT_IBL_BASE_URL)),
            username=_optional_str(raw.get("username")),
            password=_optional_str(raw.get("password", "international")),
            release_tag=str(raw.get("release_tag", DEFAULT_IBL_RELEASE_TAG)),
            project=_optional_str(raw.get("project")),
            spike_sorting_revision=_optional_str(
                raw.get("spike_sorting_revision", DEFAULT_IBL_SPIKE_SORTING_REVISION)
            ),
            alignment_event=str(raw.get("alignment_event", DEFAULT_ALIGNMENT_EVENT)),
            window_seconds=(float(window[0]), float(window[1])),
            bin_size_seconds=float(raw.get("bin_size_seconds", DEFAULT_BIN_SIZE_SECONDS)),
            context_bins=int(raw.get("context_bins", 25)),
            target_bins=int(raw.get("target_bins", 75)),
            min_fr_hz=float(raw.get("min_fr_hz", 0.2)),
            max_fr_hz=float(raw.get("max_fr_hz", 100.0)),
            cluster_label_keep=[int(v) for v in raw.get("cluster_label_keep", [1])],
            target_n_regions=int(raw.get("target_n_regions", 12)),
            min_n_regions=int(raw.get("min_n_regions", 8)),
            min_units_per_region=int(raw.get("min_units_per_region", 5)),
            max_sessions=int(raw.get("max_sessions", 24)),
            max_trials=(
                None
                if raw.get("max_trials", 128) is None
                else int(raw.get("max_trials", 128))
            ),
            max_unit_sessions=int(raw.get("max_unit_sessions", 5)),
            unit_level_top_n=int(raw.get("unit_level_top_n", 128)),
            disk_policy=str(raw.get("disk_policy", "clean_raw_after_binning")),
            min_free_disk_gb=float(raw.get("min_free_disk_gb", 8.0)),
            feature_cache_dir=_optional_str(raw.get("feature_cache_dir")),
            split_policy=str(raw.get("split_policy", "within_session,cross_session,cross_lab")),
            model_suite=[str(v) for v in raw.get("model_suite", ["psth", "glm_poisson", "var", "lds_var", "ssm_vae", "gru", "transformer", "transformer_nb_reg"])],
            neural_seeds=[int(v) for v in raw.get("neural_seeds", [7, 11, 19])],
            max_epochs=int(raw.get("max_epochs", 100)),
            patience=int(raw.get("patience", 10)),
            batch_size=int(raw.get("batch_size", 64)),
            ssm_vae_latent_dim=int(raw.get("ssm_vae_latent_dim", 8)),
            ssm_vae_hidden_dim=int(raw.get("ssm_vae_hidden_dim", 64)),
            ssm_vae_beta=float(raw.get("ssm_vae_beta", 0.2)),
            ssm_vae_free_bits=float(raw.get("ssm_vae_free_bits", 0.02)),
            ssm_vae_samples=int(raw.get("ssm_vae_samples", 10)),
            ssm_vae_score_samples=int(raw.get("ssm_vae_score_samples", 6)),
            bootstrap_samples=int(raw.get("bootstrap_samples", 500)),
            random_seed=int(raw.get("random_seed", 7)),
            output_root=_optional_str(raw.get("output_root")),
            session_eids=[str(v) for v in raw.get("session_eids", [])],
            predictions_csv=_optional_str(raw.get("predictions_csv")),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.window_seconds[1] <= self.window_seconds[0]:
            raise ValueError("window_seconds end must be greater than start.")
        expected_bins = int(round((self.window_seconds[1] - self.window_seconds[0]) / self.bin_size_seconds))
        if expected_bins != self.context_bins + self.target_bins:
            raise ValueError(
                "context_bins + target_bins must match the bin count implied by "
                f"window_seconds/bin_size_seconds ({expected_bins})."
            )
        if self.bin_size_seconds <= 0:
            raise ValueError("bin_size_seconds must be positive.")
        if self.min_fr_hz < 0 or self.max_fr_hz <= self.min_fr_hz:
            raise ValueError("Firing-rate thresholds are invalid.")
        if self.min_n_regions <= 0 or self.target_n_regions < self.min_n_regions:
            raise ValueError("target_n_regions must be >= min_n_regions > 0.")
        if self.min_units_per_region <= 0:
            raise ValueError("min_units_per_region must be positive.")
        if self.disk_policy not in {"keep_raw", "clean_raw_after_binning"}:
            raise ValueError("disk_policy must be keep_raw or clean_raw_after_binning.")
        if self.min_free_disk_gb < 0:
            raise ValueError("min_free_disk_gb must be non-negative.")
        if self.ssm_vae_latent_dim <= 0 or self.ssm_vae_hidden_dim <= 0:
            raise ValueError("SSM-VAE latent and hidden dimensions must be positive.")
        if self.ssm_vae_beta < 0 or self.ssm_vae_free_bits < 0:
            raise ValueError("SSM-VAE beta and free-bits values must be non-negative.")
        if self.ssm_vae_samples <= 0:
            raise ValueError("ssm_vae_samples must be positive.")
        if self.ssm_vae_score_samples <= 0:
            raise ValueError("ssm_vae_score_samples must be positive.")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ibl_json_ready(value):
    if isinstance(value, dict):
        return {str(k): _ibl_json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_ibl_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_ibl_json_ready(v) for v in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def load_ibl_manifest(path: Path) -> IBLManifest:
    return IBLManifest.from_dict(json.loads(Path(path).read_text()))


def write_ibl_manifest_template(output_path: Path, *, name: str | None = None) -> Path:
    payload = asdict(IBLManifest(name=name or "ibl-repeated-site-neuropixels"))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_ibl_json_ready(payload), indent=2))
    return out


def _require_ibl_dependencies() -> tuple[object, object]:
    try:
        from one.api import ONE
        from brainbox.io.one import SpikeSortingLoader
    except Exception as exc:  # pragma: no cover - exercised only when optional deps are absent
        raise RuntimeError(
            "IBL support requires optional IBL packages. Install them with "
            "`python -m pip install ONE-api ibllib` and rerun the command."
        ) from exc
    return ONE, SpikeSortingLoader


def _make_one(manifest: IBLManifest):
    ONE, _ = _require_ibl_dependencies()
    kwargs: dict[str, object] = {"base_url": manifest.base_url, "silent": True}
    if manifest.username is not None:
        kwargs["username"] = manifest.username
    if manifest.password is not None:
        kwargs["password"] = manifest.password
    return ONE(**kwargs)


def _search_repeated_site_eids(one, manifest: IBLManifest) -> tuple[list[str], list[dict[str, object]]]:
    if manifest.session_eids:
        details = [_safe_one_details(one, eid) for eid in manifest.session_eids]
        return manifest.session_eids[: manifest.max_sessions], details[: manifest.max_sessions]

    attempts: list[dict[str, object]] = []
    if manifest.release_tag:
        attempts.extend(
            [
                {"tags": manifest.release_tag, "details": True},
                {"tag": manifest.release_tag, "details": True},
                {"dataset_types": "spikes.times", "details": True},
            ]
        )
    if manifest.project:
        attempts.append({"project": manifest.project, "details": True})
    attempts.append({"details": True})

    for kwargs in attempts:
        try:
            result = one.search(**kwargs)
        except Exception:
            continue
        eids, details = _normalize_one_search_result(result)
        if not eids:
            continue
        if manifest.release_tag and kwargs.get("details") is True:
            selected = _filter_details_by_tag(eids, details, manifest.release_tag)
            if selected[0]:
                eids, details = selected
        return eids[: manifest.max_sessions], details[: manifest.max_sessions]
    raise RuntimeError("Could not find IBL repeated-site sessions with ONE search.")


def _normalize_one_search_result(result: object) -> tuple[list[str], list[dict[str, object]]]:
    if isinstance(result, tuple) and len(result) >= 2:
        eids = [str(eid) for eid in result[0]]
        details = [dict(item) if isinstance(item, dict) else {} for item in result[1]]
        return eids, details
    if isinstance(result, list):
        return [str(eid) for eid in result], [{} for _ in result]
    try:
        values = list(result)
    except TypeError:
        values = []
    return [str(eid) for eid in values], [{} for _ in values]


def _filter_details_by_tag(
    eids: list[str],
    details: list[dict[str, object]],
    tag: str,
) -> tuple[list[str], list[dict[str, object]]]:
    tag_lower = tag.lower()
    selected_eids: list[str] = []
    selected_details: list[dict[str, object]] = []
    for eid, detail in zip(eids, details):
        raw_tags = detail.get("tags") or detail.get("tag") or []
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        if any(str(item).lower() == tag_lower for item in raw_tags):
            selected_eids.append(eid)
            selected_details.append(detail)
    return selected_eids, selected_details


def _safe_one_details(one, eid: str) -> dict[str, object]:
    try:
        detail = one.get_details(eid)
        return dict(detail) if isinstance(detail, dict) else {}
    except Exception:
        return {}


def _safe_list_collections(one, eid: str) -> list[str]:
    try:
        return [str(item) for item in one.list_collections(eid)]
    except Exception:
        return []


def _safe_list_datasets(one, eid: str) -> list[str]:
    try:
        return [str(item) for item in one.list_datasets(eid)]
    except Exception:
        return []


def _probe_collections(collections: Iterable[str]) -> list[str]:
    probes = []
    for collection in collections:
        parts = str(collection).split("/")
        if not parts or parts[0] != "alf":
            continue
        for idx, part in enumerate(parts):
            if part.startswith("probe"):
                probes.append("/".join(parts[: idx + 1]))
                break
    return sorted(set(probes))


def discover_ibl_repeated_site_sessions(manifest_path: Path, *, output_root: Path | None = None) -> dict[str, object]:
    manifest = load_ibl_manifest(Path(manifest_path))
    out_root = _resolve_output_root(manifest, output_root, suffix="discovery")
    one = _make_one(manifest)
    eids, details = _search_repeated_site_eids(one, manifest)

    rows = []
    for eid, detail in zip(eids, details):
        detail = detail or _safe_one_details(one, eid)
        collections = _safe_list_collections(one, eid)
        datasets = _safe_list_datasets(one, eid)
        rows.append(
            {
                "eid": eid,
                "lab": detail.get("lab", ""),
                "subject": detail.get("subject", ""),
                "date": detail.get("date", detail.get("start_time", "")),
                "n_collections": len(collections),
                "probe_collections": ";".join(_probe_collections(collections)),
                "has_spikes": any("spikes.times" in item for item in datasets),
                "has_clusters": any("clusters." in item for item in datasets),
                "has_trials": any("trials." in item for item in datasets),
                "collections": ";".join(collections[:40]),
            }
        )

    inventory = pd.DataFrame(rows)
    inventory_path = out_root / "session_inventory.csv"
    report_path = out_root / "ibl_discovery_report.json"
    inventory.to_csv(inventory_path, index=False)
    report = {
        "manifest": asdict(manifest),
        "n_sessions": int(inventory.shape[0]),
        "n_labs": int(inventory["lab"].nunique()) if "lab" in inventory else 0,
        "inventory_csv": inventory_path,
        "sessions": rows,
    }
    report_path.write_text(json.dumps(_ibl_json_ready(report), indent=2))
    return _ibl_json_ready({"inventory_csv": inventory_path, "report_json": report_path, "report": report})


def _resolve_output_root(manifest: IBLManifest, output_root: Path | None, *, suffix: str | None = None) -> Path:
    base = Path(output_root or manifest.output_root or (Path.cwd() / "outputs" / manifest.name))
    if suffix:
        base = base / suffix
    base.mkdir(parents=True, exist_ok=True)
    return base


def _cluster_values(clusters: object, key: str, n_clusters: int, default: object) -> np.ndarray:
    if isinstance(clusters, dict):
        value = clusters.get(key, None)
    else:
        try:
            value = clusters[key]
        except Exception:
            value = getattr(clusters, key, None)
    if value is None:
        return np.full(n_clusters, default, dtype=object)
    arr = np.asarray(value)
    if arr.shape[0] < n_clusters:
        padded = np.full(n_clusters, default, dtype=arr.dtype if arr.dtype != object else object)
        padded[: arr.shape[0]] = arr
        return padded
    return arr[:n_clusters]


def _stable_region_name(acronym: object) -> str:
    text = str(acronym or "unknown").strip()
    if not text or text.lower() in {"nan", "none", "void", "root"}:
        return "unknown"
    upper = text.upper()
    if upper.startswith(("CA1", "CA2", "CA3", "DG", "SUB", "POST", "PRE", "HPF")):
        return "hippocampus"
    if upper.startswith(("LP", "PO", "POL", "RT", "LG", "MD", "VAL", "VM", "VPM", "TH")):
        return "thalamus"
    if upper.startswith(("VISA", "VISAM", "VISPM", "VISRL", "RSP", "PTLP")):
        return "posterior_parietal"
    cleaned = "".join(ch for ch in upper if ch.isalpha())
    return cleaned.lower() if cleaned else "unknown"


def filter_good_units(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    clusters: object,
    *,
    min_fr_hz: float = 0.2,
    max_fr_hz: float = 100.0,
    cluster_label_keep: Iterable[int] = (1,),
) -> pd.DataFrame:
    spike_times = np.asarray(spike_times, dtype=np.float64)
    spike_clusters = np.asarray(spike_clusters, dtype=np.int64)
    if spike_times.shape[0] != spike_clusters.shape[0]:
        raise ValueError("spike_times and spike_clusters must have matching length.")
    n_clusters = int(max(np.max(spike_clusters) + 1 if spike_clusters.size else 0, 0))
    labels = _cluster_values(clusters, "label", n_clusters, 1)
    acronyms = _cluster_values(clusters, "acronym", n_clusters, "unknown")
    if np.all(acronyms == "unknown"):
        acronyms = _cluster_values(clusters, "atlas_acronym", n_clusters, "unknown")

    duration = float(np.nanmax(spike_times) - np.nanmin(spike_times)) if spike_times.size else 0.0
    duration = max(duration, 1e-9)
    counts = np.bincount(spike_clusters, minlength=n_clusters).astype(np.float64)
    fr_hz = counts / duration
    keep_labels = {int(item) for item in cluster_label_keep}
    rows = []
    for cluster_id in range(n_clusters):
        label = labels[cluster_id]
        try:
            label_int = int(label)
        except Exception:
            label_int = 1 if str(label).lower() in {"good", "valid"} else -1
        if label_int not in keep_labels:
            continue
        if not (min_fr_hz <= fr_hz[cluster_id] <= max_fr_hz):
            continue
        acronym = acronyms[cluster_id]
        region = _stable_region_name(acronym)
        if region == "unknown":
            continue
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "label": label_int,
                "acronym": str(acronym),
                "region": region,
                "fr_hz": float(fr_hz[cluster_id]),
                "n_spikes": int(counts[cluster_id]),
            }
        )
    return pd.DataFrame(rows)


def bin_spike_counts(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    unit_ids: Iterable[int],
    event_times: np.ndarray,
    *,
    window_seconds: tuple[float, float] = DEFAULT_WINDOW_SECONDS,
    bin_size_seconds: float = DEFAULT_BIN_SIZE_SECONDS,
) -> np.ndarray:
    spike_times = np.asarray(spike_times, dtype=np.float64)
    spike_clusters = np.asarray(spike_clusters, dtype=np.int64)
    unit_ids = np.asarray(list(unit_ids), dtype=np.int64)
    event_times = np.asarray(event_times, dtype=np.float64)
    if spike_times.shape[0] != spike_clusters.shape[0]:
        raise ValueError("spike_times and spike_clusters must have matching length.")
    n_bins = int(round((window_seconds[1] - window_seconds[0]) / bin_size_seconds))
    if n_bins <= 0:
        raise ValueError("Binning window must contain at least one bin.")
    out = np.zeros((event_times.shape[0], n_bins, unit_ids.shape[0]), dtype=np.float64)
    unit_to_index = {int(unit_id): idx for idx, unit_id in enumerate(unit_ids.tolist())}
    for trial_idx, event_time in enumerate(event_times):
        start = float(event_time + window_seconds[0])
        stop = float(event_time + window_seconds[1])
        left = int(np.searchsorted(spike_times, start, side="left"))
        right = int(np.searchsorted(spike_times, stop, side="left"))
        if right <= left:
            continue
        local_times = spike_times[left:right] - start
        local_bins = np.floor(local_times / bin_size_seconds).astype(np.int64)
        local_clusters = spike_clusters[left:right]
        for cluster_id, bin_idx in zip(local_clusters, local_bins):
            unit_idx = unit_to_index.get(int(cluster_id))
            if unit_idx is not None and 0 <= bin_idx < n_bins:
                out[trial_idx, bin_idx, unit_idx] += 1.0
    return out


def aggregate_region_rates(
    unit_counts: np.ndarray,
    unit_table: pd.DataFrame,
    *,
    bin_size_seconds: float = DEFAULT_BIN_SIZE_SECONDS,
    region_panel: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if unit_counts.ndim != 3:
        raise ValueError("unit_counts must have shape [n_sequences, n_time, n_units].")
    if unit_counts.shape[2] != unit_table.shape[0]:
        raise ValueError("unit_table rows must match unit_counts units.")
    regions = list(region_panel or sorted(unit_table["region"].astype(str).unique()))
    out = np.zeros((unit_counts.shape[0], unit_counts.shape[1], len(regions)), dtype=np.float64)
    for region_idx, region in enumerate(regions):
        unit_idx = np.where(unit_table["region"].astype(str).to_numpy() == region)[0]
        if unit_idx.size:
            out[:, :, region_idx] = unit_counts[:, :, unit_idx].mean(axis=2) / bin_size_seconds
    return out, regions


def choose_region_panel(
    session_region_counts: list[dict[str, int]],
    *,
    target_n_regions: int,
    min_n_regions: int,
    min_units_per_region: int,
) -> list[str]:
    if not session_region_counts:
        raise ValueError("No sessions available for region-panel selection.")
    eligible: dict[str, int] = {}
    median_units: dict[str, float] = {}
    all_regions = sorted({region for counts in session_region_counts for region in counts})
    for region in all_regions:
        values = np.asarray([counts.get(region, 0) for counts in session_region_counts], dtype=np.float64)
        support = int(np.sum(values >= min_units_per_region))
        if support == len(session_region_counts):
            eligible[region] = support
            median_units[region] = float(np.median(values))
    ranked = sorted(eligible, key=lambda r: (eligible[r], median_units[r], r), reverse=True)
    panel = ranked[:target_n_regions]
    if len(panel) < min_n_regions:
        raise ValueError(
            f"Only {len(panel)} regions passed min_units_per_region={min_units_per_region}; "
            f"need at least {min_n_regions}."
        )
    return sorted(panel)


def zscore_from_train_context(
    arr: np.ndarray,
    train_indices: Iterable[int],
    *,
    context_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=np.float64)
    train_indices = np.asarray(list(train_indices), dtype=np.int64)
    if arr.ndim != 3:
        raise ValueError("Expected arr with shape [n_sequences, n_time, n_channels].")
    if train_indices.size == 0:
        raise ValueError("At least one training index is required for normalization.")
    context_bins = min(int(context_bins), arr.shape[1])
    ref = arr[train_indices, :context_bins, :].reshape(-1, arr.shape[2])
    mean = np.nanmean(ref, axis=0)
    std = np.nanstd(ref, axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    return (arr - mean[None, None, :]) / std[None, None, :], mean, std


def array_to_nethobench_frame(arr: np.ndarray, channel_names: list[str], *, sequence_prefix: str = "seq") -> pd.DataFrame:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("Expected arr with shape [n_sequences, n_time, n_channels].")
    if arr.shape[2] != len(channel_names):
        raise ValueError("channel_names length must match arr.shape[2].")
    n_seq, n_time, n_channels = arr.shape
    frame = pd.DataFrame(arr.reshape(-1, n_channels), columns=channel_names)
    frame.insert(0, TIME_KEY, np.tile(np.arange(n_time, dtype=int), n_seq))
    seq_ids = [f"{sequence_prefix}_{idx:05d}" for idx in range(n_seq)]
    frame.insert(0, SEQUENCE_KEY, np.repeat(seq_ids, n_time))
    return frame


def _split_indices(n_sequences: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n_sequences < 4:
        raise ValueError("At least four sequences are required for split-half and baseline scoring.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_sequences)
    split = n_sequences // 2
    return np.sort(perm[:split]), np.sort(perm[split:])


def poisson_psth_prediction(train: np.ndarray, n_sequences: int) -> np.ndarray:
    template = np.nanmean(train, axis=0, keepdims=True)
    return np.repeat(template, n_sequences, axis=0)


def var_rollout_prediction(train: np.ndarray, contexts: np.ndarray, *, context_bins: int, ridge: float = 1e-3) -> np.ndarray:
    train = np.asarray(train, dtype=np.float64)
    contexts = np.asarray(contexts, dtype=np.float64)
    if train.ndim != 3 or contexts.ndim != 3:
        raise ValueError("train and contexts must be [n_sequences, n_time, n_channels].")
    n_time = train.shape[1]
    if context_bins <= 0 or context_bins >= n_time:
        raise ValueError("context_bins must split the sequence into context and target bins.")
    X = train[:, :-1, :].reshape(-1, train.shape[2])
    Y = train[:, 1:, :].reshape(-1, train.shape[2])
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    eye = np.eye(X_aug.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0
    coef = np.linalg.solve(X_aug.T @ X_aug + ridge * eye, X_aug.T @ Y)
    pred = contexts.copy()
    for t in range(context_bins, n_time):
        prev_aug = np.concatenate([pred[:, t - 1, :], np.ones((pred.shape[0], 1))], axis=1)
        pred[:, t, :] = prev_aug @ coef
    return pred


def apply_ibl_corruption(arr: np.ndarray, corruption_name: str, *, level: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.asarray(arr, dtype=np.float64)
    out = values.copy()
    if corruption_name == "time_shuffle":
        for seq_idx in range(out.shape[0]):
            perm = rng.permutation(out.shape[1])
            out[seq_idx] = out[seq_idx, perm, :]
    elif corruption_name == "channel_permute":
        perm = rng.permutation(out.shape[2])
        out = (1.0 - level) * out + level * out[:, :, perm]
    elif corruption_name == "lag_jitter":
        max_shift = max(1, int(round(level * max(1, out.shape[1] // 8))))
        for seq_idx in range(out.shape[0]):
            for channel_idx in range(out.shape[2]):
                out[seq_idx, :, channel_idx] = np.roll(out[seq_idx, :, channel_idx], int(rng.integers(-max_shift, max_shift + 1)))
    elif corruption_name == "dropout":
        mask = rng.random(out.shape) < float(level)
        out[mask] = 0.0
    elif corruption_name == "gain_scaling":
        gains = rng.lognormal(mean=0.0, sigma=0.35 * float(level), size=out.shape[2])
        biases = rng.normal(scale=0.10 * float(level), size=out.shape[2])
        out = out * gains[None, None, :] + biases[None, None, :]
    else:
        raise ValueError(f"Unknown IBL corruption {corruption_name!r}.")
    return out


def _score_arrays(gt: np.ndarray, pred: np.ndarray, channel_names: list[str]) -> dict[str, float]:
    del channel_names
    return calculate_neuro_composites(gt, pred)


def _score_arrays_or_error(gt: np.ndarray, pred: np.ndarray, channel_names: list[str]) -> dict[str, object]:
    try:
        return {"neuro_scores": _score_arrays(gt, pred, channel_names)}
    except Exception as exc:
        return {
            "neuro_scores": {
                "FINAL_COMPOSITE_SCORE": float("nan"),
                "family_distribution": float("nan"),
                "family_temporal_spectral": float("nan"),
                "family_relational": float("nan"),
                "family_geometry": float("nan"),
                "family_state_dynamics": float("nan"),
            },
            "score_error": f"{type(exc).__name__}: {exc}",
        }


def _fidelity_arrays(gt: np.ndarray, pred: np.ndarray, channel_names: list[str]) -> dict[str, float]:
    del channel_names
    scores = {
        "Error_score": float(compute_error_score(gt, pred)),
        "MI_score": float(compute_mi_score(gt, pred)),
    }
    scores["Error_score01"] = scores["Error_score"]
    scores["MI_score01"] = scores["MI_score"]
    fidelity = float(compute_fidelity_composite(scores))
    scores["family_fidelity"] = fidelity
    scores["FIDELITY_SCORE"] = fidelity
    return scores


def _load_session_region_data(one, manifest: IBLManifest, eid: str) -> dict[str, object]:
    collections = _probe_collections(_safe_list_collections(one, eid))
    if not collections:
        raise RuntimeError(f"No probe collections found for session {eid}.")

    unit_tables = []
    unit_count_arrays = []
    event_times = _load_trial_event_times(one, eid, manifest.alignment_event)
    if manifest.max_trials is not None and event_times.shape[0] > manifest.max_trials:
        rng = np.random.default_rng(manifest.random_seed)
        event_times = np.sort(rng.choice(event_times, size=int(manifest.max_trials), replace=False))
    for probe_idx, collection in enumerate(collections):
        spike_collection = _resolve_spike_collection(one, eid, collection, manifest.spike_sorting_revision)
        actual_revision = manifest.spike_sorting_revision or "latest"
        spike_times = np.asarray(
            one.load_dataset(eid, "spikes.times.npy", collection=spike_collection, revision=manifest.spike_sorting_revision),
            dtype=np.float64,
        )
        spike_clusters = np.asarray(
            one.load_dataset(eid, "spikes.clusters.npy", collection=spike_collection, revision=manifest.spike_sorting_revision),
            dtype=np.int64,
        )
        clusters = _load_direct_clusters(one, eid, spike_collection, manifest.spike_sorting_revision)
        units = filter_good_units(
            spike_times,
            spike_clusters,
            clusters,
            min_fr_hz=manifest.min_fr_hz,
            max_fr_hz=manifest.max_fr_hz,
            cluster_label_keep=manifest.cluster_label_keep,
        )
        if units.empty:
            continue
        unit_ids = units["cluster_id"].to_numpy(dtype=np.int64)
        counts = bin_spike_counts(
            spike_times,
            spike_clusters,
            unit_ids,
            event_times,
            window_seconds=manifest.window_seconds,
            bin_size_seconds=manifest.bin_size_seconds,
        )
        units = units.copy()
        units["probe_collection"] = spike_collection
        units["probe_unit_index"] = np.arange(units.shape[0], dtype=int)
        units["actual_revision"] = actual_revision
        unit_tables.append(units)
        unit_count_arrays.append(counts)

    if not unit_tables:
        raise RuntimeError(f"No good units remained after filtering for session {eid}.")
    unit_table = pd.concat(unit_tables, ignore_index=True)
    unit_counts = np.concatenate(unit_count_arrays, axis=2)
    detail = _safe_one_details(one, eid)
    return {
        "eid": eid,
        "detail": detail,
        "event_times": event_times,
        "unit_table": unit_table,
        "unit_counts": unit_counts,
        "region_counts": unit_table.groupby("region").size().astype(int).to_dict(),
    }


def _resolve_spike_collection(one, eid: str, probe_collection: str, revision: str | None) -> str:
    datasets = [str(item) for item in one.list_datasets(eid)]
    prefix = f"{probe_collection}/"
    candidates = []
    for dataset in datasets:
        if not dataset.startswith(prefix) or not dataset.endswith("spikes.times.npy"):
            continue
        if "pykilosort" in dataset:
            collection = dataset.rsplit("/spikes.times.npy", 1)[0]
            if "/#" in collection:
                collection = collection.split("/#", 1)[0]
            candidates.append(collection)
    if candidates:
        return sorted(set(candidates), key=lambda c: ("pykilosort" not in c, len(c), c))[0]
    return probe_collection


def _load_direct_clusters(one, eid: str, collection: str, revision: str | None) -> dict[str, np.ndarray]:
    metrics = one.load_dataset(eid, "clusters.metrics.pqt", collection=collection, revision=revision)
    channels = np.asarray(one.load_dataset(eid, "clusters.channels.npy", collection=collection, revision=revision), dtype=np.int64)
    channel_ids = np.asarray(
        one.load_dataset(eid, "channels.brainLocationIds_ccf_2017.npy", collection=collection, revision=revision),
        dtype=np.int64,
    )
    n_clusters = int(max(int(metrics["cluster_id"].max()) + 1, channels.shape[0]))
    labels = np.zeros(n_clusters, dtype=np.float64)
    labels[np.asarray(metrics["cluster_id"], dtype=np.int64)] = np.asarray(metrics["label"], dtype=np.float64)
    cluster_channels = np.clip(channels[:n_clusters], 0, max(0, channel_ids.shape[0] - 1))
    atlas_ids = channel_ids[cluster_channels]
    acronyms = _atlas_ids_to_acronyms(atlas_ids)
    return {"label": labels, "acronym": acronyms}


def _atlas_ids_to_acronyms(atlas_ids: np.ndarray) -> np.ndarray:
    try:
        from iblatlas.regions import BrainRegions

        regions = BrainRegions()
        return np.asarray(regions.id2acronym(atlas_ids), dtype=object)
    except Exception:
        return np.asarray([str(item) for item in atlas_ids], dtype=object)


def _load_trial_event_times(one, eid: str, alignment_event: str) -> np.ndarray:
    attr = alignment_event.replace("trials.", "").replace("_ibl_trials.", "")
    candidates = [
        alignment_event,
        f"trials.{attr}",
        f"_ibl_trials.{attr}",
        f"_ibl_trials.{attr}.npy",
    ]
    for dataset in candidates:
        try:
            values = one.load_dataset(eid, dataset, collection="alf")
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                return arr
        except Exception:
            continue
    try:
        trials = one.load_object(eid, "trials", collection="alf")
        values = trials[attr]
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            return arr
    except Exception:
        pass
    raise RuntimeError(f"Could not load trial alignment event {alignment_event!r} for {eid}.")


def _build_region_dataset(session_data: list[dict[str, object]], manifest: IBLManifest) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    panel = choose_region_panel(
        [item["region_counts"] for item in session_data],
        target_n_regions=manifest.target_n_regions,
        min_n_regions=manifest.min_n_regions,
        min_units_per_region=manifest.min_units_per_region,
    )
    arrays = []
    rows = []
    for item in session_data:
        unit_table = item["unit_table"]
        eligible_mask = unit_table["region"].isin(panel).to_numpy()
        eligible = unit_table.loc[eligible_mask].reset_index(drop=True)
        if eligible.empty:
            continue
        counts = item["unit_counts"][:, :, eligible_mask]
        region_rates, _ = aggregate_region_rates(
            counts,
            eligible,
            bin_size_seconds=manifest.bin_size_seconds,
            region_panel=panel,
        )
        transformed = np.log1p(region_rates)
        arrays.append(transformed)
        detail = item.get("detail", {})
        for trial_idx in range(transformed.shape[0]):
            rows.append(
                {
                    "sequence_index": len(rows),
                    "eid": item["eid"],
                    "trial_index": trial_idx,
                    "lab": detail.get("lab", ""),
                    "subject": detail.get("subject", ""),
                }
            )
    if not arrays:
        raise RuntimeError("No session had enough units for the selected region panel.")
    return np.concatenate(arrays, axis=0), panel, pd.DataFrame(rows)


def _evaluate_region_dataset(
    arr: np.ndarray,
    channels: list[str],
    manifest: IBLManifest,
    *,
    output_root: Path,
) -> dict[str, object]:
    train_idx, test_idx = _split_indices(arr.shape[0], manifest.random_seed)
    norm, mean, std = zscore_from_train_context(arr, train_idx, context_bins=manifest.context_bins)
    train = norm[train_idx]
    test = norm[test_idx]

    half_a = test[::2]
    half_b = test[1::2]
    if half_a.shape[0] == 0 or half_b.shape[0] == 0:
        half_a = train[: min(train.shape[0], test.shape[0])]
        half_b = test[: half_a.shape[0]]
    n_half = min(half_a.shape[0], half_b.shape[0])
    half_a = half_a[:n_half]
    half_b = half_b[:n_half]

    poisson = poisson_psth_prediction(train, test.shape[0])
    var_pred = var_rollout_prediction(train, test.copy(), context_bins=manifest.context_bins)
    gt = test

    gt_path = output_root / "gt_region_rates.csv"
    poisson_path = output_root / "pred_poisson_region_rates.csv"
    var_path = output_root / "pred_var_region_rates.csv"
    array_to_nethobench_frame(gt, channels, sequence_prefix="ibl_test").to_csv(gt_path, index=False)
    array_to_nethobench_frame(poisson, channels, sequence_prefix="ibl_test").to_csv(poisson_path, index=False)
    array_to_nethobench_frame(var_pred, channels, sequence_prefix="ibl_test").to_csv(var_path, index=False)

    split_half_scores = {
        "neuro_scores": _score_arrays(half_a, half_b, channels),
        "fidelity_scores": _fidelity_arrays(half_a, half_b, channels),
    }
    model_scores = {
        "poisson_psth": {
            "neuro_scores": _score_arrays(gt, poisson, channels),
            "fidelity_scores": _fidelity_arrays(gt, poisson, channels),
        },
        "var": {
            "neuro_scores": _score_arrays(gt, var_pred, channels),
            "fidelity_scores": _fidelity_arrays(gt, var_pred, channels),
        },
    }
    corruption_ladder = _corruption_ladder(gt, channels, manifest.random_seed)

    region_config = {
        "sequence_key": SEQUENCE_KEY,
        "time_key": TIME_KEY,
        "regions": channels,
        "bin_size_seconds": manifest.bin_size_seconds,
        "window_seconds": manifest.window_seconds,
        "context_bins": manifest.context_bins,
        "target_bins": manifest.target_bins,
        "normalization": {
            "mean": mean,
            "std": std,
            "fit": "train sequences, context bins only",
        },
    }
    outputs = {
        "gt_region_rates_csv": gt_path,
        "pred_poisson_region_rates_csv": poisson_path,
        "pred_var_region_rates_csv": var_path,
        "ibl_region_config_json": output_root / "ibl_region_config.json",
        "ibl_split_half_ceiling_scores_json": output_root / "ibl_split_half_ceiling_scores.json",
        "ibl_model_scores_json": output_root / "ibl_model_scores.json",
        "ibl_corruption_ladder_scores_json": output_root / "ibl_corruption_ladder_scores.json",
        "family_comparison_plot": output_root / "ibl_family_comparison.png",
        "corruption_ladder_plot": output_root / "ibl_corruption_ladder.png",
    }
    outputs["ibl_region_config_json"].write_text(json.dumps(_ibl_json_ready(region_config), indent=2))
    outputs["ibl_split_half_ceiling_scores_json"].write_text(json.dumps(_ibl_json_ready(split_half_scores), indent=2))
    outputs["ibl_model_scores_json"].write_text(json.dumps(_ibl_json_ready(model_scores), indent=2))
    outputs["ibl_corruption_ladder_scores_json"].write_text(json.dumps(_ibl_json_ready(corruption_ladder), indent=2))
    _plot_family_comparison(
        outputs["family_comparison_plot"],
        split_half_scores["neuro_scores"],
        model_scores,
    )
    _plot_corruption_ladder(outputs["corruption_ladder_plot"], corruption_ladder)
    return {
        "outputs": outputs,
        "split_half_ceiling": split_half_scores,
        "model_scores": model_scores,
        "corruption_ladder": corruption_ladder,
    }


def _corruption_ladder(gt: np.ndarray, channels: list[str], seed: int) -> dict[str, object]:
    names = ["time_shuffle", "channel_permute", "lag_jitter", "dropout", "gain_scaling"]
    levels = [0.25, 0.50, 0.75, 1.0]
    out: dict[str, object] = {}
    for name_idx, name in enumerate(names):
        runs = []
        for level_idx, level in enumerate(levels):
            pred = apply_ibl_corruption(gt, name, level=level, seed=seed + 100 * name_idx + level_idx)
            run = {"level": level}
            run.update(_score_arrays_or_error(gt, pred, channels))
            runs.append(run)
        out[name] = runs
    return out


def _plot_family_comparison(output_path: Path, split_half_scores: dict[str, float], model_scores: dict[str, object]) -> None:
    family_keys = [
        "family_distribution",
        "family_temporal_spectral",
        "family_relational",
        "family_geometry",
        "family_state_dynamics",
        "FINAL_COMPOSITE_SCORE",
    ]
    labels = ["Distribution", "Temporal", "Relational", "Geometry", "State Dyn.", "Composite"]
    series = [("Split-half", split_half_scores)]
    for model_name, payload in model_scores.items():
        series.append((model_name.replace("_", " ").title(), payload["neuro_scores"]))
    x = np.arange(len(family_keys))
    width = min(0.25, 0.82 / max(1, len(series)))
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    for idx, (label, scores) in enumerate(series):
        offset = (idx - (len(series) - 1) / 2.0) * width
        ax.bar(x + offset, [float(scores.get(key, np.nan)) for key in family_keys], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("NethoBench score")
    ax.set_title("IBL repeated-site Neuropixels validation")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_corruption_ladder(output_path: Path, ladder: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    for name, runs in ladder.items():
        levels = [float(run["level"]) for run in runs]
        scores = [float(run["neuro_scores"].get("FINAL_COMPOSITE_SCORE", np.nan)) for run in runs]
        ax.plot(levels, scores, marker="o", linewidth=2.0, label=name.replace("_", " "))
    ax.set_xlabel("Corruption severity")
    ax.set_ylabel("Neuro composite")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("IBL corruption sensitivity ladder")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _evaluate_unit_level_supplement(
    session_data: list[dict[str, object]],
    manifest: IBLManifest,
    output_root: Path,
) -> dict[str, object]:
    supplement_root = output_root / "unit_level"
    supplement_root.mkdir(parents=True, exist_ok=True)
    ranked_sessions = sorted(
        session_data,
        key=lambda item: int(item["unit_table"].shape[0]),
        reverse=True,
    )[: manifest.max_unit_sessions]
    reports = []
    skipped = []
    for item in ranked_sessions:
        eid = str(item["eid"])
        unit_table = item["unit_table"].sort_values("fr_hz", ascending=False)
        if unit_table.shape[0] < min(16, manifest.unit_level_top_n):
            skipped.append({"eid": eid, "reason": "too_few_units", "n_units": int(unit_table.shape[0])})
            continue
        unit_table = unit_table.head(manifest.unit_level_top_n)
        counts = item["unit_counts"][:, :, unit_table.index.to_numpy(dtype=np.int64)]
        if counts.shape[0] < 4:
            skipped.append({"eid": eid, "reason": "too_few_trials", "n_trials": int(counts.shape[0])})
            continue
        arr = np.sqrt(counts)
        channels = [
            f"unit_{idx:03d}_cluster_{int(row.cluster_id)}"
            for idx, row in enumerate(unit_table.itertuples(index=False))
        ]
        safe_eid = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in eid)
        try:
            eval_report = _evaluate_unit_dataset(arr, channels, manifest, supplement_root, safe_eid)
        except Exception as exc:
            skipped.append({"eid": eid, "reason": "score_failed", "error": str(exc)})
            continue
        reports.append(
            {
                "eid": eid,
                "n_trials": int(arr.shape[0]),
                "n_time_bins": int(arr.shape[1]),
                "n_units": int(arr.shape[2]),
                **eval_report,
            }
        )
    summary = {
        "sessions": reports,
        "skipped": skipped,
        "note": "Unit-level channels are session-local units and are scored only within session.",
    }
    summary_path = supplement_root / "ibl_unit_level_supplement_scores.json"
    summary_path.write_text(json.dumps(_ibl_json_ready(summary), indent=2))
    summary["summary_json"] = summary_path
    return _ibl_json_ready(summary)


def _evaluate_unit_dataset(
    arr: np.ndarray,
    channels: list[str],
    manifest: IBLManifest,
    output_root: Path,
    stem: str,
) -> dict[str, object]:
    train_idx, test_idx = _split_indices(arr.shape[0], manifest.random_seed)
    norm, mean, std = zscore_from_train_context(arr, train_idx, context_bins=manifest.context_bins)
    train = norm[train_idx]
    test = norm[test_idx]
    poisson = poisson_psth_prediction(train, test.shape[0])
    var_pred = var_rollout_prediction(train, test.copy(), context_bins=manifest.context_bins)
    gt_path = output_root / f"{stem}_gt_unit_counts.csv"
    poisson_path = output_root / f"{stem}_pred_poisson_unit_counts.csv"
    var_path = output_root / f"{stem}_pred_var_unit_counts.csv"
    array_to_nethobench_frame(test, channels, sequence_prefix=f"{stem}_unit_test").to_csv(gt_path, index=False)
    array_to_nethobench_frame(poisson, channels, sequence_prefix=f"{stem}_unit_test").to_csv(poisson_path, index=False)
    array_to_nethobench_frame(var_pred, channels, sequence_prefix=f"{stem}_unit_test").to_csv(var_path, index=False)
    scores = {
        "poisson_psth": {"neuro_scores": _score_arrays(test, poisson, channels)},
        "var": {"neuro_scores": _score_arrays(test, var_pred, channels)},
    }
    scores_path = output_root / f"{stem}_unit_level_scores.json"
    scores_path.write_text(json.dumps(_ibl_json_ready(scores), indent=2))
    return {
        "gt_unit_counts_csv": gt_path,
        "pred_poisson_unit_counts_csv": poisson_path,
        "pred_var_unit_counts_csv": var_path,
        "unit_level_scores_json": scores_path,
        "model_scores": scores,
        "normalization": {
            "mean": mean,
            "std": std,
            "fit": "train trials, context bins only",
        },
    }


def export_ibl_repeated_site_benchmark(
    manifest_path: Path,
    *,
    output_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    manifest = load_ibl_manifest(Path(manifest_path))
    out_root = _resolve_output_root(manifest, output_root)
    if dry_run:
        deps_available = True
        deps_error = None
        try:
            _require_ibl_dependencies()
        except RuntimeError as exc:
            deps_available = False
            deps_error = str(exc)
        report = {
            "manifest": asdict(manifest),
            "dry_run": True,
            "optional_dependencies_available": deps_available,
            "dependency_error": deps_error,
            "expected_outputs": _expected_output_names(out_root),
        }
        report_path = out_root / "ibl_repeated_site_report.json"
        report_path.write_text(json.dumps(_ibl_json_ready(report), indent=2))
        return _ibl_json_ready({"report_json": report_path, "report": report})

    one = _make_one(manifest)
    eids, details = _search_repeated_site_eids(one, manifest)
    session_data = []
    failures = []
    for eid in eids:
        try:
            session_data.append(_load_session_region_data(one, manifest, eid))
        except Exception as exc:
            failures.append({"eid": eid, "error": str(exc)})
    if not session_data:
        raise RuntimeError(f"No IBL sessions could be loaded. Failures: {failures[:5]}")

    region_arr, region_panel, sequence_table = _build_region_dataset(session_data, manifest)
    eval_report = _evaluate_region_dataset(region_arr, region_panel, manifest, output_root=out_root)
    sequence_table_path = out_root / "ibl_sequence_table.csv"
    sequence_table.to_csv(sequence_table_path, index=False)
    report_path = out_root / "ibl_repeated_site_report.json"
    report = {
        "manifest": asdict(manifest),
        "n_loaded_sessions": len(session_data),
        "n_failed_sessions": len(failures),
        "failures": failures,
        "region_panel": region_panel,
        "n_sequences": int(region_arr.shape[0]),
        "n_time_bins": int(region_arr.shape[1]),
        "n_regions": int(region_arr.shape[2]),
        "sequence_table_csv": sequence_table_path,
        "outputs": eval_report["outputs"],
        "split_half_ceiling": eval_report["split_half_ceiling"],
        "model_scores": eval_report["model_scores"],
        "corruption_ladder": eval_report["corruption_ladder"],
        "unit_level_supplement": _evaluate_unit_level_supplement(session_data, manifest, out_root),
    }
    report_path.write_text(json.dumps(_ibl_json_ready(report), indent=2))
    report["outputs"]["ibl_repeated_site_report_json"] = report_path
    return _ibl_json_ready(report)


def _feature_cache_root(manifest: IBLManifest, out_root: Path) -> Path:
    root = Path(manifest.feature_cache_dir) if manifest.feature_cache_dir else out_root / "feature_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _free_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return float(usage.free / (1024**3))


def _assert_disk_available(path: Path, manifest: IBLManifest) -> None:
    free_gb = _free_disk_gb(path)
    if free_gb < manifest.min_free_disk_gb:
        raise RuntimeError(
            f"Free disk at {path} is {free_gb:.1f}GB, below min_free_disk_gb={manifest.min_free_disk_gb:.1f}."
        )


def _one_cache_root() -> Path:
    return Path.home() / "Downloads" / "ONE"


def _snapshot_large_one_files() -> set[Path]:
    root = _one_cache_root()
    if not root.exists():
        return set()
    names = {"spikes.times.npy", "spikes.clusters.npy", "spikes.amps.npy", "spikes.depths.npy"}
    return {p for p in root.rglob("*") if p.is_file() and p.name in names and p.stat().st_size > 10_000_000}


def _clean_owned_raw_files(before: set[Path], *, enabled: bool) -> list[str]:
    if not enabled:
        return []
    removed: list[str] = []
    after = _snapshot_large_one_files()
    for path in sorted(after - before):
        try:
            path.unlink()
            removed.append(str(path))
        except OSError:
            pass
    return removed


def _select_diverse_sessions(
    eids: list[str],
    details: list[dict[str, object]],
    *,
    max_sessions: int,
) -> tuple[list[str], list[dict[str, object]]]:
    rows = []
    for eid, detail in zip(eids, details):
        detail = detail or {}
        rows.append(
            {
                "eid": eid,
                "detail": detail,
                "lab": str(detail.get("lab", "")),
                "subject": str(detail.get("subject", "")),
                "date": str(detail.get("date", detail.get("start_time", ""))),
            }
        )
    rows.sort(key=lambda row: (row["lab"], row["subject"], row["date"], row["eid"]))
    selected: list[dict[str, object]] = []
    used_eids: set[str] = set()
    while len(selected) < max_sessions:
        added = False
        used_labs = {str(row["lab"]) for row in selected}
        for row in rows:
            if row["eid"] in used_eids:
                continue
            if len(used_labs) < len({str(item["lab"]) for item in rows}) and row["lab"] in used_labs:
                continue
            selected.append(row)
            used_eids.add(str(row["eid"]))
            added = True
            if len(selected) >= max_sessions:
                break
        if not added:
            for row in rows:
                if row["eid"] not in used_eids:
                    selected.append(row)
                    used_eids.add(str(row["eid"]))
                    added = True
                    break
        if not added:
            break
    return [str(row["eid"]) for row in selected], [dict(row["detail"]) for row in selected]


def _session_feature_paths(feature_root: Path, eid: str) -> dict[str, Path]:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in eid)
    session_root = feature_root / safe
    session_root.mkdir(parents=True, exist_ok=True)
    return {
        "root": session_root,
        "npz": session_root / "session_unit_counts.npz",
        "units": session_root / "unit_table.csv",
        "meta": session_root / "session_meta.json",
    }


def _save_session_feature(item: dict[str, object], feature_root: Path) -> dict[str, object]:
    paths = _session_feature_paths(feature_root, str(item["eid"]))
    np.savez_compressed(
        paths["npz"],
        unit_counts=np.asarray(item["unit_counts"], dtype=np.float32),
        event_times=np.asarray(item["event_times"], dtype=np.float64),
    )
    unit_table = item["unit_table"]
    unit_table.to_csv(paths["units"], index=False)
    detail = dict(item.get("detail", {}))
    meta = {
        "eid": item["eid"],
        "detail": detail,
        "region_counts": item["region_counts"],
        "unit_counts_npz": paths["npz"],
        "unit_table_csv": paths["units"],
    }
    paths["meta"].write_text(json.dumps(_ibl_json_ready(meta), indent=2))
    return _ibl_json_ready(meta)


def _load_session_feature(meta: dict[str, object]) -> dict[str, object]:
    with np.load(Path(str(meta["unit_counts_npz"]))) as data:
        unit_counts = np.asarray(data["unit_counts"], dtype=np.float64)
        event_times = np.asarray(data["event_times"], dtype=np.float64)
    unit_table = pd.read_csv(str(meta["unit_table_csv"]))
    return {
        "eid": str(meta["eid"]),
        "detail": dict(meta.get("detail", {})),
        "event_times": event_times,
        "unit_table": unit_table,
        "unit_counts": unit_counts,
        "region_counts": {str(k): int(v) for k, v in dict(meta["region_counts"]).items()},
    }


def build_ibl_repeated_site_dataset(
    manifest_path: Path,
    *,
    output_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    manifest = load_ibl_manifest(Path(manifest_path))
    out_root = _resolve_output_root(manifest, output_root)
    feature_root = _feature_cache_root(manifest, out_root)

    if dry_run:
        report = {
            "manifest": asdict(manifest),
            "dry_run": True,
            "free_disk_gb": _free_disk_gb(out_root),
            "feature_cache_dir": feature_root,
            "expected_outputs": {
                "ibl_dataset_inventory_csv": out_root / "ibl_dataset_inventory.csv",
                "ibl_feature_manifest_json": out_root / "ibl_feature_manifest.json",
                "gt_region_rates_csv": out_root / "gt_region_rates.csv",
            },
        }
        report_path = out_root / "ibl_study_report.json"
        report_path.write_text(json.dumps(_ibl_json_ready(report), indent=2))
        return _ibl_json_ready({"report_json": report_path, "report": report})

    _assert_disk_available(out_root, manifest)
    one = _make_one(manifest)
    eids, details = _search_repeated_site_eids(one, manifest)
    eids, details = _select_diverse_sessions(eids, details, max_sessions=manifest.max_sessions)
    metas: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    cleanup_records: list[dict[str, object]] = []
    for eid in eids:
        _assert_disk_available(out_root, manifest)
        before = _snapshot_large_one_files()
        try:
            item = _load_session_region_data(one, manifest, eid)
            metas.append(_save_session_feature(item, feature_root))
        except Exception as exc:
            failures.append({"eid": eid, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            removed = _clean_owned_raw_files(before, enabled=manifest.disk_policy == "clean_raw_after_binning")
            if removed:
                cleanup_records.append({"eid": eid, "removed_files": removed})

    if not metas:
        raise RuntimeError(f"No IBL session features could be built. Failures: {failures[:5]}")
    session_data = [_load_session_feature(meta) for meta in metas]
    selected_data = session_data
    reduction_note = None
    while selected_data:
        try:
            region_arr, region_panel, sequence_table = _build_region_dataset(selected_data, manifest)
            break
        except ValueError as exc:
            if len(selected_data) <= 1:
                raise
            reduction_note = str(exc)
            selected_data = selected_data[:-1]
    else:
        raise RuntimeError("No sessions remained after region-panel reduction.")

    region_npz = out_root / "ibl_region_rates_raw.npz"
    np.savez_compressed(region_npz, region_rates=np.asarray(region_arr, dtype=np.float32), regions=np.asarray(region_panel, dtype=object))
    sequence_table_path = out_root / "ibl_sequence_table.csv"
    sequence_table.to_csv(sequence_table_path, index=False)
    inventory = pd.DataFrame(
        [
            {
                "eid": meta["eid"],
                "lab": dict(meta.get("detail", {})).get("lab", ""),
                "subject": dict(meta.get("detail", {})).get("subject", ""),
                "n_regions": len(dict(meta.get("region_counts", {}))),
                "unit_counts_npz": meta["unit_counts_npz"],
                "used": str(meta["eid"]) in set(sequence_table["eid"].astype(str)),
            }
            for meta in metas
        ]
    )
    inventory_path = out_root / "ibl_dataset_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    gt_preview = array_to_nethobench_frame(region_arr, region_panel, sequence_prefix="ibl_raw")
    gt_preview.to_csv(out_root / "gt_region_rates_raw.csv", index=False)
    feature_manifest = {
        "manifest": asdict(manifest),
        "feature_cache_dir": feature_root,
        "region_rates_npz": region_npz,
        "sequence_table_csv": sequence_table_path,
        "inventory_csv": inventory_path,
        "session_feature_manifests": metas,
        "failures": failures,
        "cleanup_records": cleanup_records,
        "reduction_note": reduction_note,
        "n_loaded_sessions": len(metas),
        "n_used_sessions": int(sequence_table["eid"].nunique()),
        "n_sequences": int(region_arr.shape[0]),
        "n_time_bins": int(region_arr.shape[1]),
        "n_regions": int(region_arr.shape[2]),
        "region_panel": region_panel,
    }
    feature_manifest_path = out_root / "ibl_feature_manifest.json"
    feature_manifest_path.write_text(json.dumps(_ibl_json_ready(feature_manifest), indent=2))
    return _ibl_json_ready(
        {
            "feature_manifest_json": feature_manifest_path,
            "inventory_csv": inventory_path,
            "sequence_table_csv": sequence_table_path,
            "region_rates_npz": region_npz,
            "report": feature_manifest,
        }
    )


def _load_feature_bundle(output_root: Path) -> tuple[np.ndarray, list[str], pd.DataFrame, dict[str, object]]:
    feature_manifest_path = output_root / "ibl_feature_manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text())
    with np.load(feature_manifest["region_rates_npz"], allow_pickle=True) as data:
        arr = np.asarray(data["region_rates"], dtype=np.float64)
        regions = [str(v) for v in data["regions"].tolist()]
    sequence_table = pd.read_csv(feature_manifest["sequence_table_csv"])
    return arr, regions, sequence_table, feature_manifest


def _study_split_indices(sequence_table: pd.DataFrame, policy: str, seed: int) -> dict[str, dict[str, np.ndarray | str]]:
    rng = np.random.default_rng(seed)
    n = int(sequence_table.shape[0])
    indices = np.arange(n)
    tasks: dict[str, dict[str, np.ndarray | str]] = {}

    perm = rng.permutation(indices)
    n_train = max(1, int(0.6 * n))
    n_val = max(1, int(0.2 * n))
    tasks["within_session"] = {
        "train": np.sort(perm[:n_train]),
        "val": np.sort(perm[n_train : n_train + n_val]),
        "test": np.sort(perm[n_train + n_val :]),
        "note": "pooled trial split across available sessions",
    }

    eids = np.asarray(sorted(sequence_table["eid"].astype(str).unique()))
    if eids.size >= 3:
        eid_perm = rng.permutation(eids)
        n_train_eid = max(1, int(0.6 * eid_perm.size))
        n_val_eid = max(1, int(0.2 * eid_perm.size))
        train_eids = set(eid_perm[:n_train_eid])
        val_eids = set(eid_perm[n_train_eid : n_train_eid + n_val_eid])
        test_eids = set(eid_perm[n_train_eid + n_val_eid :])
        if test_eids:
            eid_col = sequence_table["eid"].astype(str)
            tasks["cross_session"] = {
                "train": np.where(eid_col.isin(train_eids))[0],
                "val": np.where(eid_col.isin(val_eids))[0],
                "test": np.where(eid_col.isin(test_eids))[0],
                "note": "held-out sessions",
            }

    labs = np.asarray(sorted(sequence_table["lab"].astype(str).replace("", "unknown").unique()))
    if labs.size >= 4:
        lab_perm = rng.permutation(labs)
        test_labs = set(lab_perm[-max(1, labs.size // 4) :])
        val_labs = set(lab_perm[-max(2, labs.size // 4 + 1) : -max(1, labs.size // 4)])
        train_labs = set(lab_perm) - test_labs - val_labs
        lab_col = sequence_table["lab"].astype(str).replace("", "unknown")
        tasks["cross_lab"] = {
            "train": np.where(lab_col.isin(train_labs))[0],
            "val": np.where(lab_col.isin(val_labs))[0],
            "test": np.where(lab_col.isin(test_labs))[0],
            "note": "held-out labs",
        }
    requested = {part.strip() for part in policy.split(",") if part.strip()}
    return {name: split for name, split in tasks.items() if name in requested}


def _normalize_for_split(arr: np.ndarray, train_idx: np.ndarray, context_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return zscore_from_train_context(arr, train_idx, context_bins=context_bins)


def _ridge_multistep_var(
    train: np.ndarray,
    contexts: np.ndarray,
    *,
    context_bins: int,
    lag: int = 3,
    ridge: float = 1e-3,
) -> np.ndarray:
    train = np.asarray(train, dtype=np.float64)
    X_rows = []
    Y_rows = []
    for seq in train:
        for t in range(lag, seq.shape[0]):
            X_rows.append(seq[t - lag : t].reshape(-1))
            Y_rows.append(seq[t])
    X = np.asarray(X_rows)
    Y = np.asarray(Y_rows)
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    eye = np.eye(X_aug.shape[1])
    eye[-1, -1] = 0.0
    coef = np.linalg.solve(X_aug.T @ X_aug + ridge * eye, X_aug.T @ Y)
    pred = np.asarray(contexts, dtype=np.float64).copy()
    for t in range(context_bins, pred.shape[1]):
        start = max(0, t - lag)
        hist = pred[:, start:t, :]
        if hist.shape[1] < lag:
            pad = np.repeat(hist[:, :1, :], lag - hist.shape[1], axis=1)
            hist = np.concatenate([pad, hist], axis=1)
        Xp = hist.reshape(hist.shape[0], -1)
        Xp_aug = np.concatenate([Xp, np.ones((Xp.shape[0], 1))], axis=1)
        pred[:, t, :] = Xp_aug @ coef
    return pred


def _select_var_ridge(train: np.ndarray, val: np.ndarray, context_bins: int) -> float:
    best_ridge = 1e-3
    best_err = float("inf")
    for ridge in [1e-4, 1e-3, 1e-2, 1e-1]:
        pred = _ridge_multistep_var(train, val.copy(), context_bins=context_bins, lag=3, ridge=ridge)
        err = float(np.nanmean((pred[:, context_bins:, :] - val[:, context_bins:, :]) ** 2))
        if err < best_err:
            best_err = err
            best_ridge = ridge
    return best_ridge


def _lds_var_prediction(train: np.ndarray, val: np.ndarray, test: np.ndarray, *, context_bins: int) -> tuple[np.ndarray, dict[str, object]]:
    from sklearn.decomposition import PCA

    flat = train.reshape(-1, train.shape[2])
    pca_full = PCA().fit(flat)
    csum = np.cumsum(pca_full.explained_variance_ratio_)
    n_latent = int(np.searchsorted(csum, 0.95) + 1)
    n_latent = max(1, min(8, n_latent, train.shape[2]))
    pca = PCA(n_components=n_latent).fit(flat)
    train_z = pca.transform(flat).reshape(train.shape[0], train.shape[1], n_latent)
    val_z = pca.transform(val.reshape(-1, val.shape[2])).reshape(val.shape[0], val.shape[1], n_latent)
    test_z_context = pca.transform(test.reshape(-1, test.shape[2])).reshape(test.shape[0], test.shape[1], n_latent)
    ridge = _select_var_ridge(train_z, val_z, context_bins)
    pred_z = _ridge_multistep_var(train_z, test_z_context.copy(), context_bins=context_bins, lag=3, ridge=ridge)
    pred = pca.inverse_transform(pred_z.reshape(-1, n_latent)).reshape(test.shape)
    pred[:, :context_bins, :] = test[:, :context_bins, :]
    return pred, {"n_latent": n_latent, "ridge": ridge}


def _glm_poisson_prediction(raw_arr: np.ndarray, norm_arr: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, mean: np.ndarray, std: np.ndarray, *, context_bins: int) -> tuple[np.ndarray, dict[str, object]]:
    from sklearn.linear_model import PoissonRegressor
    from sklearn.preprocessing import StandardScaler

    raw_train = raw_arr[train_idx]
    raw_test = raw_arr[test_idx]
    n_time, n_channels = raw_train.shape[1], raw_train.shape[2]
    X_rows = []
    Y_rows = []
    for seq in raw_train:
        for t in range(1, n_time):
            time_feat = [t / max(1, n_time - 1), np.sin(2 * np.pi * t / n_time), np.cos(2 * np.pi * t / n_time)]
            X_rows.append(np.concatenate([seq[t - 1], np.asarray(time_feat)]))
            Y_rows.append(np.expm1(seq[t]))
    X = np.asarray(X_rows, dtype=np.float64)
    Y = np.clip(np.asarray(Y_rows, dtype=np.float64), 0.0, None)
    y_cap = np.nanpercentile(Y, 99.5, axis=0)
    y_cap = np.where(np.isfinite(y_cap) & (y_cap > 0), y_cap, 1.0)
    if X.shape[0] > 60000:
        keep = np.linspace(0, X.shape[0] - 1, 60000, dtype=np.int64)
        X = X[keep]
        Y = Y[keep]
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    models = []
    for ch in range(n_channels):
        model = PoissonRegressor(alpha=1e-3, max_iter=50, tol=1e-4)
        model.fit(Xs, Y[:, ch])
        models.append(model)
    pred_raw = raw_test.copy()
    for t in range(context_bins, n_time):
        time_feat = np.asarray([t / max(1, n_time - 1), np.sin(2 * np.pi * t / n_time), np.cos(2 * np.pi * t / n_time)])
        Xp = np.concatenate([pred_raw[:, t - 1, :], np.repeat(time_feat[None, :], pred_raw.shape[0], axis=0)], axis=1)
        Xps = scaler.transform(Xp)
        rates = np.column_stack([model.predict(Xps) for model in models])
        rates = np.nan_to_num(rates, nan=0.0, posinf=float(np.nanmax(y_cap)), neginf=0.0)
        pred_raw[:, t, :] = np.log1p(np.minimum(np.clip(rates, 0.0, None), y_cap[None, :]))
    pred = (pred_raw - mean[None, None, :]) / std[None, None, :]
    pred[:, :context_bins, :] = norm_arr[test_idx, :context_bins, :]
    return pred, {"alpha": 1e-3, "features": "lag1_channels+time_sin_cos", "max_fit_rows": 60000}


def _torch_device():
    try:
        import torch

        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except Exception:
        return None


def _train_torch_forecaster(
    model_name: str,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    manifest: IBLManifest,
    *,
    seed: int,
    checkpoint_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError("PyTorch is required for GRU/Transformer IBL models.") from exc

    torch.manual_seed(seed)
    device = _torch_device()
    if device is None:
        raise RuntimeError("No torch device available.")
    n_channels = train.shape[2]
    context_bins = manifest.context_bins
    target_bins = train.shape[1] - context_bins
    original_train_n = int(train.shape[0])
    original_val_n = int(val.shape[0])
    if train.shape[0] > 1024:
        train = train[np.linspace(0, train.shape[0] - 1, 1024, dtype=np.int64)]
    if val.shape[0] > 256:
        val = val[np.linspace(0, val.shape[0] - 1, 256, dtype=np.int64)]

    class GRUForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_channels, 64, num_layers=2, dropout=0.1, batch_first=True)
            self.out = nn.Linear(64, n_channels)

        def forward(self, context, target=None):
            _, h = self.gru(context)
            prev = context[:, -1:, :]
            outs = []
            for idx in range(target_bins):
                y, h = self.gru(prev, h)
                pred = self.out(y)
                outs.append(pred)
                prev = target[:, idx : idx + 1, :] if self.training and target is not None else pred
            return torch.cat(outs, dim=1)

    class TransformerForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(n_channels, 64)
            layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=3)
            self.out = nn.Linear(64, target_bins * n_channels)

        def forward(self, context, target=None):
            h = self.enc(self.inp(context))
            pred = self.out(h[:, -1, :])
            return pred.reshape(context.shape[0], target_bins, n_channels)

    model = GRUForecaster() if model_name == "gru" else TransformerForecaster()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_x = torch.tensor(train[:, :context_bins, :], dtype=torch.float32)
    train_y = torch.tensor(train[:, context_bins:, :], dtype=torch.float32)
    val_x = torch.tensor(val[:, :context_bins, :], dtype=torch.float32, device=device)
    val_y = torch.tensor(val[:, context_bins:, :], dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=manifest.batch_size, shuffle=True)
    best_state = None
    best_val = float("inf")
    stale = 0
    history = []
    max_epochs = min(manifest.max_epochs, 100)
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb, yb)
            loss = torch.mean((pred - yb) ** 2)
            if model_name == "transformer_nb_reg":
                loss = loss + 0.05 * torch.mean(torch.abs(pred.mean(dim=(0, 1)) - yb.mean(dim=(0, 1))))
                loss = loss + 0.05 * torch.mean(torch.abs(torch.quantile(pred, 0.9, dim=1) - torch.quantile(yb, 0.9, dim=1)))
                if target_bins > 2:
                    pred_auto = torch.mean(pred[:, 1:, :] * pred[:, :-1, :], dim=(0, 1))
                    y_auto = torch.mean(yb[:, 1:, :] * yb[:, :-1, :], dim=(0, 1))
                    loss = loss + 0.02 * torch.mean(torch.abs(pred_auto - y_auto))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = float(torch.mean((val_pred - val_y) ** 2).detach().cpu())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_mse": val_loss})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= manifest.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "model_name": model_name,
                "seed": int(seed),
                "state_dict": model.state_dict(),
                "n_channels": int(n_channels),
                "context_bins": int(context_bins),
                "target_bins": int(target_bins),
                "hidden_dim": 64,
                "num_layers": 2 if model_name == "gru" else 3,
                "best_val_mse": float(best_val),
                "epochs": int(len(history)),
                "history": history,
            },
            checkpoint_path,
        )
    test_x = torch.tensor(test[:, :context_bins, :], dtype=torch.float32, device=device)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, test_x.shape[0], manifest.batch_size):
            preds.append(model(test_x[start : start + manifest.batch_size]).detach().cpu().numpy())
    target_pred = np.concatenate(preds, axis=0)
    pred = test.copy()
    pred[:, context_bins:, :] = target_pred
    return pred, {
        "seed": seed,
        "best_val_mse": best_val,
        "epochs": len(history),
        "device": str(device),
        "history": history,
        "train_sequences_used": int(train.shape[0]),
        "train_sequences_available": original_train_n,
        "val_sequences_used": int(val.shape[0]),
        "val_sequences_available": original_val_n,
        "checkpoint_path": (
            str(checkpoint_path) if checkpoint_path is not None else None
        ),
    }


def _train_ssm_vae_forecaster(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    manifest: IBLManifest,
    *,
    seed: int,
    checkpoint_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError("PyTorch is required for the SSM-VAE IBL model.") from exc

    torch.manual_seed(seed)
    device = _torch_device()
    if device is None:
        raise RuntimeError("No torch device available.")
    n_channels = train.shape[2]
    context_bins = manifest.context_bins
    target_bins = train.shape[1] - context_bins
    latent_dim = int(manifest.ssm_vae_latent_dim)
    hidden_dim = int(manifest.ssm_vae_hidden_dim)
    n_samples = int(manifest.ssm_vae_samples)
    original_train_n = int(train.shape[0])
    original_val_n = int(val.shape[0])
    if train.shape[0] > 1024:
        train = train[np.linspace(0, train.shape[0] - 1, 1024, dtype=np.int64)]
    if val.shape[0] > 256:
        val = val[np.linspace(0, val.shape[0] - 1, 256, dtype=np.int64)]

    class LatentSSMVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.GRU(n_channels, hidden_dim, batch_first=True)
            self.context_encoder = nn.GRU(n_channels, hidden_dim, batch_first=True)
            self.q_mu = nn.Linear(hidden_dim, latent_dim)
            self.q_logvar = nn.Linear(hidden_dim, latent_dim)
            self.prior_init = nn.Linear(hidden_dim, 2 * latent_dim)
            self.transition = nn.Sequential(
                nn.Linear(latent_dim + hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 2 * latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, n_channels),
            )
            self.emission_logvar = nn.Parameter(torch.full((n_channels,), -0.7))

        def _posterior(self, x):
            h, _ = self.encoder(x)
            mu = self.q_mu(h)
            logvar = torch.clamp(self.q_logvar(h), min=-6.0, max=3.0)
            return mu, logvar

        def _context_summary(self, context):
            _, h = self.context_encoder(context)
            return h[-1]

        def _transition_prior(self, z_prev, context_summary):
            stats = self.transition(torch.cat([z_prev, context_summary], dim=-1))
            mu, logvar = stats.chunk(2, dim=-1)
            return mu, torch.clamp(logvar, min=-6.0, max=3.0)

        def forward(self, x):
            q_mu, q_logvar = self._posterior(x)
            q_std = torch.exp(0.5 * q_logvar)
            z = q_mu + torch.randn_like(q_std) * q_std
            context_summary = self._context_summary(x[:, :context_bins, :])
            init_stats = self.prior_init(context_summary)
            init_mu, init_logvar = init_stats.chunk(2, dim=-1)
            init_logvar = torch.clamp(init_logvar, min=-6.0, max=3.0)
            prior_mus = [init_mu]
            prior_logvars = [init_logvar]
            for t in range(1, x.shape[1]):
                mu_t, logvar_t = self._transition_prior(z[:, t - 1, :], context_summary)
                prior_mus.append(mu_t)
                prior_logvars.append(logvar_t)
            prior_mu = torch.stack(prior_mus, dim=1)
            prior_logvar = torch.stack(prior_logvars, dim=1)
            recon_mu = self.decoder(z)
            return recon_mu, q_mu, q_logvar, prior_mu, prior_logvar

        def rollout_samples(self, context, *, samples: int):
            q_mu, q_logvar = self._posterior(context)
            context_summary = self._context_summary(context)
            q_last_mu = q_mu[:, -1, :]
            q_last_logvar = torch.clamp(q_logvar[:, -1, :], min=-6.0, max=3.0)
            sample_rollouts = []
            for _ in range(samples):
                z_prev = q_last_mu + torch.randn_like(q_last_mu) * torch.exp(0.5 * q_last_logvar)
                decoded = []
                for _t in range(target_bins):
                    prior_mu, prior_logvar = self._transition_prior(z_prev, context_summary)
                    z_prev = prior_mu + torch.randn_like(prior_mu) * torch.exp(0.5 * prior_logvar)
                    decoded.append(self.decoder(z_prev))
                sample_rollouts.append(torch.stack(decoded, dim=1))
            return torch.stack(sample_rollouts, dim=0)

    def kl_normal(q_mu, q_logvar, p_mu, p_logvar):
        q_var = torch.exp(q_logvar)
        p_var = torch.exp(p_logvar)
        kl = 0.5 * (p_logvar - q_logvar + (q_var + (q_mu - p_mu) ** 2) / p_var - 1.0)
        if manifest.ssm_vae_free_bits > 0:
            kl = torch.clamp(kl, min=float(manifest.ssm_vae_free_bits))
        return kl.sum(dim=-1)

    model = LatentSSMVAE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_tensor = torch.tensor(train, dtype=torch.float32)
    val_tensor = torch.tensor(val, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=manifest.batch_size, shuffle=True)
    best_state = None
    best_val = float("inf")
    stale = 0
    history = []
    max_epochs = min(manifest.max_epochs, 100)
    beta_final = float(manifest.ssm_vae_beta)
    anneal_epochs = max(1, int(round(0.3 * max_epochs)))
    weights = torch.ones(train.shape[1], dtype=torch.float32, device=device)
    weights[:context_bins] = 0.25
    for epoch in range(max_epochs):
        beta = beta_final * min(1.0, float(epoch + 1) / anneal_epochs)
        model.train()
        losses = []
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            recon_mu, q_mu, q_logvar, prior_mu, prior_logvar = model(xb)
            emission_logvar = torch.clamp(model.emission_logvar, min=-6.0, max=3.0)
            recon_nll = 0.5 * (((xb - recon_mu) ** 2) / torch.exp(emission_logvar)[None, None, :] + emission_logvar[None, None, :])
            recon_nll = torch.mean(torch.mean(recon_nll, dim=-1) * weights[None, :])
            kl = torch.mean(kl_normal(q_mu, q_logvar, prior_mu, prior_logvar))
            loss = recon_nll + beta * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            recon_mu, q_mu, q_logvar, prior_mu, prior_logvar = model(val_tensor)
            emission_logvar = torch.clamp(model.emission_logvar, min=-6.0, max=3.0)
            recon_nll = 0.5 * (((val_tensor - recon_mu) ** 2) / torch.exp(emission_logvar)[None, None, :] + emission_logvar[None, None, :])
            recon_nll = torch.mean(torch.mean(recon_nll, dim=-1) * weights[None, :])
            kl = torch.mean(kl_normal(q_mu, q_logvar, prior_mu, prior_logvar))
            val_loss = float((recon_nll + beta_final * kl).detach().cpu())
            val_mse = float(torch.mean((recon_mu[:, context_bins:, :] - val_tensor[:, context_bins:, :]) ** 2).detach().cpu())
        history.append({"epoch": epoch + 1, "train_elbo": float(np.mean(losses)), "val_elbo": val_loss, "val_target_mse": val_mse, "beta": beta})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= manifest.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "model_name": "latent_state_space_vae",
                "seed": int(seed),
                "state_dict": model.state_dict(),
                "n_channels": int(n_channels),
                "context_bins": int(context_bins),
                "target_bins": int(target_bins),
                "latent_dim": int(latent_dim),
                "hidden_dim": int(hidden_dim),
                "beta": float(beta_final),
                "free_bits": float(manifest.ssm_vae_free_bits),
                "emission_distribution": "diagonal_gaussian",
                "best_val_elbo": float(best_val),
                "epochs": int(len(history)),
                "history": history,
            },
            checkpoint_path,
        )

    test_x = torch.tensor(test[:, :context_bins, :], dtype=torch.float32, device=device)
    model.eval()
    sample_chunks = []
    with torch.no_grad():
        for start in range(0, test_x.shape[0], manifest.batch_size):
            sample_chunks.append(model.rollout_samples(test_x[start : start + manifest.batch_size], samples=n_samples).detach().cpu().numpy())
    sample_targets = np.concatenate(sample_chunks, axis=1)
    sample_predictions = np.repeat(test[None, :, :, :], n_samples, axis=0)
    sample_predictions[:, :, context_bins:, :] = sample_targets
    pred = test.copy()
    pred[:, context_bins:, :] = np.mean(sample_targets, axis=0)
    return pred, sample_predictions, {
        "seed": seed,
        "model": "latent_state_space_vae",
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "beta": beta_final,
        "free_bits": float(manifest.ssm_vae_free_bits),
        "n_samples": n_samples,
        "best_val_elbo": best_val,
        "epochs": len(history),
        "device": str(device),
        "history": history,
        "train_sequences_used": int(train.shape[0]),
        "train_sequences_available": original_train_n,
        "val_sequences_used": int(val.shape[0]),
        "val_sequences_available": original_val_n,
        "rollout_mode": "context_only_latent_prior_sampling",
        "checkpoint_path": (
            str(checkpoint_path) if checkpoint_path is not None else None
        ),
    }


def _pool_stochastic_samples(gt: np.ndarray, sample_predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(sample_predictions, dtype=np.float64)
    if samples.ndim != 4:
        raise ValueError("sample_predictions must have shape [n_samples, n_sequences, n_time, n_channels].")
    n_samples = samples.shape[0]
    gt_pooled = np.repeat(np.asarray(gt, dtype=np.float64)[None, :, :, :], n_samples, axis=0).reshape(-1, gt.shape[1], gt.shape[2])
    pred_pooled = samples.reshape(-1, samples.shape[2], samples.shape[3])
    return gt_pooled, pred_pooled


def _limit_pooled_stochastic_samples(gt_pooled: np.ndarray, pred_pooled: np.ndarray, *, n_samples: int, max_samples: int) -> tuple[np.ndarray, np.ndarray, int]:
    n_samples = int(max(1, n_samples))
    max_samples = int(max(1, max_samples))
    if n_samples <= max_samples:
        return gt_pooled, pred_pooled, n_samples
    if pred_pooled.shape[0] % n_samples != 0:
        return gt_pooled, pred_pooled, n_samples
    n_seq = pred_pooled.shape[0] // n_samples
    keep = np.unique(np.linspace(0, n_samples - 1, max_samples, dtype=np.int64))
    gt_view = gt_pooled.reshape(n_samples, n_seq, gt_pooled.shape[1], gt_pooled.shape[2])[keep]
    pred_view = pred_pooled.reshape(n_samples, n_seq, pred_pooled.shape[1], pred_pooled.shape[2])[keep]
    return (
        gt_view.reshape(-1, gt_pooled.shape[1], gt_pooled.shape[2]),
        pred_view.reshape(-1, pred_pooled.shape[1], pred_pooled.shape[2]),
        int(keep.size),
    )


def _predict_study_models(
    raw_arr: np.ndarray,
    norm_arr: np.ndarray,
    split: dict[str, np.ndarray | str],
    manifest: IBLManifest,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    checkpoint_root: Path | None = None,
) -> dict[str, dict[str, object]]:
    train_idx = np.asarray(split["train"], dtype=np.int64)
    val_idx = np.asarray(split["val"], dtype=np.int64)
    test_idx = np.asarray(split["test"], dtype=np.int64)
    train = norm_arr[train_idx]
    val = norm_arr[val_idx] if val_idx.size else train
    test = norm_arr[test_idx]
    outputs: dict[str, dict[str, object]] = {}
    for model_name in manifest.model_suite:
        try:
            if model_name == "psth":
                pred = poisson_psth_prediction(train, test.shape[0])
                meta = {"model": "train_time_bin_region_mean"}
            elif model_name == "glm_poisson":
                pred, meta = _glm_poisson_prediction(raw_arr, norm_arr, train_idx, test_idx, mean, std, context_bins=manifest.context_bins)
            elif model_name == "var":
                ridge = _select_var_ridge(train, val, manifest.context_bins)
                pred = _ridge_multistep_var(train, test.copy(), context_bins=manifest.context_bins, lag=3, ridge=ridge)
                meta = {"lag": 3, "ridge": ridge}
            elif model_name == "lds_var":
                pred, meta = _lds_var_prediction(train, val, test.copy(), context_bins=manifest.context_bins)
            elif model_name == "ssm_vae":
                seed_reports = []
                seed_preds = []
                seed_sample_preds = []
                seeds = manifest.neural_seeds if manifest.neural_seeds else [manifest.random_seed]
                for seed in seeds:
                    checkpoint_path = (
                        Path(checkpoint_root) / model_name / f"seed_{seed}.pt"
                        if checkpoint_root is not None
                        else None
                    )
                    pred_seed, sample_seed, meta_seed = _train_ssm_vae_forecaster(
                        train,
                        val,
                        test.copy(),
                        manifest,
                        seed=seed,
                        checkpoint_path=checkpoint_path,
                    )
                    seed_preds.append(pred_seed)
                    seed_sample_preds.append(sample_seed)
                    seed_reports.append(meta_seed)
                pred = np.mean(seed_preds, axis=0)
                sample_pred = np.concatenate(seed_sample_preds, axis=0)
                meta = {
                    "seeds": seed_reports,
                    "ensemble": "mean_prediction_and_pooled_stochastic_samples",
                    "n_total_samples": int(sample_pred.shape[0]),
                    "n_samples_per_seed": int(seed_sample_preds[0].shape[0]) if seed_sample_preds else 0,
                }
                outputs[model_name] = {
                    "prediction": pred,
                    "sample_prediction": sample_pred,
                    "seed_predictions": {
                        str(seed): pred_seed
                        for seed, pred_seed in zip(seeds, seed_preds)
                    },
                    "metadata": meta,
                }
                continue
            elif model_name in {"gru", "transformer", "transformer_nb_reg"}:
                seed_reports = []
                seed_preds = []
                for seed in manifest.neural_seeds:
                    checkpoint_path = (
                        Path(checkpoint_root) / model_name / f"seed_{seed}.pt"
                        if checkpoint_root is not None
                        else None
                    )
                    pred_seed, meta_seed = _train_torch_forecaster(
                        model_name,
                        train,
                        val,
                        test.copy(),
                        manifest,
                        seed=seed,
                        checkpoint_path=checkpoint_path,
                    )
                    seed_preds.append(pred_seed)
                    seed_reports.append(meta_seed)
                pred = np.mean(seed_preds, axis=0)
                meta = {"seeds": seed_reports, "ensemble": "mean"}
                seed_predictions = {
                    str(seed): pred_seed
                    for seed, pred_seed in zip(manifest.neural_seeds, seed_preds)
                }
            else:
                outputs[model_name] = {"error": f"Unknown model {model_name!r}"}
                continue
            pred[:, : manifest.context_bins, :] = test[:, : manifest.context_bins, :]
            payload: dict[str, object] = {"prediction": pred, "metadata": meta}
            if model_name in {"gru", "transformer", "transformer_nb_reg"}:
                payload["seed_predictions"] = seed_predictions
            outputs[model_name] = payload
        except Exception as exc:
            outputs[model_name] = {"error": f"{type(exc).__name__}: {exc}"}
    return outputs


def _score_prediction_bundle(gt: np.ndarray, pred: np.ndarray, channels: list[str]) -> dict[str, object]:
    out = _score_arrays_or_error(gt, pred, channels)
    if "score_error" not in out:
        try:
            out["fidelity_scores"] = _fidelity_arrays(gt, pred, channels)
        except Exception as exc:
            out["fidelity_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _flatten_scores(scores_by_task: dict[str, object]) -> pd.DataFrame:
    rows = []
    for task, payload in scores_by_task.items():
        split_half = payload.get("split_half", {})
        for metric, value in split_half.get("neuro_scores", {}).items():
            rows.append({"task": task, "model": "split_half", "metric": metric, "value": value, "kind": "ceiling"})
        for model, model_payload in payload.get("models", {}).items():
            for metric, value in model_payload.get("neuro_scores", {}).items():
                rows.append({"task": task, "model": model, "metric": metric, "value": value, "kind": "model"})
    return pd.DataFrame(rows)


def _bootstrap_ci(score_long: pd.DataFrame, *, samples: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    metric = "FINAL_COMPOSITE_SCORE"
    out = []
    sub = score_long[score_long["metric"] == metric].dropna(subset=["value"])
    for (model, kind), group in sub.groupby(["model", "kind"]):
        values = group["value"].astype(float).to_numpy()
        if values.size == 0:
            continue
        if values.size == 1 or samples <= 0:
            lo = hi = point = float(values.mean())
        else:
            boot = [float(np.mean(rng.choice(values, size=values.size, replace=True))) for _ in range(samples)]
            point = float(values.mean())
            lo, hi = [float(v) for v in np.quantile(boot, [0.025, 0.975])]
        out.append({"model": model, "kind": kind, "metric": metric, "mean": point, "ci_low": lo, "ci_high": hi, "n": int(values.size)})
    return {"bootstrap_samples": int(samples), "rows": out}


def _plot_dataset_coverage(output_path: Path, sequence_table: pd.DataFrame, regions: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    sequence_table["lab"].replace("", "unknown").value_counts().plot(kind="bar", ax=axes[0], color="#4C78A8")
    axes[0].set_title("Trials by lab")
    axes[0].set_ylabel("Sequences")
    sequence_table["eid"].value_counts().plot(kind="bar", ax=axes[1], color="#59A14F")
    axes[1].set_title("Sequences by session")
    axes[1].set_xticklabels([])
    axes[2].barh(np.arange(len(regions)), np.ones(len(regions)), color="#F28E2B")
    axes[2].set_yticks(np.arange(len(regions)))
    axes[2].set_yticklabels(regions, fontsize=8)
    axes[2].set_xticks([])
    axes[2].set_title("Common region panel")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_model_family_heatmap(output_path: Path, score_long: pd.DataFrame) -> None:
    family_metrics = [
        "family_distribution",
        "family_temporal_spectral",
        "family_relational",
        "family_geometry",
        "family_state_dynamics",
        "FINAL_COMPOSITE_SCORE",
    ]
    sub = score_long[score_long["metric"].isin(family_metrics)].copy()
    if sub.empty:
        return
    pivot = sub.pivot_table(index="model", columns="metric", values="value", aggfunc="mean").reindex(columns=family_metrics)
    fig, ax = plt.subplots(figsize=(9.0, max(3.0, 0.38 * pivot.shape[0] + 1.2)))
    im = ax.imshow(pivot.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(family_metrics)))
    ax.set_xticklabels([m.replace("family_", "").replace("_", " ").replace("FINAL COMPOSITE SCORE", "composite") for m in family_metrics], rotation=25, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax, label="NethoBench score")
    ax.set_title("IBL model family signatures")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_score_distributions(output_path: Path, score_long: pd.DataFrame) -> None:
    sub = score_long[score_long["metric"] == "FINAL_COMPOSITE_SCORE"].dropna(subset=["value"])
    if sub.empty:
        return
    models = sorted(sub["model"].unique())
    fig, ax = plt.subplots(figsize=(10.0, 4.5))
    for idx, model in enumerate(models):
        values = sub[sub["model"] == model]["value"].astype(float).to_numpy()
        x = np.full(values.shape, idx, dtype=float) + np.linspace(-0.08, 0.08, max(1, values.size))
        ax.scatter(x, values, s=45, alpha=0.85)
        ax.plot([idx - 0.22, idx + 0.22], [np.mean(values), np.mean(values)], color="black", lw=2)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Composite score")
    ax.set_title("IBL task/model score distribution")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _score_horizons(gt: np.ndarray, pred: np.ndarray, channels: list[str], horizons: list[int], context_bins: int) -> dict[str, object]:
    rows = []
    for horizon in horizons:
        stop = min(gt.shape[1], context_bins + horizon)
        if stop <= context_bins + 1:
            continue
        gt_slice = gt[:, context_bins:stop, :]
        pred_slice = pred[:, context_bins:stop, :]
        err = pred_slice - gt_slice
        gt_flat = gt_slice.reshape(-1)
        pred_flat = pred_slice.reshape(-1)
        corr = float("nan")
        if np.nanstd(gt_flat) > 1e-9 and np.nanstd(pred_flat) > 1e-9:
            corr = float(np.corrcoef(gt_flat, pred_flat)[0, 1])
        rows.append(
            {
                "horizon_bins": int(horizon),
                "horizon_seconds": float(horizon * 0.02),
                "mse": float(np.nanmean(err**2)),
                "mae": float(np.nanmean(np.abs(err))),
                "correlation": corr,
            }
        )
    return {"horizons": rows, "summary": "fast forecast-error horizon summary"}


def _ssm_vae_diagnostics(
    pred_manifest: dict[str, object],
    channels: list[str],
    manifest: IBLManifest,
    output_root: Path,
) -> dict[str, object]:
    rows = []
    sensitivity = []
    example = None
    for task_name, task_payload in pred_manifest.get("tasks", {}).items():
        gt = _frame_to_array(Path(task_payload["gt_csv"]), channels)
        for model_name, model_payload in task_payload.get("models", {}).items():
            if "stochastic_prediction_csv" not in model_payload or "stochastic_gt_csv" not in model_payload:
                continue
            n_samples = int(model_payload.get("n_stochastic_samples", manifest.ssm_vae_samples))
            pred_pooled = _frame_to_array(Path(model_payload["stochastic_prediction_csv"]), channels)
            if pred_pooled.shape[0] % max(1, n_samples) != 0:
                rows.append({"task": task_name, "model": model_name, "error": "pooled prediction count is not divisible by n_samples"})
                continue
            n_seq = pred_pooled.shape[0] // n_samples
            pred_samples = pred_pooled.reshape(n_samples, n_seq, pred_pooled.shape[1], pred_pooled.shape[2])
            target = slice(manifest.context_bins, pred_pooled.shape[1])
            sample_var = float(np.nanmean(np.nanvar(pred_samples[:, :, target, :], axis=0)))
            empirical_var = float(np.nanmean(np.nanvar(gt[:n_seq, target, :], axis=0)))
            ratio = sample_var / empirical_var if empirical_var > 1e-12 else float("nan")
            rows.append(
                {
                    "task": task_name,
                    "model": model_name,
                    "stochastic_model": model_payload.get("stochastic_model_name", f"{model_name}_samples"),
                    "n_samples": n_samples,
                    "n_sequences": n_seq,
                    "sample_target_variance": sample_var,
                    "empirical_target_variance": empirical_var,
                    "sample_to_empirical_variance_ratio": ratio,
                }
            )
            if example is None and n_seq > 0:
                example = {
                    "task": task_name,
                    "model": model_name,
                    "gt": gt[:n_seq],
                    "samples": pred_samples,
                }
            if task_name == "within_session":
                for k in [1, 3, 5, 10]:
                    if k > n_samples or k > manifest.ssm_vae_score_samples:
                        continue
                    gt_full = np.repeat(gt[:n_seq][None, :, :, :], n_samples, axis=0).reshape(-1, gt.shape[1], gt.shape[2])
                    pred_full = pred_samples.reshape(-1, pred_samples.shape[2], pred_samples.shape[3])
                    gt_k, pred_k, k_used = _limit_pooled_stochastic_samples(gt_full, pred_full, n_samples=n_samples, max_samples=k)
                    score = _score_prediction_bundle(gt_k, pred_k, channels)
                    neuro = score.get("neuro_scores", {})
                    sensitivity.append(
                        {
                            "task": task_name,
                            "model": f"{model_name}_samples",
                            "n_samples": k_used,
                            "FINAL_COMPOSITE_SCORE": neuro.get("FINAL_COMPOSITE_SCORE", float("nan")),
                            "family_distribution": neuro.get("family_distribution", float("nan")),
                            "family_temporal_spectral": neuro.get("family_temporal_spectral", float("nan")),
                            "family_relational": neuro.get("family_relational", float("nan")),
                            "family_geometry": neuro.get("family_geometry", float("nan")),
                            "family_state_dynamics": neuro.get("family_state_dynamics", float("nan")),
                            "score_error": score.get("score_error"),
                        }
                    )
    diagnostics = {
        "summary": rows,
        "sample_count_sensitivity": sensitivity,
        "interpretation": (
            "sample_to_empirical_variance_ratio near 1 indicates stochastic rollouts have target-bin diversity "
            "comparable to held-out IBL trial variability; lower values indicate under-diverse latent rollouts."
        ),
    }
    diag_path = output_root / "ibl_ssm_vae_diagnostics.json"
    diag_path.write_text(json.dumps(_ibl_json_ready(diagnostics), indent=2))
    if example is not None:
        _plot_ssm_vae_diagnostics(output_root / "ibl_ssm_vae_diagnostics.svg", example, sensitivity, channels, manifest.context_bins)
    return _ibl_json_ready({"diagnostics_json": diag_path, "diagnostics_plot": output_root / "ibl_ssm_vae_diagnostics.svg", "report": diagnostics})


def _plot_ssm_vae_diagnostics(output_path: Path, example: dict[str, object], sensitivity: list[dict[str, object]], channels: list[str], context_bins: int) -> None:
    gt = np.asarray(example["gt"], dtype=np.float64)
    samples = np.asarray(example["samples"], dtype=np.float64)
    n_show = min(5, samples.shape[0])
    ch_idx = min(2, len(channels) - 1)
    seq_idx = 0
    time = np.arange(gt.shape[1])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.5))
    ax = axes[0, 0]
    ax.plot(time, gt[seq_idx, :, ch_idx], color="black", lw=2.0, label="held-out trial")
    for sample_idx in range(n_show):
        ax.plot(time, samples[sample_idx, seq_idx, :, ch_idx], lw=1.1, alpha=0.65, label="sample" if sample_idx == 0 else None)
    ax.axvline(context_bins - 0.5, color="0.4", ls="--", lw=1.0)
    ax.set_title(f"SSM-VAE stochastic rollouts: {channels[ch_idx]}")
    ax.set_xlabel("20 ms bin")
    ax.set_ylabel("normalized log rate")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    target = slice(context_bins, gt.shape[1])
    empirical_var = np.nanvar(gt[:, target, :], axis=0).mean(axis=0)
    sample_var = np.nanvar(samples[:, :, target, :], axis=0).mean(axis=(0, 1))
    x = np.arange(len(channels))
    width = 0.38
    ax.bar(x - width / 2, empirical_var, width=width, label="held-out trial variance", color="#4C78A8")
    ax.bar(x + width / 2, sample_var, width=width, label="SSM-VAE sample variance", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=35, ha="right", fontsize=7)
    ax.set_title("Diversity by region")
    ax.set_ylabel("target-bin variance")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    sens_df = pd.DataFrame(sensitivity)
    if not sens_df.empty and "FINAL_COMPOSITE_SCORE" in sens_df:
        ax.plot(sens_df["n_samples"], sens_df["FINAL_COMPOSITE_SCORE"], marker="o", color="#54A24B")
        ax.set_ylim(0.0, 1.05)
    ax.set_title("NethoBench vs generated sample count")
    ax.set_xlabel("pooled stochastic samples")
    ax.set_ylabel("composite score")
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    mean_pred = samples[:, :, target, :].mean(axis=0).reshape(-1)
    gt_target = gt[:, target, :].reshape(-1)
    ax.hexbin(gt_target, mean_pred, gridsize=35, mincnt=1, cmap="mako" if "mako" in plt.colormaps() else "viridis")
    ax.set_title("Sample-mean fidelity sidecar")
    ax.set_xlabel("ground truth")
    ax.set_ylabel("SSM-VAE sample mean")
    lim = np.nanpercentile(np.concatenate([gt_target, mean_pred]), [1, 99])
    ax.plot(lim, lim, color="white", lw=1.0, alpha=0.8)
    fig.suptitle("Latent SSM-VAE probabilistic Neuropixels baseline", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def train_ibl_study_models(
    manifest_path: Path,
    *,
    output_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    manifest = load_ibl_manifest(Path(manifest_path))
    out_root = _resolve_output_root(manifest, output_root)
    if dry_run:
        arr_ready = (out_root / "ibl_feature_manifest.json").exists()
        report = {"dry_run": True, "feature_manifest_found": arr_ready, "model_suite": manifest.model_suite}
        report_path = out_root / "ibl_train_dry_run_report.json"
        report_path.write_text(json.dumps(_ibl_json_ready(report), indent=2))
        return _ibl_json_ready({"report_json": report_path, "report": report})
    arr, channels, sequence_table, feature_manifest = _load_feature_bundle(out_root)
    tasks = _study_split_indices(sequence_table, manifest.split_policy, manifest.random_seed)
    predictions_root = out_root / "predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)
    task_reports: dict[str, object] = {}
    for task_name, split in tasks.items():
        train_idx = np.asarray(split["train"], dtype=np.int64)
        norm, mean, std = _normalize_for_split(arr, train_idx, manifest.context_bins)
        model_outputs = _predict_study_models(
            arr,
            norm,
            split,
            manifest,
            mean,
            std,
            checkpoint_root=out_root / "models" / task_name,
        )
        test_idx = np.asarray(split["test"], dtype=np.int64)
        task_root = predictions_root / task_name
        task_root.mkdir(parents=True, exist_ok=True)
        gt = norm[test_idx]
        gt_path = task_root / "gt_region_rates.csv"
        array_to_nethobench_frame(gt, channels, sequence_prefix=f"{task_name}_gt").to_csv(gt_path, index=False)
        models_report = {}
        for model_name, payload in model_outputs.items():
            if "prediction" not in payload:
                models_report[model_name] = payload
                continue
            pred = np.asarray(payload["prediction"], dtype=np.float64)
            pred_path = task_root / f"pred_{model_name}_region_rates.csv"
            array_to_nethobench_frame(pred, channels, sequence_prefix=f"{task_name}_{model_name}").to_csv(pred_path, index=False)
            model_report = {
                "prediction_csv": pred_path,
                "metadata": payload.get("metadata", {}),
                "prediction_shape": list(pred.shape),
            }
            seed_prediction_reports: dict[str, object] = {}
            for seed, seed_prediction in payload.get("seed_predictions", {}).items():
                seed_pred = np.asarray(seed_prediction, dtype=np.float64)
                seed_path = (
                    task_root
                    / f"pred_{model_name}_seed_{seed}_region_rates.csv"
                )
                array_to_nethobench_frame(
                    seed_pred,
                    channels,
                    sequence_prefix=f"{task_name}_{model_name}_seed_{seed}",
                ).to_csv(seed_path, index=False)
                seed_prediction_reports[str(seed)] = {
                    "prediction_csv": seed_path,
                    "prediction_shape": list(seed_pred.shape),
                }
            if seed_prediction_reports:
                model_report["seed_predictions"] = seed_prediction_reports
            if "sample_prediction" in payload:
                sample_pred = np.asarray(payload["sample_prediction"], dtype=np.float64)
                gt_pooled, pred_pooled = _pool_stochastic_samples(gt, sample_pred)
                sample_gt_path = task_root / f"gt_{model_name}_samples_region_rates.csv"
                sample_pred_path = task_root / f"pred_{model_name}_samples_region_rates.csv"
                array_to_nethobench_frame(gt_pooled, channels, sequence_prefix=f"{task_name}_{model_name}_sample_gt").to_csv(sample_gt_path, index=False)
                array_to_nethobench_frame(pred_pooled, channels, sequence_prefix=f"{task_name}_{model_name}_sample").to_csv(sample_pred_path, index=False)
                model_report.update(
                    {
                        "stochastic_gt_csv": sample_gt_path,
                        "stochastic_prediction_csv": sample_pred_path,
                        "stochastic_model_name": f"{model_name}_samples",
                        "stochastic_prediction_shape": list(pred_pooled.shape),
                        "n_stochastic_samples": int(sample_pred.shape[0]),
                    }
                )
            models_report[model_name] = model_report
        task_reports[task_name] = {
            "split": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in split.items()},
            "gt_csv": gt_path,
            "models": models_report,
            "normalization": {"mean": mean, "std": std, "fit": "train sequences, context bins only"},
        }
    report = {
        "manifest": asdict(manifest),
        "feature_manifest": feature_manifest,
        "tasks": task_reports,
    }
    report_path = out_root / "ibl_model_predictions_manifest.json"
    report_path.write_text(json.dumps(_ibl_json_ready(report), indent=2))
    return _ibl_json_ready({"predictions_manifest_json": report_path, "report": report})


def score_ibl_study(
    manifest_path: Path,
    *,
    output_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    manifest = load_ibl_manifest(Path(manifest_path))
    out_root = _resolve_output_root(manifest, output_root)
    pred_manifest_path = out_root / "ibl_model_predictions_manifest.json"
    if dry_run:
        report = {"dry_run": True, "predictions_manifest_found": pred_manifest_path.exists()}
        report_path = out_root / "ibl_score_dry_run_report.json"
        report_path.write_text(json.dumps(_ibl_json_ready(report), indent=2))
        return _ibl_json_ready({"report_json": report_path, "report": report})
    arr, channels, sequence_table, feature_manifest = _load_feature_bundle(out_root)
    pred_manifest = json.loads(pred_manifest_path.read_text())
    scores_by_task: dict[str, object] = {}
    horizon_scores: dict[str, object] = {}
    for task_name, task_payload in pred_manifest["tasks"].items():
        gt = _frame_to_array(Path(task_payload["gt_csv"]), channels)
        half_a = gt[::2]
        half_b = gt[1::2]
        n_half = min(half_a.shape[0], half_b.shape[0])
        split_half = _score_prediction_bundle(half_a[:n_half], half_b[:n_half], channels) if n_half else {"score_error": "too_few_sequences"}
        models = {}
        horizon_scores[task_name] = {}
        for model_name, model_payload in task_payload.get("models", {}).items():
            if "prediction_csv" not in model_payload:
                models[model_name] = {"score_error": model_payload.get("error", "missing prediction")}
                continue
            pred = _frame_to_array(Path(model_payload["prediction_csv"]), channels)
            models[model_name] = _score_prediction_bundle(gt, pred, channels)
            horizon_scores[task_name][model_name] = _score_horizons(gt, pred, channels, [25, 50, 75], manifest.context_bins)
            for seed, seed_payload in model_payload.get(
                "seed_predictions", {}
            ).items():
                seed_name = f"{model_name}_seed_{seed}"
                seed_pred = _frame_to_array(
                    Path(seed_payload["prediction_csv"]),
                    channels,
                )
                models[seed_name] = _score_prediction_bundle(
                    gt,
                    seed_pred,
                    channels,
                )
                models[seed_name]["training_seed"] = int(seed)
                models[seed_name]["ensemble_parent"] = model_name
                horizon_scores[task_name][seed_name] = _score_horizons(
                    gt,
                    seed_pred,
                    channels,
                    [25, 50, 75],
                    manifest.context_bins,
                )
            if "stochastic_prediction_csv" in model_payload and "stochastic_gt_csv" in model_payload:
                stochastic_name = str(model_payload.get("stochastic_model_name", f"{model_name}_samples"))
                stochastic_gt = _frame_to_array(Path(model_payload["stochastic_gt_csv"]), channels)
                stochastic_pred = _frame_to_array(Path(model_payload["stochastic_prediction_csv"]), channels)
                n_stochastic_samples = int(model_payload.get("n_stochastic_samples", manifest.ssm_vae_samples))
                stochastic_gt, stochastic_pred, n_scored_samples = _limit_pooled_stochastic_samples(
                    stochastic_gt,
                    stochastic_pred,
                    n_samples=n_stochastic_samples,
                    max_samples=manifest.ssm_vae_score_samples,
                )
                models[stochastic_name] = _score_prediction_bundle(stochastic_gt, stochastic_pred, channels)
                models[stochastic_name]["scoring_note"] = (
                    f"pooled stochastic score used {n_scored_samples} evenly spaced samples "
                    f"out of {n_stochastic_samples} generated samples per context"
                )
                horizon_scores[task_name][stochastic_name] = _score_horizons(stochastic_gt, stochastic_pred, channels, [25, 50, 75], manifest.context_bins)
        scores_by_task[task_name] = {"split_half": split_half, "models": models}
    score_long = _flatten_scores(scores_by_task)
    score_long_path = out_root / "ibl_model_scores_long.csv"
    score_long.to_csv(score_long_path, index=False)
    score_json_path = out_root / "ibl_model_scores.json"
    score_json_path.write_text(json.dumps(_ibl_json_ready(scores_by_task), indent=2))
    bootstrap = _bootstrap_ci(score_long, samples=manifest.bootstrap_samples, seed=manifest.random_seed)
    bootstrap_path = out_root / "ibl_bootstrap_ci.json"
    bootstrap_path.write_text(json.dumps(_ibl_json_ready(bootstrap), indent=2))
    main_task = "within_session" if "within_session" in scores_by_task else next(iter(scores_by_task))
    main_gt = _frame_to_array(Path(pred_manifest["tasks"][main_task]["gt_csv"]), channels)
    corruption_ladder = _corruption_ladder(main_gt, channels, manifest.random_seed)
    corruption_path = out_root / "ibl_corruption_ladder_scores.json"
    corruption_path.write_text(json.dumps(_ibl_json_ready(corruption_ladder), indent=2))
    coverage_plot = out_root / "ibl_dataset_coverage.png"
    family_plot = out_root / "ibl_family_comparison_expanded.png"
    heatmap_plot = out_root / "ibl_model_family_heatmap.png"
    distribution_plot = out_root / "ibl_task_model_stripplot.png"
    corruption_plot = out_root / "ibl_corruption_ladder.png"
    _plot_dataset_coverage(coverage_plot, sequence_table, channels)
    _plot_model_family_heatmap(heatmap_plot, score_long)
    _plot_score_distributions(distribution_plot, score_long)
    _plot_corruption_ladder(corruption_plot, corruption_ladder)
    _plot_expanded_family_comparison(family_plot, scores_by_task)
    ssm_vae_diagnostics = _ssm_vae_diagnostics(pred_manifest, channels, manifest, out_root)
    study_report = {
        "manifest": asdict(manifest),
        "feature_manifest": feature_manifest,
        "predictions_manifest_json": pred_manifest_path,
        "n_sequences": int(arr.shape[0]),
        "n_time_bins": int(arr.shape[1]),
        "n_regions": int(arr.shape[2]),
        "n_sessions": int(sequence_table["eid"].nunique()),
        "n_labs": int(sequence_table["lab"].replace("", "unknown").nunique()),
        "region_panel": channels,
        "scores": scores_by_task,
        "horizon_scores": horizon_scores,
        "ssm_vae_diagnostics": ssm_vae_diagnostics.get("report", {}),
        "outputs": {
            "ibl_dataset_inventory_csv": out_root / "ibl_dataset_inventory.csv",
            "ibl_feature_manifest_json": out_root / "ibl_feature_manifest.json",
            "ibl_region_config_json": out_root / "ibl_region_config.json",
            "gt_region_rates_csv": out_root / "gt_region_rates.csv",
            "ibl_model_scores_long_csv": score_long_path,
            "ibl_model_scores_json": score_json_path,
            "ibl_bootstrap_ci_json": bootstrap_path,
            "ibl_corruption_ladder_scores_json": corruption_path,
            "dataset_coverage_plot": coverage_plot,
            "family_comparison_plot": family_plot,
            "model_family_heatmap_plot": heatmap_plot,
            "task_model_stripplot": distribution_plot,
            "corruption_ladder_plot": corruption_plot,
            "ssm_vae_diagnostics_json": ssm_vae_diagnostics.get("diagnostics_json"),
            "ssm_vae_diagnostics_plot": ssm_vae_diagnostics.get("diagnostics_plot"),
        },
        "paper_interpretation": (
            "NethoBench generalizes from widefield calcium to public high-temporal-resolution "
            "Neuropixels spiking by separating statistical, linear dynamical, recurrent, and "
            "Transformer models across structural-realism families."
        ),
    }
    region_config = {
        "sequence_key": SEQUENCE_KEY,
        "time_key": TIME_KEY,
        "regions": channels,
        "bin_size_seconds": manifest.bin_size_seconds,
        "window_seconds": manifest.window_seconds,
        "context_bins": manifest.context_bins,
        "target_bins": manifest.target_bins,
        "representation": "region-level log1p mean firing rate, z-scored per task from train context bins",
    }
    (out_root / "ibl_region_config.json").write_text(json.dumps(_ibl_json_ready(region_config), indent=2))
    norm_preview, _, _ = zscore_from_train_context(arr, np.arange(max(1, arr.shape[0] // 2)), context_bins=manifest.context_bins)
    array_to_nethobench_frame(norm_preview, channels, sequence_prefix="ibl_all_norm").to_csv(out_root / "gt_region_rates.csv", index=False)
    report_path = out_root / "ibl_study_report.json"
    report_path.write_text(json.dumps(_ibl_json_ready(study_report), indent=2))
    return _ibl_json_ready({"study_report_json": report_path, "report": study_report})


def _frame_to_array(path: Path, channels: list[str]) -> np.ndarray:
    frame = pd.read_csv(path)
    seq_ids = frame[SEQUENCE_KEY].astype(str).drop_duplicates().tolist()
    n_time = int(frame.groupby(SEQUENCE_KEY)[TIME_KEY].count().iloc[0])
    arr = frame[channels].to_numpy(dtype=np.float64).reshape(len(seq_ids), n_time, len(channels))
    return arr


def _plot_expanded_family_comparison(output_path: Path, scores_by_task: dict[str, object]) -> None:
    rows = []
    family_keys = [
        "family_distribution",
        "family_temporal_spectral",
        "family_relational",
        "family_geometry",
        "family_state_dynamics",
        "FINAL_COMPOSITE_SCORE",
    ]
    for task, payload in scores_by_task.items():
        for model, model_payload in {"split_half": payload.get("split_half", {}), **payload.get("models", {})}.items():
            scores = model_payload.get("neuro_scores", {})
            for metric in family_keys:
                rows.append({"task": task, "model": model, "metric": metric, "value": scores.get(metric, np.nan)})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    pivot = df.pivot_table(index="metric", columns="model", values="value", aggfunc="mean").reindex(family_keys)
    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    x = np.arange(pivot.shape[0])
    width = min(0.14, 0.82 / max(1, pivot.shape[1]))
    for idx, model in enumerate(pivot.columns):
        ax.bar(x + (idx - (pivot.shape[1] - 1) / 2.0) * width, pivot[model].astype(float).to_numpy(), width=width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("family_", "").replace("_", " ") for m in family_keys], rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("NethoBench score")
    ax.set_title("Expanded IBL repeated-site model comparison")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_ibl_expanded_study(
    manifest_path: Path,
    *,
    output_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    manifest = load_ibl_manifest(Path(manifest_path))
    out_root = _resolve_output_root(manifest, output_root)
    if dry_run:
        build = build_ibl_repeated_site_dataset(manifest_path, output_root=out_root, dry_run=True)
        train = train_ibl_study_models(manifest_path, output_root=out_root, dry_run=True)
        score = score_ibl_study(manifest_path, output_root=out_root, dry_run=True)
        return _ibl_json_ready({"dry_run": True, "build": build, "train": train, "score": score})
    build_ibl_repeated_site_dataset(manifest_path, output_root=out_root)
    train_ibl_study_models(manifest_path, output_root=out_root)
    return score_ibl_study(manifest_path, output_root=out_root)


def _expected_output_names(out_root: Path) -> dict[str, str]:
    names = [
        "gt_region_rates.csv",
        "pred_poisson_region_rates.csv",
        "pred_var_region_rates.csv",
        "ibl_region_config.json",
        "ibl_split_half_ceiling_scores.json",
        "ibl_model_scores.json",
        "ibl_corruption_ladder_scores.json",
        "ibl_repeated_site_report.json",
        "ibl_family_comparison.png",
        "ibl_corruption_ladder.png",
        "unit_level/ibl_unit_level_supplement_scores.json",
    ]
    return {name: str(out_root / name) for name in names}


def cli_export_main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ibl_export_nethobench",
        description="Run the IBL repeated-site Neuropixels NethoBench validation.",
    )
    parser.add_argument("--manifest", required=True, help="Path to the IBL benchmark manifest JSON.")
    parser.add_argument("--output-root", help="Optional output directory override.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and dependency availability without downloading data.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = export_ibl_repeated_site_benchmark(
        Path(args.manifest),
        output_root=Path(args.output_root) if args.output_root else None,
        dry_run=args.dry_run,
    )
    print("IBL repeated-site export complete.")
    if "outputs" in report:
        print(f"Benchmark report : {report['outputs']['ibl_repeated_site_report_json']}")
    else:
        print(f"Benchmark report : {report['report_json']}")
