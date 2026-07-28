from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from nethobench.neuro import _load_sequences
from nethobench.analysis.additional_neuro_metrics import (
    _assign_to_centers,
    _fit_kmeans_centers,
    _occupancy_similarity,
    _pooled_rows,
    _prepare_latent_state_reference,
    _transition_similarity,
)
from nethobench.analysis.score_definitions import (
    NEURO_FAMILY_METRICS,
    NEURO_FAMILY_WEIGHTS,
    compute_neuro_composite,
    compute_neuro_family_scores,
    weighted_mean_available,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "paper" / "Nethobench"
FIG_DIR = PAPER_ROOT / "figures"
TABLE_DIR = PAPER_ROOT / "generated_tables"

DESKTOP = Path.home() / "Desktop"
SCALING_WIDE = DESKTOP / "netho-seq-scaling-rerun" / "tables" / "nethobench_neurobench_scores_scores_wide.csv"
MODEL_FAMILY_SCORES = DESKTOP / "final_pruned_scores_360_1260.csv"
FIG2_MASTER = TABLE_DIR / "fig2_multiseed_scores_long.csv"

METRIC_COMPARISON_TABLES = [
    ("sequifier_convergence", DESKTOP / "nethobench-sequifier-convergence" / "results" / "all_metric_comparisons.csv"),
    (
        "biophysical_convergence",
        DESKTOP / "nethobench-sequifier-convergence-biophysical" / "results" / "all_metric_comparisons.csv",
    ),
    ("calciumgan_transfer", DESKTOP / "nethobench-calciumgan-biophysical" / "results" / "all_metric_comparisons.csv"),
]

FAMILY_COLS = [f"family_{name}" for name in NEURO_FAMILY_WEIGHTS]
FAMILY_LABELS = {
    "family_distribution": "Distribution",
    "family_temporal_spectral": "Temporal",
    "family_relational": "Relational",
    "family_geometry": "Geometry",
    "family_state_dynamics": "State dynamics",
}
METRIC_TO_FAMILY = {
    metric: family
    for family, metrics in NEURO_FAMILY_METRICS.items()
    for metric in metrics
}
DEFAULT_FAMILY_WEIGHTS = OrderedDict((f"family_{k}", float(v)) for k, v in NEURO_FAMILY_WEIGHTS.items())

COLORS = {
    "rank": "#4C78A8",
    "corr": "#6BAA75",
    "warn": "#E45756",
    "neutral": "#6F6F6F",
}
RAW_MAX_SEQUENCES = int(os.environ.get("NETHOBENCH_ROBUSTNESS_RAW_MAX_SEQ", "96"))
RAW_MAX_TIMESTEPS = int(os.environ.get("NETHOBENCH_ROBUSTNESS_RAW_MAX_TIME", "240"))


@dataclass(frozen=True)
class VariantResult:
    variant: str
    variant_group: str
    condition_id: str
    source: str
    score: float
    rank: float
    default_score: float
    default_rank: float
    spearman_to_default: float
    coverage_note: str


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _as_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def _finite_dict(row: Mapping[str, object]) -> dict[str, float]:
    return {str(k): _as_float(v) for k, v in row.items()}


def _renormalize(weights: Mapping[str, float]) -> OrderedDict[str, float]:
    vals = OrderedDict((k, max(float(v), 0.0)) for k, v in weights.items())
    total = sum(vals.values())
    if total <= 0:
        n = max(len(vals), 1)
        return OrderedDict((k, 1.0 / n) for k in vals)
    return OrderedDict((k, v / total) for k, v in vals.items())


def _compute_family_scores_with_metric_weights(
    metric_scores: Mapping[str, float],
    family_metric_weights: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for family, metric_weights in family_metric_weights.items():
        vals = {metric: _as_float(metric_scores.get(metric, np.nan)) for metric in metric_weights}
        out[f"family_{family}"] = weighted_mean_available(vals, metric_weights)
    return out


def _compute_composite_from_families(
    family_scores: Mapping[str, float],
    family_weights: Mapping[str, float],
) -> float:
    vals = {family: _as_float(family_scores.get(family, np.nan)) for family in family_weights}
    return weighted_mean_available(vals, family_weights)


def _row_family_scores(row: Mapping[str, object], family_metric_weights: Mapping[str, Mapping[str, float]] | None = None) -> dict[str, float]:
    metric_scores = _finite_dict(row)
    if family_metric_weights is None:
        families = compute_neuro_family_scores(metric_scores)
    else:
        families = _compute_family_scores_with_metric_weights(metric_scores, family_metric_weights)
    for col in FAMILY_COLS:
        if not np.isfinite(families.get(col, np.nan)):
            families[col] = _as_float(row.get(col, np.nan))
    return families


def _row_default_score(row: Mapping[str, object]) -> float:
    existing = _as_float(row.get("FINAL_COMPOSITE_SCORE", row.get("FINAL_NEURO_COMPOSITE_SCORE", row.get("composite_score", np.nan))))
    if np.isfinite(existing):
        return existing
    metric_scores = _finite_dict(row)
    comp = compute_neuro_composite(metric_scores)
    if np.isfinite(comp):
        return comp
    return _compute_composite_from_families(_row_family_scores(row), DEFAULT_FAMILY_WEIGHTS)


def _load_fig2_master() -> pd.DataFrame:
    df = _read_csv(FIG2_MASTER)
    if df.empty:
        return df
    df = df.copy()
    if "condition_id" not in df.columns:
        parts = []
        for _, r in df.iterrows():
            fields = [
                str(r.get("analysis", "analysis")),
                str(r.get("condition", r.get("model", "condition"))),
                str(r.get("model", "")),
                f"seed{r.get('seed', '')}",
                f"mag{r.get('perturbation_magnitude', '')}",
            ]
            parts.append("|".join([x for x in fields if x and x != "nan"]))
        df["condition_id"] = parts
    df["source"] = "fig2_master"
    return df


def _load_scaling_summary() -> pd.DataFrame:
    df = _read_csv(SCALING_WIDE)
    if df.empty:
        return df
    sub = df[
        (df.get("sequence_length", pd.Series(dtype=float)) == 90)
        & (df.get("brain_areas", pd.Series(dtype=float)) == 16)
        & (df.get("data_share", pd.Series(dtype=float)) == 100)
        & (df.get("training_percent", pd.Series(dtype=float)).isin([10.0, 30.0, 100.0]))
    ].copy()
    if sub.empty:
        sub = df.copy()
    score_cols = [
        c
        for c in sub.columns
        if c.endswith("_score01") or c in FAMILY_COLS or c in {"FINAL_COMPOSITE_SCORE", "FINAL_NEURO_COMPOSITE_SCORE", "composite_score", "family_fidelity"}
    ]
    group_cols = ["model_family", "training_percent"]
    if not set(group_cols).issubset(sub.columns):
        return pd.DataFrame()
    out = sub.groupby(group_cols, as_index=False)[score_cols].mean(numeric_only=True)
    out["condition_id"] = out["model_family"].astype(str) + "_tp" + out["training_percent"].astype(int).astype(str)
    out["source"] = "training_progress_mean"
    return out


def _load_raw_recompute_candidates(max_rows: int | None = None) -> pd.DataFrame:
    df = _read_csv(SCALING_WIDE)
    rows = []

    if not df.empty and {"ground_truth_path", "prediction_path", "training_percent", "seed"}.issubset(df.columns):
        sub = df[
            (df.get("sequence_length", pd.Series(dtype=float)) == 90)
            & (df.get("brain_areas", pd.Series(dtype=float)) == 16)
            & (df.get("data_share", pd.Series(dtype=float)) == 100)
            & (df.get("training_percent", pd.Series(dtype=float)).isin([10.0, 30.0, 100.0]))
        ].copy()
        if sub.empty:
            sub = df.copy()
        for _, row in sub.sort_values(["training_percent", "seed"]).iterrows():
            gt_path = Path(str(row["ground_truth_path"]))
            pred_path = Path(str(row["prediction_path"]))
            if gt_path.exists() and pred_path.exists():
                item = row.to_dict()
                item["condition_id"] = f"training_tp{int(float(row['training_percent']))}_seed{int(row['seed'])}"
                item["raw_source"] = "training_progress_raw"
                rows.append(item)

    fixed_pairs = [
        (
            "biophysical_converged_raw",
            DESKTOP / "nethobench-sequifier-convergence-biophysical" / "results" / "converged" / "ground_truth_aligned_rollout.csv",
            DESKTOP / "nethobench-sequifier-convergence-biophysical" / "results" / "converged" / "predictions_aligned_rollout.csv",
        ),
        (
            "biophysical_weakest_raw",
            DESKTOP / "nethobench-sequifier-convergence-biophysical" / "results" / "weakest" / "ground_truth_aligned_rollout.csv",
            DESKTOP / "nethobench-sequifier-convergence-biophysical" / "results" / "weakest" / "predictions_aligned_rollout.csv",
        ),
        (
            "transfer_calciumgan_converged_raw",
            DESKTOP / "nethobench-calciumgan-biophysical" / "results" / "transfer_calciumgan_converged" / "ground_truth_aligned_rollout.csv",
            DESKTOP / "nethobench-calciumgan-biophysical" / "results" / "transfer_calciumgan_converged" / "predictions_aligned_rollout.csv",
        ),
        (
            "transfer_calciumgan_weakest_raw",
            DESKTOP / "nethobench-calciumgan-biophysical" / "results" / "transfer_calciumgan_weakest" / "ground_truth_aligned_rollout.csv",
            DESKTOP / "nethobench-calciumgan-biophysical" / "results" / "transfer_calciumgan_weakest" / "predictions_aligned_rollout.csv",
        ),
        (
            "ar1_converged_raw",
            DESKTOP / "nethobench-sequifier-convergence" / "results" / "converged" / "ground_truth_aligned_rollout.csv",
            DESKTOP / "nethobench-sequifier-convergence" / "results" / "converged" / "predictions_aligned_rollout.csv",
        ),
        (
            "ar1_underfit_raw",
            DESKTOP / "nethobench-sequifier-convergence" / "results" / "underfit" / "ground_truth_aligned_rollout.csv",
            DESKTOP / "nethobench-sequifier-convergence" / "results" / "underfit" / "predictions_aligned_rollout.csv",
        ),
        (
            "ar1_weakest_raw",
            DESKTOP / "nethobench-sequifier-convergence" / "results" / "weakest" / "ground_truth_aligned_rollout.csv",
            DESKTOP / "nethobench-sequifier-convergence" / "results" / "weakest" / "predictions_aligned_rollout.csv",
        ),
    ]
    for condition_id, gt_path, pred_path in fixed_pairs:
        if gt_path.exists() and pred_path.exists():
            rows.append(
                {
                    "condition_id": condition_id,
                    "raw_source": "fixed_model_raw",
                    "ground_truth_path": str(gt_path),
                    "prediction_path": str(pred_path),
                    "FINAL_COMPOSITE_SCORE": np.nan,
                }
            )
    if max_rows is not None:
        rows = rows[:max_rows]
    return pd.DataFrame(rows)


def _load_metric_comparison_tables() -> pd.DataFrame:
    rows = []
    for source, path in METRIC_COMPARISON_TABLES:
        df = _read_csv(path)
        if df.empty or not {"run_name", "metric", "model_score"}.issubset(df.columns):
            continue
        piv = df.pivot_table(index="run_name", columns="metric", values="model_score", aggfunc="mean").reset_index()
        for _, r in piv.iterrows():
            row = r.to_dict()
            row["condition_id"] = f"{source}|{row.get('run_name')}"
            row["source"] = source
            rows.append(row)
    return pd.DataFrame(rows)


def _load_model_family_table() -> pd.DataFrame:
    df = _read_csv(MODEL_FAMILY_SCORES)
    if df.empty:
        return df
    out = df.rename(
        columns={
            "distribution": "family_distribution",
            "temporal_trjdist_only": "family_temporal_spectral",
            "relational": "family_relational",
            "geometry": "family_geometry",
            "state_dynamics": "family_state_dynamics",
            "good_final_score_pruned": "FINAL_COMPOSITE_SCORE",
        }
    ).copy()
    out["condition_id"] = out["model"].astype(str) + "_h" + out["horizon"].astype(str)
    out["source"] = "model_family_table"
    return out


def build_score_matrix() -> pd.DataFrame:
    frames = [_load_fig2_master(), _load_scaling_summary(), _load_metric_comparison_tables(), _load_model_family_table()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    for idx, row in df.iterrows():
        families = _row_family_scores(row)
        for key, value in families.items():
            if key not in df.columns or not np.isfinite(_as_float(df.at[idx, key])):
                df.at[idx, key] = value
        if not np.isfinite(_as_float(row.get("FINAL_COMPOSITE_SCORE", np.nan))):
            df.at[idx, "FINAL_COMPOSITE_SCORE"] = _row_default_score({**row.to_dict(), **families})
    df = df[np.isfinite(pd.to_numeric(df["FINAL_COMPOSITE_SCORE"], errors="coerce"))].copy()
    df.to_csv(TABLE_DIR / "supp_metric_robustness_score_matrix.csv", index=False)
    return df


def _equal_submetric_weights() -> OrderedDict[str, OrderedDict[str, float]]:
    out: OrderedDict[str, OrderedDict[str, float]] = OrderedDict()
    for family, metrics in NEURO_FAMILY_METRICS.items():
        n = len(metrics)
        out[family] = OrderedDict((metric, 1.0 / n) for metric in metrics)
    return out


def _leave_one_submetric_weights(metric_to_omit: str) -> OrderedDict[str, OrderedDict[str, float]]:
    out: OrderedDict[str, OrderedDict[str, float]] = OrderedDict()
    for family, metrics in NEURO_FAMILY_METRICS.items():
        keep = [m for m in metrics if m != metric_to_omit]
        if not keep:
            out[family] = OrderedDict(metrics)
        else:
            out[family] = OrderedDict((metric, 1.0 / len(keep)) for metric in keep)
    return out


def _variant_score(row: Mapping[str, object], variant: str) -> tuple[float, str, str]:
    default_families = _row_family_scores(row)
    default_metric_weights = NEURO_FAMILY_METRICS
    default_family_weights = DEFAULT_FAMILY_WEIGHTS

    if variant == "default_weights":
        return _compute_composite_from_families(default_families, default_family_weights), "family_weights", "default"

    if variant == "equal_family_weights":
        weights = OrderedDict((family, 1.0 / len(default_family_weights)) for family in default_family_weights)
        return _compute_composite_from_families(default_families, weights), "family_weights", "all families equal"

    if variant.startswith("leave_family_"):
        family = variant.removeprefix("leave_family_")
        weights = OrderedDict(default_family_weights)
        weights[f"family_{family}"] = 0.0
        return _compute_composite_from_families(default_families, _renormalize(weights)), "family_weights", f"excluded {family}"

    if variant.startswith("double_family_"):
        family = variant.removeprefix("double_family_")
        weights = OrderedDict(default_family_weights)
        weights[f"family_{family}"] = weights.get(f"family_{family}", 0.0) * 2.0
        return _compute_composite_from_families(default_families, _renormalize(weights)), "family_weights", f"upweighted {family}"

    if variant == "submetric_equal_weights":
        families = _row_family_scores(row, _equal_submetric_weights())
        return _compute_composite_from_families(families, default_family_weights), "submetric_weights", "equalized within-family submetric weights"

    if variant.startswith("leave_submetric_"):
        metric = variant.removeprefix("leave_submetric_")
        families = _row_family_scores(row, _leave_one_submetric_weights(metric))
        return _compute_composite_from_families(families, default_family_weights), "submetric_weights", f"excluded {metric}"

    if variant == "distribution_jsd":
        return _compute_composite_from_families(default_families, default_family_weights), "distribution_divergence", "current KL_or_JSD score"

    if variant == "distribution_symmetric_kl":
        metric_scores = _finite_dict(row)
        alt = _as_float(metric_scores.get("KL_score01", np.nan))
        if not np.isfinite(alt):
            return np.nan, "distribution_divergence", "requires KL_score01 or raw GT/Pred recomputation"
        metric_scores["KL_or_JSD_score01"] = alt
        families = compute_neuro_family_scores(metric_scores)
        for col in FAMILY_COLS:
            if not np.isfinite(families.get(col, np.nan)):
                families[col] = _as_float(row.get(col, np.nan))
        return _compute_composite_from_families(families, default_family_weights), "distribution_divergence", "KL_score01 substituted for KL_or_JSD_score01"

    if variant == "topology_excluded":
        geom = _as_float(row.get("SubspaceAngle_score01", np.nan))
        if not np.isfinite(geom):
            return np.nan, "topology", "requires SubspaceAngle_score01"
        families = dict(default_families)
        families["family_geometry"] = geom
        return _compute_composite_from_families(families, default_family_weights), "topology", "geometry family uses subspace angle only"

    if variant == "topology_only":
        geom = _as_float(row.get("MANI_score01", np.nan))
        if not np.isfinite(geom):
            return np.nan, "topology", "requires MANI_score01"
        families = dict(default_families)
        families["family_geometry"] = geom
        return _compute_composite_from_families(families, default_family_weights), "topology", "geometry family uses MANI only"

    if variant == "state_k_11":
        keys = [
            "LatentStateOccupancyK11_score01",
            "LatentStateTransitionLag1K11_score01",
            "LatentStateTransitionLag2K11_score01",
            "LatentStateTransitionLag3K11_score01",
        ]
        state = weighted_mean_available({k: _as_float(row.get(k, np.nan)) for k in keys}, {k: 1.0 for k in keys})
        if not np.isfinite(state):
            return np.nan, "state_hyperparameters", "requires K11 occupancy/transition scores"
        families = dict(default_families)
        families["family_state_dynamics"] = state
        return _compute_composite_from_families(families, default_family_weights), "state_hyperparameters", "state family restricted to K=11"

    if variant == "state_k_12":
        state = _as_float(row.get("LatentStateOccupancyK12_score01", np.nan))
        if not np.isfinite(state):
            return np.nan, "state_hyperparameters", "requires K12 occupancy score"
        families = dict(default_families)
        families["family_state_dynamics"] = state
        return _compute_composite_from_families(families, default_family_weights), "state_hyperparameters", "state family restricted to K=12 occupancy"

    if variant.startswith("transition_lags_"):
        suffix = variant.removeprefix("transition_lags_")
        lag_map = {
            "1": ["LatentStateTransitionLag1K11_score01"],
            "1_2_3": [
                "LatentStateTransitionLag1K11_score01",
                "LatentStateTransitionLag2K11_score01",
                "LatentStateTransitionLag3K11_score01",
            ],
        }
        keys = lag_map.get(suffix)
        if keys is None:
            return np.nan, "state_hyperparameters", "requires raw GT/Pred recomputation for unavailable lag set"
        trans = weighted_mean_available({k: _as_float(row.get(k, np.nan)) for k in keys}, {k: 1.0 for k in keys})
        occ = weighted_mean_available(
            {
                "LatentStateOccupancyK11_score01": _as_float(row.get("LatentStateOccupancyK11_score01", np.nan)),
                "LatentStateOccupancyK12_score01": _as_float(row.get("LatentStateOccupancyK12_score01", np.nan)),
            },
            {"LatentStateOccupancyK11_score01": 1.0, "LatentStateOccupancyK12_score01": 1.0},
        )
        state = weighted_mean_available({"occupancy": occ, "transition": trans}, {"occupancy": 0.4, "transition": 0.6})
        if not np.isfinite(state):
            return np.nan, "state_hyperparameters", "requires state occupancy/transition scores"
        families = dict(default_families)
        families["family_state_dynamics"] = state
        return _compute_composite_from_families(families, default_family_weights), "state_hyperparameters", f"transition lags {suffix.replace('_', ',')}"

    if variant.startswith("pca_dim_") or variant in {"state_k_8", "state_k_16", "transition_lags_1_2_4", "transition_lags_2_4_8"}:
        return np.nan, "raw_recompute_required", "requires raw GT/Pred recomputation; not derivable from scalar score tables"

    return np.nan, "unknown", "unknown variant"


def _variant_names() -> list[str]:
    variants = [
        "default_weights",
        "equal_family_weights",
        "submetric_equal_weights",
        "distribution_jsd",
        "distribution_symmetric_kl",
        "topology_excluded",
        "topology_only",
        "state_k_11",
        "state_k_12",
        "transition_lags_1",
        "transition_lags_1_2_3",
        "transition_lags_1_2_4",
        "transition_lags_2_4_8",
        "state_k_8",
        "state_k_16",
        "pca_dim_2",
        "pca_dim_3",
        "pca_dim_5",
        "pca_dim_8",
    ]
    variants += [f"leave_family_{family}" for family in NEURO_FAMILY_WEIGHTS]
    variants += [f"double_family_{family}" for family in NEURO_FAMILY_WEIGHTS]
    variants += [f"leave_submetric_{metric}" for metric in METRIC_TO_FAMILY]
    return variants


def _distribution_score_from_arrays(gt: np.ndarray, pred: np.ndarray, *, mode: str) -> float:
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    n_reg = min(gt.shape[2], pred.shape[2])
    eps = 1e-12
    vals = []
    for r in range(n_reg):
        g = gt[:, :, r].reshape(-1)
        p = pred[:, :, r].reshape(-1)
        m = np.isfinite(g) & np.isfinite(p)
        g = g[m]
        p = p[m]
        if g.size < 50 or p.size < 50:
            continue
        lo, hi = np.nanpercentile(g, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        bins = np.linspace(lo, hi, 41)
        hg, _ = np.histogram(g, bins=bins)
        hp, _ = np.histogram(p, bins=bins)
        q = hg.astype(float) + eps
        s = hp.astype(float) + eps
        q /= q.sum()
        s /= s.sum()
        if mode == "jsd":
            mid = 0.5 * (q + s)
            div = 0.5 * np.sum(q * np.log(q / mid)) + 0.5 * np.sum(s * np.log(s / mid))
        elif mode == "symmetric_kl":
            div = 0.5 * np.sum(q * np.log(q / s)) + 0.5 * np.sum(s * np.log(s / q))
        else:
            raise ValueError(f"Unknown distribution divergence mode: {mode}")
        vals.append(1.0 / (1.0 + float(div)))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else np.nan


def _subspace_score_from_arrays(gt: np.ndarray, pred: np.ndarray, *, n_components: int) -> float:
    gt_flat = _pooled_rows(gt)
    pred_flat = _pooled_rows(pred)
    if gt_flat.shape[0] < 10 or pred_flat.shape[0] < 10:
        return np.nan
    n = min(gt_flat.shape[0], pred_flat.shape[0])
    gt_flat = gt_flat[:n]
    pred_flat = pred_flat[:n]
    gt_flat = gt_flat - np.mean(gt_flat, axis=0, keepdims=True)
    pred_flat = pred_flat - np.mean(pred_flat, axis=0, keepdims=True)
    try:
        _, _, vg = np.linalg.svd(gt_flat, full_matrices=False)
        _, _, vp = np.linalg.svd(pred_flat, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.nan
    k = min(int(n_components), vg.shape[0], vp.shape[0])
    if k < 1:
        return np.nan
    vals = np.linalg.svd(vg[:k] @ vp[:k].T, compute_uv=False)
    vals = np.clip(vals, 0.0, 1.0)
    return float(np.mean(vals))


def _load_and_align_for_raw(gt_path: Path, pred_path: Path) -> tuple[np.ndarray, np.ndarray]:
    pred, pred_regions = _load_sequences(pred_path)
    max_time = int(pred.shape[1])
    max_seq = int(pred.shape[0])
    gt_path = Path(gt_path)
    header = pd.read_csv(gt_path, nrows=0)
    if {"sequenceId", "itemPosition"}.issubset(header.columns):
        usecols = [c for c in header.columns if c in {"sequenceId", "itemPosition"} or c in pred_regions]
        if "aligned" in gt_path.name or gt_path.stat().st_size < 250_000_000:
            gt_df = pd.read_csv(gt_path, usecols=usecols)
            gt_df = gt_df[gt_df["sequenceId"] < max_seq]
        else:
            chunks = []
            for chunk in pd.read_csv(gt_path, usecols=usecols, chunksize=250_000):
                sub = chunk[(chunk["sequenceId"] < max_seq) & (chunk["itemPosition"] < max_time)]
                if not sub.empty:
                    chunks.append(sub)
            if not chunks:
                raise ValueError(f"No GT rows overlap prediction shape for {gt_path}")
            gt_df = pd.concat(chunks, ignore_index=True)
        gt_df = gt_df.sort_values(["sequenceId", "itemPosition"]).reset_index(drop=True)
        gt_regions = [c for c in gt_df.columns if c not in {"sequenceId", "itemPosition"}]
        seq_lengths = gt_df.groupby("sequenceId").size()
        common_seq = sorted(set(seq_lengths[seq_lengths >= max_time].index.tolist()).intersection(range(max_seq)))
        if not common_seq:
            raise ValueError(f"No complete GT sequences with {max_time} timepoints in {gt_path}")
        gt_df = gt_df[gt_df["sequenceId"].isin(common_seq)]
        gt_df = gt_df.groupby("sequenceId", group_keys=False).head(max_time)
        n_seq = len(common_seq)
        gt = gt_df[gt_regions].to_numpy(dtype=np.float64).reshape(n_seq, max_time, len(gt_regions))
        pred = pred[common_seq, :, :]
    else:
        gt, gt_regions = _load_sequences(gt_path)

    overlap = [region for region in gt_regions if region in pred_regions]
    if not overlap:
        raise ValueError("No overlapping regions for raw robustness pair.")
    gt_idx = [gt_regions.index(region) for region in overlap]
    pred_idx = [pred_regions.index(region) for region in overlap]
    n_seq = min(gt.shape[0], pred.shape[0])
    n_time = min(gt.shape[1], pred.shape[1])
    return gt[:n_seq, :n_time, :][:, :, gt_idx], pred[:n_seq, :n_time, :][:, :, pred_idx]


def _reference_for_pca(gt: np.ndarray, pred: np.ndarray, pca_dim: int) -> dict[str, object] | None:
    gt_flat = _pooled_rows(gt)
    pred_flat = _pooled_rows(pred)
    return _prepare_latent_state_reference(gt, pred, gt_flat, pred_flat, n_components=pca_dim)


def _state_family_from_reference(
    ref: dict[str, object] | None,
    *,
    k_values: tuple[int, ...] = (11, 12),
    transition_k: int = 11,
    transition_lags: tuple[int, ...] = (1, 2, 3),
) -> float:
    if ref is None:
        return np.nan

    def occupancy(k: int) -> float:
        centers = _fast_kmeans_centers(ref["gt_proj_fit"], k, random_state=k)
        gt_assign = _assign_to_centers(ref["gt_proj_eval"], centers)
        pred_assign = _assign_to_centers(ref["pred_proj_eval"], centers)
        return _occupancy_similarity(gt_assign, pred_assign, k)

    def transition(k: int, lag: int) -> float:
        centers = _fast_kmeans_centers(ref["gt_proj_fit"], k, random_state=k)
        gt_seq_assign = [_assign_to_centers(seq, centers) for seq in ref["gt_seq_proj"]]
        pred_seq_assign = [_assign_to_centers(seq, centers) for seq in ref["pred_seq_proj"]]
        return _transition_similarity(gt_seq_assign, pred_seq_assign, k, lag)

    occ_scores = [occupancy(k) for k in k_values]
    trans_scores = [transition(transition_k, lag) for lag in transition_lags]
    pieces = occ_scores + trans_scores
    weights = {f"x{i}": 1.0 for i in range(len(pieces))}
    vals = {f"x{i}": score for i, score in enumerate(pieces)}
    return weighted_mean_available(vals, weights)


def _fast_kmeans_centers(features: np.ndarray, n_clusters: int, *, random_state: int) -> np.ndarray | None:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < max(12, n_clusters):
        return None
    max_fit_points = 5000
    fit_features = features
    if features.shape[0] > max_fit_points:
        rng = np.random.default_rng(random_state)
        fit_features = features[rng.choice(features.shape[0], size=max_fit_points, replace=False)]
    model = MiniBatchKMeans(
        n_clusters=int(min(n_clusters, fit_features.shape[0])),
        n_init=3,
        max_iter=80,
        random_state=random_state,
        batch_size=min(2048, fit_features.shape[0]),
    )
    model.fit(fit_features)
    return np.asarray(model.cluster_centers_, dtype=np.float64)


def _build_raw_hyperparameter_variant_rows() -> pd.DataFrame:
    candidates = _load_raw_recompute_candidates()
    if candidates.empty:
        return pd.DataFrame()
    rows = []
    for _, source_row in candidates.iterrows():
        gt_path = Path(str(source_row["ground_truth_path"]))
        pred_path = Path(str(source_row["prediction_path"]))
        condition_id = str(source_row.get("condition_id", f"raw_tp{source_row.get('training_percent')}_seed{source_row.get('seed')}"))
        print(f"[raw robustness] loading {condition_id}", flush=True)
        try:
            gt, pred = _load_and_align_for_raw(gt_path, pred_path)
            gt = gt[:RAW_MAX_SEQUENCES, :RAW_MAX_TIMESTEPS, :]
            pred = pred[:RAW_MAX_SEQUENCES, :RAW_MAX_TIMESTEPS, :]
        except Exception as exc:
            rows.append(
                {
                    "variant": "raw_recompute_error",
                    "variant_group": "raw_recompute_required",
                    "condition_id": condition_id,
                    "source": "raw_hyperparameter_recompute",
                    "score": np.nan,
                    "rank": np.nan,
                    "default_score": _as_float(source_row.get("FINAL_COMPOSITE_SCORE", np.nan)),
                    "default_rank": np.nan,
                    "spearman_to_default": np.nan,
                    "coverage_note": str(exc),
                }
            )
            continue

        print(f"[raw robustness] recomputing variants for {condition_id} shape={gt.shape}", flush=True)
        default_families = _row_family_scores(source_row.to_dict())
        refs = {k: _reference_for_pca(gt, pred, k) for k in (2, 3, 5, 8)}
        if not np.isfinite(default_families.get("family_state_dynamics", np.nan)):
            default_families["family_state_dynamics"] = _state_family_from_reference(refs[3])
        if not np.isfinite(default_families.get("family_distribution", np.nan)):
            metric_scores = _finite_dict(source_row.to_dict())
            default_families["family_distribution"] = compute_neuro_family_scores(metric_scores).get("family_distribution", np.nan)
        default_score = _row_default_score({**source_row.to_dict(), **default_families})
        subspace_scores = {k: _subspace_score_from_arrays(gt, pred, n_components=k) for k in (2, 3, 5, 8)}
        def _families_with_pca(k: int) -> dict[str, float]:
            fam = dict(default_families)
            state = _state_family_from_reference(refs[k])
            geom = weighted_mean_available(
                {
                    "MANI_score01": _as_float(source_row.get("MANI_score01", np.nan)),
                    "SubspaceAngle_score01": subspace_scores[k],
                },
                {"MANI_score01": 0.50, "SubspaceAngle_score01": 0.50},
            )
            if np.isfinite(state):
                fam["family_state_dynamics"] = state
            if np.isfinite(geom):
                fam["family_geometry"] = geom
            return fam
        variant_specs = {
            "default_weights": ("raw_default", default_families),
            "distribution_jsd": (
                "distribution_divergence",
                {**default_families, "family_distribution": _distribution_score_from_arrays(gt, pred, mode="jsd")},
            ),
            "distribution_symmetric_kl": (
                "distribution_divergence",
                {**default_families, "family_distribution": _distribution_score_from_arrays(gt, pred, mode="symmetric_kl")},
            ),
            "pca_dim_2": (
                "raw_hyperparameters",
                _families_with_pca(2),
            ),
            "pca_dim_3": (
                "raw_hyperparameters",
                _families_with_pca(3),
            ),
            "pca_dim_5": (
                "raw_hyperparameters",
                _families_with_pca(5),
            ),
            "pca_dim_8": (
                "raw_hyperparameters",
                _families_with_pca(8),
            ),
            "state_k_8": (
                "state_hyperparameters",
                {**default_families, "family_state_dynamics": _state_family_from_reference(refs[3], k_values=(8,), transition_k=8)},
            ),
            "state_k_11": (
                "state_hyperparameters",
                {**default_families, "family_state_dynamics": _state_family_from_reference(refs[3], k_values=(11,), transition_k=11)},
            ),
            "state_k_12": (
                "state_hyperparameters",
                {**default_families, "family_state_dynamics": _state_family_from_reference(refs[3], k_values=(12,), transition_k=12)},
            ),
            "state_k_16": (
                "state_hyperparameters",
                {**default_families, "family_state_dynamics": _state_family_from_reference(refs[3], k_values=(16,), transition_k=16)},
            ),
            "transition_lags_1": (
                "state_hyperparameters",
                {**default_families, "family_state_dynamics": _state_family_from_reference(refs[3], transition_lags=(1,))},
            ),
            "transition_lags_1_2_3": (
                "state_hyperparameters",
                {**default_families, "family_state_dynamics": _state_family_from_reference(refs[3], transition_lags=(1, 2, 3))},
            ),
            "transition_lags_1_2_4": (
                "state_hyperparameters",
                {**default_families, "family_state_dynamics": _state_family_from_reference(refs[3], transition_lags=(1, 2, 4))},
            ),
            "transition_lags_2_4_8": (
                "state_hyperparameters",
                {**default_families, "family_state_dynamics": _state_family_from_reference(refs[3], transition_lags=(2, 4, 8))},
            ),
        }
        for variant, (group, families) in variant_specs.items():
            score = _compute_composite_from_families(families, DEFAULT_FAMILY_WEIGHTS)
            rows.append(
                {
                    "variant": variant,
                    "variant_group": group,
                    "condition_id": condition_id,
                    "source": "raw_hyperparameter_recompute",
                    "score": score,
                    "rank": np.nan,
                    "default_score": default_score,
                    "default_rank": np.nan,
                    "spearman_to_default": np.nan,
                    "coverage_note": f"raw recompute from {pred_path.name}",
                }
            )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    raw.to_csv(TABLE_DIR / "supp_metric_raw_hyperparameter_variants_all_rows.csv", index=False)
    out = []
    for variant, sub in raw[np.isfinite(pd.to_numeric(raw["score"], errors="coerce"))].groupby("variant"):
        ranks = sub["score"].rank(ascending=False, method="min")
        default_ranks = sub["default_score"].rank(ascending=False, method="min")
        rho = spearmanr(default_ranks, ranks).correlation if len(sub) >= 2 else np.nan
        for (idx, row), rank, default_rank in zip(sub.iterrows(), ranks, default_ranks):
            item = row.to_dict()
            item["rank"] = float(rank)
            item["default_rank"] = float(default_rank)
            item["spearman_to_default"] = float(rho) if np.isfinite(rho) else np.nan
            out.append(item)
    raw_out = pd.DataFrame(out)
    raw_out.to_csv(TABLE_DIR / "supp_metric_raw_hyperparameter_variants.csv", index=False)
    return raw_out


def build_robustness_variants(score_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if score_matrix.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for variant in _variant_names():
        scores = []
        groups = []
        notes = []
        for _, row in score_matrix.iterrows():
            score, group, note = _variant_score(row.to_dict(), variant)
            scores.append(score)
            groups.append(group)
            notes.append(note)
        common = np.isfinite(scores)
        if common.sum() < 2:
            skipped.append(
                {
                    "variant": variant,
                    "variant_group": groups[0] if groups else "unknown",
                    "available_conditions": int(common.sum()),
                    "reason": notes[0] if notes else "no finite scores",
                }
            )
            continue

        variant_scores = pd.Series(scores, index=score_matrix["condition_id"])
        default_scores = pd.to_numeric(score_matrix["FINAL_COMPOSITE_SCORE"], errors="coerce")
        default_ranks = default_scores.rank(ascending=False, method="min")
        ranks = variant_scores.rank(ascending=False, method="min")
        common_idx = common & np.isfinite(default_scores.to_numpy(float))
        rho = spearmanr(default_ranks[common_idx], ranks[common_idx]).correlation if common_idx.sum() >= 2 else np.nan
        for idx, row in score_matrix.reset_index(drop=True).iterrows():
            score = _as_float(scores[idx])
            if not np.isfinite(score):
                continue
            rows.append(
                {
                    "variant": variant,
                    "variant_group": groups[idx],
                    "condition_id": row["condition_id"],
                    "source": row.get("source", ""),
                    "score": score,
                    "rank": _as_float(ranks.iloc[idx]),
                    "default_score": _as_float(default_scores.iloc[idx]),
                    "default_rank": _as_float(default_ranks.iloc[idx]),
                    "spearman_to_default": _as_float(rho),
                    "coverage_note": notes[idx],
                }
            )

    variant_df = pd.DataFrame(rows)
    raw_variants = _build_raw_hyperparameter_variant_rows() if os.environ.get("NETHOBENCH_ROBUSTNESS_RAW", "1") == "1" else pd.DataFrame()
    if not raw_variants.empty:
        variant_df = pd.concat([variant_df, raw_variants], ignore_index=True, sort=False)
        available = set(raw_variants["variant"].dropna().astype(str))
        skipped = [row for row in skipped if row["variant"] not in available]
    skipped_df = pd.DataFrame(skipped, columns=["variant", "variant_group", "available_conditions", "reason"])
    summary = (
        variant_df.groupby(["variant", "variant_group"], as_index=False)
        .agg(
            n_conditions=("condition_id", "nunique"),
            spearman_to_default=("spearman_to_default", "first"),
            mean_abs_score_delta=("score", lambda s: np.nan),
        )
    )
    if not variant_df.empty:
        deltas = variant_df.assign(abs_delta=lambda d: np.abs(d["score"] - d["default_score"]))
        summary = (
            deltas.groupby(["variant", "variant_group"], as_index=False)
            .agg(
                n_conditions=("condition_id", "nunique"),
                spearman_to_default=("spearman_to_default", "first"),
                mean_abs_score_delta=("abs_delta", "mean"),
                max_abs_score_delta=("abs_delta", "max"),
            )
            .sort_values(["variant_group", "variant"])
        )
    variant_df.to_csv(TABLE_DIR / "supp_metric_robustness_variants_long.csv", index=False)
    summary.to_csv(TABLE_DIR / "supp_metric_robustness_rank_summary.csv", index=False)
    skipped_df.to_csv(TABLE_DIR / "supp_metric_robustness_skipped_variants.csv", index=False)
    return variant_df, summary, skipped_df


def build_dirichlet_weight_summary(score_matrix: pd.DataFrame, n_samples: int = 10_000) -> pd.DataFrame:
    rows = []
    usable = score_matrix[np.isfinite(pd.to_numeric(score_matrix["FINAL_COMPOSITE_SCORE"], errors="coerce"))].copy()
    if usable.empty or not all(c in usable.columns for c in FAMILY_COLS):
        return pd.DataFrame()
    family_vals = usable[FAMILY_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = np.isfinite(family_vals).any(axis=1)
    usable = usable.loc[valid].reset_index(drop=True)
    family_vals = family_vals[valid]
    if usable.shape[0] < 2:
        return pd.DataFrame()

    default_w = np.asarray([DEFAULT_FAMILY_WEIGHTS[c] for c in FAMILY_COLS], dtype=float)
    default_w = default_w / default_w.sum()
    default_scores = np.asarray([_compute_composite_from_families(dict(zip(FAMILY_COLS, vals)), DEFAULT_FAMILY_WEIGHTS) for vals in family_vals])
    default_rank = pd.Series(default_scores, index=usable["condition_id"]).rank(ascending=False, method="min")

    rng = np.random.default_rng(20260505)
    weights = rng.dirichlet(default_w * 80.0, size=n_samples)
    rank_store = np.zeros((usable.shape[0], n_samples), dtype=float)
    top_store = np.zeros((usable.shape[0], n_samples), dtype=float)
    rho = np.zeros(n_samples, dtype=float)
    for i, w in enumerate(weights):
        scores = np.array([weighted_mean_available(dict(zip(FAMILY_COLS, vals)), dict(zip(FAMILY_COLS, w))) for vals in family_vals])
        ranks = pd.Series(scores, index=usable["condition_id"]).rank(ascending=False, method="min")
        rank_store[:, i] = ranks.to_numpy(float)
        top_store[:, i] = (ranks.to_numpy(float) == 1.0).astype(float)
        rho[i] = spearmanr(default_rank.loc[ranks.index], ranks).correlation
    for row_idx, cid in enumerate(usable["condition_id"]):
        rows.append(
            {
                "condition_id": cid,
                "source": usable.loc[row_idx, "source"],
                "default_rank": float(default_rank[cid]),
                "mean_rank": float(np.mean(rank_store[row_idx])),
                "rank_q05": float(np.quantile(rank_store[row_idx], 0.05)),
                "rank_q95": float(np.quantile(rank_store[row_idx], 0.95)),
                "top_probability": float(np.mean(top_store[row_idx])),
                "n_weight_samples": int(n_samples),
                "mean_spearman_to_default": float(np.nanmean(rho)),
                "q05_spearman_to_default": float(np.nanquantile(rho, 0.05)),
                "q95_spearman_to_default": float(np.nanquantile(rho, 0.95)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "supp_metric_dirichlet_weight_rank_summary.csv", index=False)
    return out


def _score_columns(df: pd.DataFrame, min_non_na: int = 3) -> list[str]:
    cols = []
    for col in df.columns:
        if col in {"condition_id", "source"}:
            continue
        if col.endswith("_score01") or col in FAMILY_COLS or col in {"FINAL_COMPOSITE_SCORE", "FINAL_NEURO_COMPOSITE_SCORE", "composite_score", "FIDELITY_SCORE", "family_fidelity"}:
            if pd.to_numeric(df[col], errors="coerce").notna().sum() >= min_non_na:
                cols.append(col)
    return cols


def build_correlation_and_redundancy(score_matrix: pd.DataFrame, variant_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = _score_columns(score_matrix, min_non_na=3)
    data = score_matrix[cols].apply(pd.to_numeric, errors="coerce")
    metric_corr = data.corr(method="spearman")
    metric_corr.to_csv(TABLE_DIR / "supp_metric_correlation_matrix.csv")

    family_cols = [c for c in FAMILY_COLS + ["FINAL_COMPOSITE_SCORE", "FIDELITY_SCORE", "family_fidelity"] if c in data.columns]
    family_corr = data[family_cols].corr(method="spearman")
    family_corr.to_csv(TABLE_DIR / "supp_family_correlation_matrix.csv")

    rows = []
    labels = list(metric_corr.columns)
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            rho = _as_float(metric_corr.loc[a, b])
            if np.isfinite(rho) and abs(rho) > 0.85:
                rows.append({"analysis": "high_correlation_pair", "metric_a": a, "metric_b": b, "spearman_rho": rho})

    if not variant_df.empty:
        loo = variant_df[variant_df["variant"].str.startswith("leave_family_")].copy()
        if not loo.empty:
            loo["rank_shift_abs"] = np.abs(loo["rank"] - loo["default_rank"])
            for _, r in loo.iterrows():
                rows.append(
                    {
                        "analysis": "leave_one_family_out_rank_shift",
                        "variant": r["variant"],
                        "condition_id": r["condition_id"],
                        "rank_shift_abs": _as_float(r["rank_shift_abs"]),
                    }
                )

    fam_available = [c for c in FAMILY_COLS if c in data.columns and data[c].notna().sum() >= 8]
    if len(fam_available) >= 3:
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import KFold, cross_val_score
        except Exception:
            LinearRegression = None
        if LinearRegression is not None:
            fam_data = data[fam_available].dropna()
            n_splits = min(5, max(2, fam_data.shape[0] // 4))
            if fam_data.shape[0] >= n_splits and n_splits >= 2:
                for target in fam_available:
                    predictors = [c for c in fam_available if c != target]
                    X = fam_data[predictors].to_numpy(float)
                    y = fam_data[target].to_numpy(float)
                    cv = KFold(n_splits=n_splits, shuffle=True, random_state=20260505)
                    vals = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")
                    rows.append(
                        {
                            "analysis": "family_redundancy_regression",
                            "target_family": target,
                            "cv_r2_mean": float(np.nanmean(vals)),
                            "cv_r2_std": float(np.nanstd(vals)),
                            "n_rows": int(fam_data.shape[0]),
                        }
                    )

    redundancy = pd.DataFrame(rows)
    redundancy.to_csv(TABLE_DIR / "supp_metric_redundancy_summary.csv", index=False)
    return metric_corr, family_corr, redundancy


def _select_heatmap_conditions(
    variant_df: pd.DataFrame,
    max_conditions: int = 24,
    source_filter: str | None = None,
) -> list[str]:
    default = variant_df[variant_df["variant"] == "default_weights"].copy()
    if source_filter is not None:
        default = default[default["source"] == source_filter].copy()
    if default.empty:
        return []
    priority_sources = (
        [source_filter]
        if source_filter is not None
        else [
            "sequifier_convergence",
            "biophysical_convergence",
            "calciumgan_transfer",
            "training_progress_mean",
            "model_family_table",
        ]
    )
    selected = []
    for source in priority_sources:
        vals = default[default["source"] == source].sort_values("default_rank")["condition_id"].tolist()
        for cid in vals:
            if cid not in selected:
                selected.append(cid)
            if len(selected) >= max_conditions:
                return selected
    for cid in default.sort_values("default_rank")["condition_id"]:
        if cid not in selected:
            selected.append(cid)
        if len(selected) >= max_conditions:
            break
    return selected


def _draw_rank_heatmap_panel(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    condition_order: list[str],
    variant_order: list[str],
    title: str,
) -> mpl.image.AxesImage | None:
    if plot_df.empty or not condition_order or not variant_order:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return None
    piv = (
        plot_df.pivot_table(index="variant", columns="condition_id", values="rank", aggfunc="mean")
        .reindex(index=variant_order, columns=condition_order)
    )
    arr = piv.to_numpy(float)
    if not np.isfinite(arr).any():
        ax.axis("off")
        ax.text(0.5, 0.5, "No finite ranks", ha="center", va="center", transform=ax.transAxes)
        return None
    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad("#f2f2f2")
    im = ax.imshow(np.ma.masked_invalid(arr), aspect="auto", cmap=cmap, vmin=1, vmax=np.nanmax(arr))
    ax.set_xticks(np.arange(len(condition_order)))
    ax.set_xticklabels(condition_order, rotation=55, ha="right", fontsize=6.5)
    ax.set_yticks(np.arange(len(variant_order)))
    rho_lookup = (
        summary.sort_values(["variant", "n_conditions"], ascending=[True, False])
        .drop_duplicates("variant")
        .set_index("variant")
    )
    rho = rho_lookup.reindex(variant_order)["spearman_to_default"]
    labels = [f"{v}  rho={rho.loc[v]:.2f}" if np.isfinite(rho.loc[v]) else v for v in variant_order]
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_title(title, loc="left", fontsize=11, weight="bold")
    ax.set_xlabel("Model / condition")
    ax.set_ylabel("Benchmark variant")
    return im


def plot_robustness_heatmap(variant_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    if variant_df.empty:
        return
    scalar_groups = {
        "family_weights",
        "submetric_weights",
        "distribution_divergence",
        "topology",
        "state_hyperparameters",
    }
    raw_groups = {"raw_hyperparameters", "raw_default"}
    scalar_df = variant_df[
        (variant_df["source"] != "raw_hyperparameter_recompute")
        & variant_df["variant_group"].isin(scalar_groups)
    ].copy()
    raw_df = variant_df[
        (variant_df["source"] == "raw_hyperparameter_recompute")
        & variant_df["variant_group"].isin(raw_groups)
    ].copy()

    scalar_conditions = _select_heatmap_conditions(scalar_df, max_conditions=24)
    raw_conditions = _select_heatmap_conditions(raw_df, max_conditions=22, source_filter="raw_hyperparameter_recompute")

    scalar_order = (
        scalar_df[scalar_df["condition_id"].isin(scalar_conditions)][["variant", "variant_group"]]
        .drop_duplicates()
        .sort_values(["variant_group", "variant"])
        ["variant"]
        .tolist()
    )
    raw_order = (
        raw_df[raw_df["condition_id"].isin(raw_conditions)][["variant", "variant_group"]]
        .drop_duplicates()
        .sort_values(["variant_group", "variant"])
        ["variant"]
        .tolist()
    )
    if not scalar_order and not raw_order:
        return

    fig_w = max(11.5, 0.29 * max(len(scalar_conditions), len(raw_conditions)) + 6.8)
    fig_h = max(8.2, 0.24 * (len(scalar_order) + len(raw_order)) + 3.0)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(fig_w, fig_h),
        gridspec_kw={
            "height_ratios": [max(len(scalar_order), 1), max(len(raw_order), 1)],
            "hspace": 0.55,
        },
    )
    im0 = _draw_rank_heatmap_panel(
        axes[0],
        scalar_df[scalar_df["condition_id"].isin(scalar_conditions)],
        summary,
        condition_order=scalar_conditions,
        variant_order=scalar_order,
        title="Benchmark robustness: scalar recombination variants",
    )
    im1 = _draw_rank_heatmap_panel(
        axes[1],
        raw_df[raw_df["condition_id"].isin(raw_conditions)],
        summary,
        condition_order=raw_conditions,
        variant_order=raw_order,
        title="Benchmark robustness: raw hyperparameter recomputation variants",
    )
    for ax, im in zip(axes, [im0, im1]):
        if im is not None:
            fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01, label="Rank (1 = best)")
    fig.suptitle("Rank stability under metric variants", fontsize=13, weight="bold", y=0.995)
    _save_fig(fig, "supp_metric_robustness_rank_heatmap")


def plot_main_summary(summary: pd.DataFrame, dirichlet: pd.DataFrame) -> None:
    if summary.empty:
        return
    groups = ["family_weights", "submetric_weights", "distribution_divergence", "topology", "state_hyperparameters"]
    sub = summary[summary["variant_group"].isin(groups)].copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    group_summary = sub.groupby("variant_group", as_index=False).agg(
        mean_rho=("spearman_to_default", "mean"),
        min_rho=("spearman_to_default", "min"),
        max_delta=("max_abs_score_delta", "max"),
        n_variants=("variant", "nunique"),
    )
    y = np.arange(len(group_summary))
    ax.errorbar(
        group_summary["mean_rho"],
        y,
        xerr=[
            group_summary["mean_rho"] - group_summary["min_rho"],
            np.maximum(0, 1.0 - group_summary["mean_rho"]),
        ],
        fmt="o",
        color=COLORS["rank"],
        ecolor="#BFC7D5",
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(group_summary["variant_group"], fontsize=9)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Rank correlation with default")
    ax.set_title("Robustness by variant class")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    if not dirichlet.empty:
        top = dirichlet.sort_values("top_probability", ascending=False).head(12).iloc[::-1]
        ax.barh(top["condition_id"], top["top_probability"], color="#7EB77F")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Top-rank probability")
        ax.set_title("Dirichlet weight ensemble")
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", labelsize=7)
    else:
        ax.axis("off")
    fig.tight_layout()
    _save_fig(fig, "main_metric_robustness_summary")


def _save_fig(fig: plt.Figure, stem: str) -> None:
    svg = FIG_DIR / f"{stem}.svg"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_correlation_heatmaps(metric_corr: pd.DataFrame, family_corr: pd.DataFrame) -> None:
    if not metric_corr.empty:
        corr = metric_corr.fillna(0.0)
        labels = list(corr.columns)
        order = labels
        if len(labels) >= 3:
            dist = 1.0 - np.abs(corr.to_numpy(float))
            np.fill_diagonal(dist, 0.0)
            link = linkage(squareform(dist, checks=False), method="average")
            leaves = dendrogram(link, no_plot=True)["leaves"]
            order = [labels[i] for i in leaves]
        mat = corr.loc[order, order]
        fig, ax = plt.subplots(figsize=(max(8.0, 0.28 * len(order)), max(7.0, 0.25 * len(order))))
        im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(order)))
        ax.set_yticks(np.arange(len(order)))
        ax.set_xticklabels(order, rotation=65, ha="right", fontsize=6)
        ax.set_yticklabels(order, fontsize=6)
        ax.set_title("Metric correlation matrix across available evaluations", loc="left", fontsize=12, weight="bold")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="Spearman rho")
        _save_fig(fig, "supp_metric_correlation_heatmap")

    if not family_corr.empty:
        labels = list(family_corr.columns)
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        im = ax.imshow(family_corr.to_numpy(float), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels([FAMILY_LABELS.get(x, x) for x in labels], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([FAMILY_LABELS.get(x, x) for x in labels], fontsize=8)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = _as_float(family_corr.iloc[i, j])
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title("Family and fidelity correlations", loc="left", fontsize=12, weight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman rho")
        _save_fig(fig, "supp_family_correlation_heatmap")


def plot_redundancy_cluster(metric_corr: pd.DataFrame, redundancy: pd.DataFrame) -> None:
    if metric_corr.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    corr = metric_corr.fillna(0.0)
    labels = list(corr.columns)
    if len(labels) >= 3:
        dist = 1.0 - np.abs(corr.to_numpy(float))
        np.fill_diagonal(dist, 0.0)
        link = linkage(squareform(dist, checks=False), method="average")
        dendrogram(link, labels=labels, leaf_rotation=90, leaf_font_size=6, ax=axes[0], color_threshold=0.7)
        axes[0].set_title("Metric clustering by |Spearman rho|")
    else:
        axes[0].axis("off")

    reg = redundancy[redundancy["analysis"] == "family_redundancy_regression"].copy() if not redundancy.empty else pd.DataFrame()
    if not reg.empty:
        reg = reg.sort_values("cv_r2_mean")
        axes[1].barh([FAMILY_LABELS.get(x, x) for x in reg["target_family"]], reg["cv_r2_mean"], color="#8DB7E8")
        axes[1].set_xlim(min(-0.1, float(reg["cv_r2_mean"].min()) - 0.05), 1.0)
        axes[1].set_xlabel("CV R2 predicted from other families")
        axes[1].set_title("Family redundancy regression")
        axes[1].grid(axis="x", alpha=0.25)
    else:
        axes[1].axis("off")
    fig.tight_layout()
    _save_fig(fig, "supp_metric_redundancy_cluster")


def write_manifest(
    score_matrix: pd.DataFrame,
    variant_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    summary: pd.DataFrame,
    dirichlet: pd.DataFrame,
) -> None:
    manifest = {
        "generated_tables": [
            str(TABLE_DIR / "supp_metric_robustness_score_matrix.csv"),
            str(TABLE_DIR / "supp_metric_robustness_variants_long.csv"),
            str(TABLE_DIR / "supp_metric_robustness_rank_summary.csv"),
            str(TABLE_DIR / "supp_metric_robustness_skipped_variants.csv"),
            str(TABLE_DIR / "supp_metric_dirichlet_weight_rank_summary.csv"),
            str(TABLE_DIR / "supp_metric_correlation_matrix.csv"),
            str(TABLE_DIR / "supp_family_correlation_matrix.csv"),
            str(TABLE_DIR / "supp_metric_redundancy_summary.csv"),
        ],
        "generated_figures": [
            str(FIG_DIR / "supp_metric_robustness_rank_heatmap.svg"),
            str(FIG_DIR / "supp_metric_robustness_rank_heatmap.pdf"),
            str(FIG_DIR / "supp_metric_correlation_heatmap.svg"),
            str(FIG_DIR / "supp_family_correlation_heatmap.svg"),
            str(FIG_DIR / "supp_metric_redundancy_cluster.svg"),
            str(FIG_DIR / "main_metric_robustness_summary.svg"),
        ],
        "source_tables": {
            "fig2_master": str(FIG2_MASTER),
            "scaling_wide": str(SCALING_WIDE),
            "model_family_scores": str(MODEL_FAMILY_SCORES),
            "metric_comparison_tables": [{"source": source, "path": str(path), "exists": path.exists()} for source, path in METRIC_COMPARISON_TABLES],
        },
        "default_family_weights": {k: float(v) for k, v in DEFAULT_FAMILY_WEIGHTS.items()},
        "n_score_rows": int(score_matrix.shape[0]),
        "n_variant_rows": int(variant_df.shape[0]),
        "n_ranked_variants": int(summary["variant"].nunique()) if not summary.empty else 0,
        "n_dirichlet_conditions": int(dirichlet.shape[0]) if not dirichlet.empty else 0,
        "skipped_variants": skipped_df.to_dict("records") if not skipped_df.empty else [],
        "interpretation_note": (
            "Family-weight and submetric-weight variants are exact recombinations of current scalar scores. "
            "Distribution-divergence, PCA-dimensionality, state-K, and transition-lag variants are recomputed "
            "from available raw GT/Pred pairs and merged with the scalar robustness table. Raw variants use a "
            f"bounded evaluation window of at most {RAW_MAX_SEQUENCES} sequences and {RAW_MAX_TIMESTEPS} timepoints "
            "per condition to keep the robustness sweep tractable."
        ),
    }
    (TABLE_DIR / "supp_metric_robustness_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> int:
    _ensure_dirs()
    score_matrix = build_score_matrix()
    if score_matrix.empty:
        raise RuntimeError("No usable score tables found for robustness analysis.")
    variant_df, summary, skipped_df = build_robustness_variants(score_matrix)
    dirichlet = build_dirichlet_weight_summary(score_matrix)
    metric_corr, family_corr, redundancy = build_correlation_and_redundancy(score_matrix, variant_df)
    plot_robustness_heatmap(variant_df, summary)
    plot_main_summary(summary, dirichlet)
    plot_correlation_heatmaps(metric_corr, family_corr)
    plot_redundancy_cluster(metric_corr, redundancy)
    write_manifest(score_matrix, variant_df, skipped_df, summary, dirichlet)

    print(f"Wrote robustness score matrix: {TABLE_DIR / 'supp_metric_robustness_score_matrix.csv'}")
    print(f"Wrote robustness variants: {TABLE_DIR / 'supp_metric_robustness_variants_long.csv'}")
    print(f"Wrote robustness figures to: {FIG_DIR}")
    if not skipped_df.empty:
        print("Skipped scalar-only variants requiring raw recomputation:")
        print(skipped_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
