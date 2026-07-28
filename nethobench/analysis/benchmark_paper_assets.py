from __future__ import annotations

import argparse
import json
import math
import textwrap
import warnings
import io
import contextlib
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.stats import entropy, kurtosis, skew
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from nethobench.analysis.additional_neuro_metrics import (
    _align_arrays,
    _lagged_covariance,
    _mi_matrix,
    _pooled_rows,
    _standardize_columns,
    _var1_coefficients,
    compute_additional_structural_metrics,
)
from nethobench.analysis.direct_neuro_metrics import (
    compute_graph_score01,
    compute_manifold_score01,
    compute_moment_score01,
    compute_trajectory_score01,
)
from nethobench.analysis.score_definitions import (
    FIDELITY_METRICS,
    NEURO_FAMILY_METRICS,
    NEURO_FAMILY_WEIGHTS,
    compute_neuro_composite,
    compute_neuro_family_scores,
)
from nethobench.neuro import _load_sequences


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "nethobench" / "notebooks" / "neuro_metrics.ipynb"
DEFAULT_OUTPUT_ROOT = Path.home() / "Desktop" / "benchmark_paper"
REFERENCE_STYLE_PATHS = [
    Path.home() / "Downloads" / "fig_2_neurips.svg",
    Path.home() / "Downloads" / "fig_3_neurips.svg",
]
DEFAULT_RESULT_ROOTS = OrderedDict(
    [
        ("ar1_ceiling", Path.home() / "Desktop" / "nethobench-sequifier-convergence"),
        ("biophysical_ceiling", Path.home() / "Desktop" / "nethobench-sequifier-convergence-biophysical"),
        ("calciumgan_transfer", Path.home() / "Desktop" / "nethobench-calciumgan-biophysical"),
    ]
)
OUTPUT_SUBDIRS = ("figures", "tables", "text")
PLOT_DPI = 200


PALETTE = {
    "ink": "#1F2430",
    "muted": "#6B7280",
    "grid": "#D9DEE7",
    "gt": "#1F5AA6",
    "pred": "#D55E00",
    "oracle": "#2E8B57",
    "fill_blue": "#D9E8FB",
    "fill_teal": "#D8F1EF",
    "fill_orange": "#FBE7D3",
    "fill_rose": "#F9E0E5",
    "fill_gold": "#F5E8C8",
    "distribution": "#4C78A8",
    "temporal_spectral": "#54A24B",
    "relational": "#E45756",
    "geometry": "#72B7B2",
    "state_dynamics": "#9D755D",
    "fidelity": "#B279A2",
}


METRIC_DETAILS = {
    "KL_or_JSD_score01": {
        "family": "distribution",
        "compare": "Per-sequence, per-region marginal value histograms between GT and prediction.",
        "aggregation": "Symmetric KL per region, geometric mean similarity per sequence, then average of sequence mean and sequence q10.",
        "direction": "Higher is better; 1 means matching marginals.",
        "sensitivities": ["mean_shift", "variance_scale", "additive_noise", "temporal_shuffle"],
        "formula": r"$\mathrm{sim}_{s,r}=1/(1+\mathrm{KL}_{sym}(p_{s,r},q_{s,r}));\ \mathrm{score}=0.5(\mathrm{mean}_s+\mathrm{q10}_s)$",
    },
    "QNT_score01": {
        "family": "distribution",
        "compare": "Tail quantiles of each region within each sequence.",
        "aggregation": "Mean tail-quantile distance per region, top-25% worst regions per sequence, then mean across sequences.",
        "direction": "Higher is better; 1 means matching tail quantiles.",
        "sensitivities": ["variance_scale", "additive_noise", "mean_shift"],
        "formula": r"$D_i=\mathrm{topqmean}_r(d^{tail}_{i,r});\ \mathrm{QNT}=1/(1+\mathrm{mean}_i D_i)$",
    },
    "MOM_score01": {
        "family": "distribution",
        "compare": "Variance, skewness, and kurtosis structure of GT and prediction.",
        "aggregation": "Direct moment agreement score from the benchmark moment replacement.",
        "direction": "Higher is better; 1 means matching moment structure.",
        "sensitivities": ["variance_scale", "additive_noise", "latent_rotation"],
        "formula": r"Composite agreement over variance, skewness, and kurtosis.",
    },
    "Mean_score01": {
        "family": "distribution",
        "compare": "Region-wise mean shifts between GT and prediction.",
        "aggregation": "Normalize mean differences by GT IQR, average top-10% worst regions per sequence, then mean across sequences.",
        "direction": "Higher is better; 1 means matching mean structure.",
        "sensitivities": ["mean_shift", "variance_scale", "additive_noise"],
        "formula": r"$D_i^{top10}=\mathrm{mean}$ of top-$10\%$ normalized region mean shifts; $\mathrm{score}=1/(1+\mathrm{mean}_i D_i^{top10})$",
    },
    "TRJDIST_score01": {
        "family": "temporal_spectral",
        "compare": "Trajectory occupancy, velocity, and path geometry in a shared GT-defined latent space.",
        "aggregation": "Direct trajectory distribution score from pooled GT PCA occupancy and path-feature agreement.",
        "direction": "Higher is better; 1 means matching latent trajectory distributions.",
        "sensitivities": ["temporal_shuffle", "latent_rotation", "oversmoothing", "additive_noise"],
        "formula": r"Composite agreement over latent occupancy, speed/turning statistics, and path geometry.",
    },
    "GRAPH_score01": {
        "family": "relational",
        "compare": "Thresholded cross-region graph structure derived from GT and prediction.",
        "aggregation": "Composite agreement over top-edge topology, weighted degree, and clustering.",
        "direction": "Higher is better; 1 means matching graph structure.",
        "sensitivities": ["region_mix", "region_permutation", "latent_rotation"],
        "formula": r"Composite graph agreement over edge set, weighted degree, and clustering.",
    },
    "CrossRegionMI_score01": {
        "family": "relational",
        "compare": "Pairwise mutual-information matrix across regions.",
        "aggregation": "Upper-triangular matrix similarity between GT and prediction MI matrices.",
        "direction": "Higher is better; 1 means matching nonlinear inter-region dependence.",
        "sensitivities": ["additive_noise", "temporal_shuffle", "region_mix"],
        "formula": r"$\mathrm{score}=\mathrm{sim}(\mathrm{MI}_{GT},\mathrm{MI}_{Pred})$",
    },
    "LaggedCovariance_score01": {
        "family": "relational",
        "compare": "Lagged covariance matrices at lags 1, 2, and 4.",
        "aggregation": "Matrix similarity at each lag, then average across lags.",
        "direction": "Higher is better; 1 means matching lagged covariance structure.",
        "sensitivities": ["temporal_shuffle", "oversmoothing", "region_mix"],
        "formula": r"$\mathrm{score}=\mathrm{mean}_{\ell\in\{1,2,4\}}\mathrm{sim}(C^{(\ell)}_{GT},C^{(\ell)}_{Pred})$",
    },
    "ImpulseResponse_score01": {
        "family": "relational",
        "compare": "VAR(1) coefficient matrix approximating cross-region impulse responses.",
        "aggregation": "Upper-triangular matrix similarity between GT and prediction VAR coefficient matrices.",
        "direction": "Higher is better; 1 means matching first-order linear dynamics.",
        "sensitivities": ["temporal_shuffle", "oversmoothing", "latent_rotation"],
        "formula": r"$\mathrm{score}=\mathrm{sim}(A_{GT},A_{Pred})$ for ridge-regularized VAR(1) coefficients",
    },
    "MANI_score01": {
        "family": "geometry",
        "compare": "Manifold topology and local neighborhood geometry in GT-defined latent space.",
        "aggregation": "Direct manifold score combining persistent homology lifetime agreement with local geometry.",
        "direction": "Higher is better; 1 means matching latent manifold structure.",
        "sensitivities": ["latent_rotation", "region_mix", "additive_noise"],
        "formula": r"Composite topology and local-neighborhood geometry agreement.",
    },
    "SubspaceAngle_score01": {
        "family": "geometry",
        "compare": "Principal covariance subspaces of GT and prediction.",
        "aggregation": "Cosine-squared agreement of leading GT and prediction covariance subspaces.",
        "direction": "Higher is better; 1 means aligned dominant subspaces.",
        "sensitivities": ["latent_rotation", "region_permutation", "region_mix"],
        "formula": r"$\mathrm{score}=\mathrm{mean}(\cos^2 \theta_k)$ over principal subspace angles",
    },
    "LatentStateOccupancyK11_score01": {
        "family": "state_dynamics",
        "compare": "Occupancy of GT-defined latent states with $K=11$.",
        "aggregation": "Similarity of state-frequency vectors in a shared GT state partition.",
        "direction": "Higher is better; 1 means matching state occupancy.",
        "sensitivities": ["temporal_shuffle", "latent_rotation", "region_mix"],
        "formula": r"$\mathrm{score}=\mathrm{sim}(\pi^{K=11}_{GT},\pi^{K=11}_{Pred})$",
    },
    "LatentStateOccupancyK12_score01": {
        "family": "state_dynamics",
        "compare": "Occupancy of GT-defined latent states with $K=12$.",
        "aggregation": "Similarity of state-frequency vectors in a shared GT state partition.",
        "direction": "Higher is better; 1 means matching state occupancy.",
        "sensitivities": ["temporal_shuffle", "latent_rotation", "region_mix"],
        "formula": r"$\mathrm{score}=\mathrm{sim}(\pi^{K=12}_{GT},\pi^{K=12}_{Pred})$",
    },
    "LatentStateTransitionLag1K11_score01": {
        "family": "state_dynamics",
        "compare": "Lag-1 transition matrix in GT-defined latent states ($K=11$).",
        "aggregation": "Similarity of GT and prediction transition matrices.",
        "direction": "Higher is better; 1 means matching short-lag state dynamics.",
        "sensitivities": ["temporal_shuffle", "oversmoothing", "additive_noise"],
        "formula": r"$\mathrm{score}=\mathrm{sim}(T^{(\ell=1,K=11)}_{GT},T^{(\ell=1,K=11)}_{Pred})$",
    },
    "LatentStateTransitionLag2K11_score01": {
        "family": "state_dynamics",
        "compare": "Lag-2 transition matrix in GT-defined latent states ($K=11$).",
        "aggregation": "Similarity of GT and prediction transition matrices.",
        "direction": "Higher is better; 1 means matching medium-lag state dynamics.",
        "sensitivities": ["temporal_shuffle", "oversmoothing", "additive_noise"],
        "formula": r"$\mathrm{score}=\mathrm{sim}(T^{(\ell=2,K=11)}_{GT},T^{(\ell=2,K=11)}_{Pred})$",
    },
    "LatentStateTransitionLag3K11_score01": {
        "family": "state_dynamics",
        "compare": "Lag-3 transition matrix in GT-defined latent states ($K=11$).",
        "aggregation": "Similarity of GT and prediction transition matrices.",
        "direction": "Higher is better; 1 means matching longer-lag state dynamics.",
        "sensitivities": ["temporal_shuffle", "oversmoothing", "additive_noise"],
        "formula": r"$\mathrm{score}=\mathrm{sim}(T^{(\ell=3,K=11)}_{GT},T^{(\ell=3,K=11)}_{Pred})$",
    },
    "Error_score01": {
        "family": "fidelity",
        "compare": "Pointwise signal fidelity between GT and prediction.",
        "aggregation": "Fidelity benchmark score, auxiliary to the final neuro composite.",
        "direction": "Higher is better.",
        "sensitivities": ["additive_noise", "mean_shift", "variance_scale"],
        "formula": r"Auxiliary fidelity score from the companion fidelity benchmark.",
    },
    "MI_score01": {
        "family": "fidelity",
        "compare": "Mutual-information-based fidelity benchmark scalar.",
        "aggregation": "Fidelity benchmark score, auxiliary to the final neuro composite.",
        "direction": "Higher is better.",
        "sensitivities": ["additive_noise", "temporal_shuffle"],
        "formula": r"Auxiliary nonlinear fidelity score from the companion fidelity benchmark.",
    },
}


EXPERIMENT_GROUPS = OrderedDict(
    [
        (
            "ar1_ceiling",
            {
                "title": "AR-1 world ceiling and convergence",
                "condition_labels": {
                    "weakest": "Weakest",
                    "underfit": "Underfit",
                    "converged": "Converged",
                },
            },
        ),
        (
            "biophysical_ceiling",
            {
                "title": "Biophysical world ceiling",
                "condition_labels": {
                    "weakest": "Weakest",
                    "converged": "Converged",
                },
            },
        ),
        (
            "baseline_models",
            {
                "title": "CalciumGAN vs DG baseline",
                "condition_labels": {
                    "calciumgan": "CalciumGAN",
                    "dg_baseline": "DG baseline",
                },
            },
        ),
        (
            "transfer",
            {
                "title": "Transfer into benchmark worlds",
                "condition_labels": {
                    "transfer_calciumgan_weakest": "CalciumGAN→Weakest",
                    "transfer_calciumgan_converged": "CalciumGAN→Converged",
                    "transfer_dg_baseline_weakest": "DG→Weakest",
                    "transfer_dg_baseline_converged": "DG→Converged",
                },
            },
        ),
    ]
)


@dataclass
class PaperConfig:
    output_root: Path
    result_roots: OrderedDict[str, Path]
    notebook_path: Path
    reference_style_paths: list[Path]
    smoke: bool = False


def _load_config(config_path: Path | None = None, output_root: Path | None = None, smoke: bool = False) -> PaperConfig:
    cfg = {
        "output_root": str(output_root or DEFAULT_OUTPUT_ROOT),
        "result_roots": {k: str(v) for k, v in DEFAULT_RESULT_ROOTS.items()},
        "notebook_path": str(NOTEBOOK_PATH),
        "reference_style_paths": [str(p) for p in REFERENCE_STYLE_PATHS],
        "smoke": bool(smoke),
    }
    if config_path is not None:
        user_cfg = json.loads(Path(config_path).read_text())
        cfg.update(user_cfg)
    return PaperConfig(
        output_root=Path(cfg["output_root"]).expanduser(),
        result_roots=OrderedDict((k, Path(v).expanduser()) for k, v in cfg["result_roots"].items()),
        notebook_path=Path(cfg["notebook_path"]).expanduser(),
        reference_style_paths=[Path(p).expanduser() for p in cfg["reference_style_paths"]],
        smoke=bool(cfg.get("smoke", False)),
    )


def _sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_output_dirs(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    out = {"root": root}
    for name in OUTPUT_SUBDIRS:
        sub = root / name
        sub.mkdir(parents=True, exist_ok=True)
        out[name] = sub
    return out


def _validate_notebook(nb, notebook_path: Path) -> None:
    markers = [
        "# === Distribution realism (KL only): baseline + corruption degradation ===",
        "# === Mean difference metric (Top-10% regions) ===",
        "# === Quantile / tail realism (simple, strict, benchmark-friendly) ===",
        "# === TRAJECTORY DISTRIBUTION realism (FIXED: global GT-PCA basis, pooled across sequences) ===",
        "# === ADDITIONAL STRUCTURAL METRICS (cross-region information, lagged dynamics, subspaces, states) ===",
        "# === FINAL NEURO SCORE COMPOSITE ===",
        "# === UNIFIED CORRUPTION SENSITIVITY DASHBOARD (active neuro score metrics) ===",
    ]
    joined = "\n".join(cell.get("source", "") for cell in nb.cells)
    missing = [marker for marker in markers if marker not in joined]
    if missing:
        raise RuntimeError(f"Notebook markers missing from {notebook_path}: {missing}")


def _find_cell_source(nb, marker: str) -> str:
    for cell in nb.cells:
        source = cell.get("source", "")
        if marker in source:
            return source
    raise KeyError(f"Could not find notebook cell containing marker: {marker}")


def _slice_before(source: str, marker: str) -> str:
    if marker in source:
        return source.split(marker, 1)[0]
    return source


def _load_notebook_score_helpers(notebook_path: Path) -> dict[str, object]:
    nb = nbformat.read(notebook_path, as_version=4)
    _validate_notebook(nb, notebook_path)
    namespace: dict[str, object] = {}
    for marker, cutoff in [
        ("# === Distribution realism (KL only): baseline + corruption degradation ===", "# Align once and keep names compatible with previous sections"),
        ("# === Mean difference metric (Top-10% regions) ===", "# Baseline"),
        ("# === Quantile / tail realism (simple, strict, benchmark-friendly) ===", "# ----------------------------\n# 3) Run"),
    ]:
        source = _find_cell_source(nb, marker)
        source = _slice_before(source, cutoff)
        exec(compile(source, str(notebook_path), "exec"), namespace)
    needed = ["_compute_kl_metrics", "compute_mean_score01_top10", "compute_quantile_score01_simple"]
    missing = [name for name in needed if name not in namespace]
    if missing:
        raise RuntimeError(f"Failed to load notebook helper functions: {missing}")
    return namespace


def _flatten_metric_names() -> list[str]:
    return [metric for family_metrics in NEURO_FAMILY_METRICS.values() for metric in family_metrics]


def _style_axis(ax, *, grid: bool = True) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(PALETTE["grid"])
    ax.spines["bottom"].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=9)
    if grid:
        ax.grid(True, color=PALETTE["grid"], linewidth=0.7, alpha=0.6)
        ax.set_axisbelow(True)


def _save_svg(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    ET.parse(path)


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(df.to_json(orient="records", indent=2))
    else:
        df.to_csv(path, index=False)


def _bootstrap_ci(values: np.ndarray, seed: int = 0, n_boot: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        boots[i] = np.mean(sample)
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _aggregate_condition_scores(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        values = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        key_values = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        ci_low, ci_high = _bootstrap_ci(values, seed=17 + values.size) if values.size >= 5 else (np.nan, np.nan)
        rows.append(
            {
                **key_values,
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "se": float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else np.nan,
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "skew_abs": float(abs(skew(values))) if values.size > 2 else 0.0,
                "use_robust_summary": bool(values.size > 2 and abs(skew(values)) > 1.0),
            }
        )
    return pd.DataFrame(rows)


def _read_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_experiment_tables(config: PaperConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    metric_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    missing: list[str] = []

    for group_name, root in config.result_roots.items():
        if not root.exists():
            missing.append(f"Missing experiment root: {root}")
            continue

        if group_name in {"ar1_ceiling", "biophysical_ceiling"}:
            metric_path = root / "results" / "all_metric_comparisons.csv"
            family_path = root / "results" / "all_family_comparisons.csv"
            if metric_path.exists():
                df = pd.read_csv(metric_path)
                df["experiment_group"] = group_name
                df["source_path"] = str(metric_path)
                metric_rows.extend(df.to_dict("records"))
            else:
                missing.append(f"Missing summary file: {metric_path}")
            if family_path.exists():
                df = pd.read_csv(family_path)
                df["experiment_group"] = group_name
                df["source_path"] = str(family_path)
                family_rows.extend(df.to_dict("records"))
            else:
                missing.append(f"Missing summary file: {family_path}")

            result_dir = root / "results"
            for run_dir in sorted(result_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                summary_path = run_dir / "summary.json"
                if not summary_path.exists():
                    continue
                data = _read_summary(summary_path)
                run_name = str(data.get("run_name", run_dir.name))
                model_scores = data.get("model_scores", {})
                oracle_scores = data.get("oracle_scores", {})
                fidelity_scores = data.get("model_fidelity_scores", {})
                run_rows.append(
                    {
                        "experiment_group": group_name,
                        "run_name": run_name,
                        "condition_label": EXPERIMENT_GROUPS[group_name]["condition_labels"].get(run_name, run_name),
                        "source_path": str(summary_path),
                        "final_composite": float(model_scores.get("FINAL_COMPOSITE_SCORE", np.nan)),
                        "oracle_final_composite": float(oracle_scores.get("FINAL_COMPOSITE_SCORE", np.nan)),
                        "composite_ratio_to_oracle": float(
                            model_scores.get("FINAL_COMPOSITE_SCORE", np.nan) / max(oracle_scores.get("FINAL_COMPOSITE_SCORE", np.nan), 1e-12)
                        )
                        if np.isfinite(model_scores.get("FINAL_COMPOSITE_SCORE", np.nan))
                        and np.isfinite(oracle_scores.get("FINAL_COMPOSITE_SCORE", np.nan))
                        else np.nan,
                        "fidelity_score": float(fidelity_scores.get("FIDELITY_SCORE", np.nan)),
                    }
                )

        if group_name == "calciumgan_transfer":
            result_dir = root / "results"
            base_runs = ["calciumgan", "dg_baseline"]
            transfer_runs = [
                "transfer_calciumgan_weakest",
                "transfer_calciumgan_converged",
                "transfer_dg_baseline_weakest",
                "transfer_dg_baseline_converged",
            ]
            for run_name in base_runs:
                run_dir = result_dir / run_name
                metric_path = run_dir / "metric_comparison.csv"
                family_path = run_dir / "family_comparison.csv"
                summary_path = run_dir / "summary.json"
                if metric_path.exists():
                    df = pd.read_csv(metric_path)
                    df["run_name"] = run_name
                    df["experiment_group"] = "baseline_models"
                    df["source_path"] = str(metric_path)
                    metric_rows.extend(df.to_dict("records"))
                else:
                    missing.append(f"Missing summary file: {metric_path}")
                if family_path.exists():
                    df = pd.read_csv(family_path)
                    df["run_name"] = run_name
                    df["experiment_group"] = "baseline_models"
                    df["source_path"] = str(family_path)
                    family_rows.extend(df.to_dict("records"))
                else:
                    missing.append(f"Missing summary file: {family_path}")
                if summary_path.exists():
                    data = _read_summary(summary_path)
                    model_scores = data.get("model_scores", {})
                    oracle_scores = data.get("oracle_scores", {})
                    fidelity_scores = data.get("model_fidelity_scores", {})
                    run_rows.append(
                        {
                            "experiment_group": "baseline_models",
                            "run_name": run_name,
                            "condition_label": EXPERIMENT_GROUPS["baseline_models"]["condition_labels"].get(run_name, run_name),
                            "source_path": str(summary_path),
                            "final_composite": float(model_scores.get("FINAL_COMPOSITE_SCORE", np.nan)),
                            "oracle_final_composite": float(oracle_scores.get("FINAL_COMPOSITE_SCORE", np.nan)),
                            "composite_ratio_to_oracle": float(
                                model_scores.get("FINAL_COMPOSITE_SCORE", np.nan) / max(oracle_scores.get("FINAL_COMPOSITE_SCORE", np.nan), 1e-12)
                            )
                            if np.isfinite(model_scores.get("FINAL_COMPOSITE_SCORE", np.nan))
                            and np.isfinite(oracle_scores.get("FINAL_COMPOSITE_SCORE", np.nan))
                            else np.nan,
                            "fidelity_score": float(fidelity_scores.get("FIDELITY_SCORE", np.nan)),
                        }
                    )
                else:
                    missing.append(f"Missing summary file: {summary_path}")

            for run_name in transfer_runs:
                run_dir = result_dir / run_name
                metric_path = run_dir / "metric_comparison_rollout.csv"
                family_path = run_dir / "family_comparison_rollout.csv"
                summary_path = run_dir / "summary_rollout.json"
                if metric_path.exists():
                    df = pd.read_csv(metric_path)
                    df["run_name"] = run_name
                    df["experiment_group"] = "transfer"
                    df["source_path"] = str(metric_path)
                    metric_rows.extend(df.to_dict("records"))
                else:
                    missing.append(f"Missing summary file: {metric_path}")
                if family_path.exists():
                    df = pd.read_csv(family_path)
                    df["run_name"] = run_name
                    df["experiment_group"] = "transfer"
                    df["source_path"] = str(family_path)
                    family_rows.extend(df.to_dict("records"))
                else:
                    missing.append(f"Missing summary file: {family_path}")
                if summary_path.exists():
                    data = _read_summary(summary_path)
                    model_scores = data.get("model_scores", {})
                    oracle_scores = data.get("oracle_scores", {})
                    fidelity_scores = data.get("model_fidelity_scores", {})
                    run_rows.append(
                        {
                            "experiment_group": "transfer",
                            "run_name": run_name,
                            "condition_label": EXPERIMENT_GROUPS["transfer"]["condition_labels"].get(run_name, run_name),
                            "source_path": str(summary_path),
                            "final_composite": float(model_scores.get("FINAL_COMPOSITE_SCORE", np.nan)),
                            "oracle_final_composite": float(oracle_scores.get("FINAL_COMPOSITE_SCORE", np.nan)),
                            "composite_ratio_to_oracle": float(
                                model_scores.get("FINAL_COMPOSITE_SCORE", np.nan) / max(oracle_scores.get("FINAL_COMPOSITE_SCORE", np.nan), 1e-12)
                            )
                            if np.isfinite(model_scores.get("FINAL_COMPOSITE_SCORE", np.nan))
                            and np.isfinite(oracle_scores.get("FINAL_COMPOSITE_SCORE", np.nan))
                            else np.nan,
                            "fidelity_score": float(fidelity_scores.get("FIDELITY_SCORE", np.nan)),
                        }
                    )
                else:
                    missing.append(f"Missing summary file: {summary_path}")

    metrics_df = pd.DataFrame(metric_rows)
    families_df = pd.DataFrame(family_rows)
    run_df = pd.DataFrame(run_rows)
    for df in (metrics_df, families_df):
        if not df.empty:
            df["condition_label"] = [
                EXPERIMENT_GROUPS.get(group, {}).get("condition_labels", {}).get(run_name, run_name)
                for group, run_name in zip(df["experiment_group"], df["run_name"])
            ]
    return metrics_df, families_df, run_df, missing


def _pick_reference_summary(config: PaperConfig) -> dict[str, object]:
    preferred = [
        config.result_roots.get("biophysical_ceiling") / "results" / "converged" / "summary.json",
        config.result_roots.get("ar1_ceiling") / "results" / "converged" / "summary.json",
    ]
    for path in preferred:
        if path is not None and path.exists():
            data = _read_summary(path)
            return {
                "summary_path": path,
                "summary": data,
                "gt_path": Path(data["aligned_ground_truth_path"]),
                "pred_path": Path(data["aligned_predictions_path"]),
                "run_name": str(data.get("run_name", "reference")),
                "experiment_group": "biophysical_ceiling" if "biophysical" in str(path) else "ar1_ceiling",
            }
    raise FileNotFoundError("Could not find a converged reference summary in the configured Desktop result roots.")


def _load_aligned_arrays(gt_path: Path, pred_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    gt_arr, gt_regions = _load_sequences(gt_path)
    pred_arr, pred_regions = _load_sequences(pred_path)
    shared = [name for name in gt_regions if name in pred_regions]
    if not shared:
        raise ValueError(f"No overlapping regions between {gt_path} and {pred_path}")
    gt_idx = [gt_regions.index(name) for name in shared]
    pred_idx = [pred_regions.index(name) for name in shared]
    gt = np.asarray(gt_arr[:, :, gt_idx], dtype=np.float64)
    pred = np.asarray(pred_arr[:, :, pred_idx], dtype=np.float64)
    keep = min(gt.shape[1], pred.shape[1])
    return gt[:, :keep, :], pred[:, :keep, :], shared


def _subsample_reference(gt_arr: np.ndarray, pred_arr: np.ndarray, *, max_seq: int = 72, max_time: int = 256) -> tuple[np.ndarray, np.ndarray]:
    gt_arr, pred_arr = _align_arrays(gt_arr, pred_arr)
    if gt_arr.shape[0] > max_seq:
        seq_idx = np.linspace(0, gt_arr.shape[0] - 1, max_seq, dtype=int)
        gt_arr = gt_arr[seq_idx]
        pred_arr = pred_arr[seq_idx]
    if gt_arr.shape[1] > max_time:
        time_idx = np.linspace(0, gt_arr.shape[1] - 1, max_time, dtype=int)
        gt_arr = gt_arr[:, time_idx, :]
        pred_arr = pred_arr[:, time_idx, :]
    return gt_arr, pred_arr


def _family_score_functions(helpers: dict[str, object]) -> Callable[[np.ndarray, np.ndarray], dict[str, float]]:
    kl_fn = helpers["_compute_kl_metrics"]
    mean_fn = helpers["compute_mean_score01_top10"]
    qnt_fn = helpers["compute_quantile_score01_simple"]

    def _score_bundle(gt_arr: np.ndarray, pred_arr: np.ndarray) -> dict[str, float]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scores: dict[str, float] = {}
            def _safe(callable_):
                try:
                    return callable_()
                except Exception:
                    return np.nan

            scores["KL_or_JSD_score01"] = _safe(
                lambda: float(kl_fn(gt_arr, pred_arr, bins=60, support_q=(0.001, 0.999)).get("KL_score01_avg", np.nan))
            )
            scores["Mean_score01"] = _safe(lambda: float(mean_fn(gt_arr, pred_arr).get("Mean_score01", np.nan)))

            def _quiet_qnt():
                sink = io.StringIO()
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    return qnt_fn(gt_arr, pred_arr)

            qnt_out = _safe(_quiet_qnt)
            scores["QNT_score01"] = (
                float(qnt_out.get("scores", {}).get("QNT_score01", np.nan))
                if isinstance(qnt_out, dict)
                else np.nan
            )
            scores["MOM_score01"] = _safe(lambda: float(compute_moment_score01(gt_arr, pred_arr).get("scores", {}).get("MOM_score01", np.nan)))
            scores["GRAPH_score01"] = _safe(lambda: float(compute_graph_score01(gt_arr, pred_arr).get("scores", {}).get("GRAPH_score01", np.nan)))
            scores["MANI_score01"] = _safe(lambda: float(compute_manifold_score01(gt_arr, pred_arr).get("scores", {}).get("MANI_score01", np.nan)))
            scores["TRJDIST_score01"] = _safe(lambda: float(compute_trajectory_score01(gt_arr, pred_arr).get("scores", {}).get("TRJDIST_score01", np.nan)))
            extra = _safe(lambda: compute_additional_structural_metrics(gt_arr, pred_arr).get("scores", {}))
            extra = extra if isinstance(extra, dict) else {}
            for key in [
                "CrossRegionMI_score01",
                "SubspaceAngle_score01",
                "LaggedCovariance_score01",
                "ImpulseResponse_score01",
                "LatentStateOccupancyK11_score01",
                "LatentStateOccupancyK12_score01",
                "LatentStateTransitionLag1K11_score01",
                "LatentStateTransitionLag2K11_score01",
                "LatentStateTransitionLag3K11_score01",
            ]:
                scores[key] = float(extra.get(key, np.nan))
            return scores

    return _score_bundle


def _global_iqr(arr: np.ndarray) -> float:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    q25, q75 = np.quantile(flat, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.nanstd(flat))
    return scale if np.isfinite(scale) and scale > 1e-12 else 1.0


SPIKE_LEVEL_CFG = {
    1: (0.01, 0.50),
    2: (0.02, 0.75),
    3: (0.04, 1.00),
    4: (0.06, 1.25),
    5: (0.08, 1.50),
}


def _corruption_registry(smoke: bool = False) -> tuple[list[dict[str, object]], list[str], dict[str, str]]:
    def mean_shift_region_iqr(pred_arr, level, rng, scale):
        return np.asarray(pred_arr, dtype=np.float64) + float(level) * scale["region_iqr"][None, None, :]

    def global_mean_shift(pred_arr, level, rng, scale):
        return np.asarray(pred_arr, dtype=np.float64) + float(level) * float(scale["global_iqr"])

    def region_mean_shift(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        pattern = np.linspace(-1.0, 1.0, out.shape[2], dtype=np.float64)
        out += float(level) * scale["region_iqr"][None, None, :] * pattern[None, None, :]
        return out

    def sequence_mean_shift(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        offsets = rng.normal(loc=0.0, scale=float(level) * float(scale["global_iqr"]), size=(out.shape[0], 1, 1))
        out += offsets
        return out

    def slow_drift(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        t = np.linspace(0.0, 1.0, out.shape[1], dtype=np.float64).reshape(1, out.shape[1], 1)
        out += float(level) * float(scale["global_iqr"]) * t
        return out

    def variance_scale_kl(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        center = np.nanmedian(out, axis=1, keepdims=True)
        return center + (1.0 + float(level)) * (out - center)

    def variance_scale_qnt(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        center = np.nanmedian(out, axis=1, keepdims=True)
        return center + float(level) * (out - center)

    def tail_spikes_kl(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        m = float(level)
        spike_prob = min(0.02 + 0.08 * m, 0.35)
        mask = rng.random(out.shape) < spike_prob
        spikes = rng.laplace(loc=0.0, scale=m, size=out.shape) * scale["region_iqr"][None, None, :]
        out += mask * spikes
        return out

    def tail_spikes_qnt(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        p_spike, mag = SPIKE_LEVEL_CFG[int(round(float(level)))]
        mask = rng.random(out.shape) < p_spike
        spikes = rng.laplace(loc=0.0, scale=1.0, size=out.shape) * (mag * scale["region_iqr"][None, None, :])
        out += mask * spikes
        return out

    def tail_spikes_mom(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        p_spike, mag = SPIKE_LEVEL_CFG[int(round(float(level)))]
        mask = rng.random(out.shape) < p_spike
        heavy = rng.standard_t(df=3, size=out.shape)
        out += mask * (heavy * (mag * scale["region_iqr"][None, None, :]))
        return out

    def one_sided_spikes(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        p_spike, mag = SPIKE_LEVEL_CFG[int(round(float(level)))]
        mask = rng.random(out.shape) < p_spike
        amp = np.abs(rng.standard_t(df=3, size=out.shape))
        out += mask * (amp * (mag * scale["region_iqr"][None, None, :]))
        return out

    def additive_noise_iqr(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        noise = rng.normal(0.0, 1.0, size=out.shape)
        out += noise * (float(level) * scale["region_iqr"][None, None, :])
        return out

    def temporal_shuffle(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        frac = min(max(float(level), 0.0), 1.0)
        n_time = out.shape[1]
        n_swap = max(1, int(round(frac * n_time)))
        for seq in range(out.shape[0]):
            idx = rng.choice(n_time, size=n_swap, replace=False)
            shuffled = idx.copy()
            rng.shuffle(shuffled)
            out[seq, idx, :] = out[seq, shuffled, :]
        return out

    def temporal_shuffle_per_region(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        n_seq, T, n_reg = out.shape
        m = int(np.floor(float(level) * T))
        if m <= 1:
            return out
        for i in range(n_seq):
            for r in range(n_reg):
                idx = rng.choice(T, size=m, replace=False)
                perm = rng.permutation(idx)
                out[i, idx, r] = out[i, perm, r]
        return out

    def temporal_circular_lag(pred_arr, level, rng, scale):
        return np.roll(np.asarray(pred_arr, dtype=np.float64), shift=int(level), axis=1)

    def temporal_nanpad_lag(pred_arr, level, rng, scale):
        lag = int(level)
        out = np.full_like(np.asarray(pred_arr, dtype=np.float64), np.nan)
        if lag == 0:
            return np.asarray(pred_arr, dtype=np.float64).copy()
        if lag > 0:
            if lag < out.shape[1]:
                out[:, lag:, :] = pred_arr[:, :-lag, :]
        else:
            k = -lag
            if k < out.shape[1]:
                out[:, :-k, :] = pred_arr[:, k:, :]
        return out

    def temporal_block_shuffle(pred_arr, level, rng, scale):
        out = np.empty_like(np.asarray(pred_arr, dtype=np.float64))
        n_seq, t_len, _ = out.shape
        b = int(level)
        if b <= 1:
            return np.asarray(pred_arr, dtype=np.float64).copy()
        for s in range(n_seq):
            starts = np.arange(0, t_len, b)
            blocks = [pred_arr[s, st : min(st + b, t_len), :].copy() for st in starts]
            perm = rng.permutation(len(blocks))
            pos = 0
            for idx in perm:
                blk = blocks[idx]
                n = blk.shape[0]
                out[s, pos : pos + n, :] = blk
                pos += n
        return out

    def local_time_jitter(pred_arr, level, rng, scale):
        out = np.empty_like(np.asarray(pred_arr, dtype=np.float64))
        n_seq, t_len, _ = out.shape
        L = int(level)
        if L <= 0:
            return np.asarray(pred_arr, dtype=np.float64).copy()
        for s in range(n_seq):
            jitter = rng.integers(-L, L + 1, size=t_len)
            idx = np.clip(np.arange(t_len) + jitter, 0, t_len - 1)
            out[s] = pred_arr[s, idx, :]
        return out

    def region_specific_lags(pred_arr, level, rng, scale):
        out = np.full_like(np.asarray(pred_arr, dtype=np.float64), np.nan)
        n_seq, T, n_reg = out.shape
        lag_max = int(level)
        lags = rng.integers(-lag_max, lag_max + 1, size=n_reg)
        for r in range(n_reg):
            lag = int(lags[r])
            if lag == 0:
                out[:, :, r] = pred_arr[:, :, r]
            elif lag > 0:
                if lag < T:
                    out[:, lag:, r] = pred_arr[:, : T - lag, r]
            else:
                k = -lag
                if k < T:
                    out[:, : T - k, r] = pred_arr[:, k:, r]
        return out

    def fixed_lag_jitter_desync(pred_arr, level, rng, scale):
        out = np.full_like(np.asarray(pred_arr, dtype=np.float64), np.nan)
        n_seq, T, n_reg = out.shape
        jmax = int(level)
        for i in range(n_seq):
            jit = rng.integers(-jmax, jmax + 1, size=n_reg)
            for r in range(n_reg):
                lag = int(jit[r])
                if lag == 0:
                    out[i, :, r] = pred_arr[i, :, r]
                elif lag > 0:
                    if lag < T:
                        out[i, lag:, r] = pred_arr[i, : T - lag, r]
                else:
                    k = -lag
                    if k < T:
                        out[i, : T - k, r] = pred_arr[i, k:, r]
        return out

    def region_permutation(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        n_reg = out.shape[2]
        k = max(2, int(np.ceil(float(level) * n_reg)))
        k = min(k, n_reg)
        idx = rng.choice(n_reg, size=k, replace=False)
        perm = rng.permutation(idx)
        if np.all(perm == idx) and k > 1:
            perm = np.roll(perm, 1)
        out[:, :, idx] = out[:, :, perm]
        return out

    def region_permute_frac(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        n_reg = out.shape[2]
        k = max(1, int(np.round(float(level) * n_reg)))
        k = min(k, n_reg)
        sel = rng.choice(n_reg, size=k, replace=False)
        perm = sel[rng.permutation(k)]
        out[:, :, sel] = out[:, :, perm]
        return out

    def region_mixing(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        partner = np.roll(out, shift=1, axis=2)
        lam = float(level)
        return (1.0 - lam) * out + lam * partner

    def region_mix_lambda(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        perm = rng.permutation(out.shape[2])
        lam = float(level)
        return (1.0 - lam) * out + lam * out[:, :, perm]

    def anti_sync_mixing(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        partner = np.roll(out, shift=1, axis=2)
        lam = float(level)
        return (1.0 - lam) * out - lam * partner

    def phase_scramble(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        n_seq, T, n_reg = out.shape
        k = max(1, int(np.ceil(float(level) * n_reg)))
        k = min(k, n_reg)
        for i in range(n_seq):
            chosen = rng.choice(n_reg, size=k, replace=False)
            for r in chosen:
                x = out[i, :, r]
                if not np.isfinite(x).all():
                    continue
                X = np.fft.rfft(x)
                mag = np.abs(X)
                ph = np.angle(X)
                rand_ph = rng.uniform(-np.pi, np.pi, size=ph.shape)
                rand_ph[0] = ph[0]
                if T % 2 == 0 and rand_ph.size > 1:
                    rand_ph[-1] = ph[-1]
                X_new = mag * np.exp(1j * rand_ph)
                out[i, :, r] = np.real(np.fft.irfft(X_new, n=T))
        return out

    def latent_contamination_pca(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        n_seq, T, R = out.shape
        t = np.linspace(0.0, 2.0 * np.pi, T, endpoint=False)
        w = rng.normal(0.0, 1.0, size=(R,))
        w /= np.linalg.norm(w) + 1e-12
        for i in range(n_seq):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            u = np.sin(t + phase).reshape(T, 1)
            out[i] += float(level) * u * w.reshape(1, R) * scale["global_iqr"]
        return out

    def latent_contamination_cca(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        n_seq, T, R = out.shape
        t = np.linspace(-1.0, 1.0, T, dtype=np.float64)
        w = rng.normal(0.0, 1.0, size=(R,))
        w /= np.linalg.norm(w) + 1e-12
        for i in range(n_seq):
            u = (t + rng.normal(scale=0.05, size=T)).reshape(T, 1)
            out[i] += float(level) * u * w.reshape(1, R) * scale["global_iqr"]
        return out

    def latent_contamination_amp(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        n_seq, T, R = out.shape
        t = np.linspace(0.0, 2.0 * np.pi, T, endpoint=False)
        w = rng.normal(0.0, 1.0, size=(R,))
        w /= np.linalg.norm(w) + 1e-12
        drift = np.linspace(-1.0, 1.0, T).reshape(T, 1)
        for i in range(n_seq):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            u = (np.sin(t + phase).reshape(T, 1) + 0.5 * drift)
            out[i] += float(level) * u * w.reshape(1, R) * scale["global_iqr"]
        return out

    def ar_oversmoothing(pred_arr, level, rng, scale):
        lam = float(level)
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        sm = out.copy()
        for i in range(sm.shape[0]):
            for r in range(sm.shape[2]):
                for t in range(1, sm.shape[1]):
                    sm[i, t, r] = (1.0 - lam) * out[i, t, r] + lam * sm[i, t - 1, r]
        return sm

    def oversmoothing_lambda(pred_arr, level, rng, scale):
        out = np.asarray(pred_arr, dtype=np.float64).copy()
        lam = float(level)
        for t in range(1, out.shape[1]):
            out[:, t, :] = (1.0 - lam) * out[:, t, :] + lam * out[:, t - 1, :]
        return out

    specs = [
        dict(name="mean_shift", label="mean_shift", group="mean_bias", levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=mean_shift_region_iqr),
        dict(name="global_mean_shift", label="global_mean_shift", group="mean_bias", levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=global_mean_shift),
        dict(name="region_mean_shift", label="region_mean_shift", group="mean_bias", levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=region_mean_shift),
        dict(name="sequence_mean_shift", label="sequence_mean_shift", group="mean_bias", levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=sequence_mean_shift),
        dict(name="slow_drift", label="slow_drift", group="mean_bias", levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=slow_drift),
        dict(name="variance_scale_kl", label="variance_scale (KL)", group="scale_tail", levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=variance_scale_kl),
        dict(name="variance_scale_qnt", label="variance_scale (QNT)", group="scale_tail", levels=[1.10, 1.25, 1.50, 1.75, 2.00], apply=variance_scale_qnt),
        dict(name="tail_spikes_kl", label="tail_spikes (KL)", group="scale_tail", levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=tail_spikes_kl),
        dict(name="tail_spikes_qnt", label="tail_spikes (QNT)", group="scale_tail", levels=[1, 2, 3, 4, 5], apply=tail_spikes_qnt),
        dict(name="tail_spikes_mom", label="tail_spikes (MOM)", group="scale_tail", levels=[1, 2, 3, 4, 5], apply=tail_spikes_mom),
        dict(name="one_sided_spikes", label="one_sided_spikes", group="scale_tail", levels=[1, 2, 3, 4, 5], apply=one_sided_spikes),
        dict(name="additive_noise", label="additive_noise", group="noise", levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=additive_noise_iqr),
        dict(name="independent_region_noise", label="independent_region_noise", group="noise", levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=additive_noise_iqr),
        dict(name="region_noise", label="region_noise", group="noise", levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=additive_noise_iqr),
        dict(name="additive_noise_iqr", label="additive_noise_iqr", group="noise", levels=[0.25, 0.75, 1.00], apply=additive_noise_iqr),
        dict(name="temporal_shuffle", label="temporal_shuffle", group="temporal", levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=temporal_shuffle),
        dict(name="temporal_shuffle_per_region", label="temporal_shuffle_per_region", group="temporal", levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=temporal_shuffle_per_region),
        dict(name="temporal_circular_lag", label="temporal_circular_lag", group="temporal", levels=[1, 4, 16], apply=temporal_circular_lag),
        dict(name="temporal_nanpad_lag", label="temporal_nanpad_lag", group="temporal", levels=[1, 4, 16], apply=temporal_nanpad_lag),
        dict(name="temporal_block_shuffle", label="temporal_block_shuffle", group="temporal", levels=[5, 20, 40], apply=temporal_block_shuffle),
        dict(name="local_time_jitter", label="local_time_jitter", group="temporal", levels=[1, 4, 16], apply=local_time_jitter),
        dict(name="region_specific_lags", label="region_specific_lags", group="temporal", levels=[1, 2, 4, 8, 12], apply=region_specific_lags),
        dict(name="fixed_lag_jitter_desync", label="fixed_lag_jitter_desync", group="temporal", levels=[1, 2, 4, 8, 12], apply=fixed_lag_jitter_desync),
        dict(name="region_permutation", label="region_permutation", group="region_mapping", levels=[0.20, 0.40, 0.60, 0.80, 1.00], apply=region_permutation),
        dict(name="region_permute_frac", label="region_permute_frac", group="region_mapping", levels=[0.25, 0.50, 1.00], apply=region_permute_frac),
        dict(name="region_mixing", label="region_mixing", group="region_mapping", levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=region_mixing),
        dict(name="region_mix_lambda", label="region_mix_lambda", group="region_mapping", levels=[0.25, 0.50, 1.00], apply=region_mix_lambda),
        dict(name="anti_sync_mixing", label="anti_sync_mixing", group="region_mapping", levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=anti_sync_mixing),
        dict(name="phase_scramble", label="phase_scramble", group="spectral_latent", levels=[0.20, 0.40, 0.60, 0.80, 1.00], apply=phase_scramble),
        dict(name="latent_contamination_pca", label="latent_contamination (PCA)", group="spectral_latent", levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=latent_contamination_pca),
        dict(name="latent_contamination_cca", label="latent_contamination (CCA)", group="spectral_latent", levels=[0.10, 0.25, 0.50, 0.75, 1.00], apply=latent_contamination_cca),
        dict(name="latent_contamination_amp", label="latent_contamination_amp", group="spectral_latent", levels=[0.25, 0.75, 1.50], apply=latent_contamination_amp),
        dict(name="ar_oversmoothing", label="ar_oversmoothing", group="smoothing", levels=[0.10, 0.25, 0.50, 0.75, 0.90], apply=ar_oversmoothing),
        dict(name="oversmoothing_lambda", label="oversmoothing_lambda", group="smoothing", levels=[0.25, 0.75, 0.90], apply=oversmoothing_lambda),
    ]
    logic_group_order = ["mean_bias", "scale_tail", "noise", "temporal", "region_mapping", "spectral_latent", "smoothing"]
    logic_group_label = {
        "mean_bias": "Mean/Bias",
        "scale_tail": "Scale/Tails",
        "noise": "Noise",
        "temporal": "Temporal",
        "region_mapping": "Region Mapping",
        "spectral_latent": "Spectral/Latent",
        "smoothing": "Smoothing",
    }
    specs_by_name = OrderedDict((spec["name"], spec) for spec in specs)
    family_order = [spec["name"] for spec in specs]
    return specs, logic_group_order, logic_group_label


def _corruption_colors(specs: list[dict[str, object]]) -> dict[str, str]:
    cmaps = [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c]
    color_list = []
    for cmap in cmaps:
        color_list.extend(cmap(np.linspace(0, 1, 20)))
    return {spec["name"]: color_list[i % len(color_list)] for i, spec in enumerate(specs)}


def _compute_corruption_dashboard(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    score_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    *,
    smoke: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[dict[str, object]], dict[str, float], list[str], dict[str, str]]:
    gt_arr, pred_arr = _align_arrays(gt_arr, pred_arr)
    specs, logic_group_order, logic_group_label = _corruption_registry(smoke=smoke)
    baseline_scores = score_fn(gt_arr, pred_arr)
    score_order = [score for score in _flatten_metric_names() if np.isfinite(baseline_scores.get(score, np.nan))]
    scale_ctx = {
        "region_iqr": np.asarray(
            [
                max(float(np.nanquantile(gt_arr[:, :, r], 0.75) - np.nanquantile(gt_arr[:, :, r], 0.25)), 1e-12)
                if np.isfinite(np.nanquantile(gt_arr[:, :, r], 0.75) - np.nanquantile(gt_arr[:, :, r], 0.25))
                else float(np.nanstd(gt_arr[:, :, r]))
                for r in range(gt_arr.shape[2])
            ],
            dtype=np.float64,
        ),
        "global_iqr": _global_iqr(gt_arr),
    }

    rows = []
    for spec in specs:
        for score_name in score_order:
            rows.append(
                {
                    "score": score_name,
                    "family": spec["name"],
                    "level": 0.0,
                    "relative_magnitude": 0.0,
                    "score_value": float(baseline_scores[score_name]),
                    "baseline": float(baseline_scores[score_name]),
                    "score_drop_abs": 0.0,
                    "score_drop_rel": 0.0,
                }
            )

    for fam_idx, spec in enumerate(specs):
        levels = spec["levels"]
        for level_idx, level in enumerate(levels, start=1):
            rng = np.random.default_rng(20260418 + 100 * fam_idx + level_idx)
            corrupted = spec["apply"](pred_arr, level, rng, scale_ctx)
            values = score_fn(gt_arr, corrupted)
            rel_mag = level_idx / float(len(levels))
            for score_name in score_order:
                baseline = float(baseline_scores.get(score_name, np.nan))
                value = float(values.get(score_name, np.nan))
                drop_abs = float(np.clip(baseline - value, 0.0, 1.0)) if np.isfinite(baseline) and np.isfinite(value) else np.nan
                drop_rel = float(np.clip(drop_abs / (baseline + 1e-12), 0.0, 1.0)) if np.isfinite(drop_abs) else np.nan
                rows.append(
                    {
                        "score": score_name,
                        "family": spec["name"],
                        "logic_group": spec["group"],
                        "level": float(level),
                        "relative_magnitude": float(rel_mag),
                        "score_value": value,
                        "baseline": baseline,
                        "score_drop_abs": drop_abs,
                        "score_drop_rel": drop_rel,
                    }
                )

    master = pd.DataFrame(rows)
    worst = (
        master.sort_values(["score", "family", "score_drop_rel", "score_drop_abs"], ascending=[True, True, False, False])
        .groupby(["score", "family"], as_index=False)
        .first()
    )
    return master, worst, score_order, specs, baseline_scores, logic_group_order, logic_group_label


def _plot_corruption_line_dashboard(master: pd.DataFrame, specs: list[dict[str, object]], score_order: list[str], baseline_scores: dict[str, float], path: Path) -> None:
    colors = _corruption_colors(specs)
    labels = {spec["name"]: spec["label"] for spec in specs}
    ncols = 3
    nrows = int(math.ceil(len(score_order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 4.1 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, score_name in zip(axes, score_order):
        sub = master[master["score"] == score_name]
        for spec in specs:
            fam = sub[sub["family"] == spec["name"]].sort_values("relative_magnitude")
            ax.plot(
                fam["relative_magnitude"],
                fam["score_value"],
                color=colors[spec["name"]],
                linewidth=1.8,
                marker="o",
                markersize=3.5,
                label=labels[spec["name"]],
            )
        ax.axhline(baseline_scores[score_name], color=PALETTE["ink"], linestyle="--", linewidth=1.0)
        ax.set_title(score_name, fontsize=11, color=PALETTE["ink"])
        ax.set_xlabel("Relative corruption magnitude", fontsize=10, color=PALETTE["ink"])
        ax.set_ylabel("Score01", fontsize=10, color=PALETTE["ink"])
        ax.set_ylim(0.0, 1.02)
        _style_axis(ax)
    for ax in axes[len(score_order):]:
        ax.axis("off")
    legend_handles = [Line2D([0], [0], color=colors[spec["name"]], marker="o", linewidth=1.8, label=labels[spec["name"]]) for spec in specs]
    legend_handles.append(Line2D([0], [0], color=PALETTE["ink"], linestyle="--", linewidth=1.0, label="Baseline"))
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Unified corruption sensitivity across active neuro metrics", fontsize=16, color=PALETTE["ink"], y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_svg(fig, path)


def _plot_corruption_polar_dashboard(worst: pd.DataFrame, specs: list[dict[str, object]], score_order: list[str], path: Path) -> None:
    family_order = [spec["name"] for spec in specs]
    specs_by_name = OrderedDict((spec["name"], spec) for spec in specs)
    theta = np.linspace(0.0, 2.0 * np.pi, len(family_order), endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])
    family_colors = _corruption_colors(specs)
    ncols = 2
    nrows = int(math.ceil(len(score_order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.8 * ncols, 6.8 * nrows), subplot_kw={"projection": "polar"})
    axes = np.atleast_1d(axes).ravel()
    for ax, score_name in zip(axes, score_order):
        sub = worst[worst["score"] == score_name]
        radii = []
        for family_name in family_order:
            match = sub[sub["family"] == family_name]
            radii.append(float(match.iloc[0]["score_drop_rel"]) if not match.empty else 0.0)
        radii = np.asarray(radii, dtype=float)
        radii_closed = np.concatenate([radii, radii[:1]])
        ax.plot(theta_closed, radii_closed, color="#2f2f2f", linewidth=1.0)
        ax.fill(theta_closed, radii_closed, color="#9e9e9e", alpha=0.08)
        for ang, radius, family_name in zip(theta, radii, family_order):
            ax.plot([ang, ang], [0.0, radius], color=family_colors[family_name], linewidth=1.8, alpha=0.85)
            ax.scatter([ang], [radius], color=family_colors[family_name], s=36, zorder=3)
        ax.set_xticks(theta)
        ax.set_xticklabels([specs_by_name[name]["label"] for name in family_order], fontsize=6, color=PALETTE["muted"])
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.25, 0.50, 0.75, 1.00])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color=PALETTE["muted"])
        ax.set_title(score_name, fontsize=11, color=PALETTE["ink"], pad=18)
        ax.grid(alpha=0.25)
    for ax in axes[len(score_order):]:
        ax.axis("off")
    fig.suptitle("Polar sensitivity map (ordered by corruption logic family)", fontsize=16, color=PALETTE["ink"], y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_svg(fig, path)


def _plot_corruption_overlaid_polar(worst: pd.DataFrame, specs: list[dict[str, object]], score_order: list[str], path: Path) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, len(specs), endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    cmap = plt.cm.get_cmap("tab20", max(len(score_order), 1))
    for score_name in score_order:
        sub = worst[worst["score"] == score_name]
        radii = []
        for spec in specs:
            match = sub[sub["family"] == spec["name"]]
            radii.append(float(match.iloc[0]["score_drop_rel"]) if not match.empty else 0.0)
        radii_closed = np.concatenate([np.asarray(radii), np.asarray(radii[:1])])
        color = cmap(score_order.index(score_name))
        ax.plot(theta_closed, radii_closed, linewidth=2.0, alpha=0.80, color=color, label=score_name)
    ax.set_xticks(theta)
    ax.set_xticklabels([spec["label"] for spec in specs], fontsize=9, color=PALETTE["muted"])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color=PALETTE["muted"])
    ax.set_title("Overlaid polar sensitivity map across all scores", fontsize=16, color=PALETTE["ink"], pad=22)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1.10), frameon=False, fontsize=8)
    _save_svg(fig, path)


def _top5_corruption_summary(worst: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score_name, sub in worst.groupby("score"):
        top = sub.sort_values(["score_drop_rel", "score_drop_abs"], ascending=False).head(5)
        rows.extend(top.assign(rank=np.arange(1, len(top) + 1)).to_dict("records"))
    return pd.DataFrame(rows)


def _plot_comparison_metric_panel(df: pd.DataFrame, title: str, path: Path) -> None:
    metrics = _flatten_metric_names()
    conditions = list(dict.fromkeys(df["condition_label"]))
    fig, ax = plt.subplots(figsize=(max(11, 0.55 * len(metrics) + 2), 5.5))
    for idx, condition in enumerate(conditions):
        sub = df[df["condition_label"] == condition].copy()
        sub["metric"] = pd.Categorical(sub["metric"], categories=metrics, ordered=True)
        sub = sub.sort_values("metric")
        ax.plot(sub["metric"].astype(str), sub["ratio_to_oracle"], marker="o", linewidth=1.8, label=condition)
        if "n" in sub.columns:
            errs = sub["se"].to_numpy(dtype=float)
            if np.isfinite(errs).any():
                ax.fill_between(
                    np.arange(len(sub)),
                    sub["ratio_to_oracle"] - np.nan_to_num(errs, nan=0.0),
                    sub["ratio_to_oracle"] + np.nan_to_num(errs, nan=0.0),
                    alpha=0.10,
                )
    ax.axhline(1.0, color=PALETTE["ink"], linestyle="--", linewidth=1.0)
    ax.set_title(title, fontsize=15, color=PALETTE["ink"])
    ax.set_ylabel("Model / oracle", fontsize=11, color=PALETTE["ink"])
    ax.set_xlabel("Metric", fontsize=11, color=PALETTE["ink"])
    ax.set_ylim(bottom=0.0)
    _style_axis(ax)
    ax.tick_params(axis="x", rotation=70)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save_svg(fig, path)


def _plot_comparison_family_panel(df: pd.DataFrame, title: str, path: Path) -> None:
    families = [f"family_{name}" for name in NEURO_FAMILY_WEIGHTS]
    conditions = list(dict.fromkeys(df["condition_label"]))
    fig, ax = plt.subplots(figsize=(max(8.5, 0.9 * len(families) + 2), 5.0))
    for condition in conditions:
        sub = df[df["condition_label"] == condition].copy()
        sub["family"] = pd.Categorical(sub["family"], categories=families, ordered=True)
        sub = sub.sort_values("family")
        ax.plot(sub["family"].astype(str), sub["ratio_to_oracle"], marker="o", linewidth=2.0, label=condition)
    ax.axhline(1.0, color=PALETTE["ink"], linestyle="--", linewidth=1.0)
    ax.set_title(title, fontsize=15, color=PALETTE["ink"])
    ax.set_ylabel("Model / oracle", fontsize=11, color=PALETTE["ink"])
    ax.set_xlabel("Family", fontsize=11, color=PALETTE["ink"])
    ax.set_ylim(bottom=0.0)
    _style_axis(ax)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save_svg(fig, path)


def _plot_comparison_compact_panel(df: pd.DataFrame, title: str, path: Path) -> None:
    df = df.sort_values("final_composite", ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(df) + 1.5)))
    y = np.arange(len(df))
    ax.hlines(y, 0, df["composite_ratio_to_oracle"], color=PALETTE["grid"], linewidth=2)
    ax.scatter(df["composite_ratio_to_oracle"], y, s=70, color=PALETTE["distribution"])
    for idx, row in enumerate(df.itertuples()):
        ax.text(row.composite_ratio_to_oracle + 0.02, idx, f"{row.condition_label} (n={getattr(row, 'n', 1)})", va="center", fontsize=10, color=PALETTE["ink"])
    ax.axvline(1.0, color=PALETTE["ink"], linestyle="--", linewidth=1.0)
    ax.set_yticks([])
    ax.set_xlabel("Final composite / oracle", fontsize=11, color=PALETTE["ink"])
    ax.set_title(title, fontsize=15, color=PALETTE["ink"])
    ax.set_ylim(-0.5, len(df) - 0.5)
    _style_axis(ax)
    fig.tight_layout()
    _save_svg(fig, path)


def _draw_schematic_box(ax, xy, wh, title, lines, facecolor):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", linewidth=1.0, edgecolor=PALETTE["ink"], facecolor=facecolor)
    ax.add_patch(box)
    ax.text(x + 0.02, y + h - 0.05, title, fontsize=12, color=PALETTE["ink"], fontweight="bold", va="top")
    ax.text(x + 0.02, y + h - 0.10, "\n".join(lines), fontsize=9.5, color=PALETTE["ink"], va="top")


def _draw_arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=1.0, color=PALETTE["ink"]))


def _plot_overview_schematic(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _draw_schematic_box(ax, (0.04, 0.58), (0.18, 0.24), "Inputs", ["GT sequences", "Prediction sequences", "Shared regions/time"], PALETTE["fill_blue"])
    _draw_schematic_box(ax, (0.28, 0.74), (0.19, 0.16), "Distribution", ["KL/JSD", "Quantiles", "Moments", "Mean shift"], PALETTE["fill_blue"])
    _draw_schematic_box(ax, (0.28, 0.50), (0.19, 0.16), "Temporal", ["TRJDIST", "latent occupancy", "speed + path"], PALETTE["fill_teal"])
    _draw_schematic_box(ax, (0.28, 0.26), (0.19, 0.16), "Relational", ["Graph topology", "Cross-region MI", "Lagged covariance", "Impulse response"], PALETTE["fill_orange"])
    _draw_schematic_box(ax, (0.54, 0.62), (0.18, 0.14), "Geometry", ["Manifold", "Subspace angle"], PALETTE["fill_rose"])
    _draw_schematic_box(ax, (0.54, 0.34), (0.18, 0.20), "State dynamics", ["State occupancy K=11/K=12", "Transitions lag 1/2/3"], PALETTE["fill_gold"])
    _draw_schematic_box(ax, (0.79, 0.47), (0.16, 0.20), "Final composite", ["Weighted family means", "Notebook-defined family weights", "Final score01"], "#F7F7F7")
    for end_y in (0.82, 0.58, 0.34):
        _draw_arrow(ax, (0.22, 0.70), (0.28, end_y))
    _draw_arrow(ax, (0.47, 0.82), (0.54, 0.70))
    _draw_arrow(ax, (0.47, 0.58), (0.54, 0.70))
    _draw_arrow(ax, (0.47, 0.34), (0.54, 0.44))
    _draw_arrow(ax, (0.72, 0.69), (0.79, 0.57))
    _draw_arrow(ax, (0.72, 0.44), (0.79, 0.57))
    ax.text(0.04, 0.92, "Active Neuro Benchmark: metric families and scalar outputs", fontsize=18, color=PALETTE["ink"], fontweight="bold")
    ax.text(0.04, 0.88, "Vector schematic generated from the current notebook-defined metric families.", fontsize=10, color=PALETTE["muted"])
    _save_svg(fig, path)


def _plot_family_cluster_schematic(family_name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    metric_names = list(NEURO_FAMILY_METRICS[family_name].keys())
    family_title = family_name.replace("_", " ").title()
    family_color = PALETTE[family_name]
    _draw_schematic_box(ax, (0.04, 0.28), (0.20, 0.42), "GT vs Pred", ["Shared aligned arrays", "One score per metric", "Notebook formulas"], PALETTE["fill_blue"])
    x_positions = np.linspace(0.34, 0.78, len(metric_names))
    for x, metric_name in zip(x_positions, metric_names):
        detail = METRIC_DETAILS[metric_name]
        _draw_schematic_box(
            ax,
            (float(x), 0.24),
            (0.16, 0.50),
            metric_name,
            textwrap.wrap(detail["compare"], width=23)[:3] + [detail["direction"]],
            "#FFFFFF",
        )
        _draw_arrow(ax, (0.24, 0.49), (float(x), 0.49))
    _draw_schematic_box(ax, (0.80, 0.34), (0.15, 0.30), f"{family_title}\nfamily", ["Notebook weight", "Weighted mean over available metrics"], family_color + "22")
    for x in x_positions:
        _draw_arrow(ax, (float(x) + 0.16, 0.49), (0.80, 0.49))
    ax.text(0.04, 0.90, f"{family_title} metric schematic", fontsize=18, color=PALETTE["ink"], fontweight="bold")
    ax.text(0.04, 0.85, "Publication-style overview of what the family computes.", fontsize=10, color=PALETTE["muted"])
    _save_svg(fig, path)


def _plot_composite_schematic(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    x = 0.08
    for family_name, weight in NEURO_FAMILY_WEIGHTS.items():
        _draw_schematic_box(
            ax,
            (x, 0.48),
            (0.14, 0.22),
            family_name.replace("_", "\n"),
            [f"weight = {weight:.2f}", "family mean over", "available metrics"],
            PALETTE[family_name] + "22",
        )
        _draw_arrow(ax, (x + 0.07, 0.48), (0.52, 0.28))
        x += 0.17
    _draw_schematic_box(ax, (0.44, 0.08), (0.18, 0.18), "FINAL_COMPOSITE_SCORE", ["Weighted mean over family scores", "Matches the active notebook composite"], "#F7F7F7")
    ax.text(0.04, 0.90, "Composite schematic", fontsize=18, color=PALETTE["ink"], fontweight="bold")
    ax.text(0.04, 0.84, "The final score is not a new metric; it is the notebook-defined weighted aggregation of family composites.", fontsize=10, color=PALETTE["muted"])
    _save_svg(fig, path)


def _representative_sequence(gt_arr: np.ndarray, pred_arr: np.ndarray, helpers: dict[str, object]) -> int:
    kl = helpers["_compute_kl_metrics"](gt_arr, pred_arr)
    seq_scores = np.asarray(kl.get("kl_geo_seq", []), dtype=float)
    valid_idx = np.where(np.isfinite(seq_scores))[0]
    if valid_idx.size == 0:
        return 0
    ranked = valid_idx[np.argsort(seq_scores[valid_idx])]
    return int(ranked[len(ranked) // 2])


def _distribution_examples(gt_arr: np.ndarray, pred_arr: np.ndarray, region_names: list[str], seq_idx: int, region_idx: int, path: Path) -> None:
    gt_vals = gt_arr[seq_idx, :, region_idx]
    pred_vals = pred_arr[seq_idx, :, region_idx]
    quantiles = [0.1, 0.5, 0.9]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    ax.hist(gt_vals, bins=30, density=True, alpha=0.55, color=PALETTE["gt"], label="GT")
    ax.hist(pred_vals, bins=30, density=True, alpha=0.45, color=PALETTE["pred"], label="Pred")
    ax.axvline(np.mean(gt_vals), color=PALETTE["gt"], linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(pred_vals), color=PALETTE["pred"], linestyle="--", linewidth=1.5)
    ax.set_title("Marginal distribution + mean")
    ax.legend(frameon=False)
    _style_axis(ax)

    ax = axes[1]
    gt_sorted = np.sort(gt_vals)
    pred_sorted = np.sort(pred_vals)
    gt_ecdf = np.linspace(0, 1, gt_sorted.size)
    pred_ecdf = np.linspace(0, 1, pred_sorted.size)
    ax.plot(gt_sorted, gt_ecdf, color=PALETTE["gt"], linewidth=2)
    ax.plot(pred_sorted, pred_ecdf, color=PALETTE["pred"], linewidth=2)
    for q in quantiles:
        ax.axvline(np.quantile(gt_vals, q), color=PALETTE["gt"], linestyle=":", linewidth=1)
        ax.axvline(np.quantile(pred_vals, q), color=PALETTE["pred"], linestyle=":", linewidth=1)
    ax.set_title("Quantiles and tails")
    _style_axis(ax)

    ax = axes[2]
    stats_names = ["Mean", "Std", "Skew", "Kurtosis"]
    gt_stats = [np.mean(gt_vals), np.std(gt_vals), skew(gt_vals), kurtosis(gt_vals)]
    pred_stats = [np.mean(pred_vals), np.std(pred_vals), skew(pred_vals), kurtosis(pred_vals)]
    x = np.arange(len(stats_names))
    width = 0.36
    ax.bar(x - width / 2, gt_stats, width, color=PALETTE["gt"], alpha=0.75, label="GT")
    ax.bar(x + width / 2, pred_stats, width, color=PALETTE["pred"], alpha=0.75, label="Pred")
    ax.set_xticks(x)
    ax.set_xticklabels(stats_names)
    ax.set_title("Shared distribution descriptors")
    _style_axis(ax)
    fig.suptitle(f"Distribution-family examples | seq={seq_idx} | region={region_names[region_idx]}", fontsize=15, color=PALETTE["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_svg(fig, path)


def _moments_examples(gt_arr: np.ndarray, pred_arr: np.ndarray, region_names: list[str], path: Path) -> None:
    gt_flat = gt_arr.reshape(-1, gt_arr.shape[-1])
    pred_flat = pred_arr.reshape(-1, pred_arr.shape[-1])
    gt_sk = np.array([skew(gt_flat[:, i]) for i in range(gt_flat.shape[1])])
    pr_sk = np.array([skew(pred_flat[:, i]) for i in range(pred_flat.shape[1])])
    gt_ku = np.array([kurtosis(gt_flat[:, i]) for i in range(gt_flat.shape[1])])
    pr_ku = np.array([kurtosis(pred_flat[:, i]) for i in range(pred_flat.shape[1])])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].scatter(gt_sk, pr_sk, color=PALETTE["distribution"], s=40)
    lims = [min(gt_sk.min(), pr_sk.min()), max(gt_sk.max(), pr_sk.max())]
    axes[0].plot(lims, lims, color=PALETTE["ink"], linestyle="--", linewidth=1)
    axes[0].set_xlabel("GT skew")
    axes[0].set_ylabel("Pred skew")
    axes[0].set_title("Per-region skew agreement")
    _style_axis(axes[0])

    axes[1].scatter(gt_ku, pr_ku, color=PALETTE["distribution"], s=40)
    lims = [min(gt_ku.min(), pr_ku.min()), max(gt_ku.max(), pr_ku.max())]
    axes[1].plot(lims, lims, color=PALETTE["ink"], linestyle="--", linewidth=1)
    axes[1].set_xlabel("GT kurtosis")
    axes[1].set_ylabel("Pred kurtosis")
    axes[1].set_title("Per-region kurtosis agreement")
    _style_axis(axes[1])
    fig.suptitle("Higher-order-moment examples", fontsize=15, color=PALETTE["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_svg(fig, path)


def _temporal_examples(gt_arr: np.ndarray, pred_arr: np.ndarray, seq_idx: int, path: Path) -> None:
    pooled_gt = gt_arr.reshape(-1, gt_arr.shape[-1])
    pca = PCA(n_components=2, random_state=0).fit(pooled_gt)
    gt_emb = pca.transform(gt_arr[seq_idx])
    pred_emb = pca.transform(pred_arr[seq_idx])
    gt_speed = np.linalg.norm(np.diff(gt_emb, axis=0), axis=1)
    pred_speed = np.linalg.norm(np.diff(pred_emb, axis=0), axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(gt_emb[:, 0], gt_emb[:, 1], color=PALETTE["gt"], linewidth=1.6, alpha=0.9, label="GT")
    axes[0].plot(pred_emb[:, 0], pred_emb[:, 1], color=PALETTE["pred"], linewidth=1.3, alpha=0.85, label="Pred")
    axes[0].set_title("Latent trajectory in GT PCA basis")
    axes[0].legend(frameon=False)
    _style_axis(axes[0])

    axes[1].hist(gt_speed, bins=30, density=True, alpha=0.55, color=PALETTE["gt"])
    axes[1].hist(pred_speed, bins=30, density=True, alpha=0.45, color=PALETTE["pred"])
    axes[1].set_title("Latent speed distribution")
    _style_axis(axes[1])

    bins = 20
    h_gt, xedges, yedges = np.histogram2d(gt_emb[:, 0], gt_emb[:, 1], bins=bins)
    h_pr, _, _ = np.histogram2d(pred_emb[:, 0], pred_emb[:, 1], bins=[xedges, yedges])
    diff = h_gt / max(h_gt.sum(), 1.0) - h_pr / max(h_pr.sum(), 1.0)
    im = axes[2].imshow(diff.T, origin="lower", cmap="coolwarm", aspect="auto")
    axes[2].set_title("Occupancy difference (GT - Pred)")
    fig.colorbar(im, ax=axes[2], fraction=0.045)
    fig.suptitle(f"Temporal/trajectory examples | seq={seq_idx}", fontsize=15, color=PALETTE["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_svg(fig, path)


def _shared_ring_positions(n_nodes: int, radius: float = 1.0, center: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n_nodes, endpoint=False)
    cx, cy = center
    return np.column_stack([cx + radius * np.cos(theta), cy + radius * np.sin(theta)])


def _top_edges_from_matrix(mat: np.ndarray | None, top_k: int = 18) -> list[tuple[int, int, float]]:
    if mat is None:
        return []
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return []
    iu = np.triu_indices(arr.shape[0], k=1)
    vals = np.abs(arr[iu])
    mask = np.isfinite(vals)
    if not np.any(mask):
        return []
    pairs = list(zip(iu[0][mask], iu[1][mask], vals[mask]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[: min(top_k, len(pairs))]


def _draw_shared_graph_overlay(ax, positions: np.ndarray, gt_edges, pred_edges, title: str) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    gt_map = {(i, j): w for i, j, w in gt_edges}
    pr_map = {(i, j): w for i, j, w in pred_edges}
    all_pairs = sorted(set(gt_map) | set(pr_map))
    for i, j in all_pairs:
        x1, y1 = positions[i]
        x2, y2 = positions[j]
        if (i, j) in gt_map:
            w = gt_map[(i, j)]
            ax.plot([x1, x2], [y1, y2], color=PALETTE["gt"], linewidth=0.8 + 2.2 * float(w), alpha=0.45, solid_capstyle="round", zorder=1)
        if (i, j) in pr_map:
            w = pr_map[(i, j)]
            ax.plot([x1, x2], [y1, y2], color=PALETTE["pred"], linewidth=0.8 + 2.2 * float(w), alpha=0.45, solid_capstyle="round", zorder=2)
    ax.scatter(positions[:, 0], positions[:, 1], s=160, color="#C7CCD6", edgecolor=PALETTE["ink"], linewidth=0.8, zorder=5)
    for idx, (x, y) in enumerate(positions):
        ax.text(x, y, str(idx + 1), ha="center", va="center", fontsize=8, color=PALETTE["ink"], zorder=6)
    ax.set_title(title, color=PALETTE["ink"], fontsize=11)


def _draw_state_partition(ax, centers: np.ndarray, labels: list[str], face_alpha: float = 0.16) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    for idx, (x, y) in enumerate(centers):
        patch = plt.Circle((x, y), 0.22, facecolor=PALETTE["fill_blue"], edgecolor=PALETTE["ink"], linewidth=1.0, alpha=face_alpha, zorder=1)
        ax.add_patch(patch)
        ax.scatter([x], [y], s=110, color="#C7CCD6", edgecolor=PALETTE["ink"], linewidth=0.7, zorder=3)
        ax.text(x, y, labels[idx], ha="center", va="center", fontsize=8, color=PALETTE["ink"], zorder=4)


def _draw_transition_overlay(ax, centers: np.ndarray, gt_mat: np.ndarray, pr_mat: np.ndarray, title: str, top_k: int = 12) -> None:
    _draw_state_partition(ax, centers, [str(i + 1) for i in range(len(centers))])
    gt_pairs = _top_edges_from_matrix(gt_mat, top_k=top_k)
    pr_pairs = _top_edges_from_matrix(pr_mat, top_k=top_k)
    gt_map = {(i, j): w for i, j, w in gt_pairs}
    pr_map = {(i, j): w for i, j, w in pr_pairs}
    all_pairs = sorted(set(gt_map) | set(pr_map))
    for i, j in all_pairs:
        if i == j:
            continue
        x1, y1 = centers[i]
        x2, y2 = centers[j]
        dx, dy = x2 - x1, y2 - y1
        if (i, j) in gt_map:
            w = gt_map[(i, j)]
            arr = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                connectionstyle="arc3,rad=0.14",
                arrowstyle="-|>",
                mutation_scale=8 + 8 * float(w),
                linewidth=0.8 + 3.0 * float(w),
                color=PALETTE["gt"],
                alpha=0.45,
                zorder=2,
                shrinkA=15,
                shrinkB=15,
            )
            ax.add_patch(arr)
        if (i, j) in pr_map:
            w = pr_map[(i, j)]
            arr = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                connectionstyle="arc3,rad=-0.14",
                arrowstyle="-|>",
                mutation_scale=8 + 8 * float(w),
                linewidth=0.8 + 3.0 * float(w),
                color=PALETTE["pred"],
                alpha=0.45,
                zorder=2,
                shrinkA=15,
                shrinkB=15,
            )
            ax.add_patch(arr)
    ax.set_title(title, color=PALETTE["ink"], fontsize=11)


def _relational_examples(gt_arr: np.ndarray, pred_arr: np.ndarray, path: Path) -> None:
    gt_aligned, pred_aligned = _align_arrays(gt_arr, pred_arr)
    gt_flat = _standardize_columns(_pooled_rows(gt_aligned))
    pred_flat = _standardize_columns(_pooled_rows(pred_aligned))
    mi_gt, mi_pr = _mi_matrix(gt_flat), _mi_matrix(pred_flat)
    lag_gt, lag_pr = _lagged_covariance(gt_flat, 1), _lagged_covariance(pred_flat, 1)
    var_gt, var_pr = _var1_coefficients(gt_flat), _var1_coefficients(pred_flat)
    positions = _shared_ring_positions(gt_flat.shape[1], radius=1.0)

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.6))
    _draw_shared_graph_overlay(axes[0, 0], positions, _top_edges_from_matrix(mi_gt), _top_edges_from_matrix(mi_pr), "Shared-node MI graph")
    _draw_shared_graph_overlay(axes[0, 1], positions, _top_edges_from_matrix(lag_gt), _top_edges_from_matrix(lag_pr), "Shared-node lag-cov graph")
    _draw_shared_graph_overlay(axes[0, 2], positions, _top_edges_from_matrix(var_gt), _top_edges_from_matrix(var_pr), "Shared-node VAR graph")

    for ax, gt_mat, pr_mat, title in [
        (axes[1, 0], mi_gt, mi_pr, "MI matrix difference"),
        (axes[1, 1], lag_gt, lag_pr, "Lag-1 covariance difference"),
        (axes[1, 2], var_gt, var_pr, "VAR(1) coefficient difference"),
    ]:
        diff = np.asarray(gt_mat) - np.asarray(pr_mat)
        im = ax.imshow(diff, cmap="coolwarm", aspect="auto")
        ax.set_title(title, color=PALETTE["ink"], fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Relational-family examples", fontsize=15, color=PALETTE["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_svg(fig, path)


def _geometry_examples(gt_arr: np.ndarray, pred_arr: np.ndarray, path: Path) -> None:
    gt_aligned, pred_aligned = _align_arrays(gt_arr, pred_arr)
    gt_flat = _pooled_rows(gt_aligned)
    pred_flat = _pooled_rows(pred_aligned)
    pca = PCA(n_components=2, random_state=0).fit(gt_flat)
    gt2 = pca.transform(gt_flat)
    pr2 = pca.transform(pred_flat)
    gt_cov = np.cov(_standardize_columns(gt_flat), rowvar=False)
    pr_cov = np.cov(_standardize_columns(pred_flat), rowvar=False)
    gt_spec = np.sort(np.linalg.eigvalsh(gt_cov))[::-1]
    pr_spec = np.sort(np.linalg.eigvalsh(pr_cov))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].scatter(gt2[:: max(1, len(gt2) // 1200), 0], gt2[:: max(1, len(gt2) // 1200), 1], s=6, alpha=0.35, color=PALETTE["gt"], label="GT")
    axes[0].scatter(pr2[:: max(1, len(pr2) // 1200), 0], pr2[:: max(1, len(pr2) // 1200), 1], s=6, alpha=0.35, color=PALETTE["pred"], label="Pred")
    axes[0].legend(frameon=False, markerscale=2)
    axes[0].set_title("Shared PCA projection")
    _style_axis(axes[0])

    axes[1].plot(gt_spec / gt_spec.sum(), color=PALETTE["gt"], linewidth=2, label="GT")
    axes[1].plot(pr_spec / pr_spec.sum(), color=PALETTE["pred"], linewidth=2, label="Pred")
    axes[1].set_title("Covariance eigenspectrum")
    axes[1].legend(frameon=False)
    _style_axis(axes[1])

    gt_local = np.mean(np.linalg.norm(gt2[:500] - gt2[:500].mean(axis=0), axis=1))
    pr_local = np.mean(np.linalg.norm(pr2[:500] - pr2[:500].mean(axis=0), axis=1))
    axes[2].bar(["GT", "Pred"], [gt_local, pr_local], color=[PALETTE["gt"], PALETTE["pred"]], alpha=0.8)
    axes[2].set_title("Local geometry proxy")
    _style_axis(axes[2])
    fig.suptitle("Geometry-family examples", fontsize=15, color=PALETTE["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_svg(fig, path)


def _state_dynamics_examples(gt_arr: np.ndarray, pred_arr: np.ndarray, path: Path) -> None:
    gt_aligned, pred_aligned = _align_arrays(gt_arr, pred_arr)
    gt_flat = _pooled_rows(gt_aligned)
    pred_flat = _pooled_rows(pred_aligned)
    pca = PCA(n_components=min(4, gt_flat.shape[1]), random_state=0).fit(gt_flat)
    gt_lat = pca.transform(gt_flat)
    pr_lat = pca.transform(pred_flat)
    kmeans = MiniBatchKMeans(n_clusters=11, random_state=0, batch_size=512, n_init=5).fit(gt_lat)
    gt_state = kmeans.predict(gt_lat)
    pr_state = kmeans.predict(pr_lat)

    def occupancy(state):
        counts = np.bincount(state, minlength=11).astype(float)
        return counts / max(counts.sum(), 1.0)

    def transition(state, lag):
        mat = np.zeros((11, 11), dtype=float)
        for a, b in zip(state[:-lag], state[lag:]):
            mat[a, b] += 1
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return mat / row_sums

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4))
    occ_gt = occupancy(gt_state)
    occ_pr = occupancy(pr_state)
    x = np.arange(11)
    axes[0, 0].bar(x - 0.18, occ_gt, width=0.36, color=PALETTE["gt"], alpha=0.8, label="GT")
    axes[0, 0].bar(x + 0.18, occ_pr, width=0.36, color=PALETTE["pred"], alpha=0.8, label="Pred")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_title("State occupancy (K=11)")
    _style_axis(axes[0, 0])

    centers = _shared_ring_positions(11, radius=1.0)
    gt_t1 = transition(gt_state, 1)
    pr_t1 = transition(pr_state, 1)
    gt_t2 = transition(gt_state, 2)
    pr_t2 = transition(pr_state, 2)
    _draw_transition_overlay(axes[0, 1], centers, gt_t1, pr_t1, "Shared parcellation: lag-1 transitions")
    _draw_transition_overlay(axes[1, 0], centers, gt_t2, pr_t2, "Shared parcellation: lag-2 transitions")
    diff = gt_t1 - pr_t1
    im = axes[1, 1].imshow(diff, cmap="coolwarm", aspect="auto")
    axes[1, 1].set_title("Lag-1 transition difference", color=PALETTE["ink"], fontsize=11)
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    fig.suptitle("State-dynamics examples", fontsize=15, color=PALETTE["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_svg(fig, path)


def _composite_examples(score_bundle: dict[str, float], path: Path) -> None:
    metrics_df = pd.DataFrame(
        [
            {"family": family_name, "metric": metric_name, "value": score_bundle.get(metric_name, np.nan)}
            for family_name, metric_weights in NEURO_FAMILY_METRICS.items()
            for metric_name in metric_weights
        ]
    )
    family_scores = compute_neuro_family_scores(score_bundle)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    for family_name, sub in metrics_df.groupby("family"):
        axes[0].bar(
            sub["metric"],
            sub["value"],
            color=PALETTE[family_name],
            alpha=0.85,
            label=family_name,
        )
    axes[0].tick_params(axis="x", rotation=70)
    axes[0].set_title("Metric terms entering the composite")
    axes[0].set_ylim(0.0, 1.02)
    _style_axis(axes[0])
    family_names = list(NEURO_FAMILY_WEIGHTS)
    family_vals = [family_scores.get(f"family_{name}", np.nan) for name in family_names]
    axes[1].bar(family_names, family_vals, color=[PALETTE[name] for name in family_names], alpha=0.85)
    final_score = compute_neuro_composite(score_bundle)
    axes[1].axhline(final_score, color=PALETTE["ink"], linestyle="--", linewidth=1.5, label=f"Final = {final_score:.3f}")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_title("Family composites and final score")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend(frameon=False)
    _style_axis(axes[1])
    fig.suptitle("Composite examples", fontsize=15, color=PALETTE["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_svg(fig, path)


def _build_metric_registry_df() -> pd.DataFrame:
    rows = []
    for family_name, metric_weights in NEURO_FAMILY_METRICS.items():
        for metric_name, weight in metric_weights.items():
            detail = METRIC_DETAILS[metric_name]
            rows.append(
                {
                    "family": family_name,
                    "metric": metric_name,
                    "weight_within_family": float(weight),
                    "family_weight": float(NEURO_FAMILY_WEIGHTS[family_name]),
                    "compare": detail["compare"],
                    "aggregation": detail["aggregation"],
                    "direction": detail["direction"],
                    "expected_sensitivities": ", ".join(detail["sensitivities"]),
                    "formula": detail["formula"],
                }
            )
    for metric_name, weight in FIDELITY_METRICS.items():
        detail = METRIC_DETAILS[metric_name]
        rows.append(
            {
                "family": "fidelity_auxiliary",
                "metric": metric_name,
                "weight_within_family": float(weight),
                "family_weight": np.nan,
                "compare": detail["compare"],
                "aggregation": detail["aggregation"],
                "direction": detail["direction"],
                "expected_sensitivities": ", ".join(detail["sensitivities"]),
                "formula": detail["formula"],
            }
        )
    return pd.DataFrame(rows)


def _write_metric_handbook(path: Path, registry_df: pd.DataFrame, notebook_path: Path, reference_summary: dict[str, object]) -> None:
    lines = [
        "# Neuro Benchmark Metric Handbook",
        "",
        f"- Notebook source: `{notebook_path}`",
        f"- Reference example run: `{reference_summary['summary_path']}`",
        "- This handbook documents the active notebook-defined neuro benchmark, not earlier deprecated metric sets.",
        "",
        "## Family weights",
        "",
    ]
    for family_name, weight in NEURO_FAMILY_WEIGHTS.items():
        lines.append(f"- `family_{family_name}`: weight `{weight:.2f}`")
    lines += [
        "",
        "## Metric descriptions",
        "",
    ]
    for family_name, metric_weights in NEURO_FAMILY_METRICS.items():
        lines.append(f"### {family_name.replace('_', ' ').title()}")
        lines.append("")
        for metric_name in metric_weights:
            detail = METRIC_DETAILS[metric_name]
            lines.append(f"#### `{metric_name}`")
            lines.append("")
            lines.append(f"- Compares: {detail['compare']}")
            lines.append(f"- Aggregation: {detail['aggregation']}")
            lines.append(f"- Direction: {detail['direction']}")
            lines.append(f"- Expected corruption sensitivities: {', '.join(detail['sensitivities'])}")
            lines.append(f"- Schematic formula: {detail['formula']}")
            lines.append("")
    lines += [
        "## Auxiliary fidelity metrics",
        "",
        "- These are present in the Desktop results and useful for paper context, but they do not enter the active final neuro composite.",
        "",
    ]
    for metric_name in FIDELITY_METRICS:
        detail = METRIC_DETAILS[metric_name]
        lines.append(f"### `{metric_name}`")
        lines.append("")
        lines.append(f"- Compares: {detail['compare']}")
        lines.append(f"- Aggregation: {detail['aggregation']}")
        lines.append(f"- Direction: {detail['direction']}")
        lines.append("")
    path.write_text("\n".join(lines))


def _write_findings_inventory(
    path: Path,
    top5_df: pd.DataFrame,
    run_stats: pd.DataFrame,
    missing_items: list[str],
    covered_panels: set[str],
) -> None:
    lines = [
        "# Findings Inventory",
        "",
        "## Corruption sensitivity",
        "",
    ]
    for score_name, sub in top5_df.groupby("score"):
        top = sub.sort_values(["score_drop_rel", "score_drop_abs"], ascending=False).head(3)
        summaries = [
            f"{row.family} (rel drop={row.score_drop_rel:.3f}, abs drop={row.score_drop_abs:.3f})"
            for row in top.itertuples()
        ]
        lines.append(f"- `{score_name}`: " + "; ".join(summaries))

    lines += ["", "## Ceiling and transfer studies", ""]
    for experiment_group, sub in run_stats.groupby("experiment_group"):
        title = EXPERIMENT_GROUPS[experiment_group]["title"]
        ranked = sub.sort_values("final_composite", ascending=False)
        lines.append(f"### {title}")
        lines.append("")
        for row in ranked.itertuples():
            n_label = f"n={row.n}" if hasattr(row, "n") else "n=1"
            ratio = f"{row.composite_ratio_to_oracle:.3f}" if np.isfinite(row.composite_ratio_to_oracle) else "nan"
            lines.append(f"- {row.condition_label}: final composite `{row.final_composite:.3f}` | ratio to oracle `{ratio}` | {n_label}")
        lines.append("")

    lines += ["## Missing or underpowered evidence", ""]
    singletons = run_stats[run_stats["n"] == 1] if "n" in run_stats.columns else run_stats
    if not singletons.empty:
        lines.append("- Conditions with no interval estimate beyond point value (`n=1`):")
        for row in singletons.itertuples():
            lines.append(f"  - {row.experiment_group}: {row.condition_label}")
    else:
        lines.append("- No single-run conditions detected.")

    uncovered = set(_flatten_metric_names()) - covered_panels
    if uncovered:
        lines.append("- Metrics lacking a dedicated explanatory family panel:")
        for metric_name in sorted(uncovered):
            lines.append(f"  - {metric_name}")
    else:
        lines.append("- All primary neuro metrics are covered by a dedicated family or composite explanatory panel.")

    if missing_items:
        lines.append("- Missing artifacts or optional sources not found:")
        for item in missing_items:
            lines.append(f"  - {item}")
    else:
        lines.append("- No configured Desktop result roots or required comparison files were missing.")

    path.write_text("\n".join(lines))


def _build_manifest(output_root: Path, created_files: list[Path], config: PaperConfig, extra_sources: list[Path]) -> dict[str, object]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "notebook_path": str(config.notebook_path),
        "reference_style_paths": [str(p) for p in config.reference_style_paths],
        "result_roots": {k: str(v) for k, v in config.result_roots.items()},
        "smoke": config.smoke,
        "files": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(created_files)
            if path.is_file()
        ],
        "source_paths": [str(p) for p in extra_sources],
    }


def generate_benchmark_paper_assets(
    output_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, object]:
    config = _load_config(Path(config_path).expanduser() if config_path else None, Path(output_root).expanduser() if output_root else None, smoke)
    nb = nbformat.read(config.notebook_path, as_version=4)
    _validate_notebook(nb, config.notebook_path)

    for root in config.result_roots.values():
        if not root.exists():
            raise FileNotFoundError(f"Configured experiment root does not exist: {root}")

    output_dirs = _ensure_output_dirs(config.output_root)
    created_files: list[Path] = []
    helper_namespace = _load_notebook_score_helpers(config.notebook_path)
    score_fn = _family_score_functions(helper_namespace)
    metric_registry_df = _build_metric_registry_df()
    metrics_cmp_df, families_cmp_df, run_df, missing_items = _load_experiment_tables(config)

    reference = _pick_reference_summary(config)
    gt_ref, pred_ref, region_names = _load_aligned_arrays(reference["gt_path"], reference["pred_path"])
    example_seq = _representative_sequence(gt_ref, pred_ref, helper_namespace)
    example_region = 0

    gt_corr, pred_corr = _subsample_reference(
        gt_ref,
        pred_ref,
        max_seq=12 if config.smoke else 72,
        max_time=64 if config.smoke else 256,
    )
    master_df, worst_df, score_order, corr_specs, baseline_scores, logic_group_order, logic_group_label = _compute_corruption_dashboard(
        gt_corr,
        pred_corr,
        score_fn,
        smoke=config.smoke,
    )
    top5_df = _top5_corruption_summary(worst_df)

    metric_registry_path = output_dirs["tables"] / "metric_registry.csv"
    _save_df(metric_registry_df, metric_registry_path)
    created_files.append(metric_registry_path)

    if not metrics_cmp_df.empty:
        metrics_cmp_path = output_dirs["tables"] / "comparison_metrics_all.csv"
        _save_df(metrics_cmp_df, metrics_cmp_path)
        created_files.append(metrics_cmp_path)
    if not families_cmp_df.empty:
        families_cmp_path = output_dirs["tables"] / "comparison_families_all.csv"
        _save_df(families_cmp_df, families_cmp_path)
        created_files.append(families_cmp_path)
    if not run_df.empty:
        run_path = output_dirs["tables"] / "comparison_run_summaries.csv"
        _save_df(run_df, run_path)
        created_files.append(run_path)

    master_path = output_dirs["tables"] / "corruption_master.csv"
    worst_path = output_dirs["tables"] / "corruption_family_worst.csv"
    top5_path = output_dirs["tables"] / "corruption_top5_by_metric.csv"
    _save_df(master_df, master_path)
    _save_df(worst_df, worst_path)
    _save_df(top5_df, top5_path)
    created_files.extend([master_path, worst_path, top5_path])

    handbook_path = output_dirs["text"] / "metric_handbook.md"
    _write_metric_handbook(handbook_path, metric_registry_df, config.notebook_path, reference)
    created_files.append(handbook_path)

    covered_panels = set(_flatten_metric_names())

    # Schematics
    overview_svg = output_dirs["figures"] / "overview_schematic.svg"
    _plot_overview_schematic(overview_svg)
    created_files.append(overview_svg)
    for family_name in NEURO_FAMILY_METRICS:
        path = output_dirs["figures"] / f"{family_name}_schematic.svg"
        _plot_family_cluster_schematic(family_name, path)
        created_files.append(path)
    composite_schematic = output_dirs["figures"] / "composite_schematic.svg"
    _plot_composite_schematic(composite_schematic)
    created_files.append(composite_schematic)

    # Examples
    distribution_svg = output_dirs["figures"] / "distribution_examples.svg"
    _distribution_examples(gt_ref, pred_ref, region_names, example_seq, example_region, distribution_svg)
    created_files.append(distribution_svg)
    moments_svg = output_dirs["figures"] / "moments_examples.svg"
    _moments_examples(gt_ref, pred_ref, region_names, moments_svg)
    created_files.append(moments_svg)
    temporal_svg = output_dirs["figures"] / "temporal_examples.svg"
    _temporal_examples(gt_ref, pred_ref, example_seq, temporal_svg)
    created_files.append(temporal_svg)
    relational_svg = output_dirs["figures"] / "relational_examples.svg"
    _relational_examples(gt_ref, pred_ref, relational_svg)
    created_files.append(relational_svg)
    geometry_svg = output_dirs["figures"] / "geometry_examples.svg"
    _geometry_examples(gt_ref, pred_ref, geometry_svg)
    created_files.append(geometry_svg)
    state_svg = output_dirs["figures"] / "state_dynamics_examples.svg"
    _state_dynamics_examples(gt_ref, pred_ref, state_svg)
    created_files.append(state_svg)
    composite_svg = output_dirs["figures"] / "composite_examples.svg"
    _composite_examples(baseline_scores, composite_svg)
    created_files.append(composite_svg)

    # Corruption figures
    line_dashboard = output_dirs["figures"] / "corruption_line_dashboard.svg"
    polar_dashboard = output_dirs["figures"] / "corruption_polar_dashboard.svg"
    polar_overlay = output_dirs["figures"] / "corruption_polar_overlay.svg"
    _plot_corruption_line_dashboard(master_df, corr_specs, score_order, baseline_scores, line_dashboard)
    _plot_corruption_polar_dashboard(worst_df, corr_specs, score_order, polar_dashboard)
    _plot_corruption_overlaid_polar(worst_df, corr_specs, score_order, polar_overlay)
    created_files.extend([line_dashboard, polar_dashboard, polar_overlay])

    # Comparison figures
    aggregated_metric_stats = pd.DataFrame()
    aggregated_family_stats = pd.DataFrame()
    aggregated_run_stats = pd.DataFrame()
    if not metrics_cmp_df.empty:
        aggregated_metric_stats = _aggregate_condition_scores(
            metrics_cmp_df,
            ["experiment_group", "condition_label", "metric"],
            "ratio_to_oracle",
        ).rename(columns={"mean": "ratio_to_oracle"})
    if not families_cmp_df.empty:
        aggregated_family_stats = _aggregate_condition_scores(
            families_cmp_df,
            ["experiment_group", "condition_label", "family"],
            "ratio_to_oracle",
        ).rename(columns={"mean": "ratio_to_oracle"})
    if not run_df.empty:
        aggregated_run_stats = _aggregate_condition_scores(
            run_df,
            ["experiment_group", "condition_label"],
            "final_composite",
        ).rename(columns={"mean": "final_composite"})
        if "oracle_final_composite" in run_df.columns:
            ratio_stats = _aggregate_condition_scores(
                run_df,
                ["experiment_group", "condition_label"],
                "composite_ratio_to_oracle",
            ).rename(columns={"mean": "composite_ratio_to_oracle"})
            aggregated_run_stats = aggregated_run_stats.merge(
                ratio_stats[["experiment_group", "condition_label", "composite_ratio_to_oracle"]],
                on=["experiment_group", "condition_label"],
                how="left",
            )

    if not aggregated_metric_stats.empty:
        for experiment_group, meta in EXPERIMENT_GROUPS.items():
            sub = aggregated_metric_stats[aggregated_metric_stats["experiment_group"] == experiment_group]
            if sub.empty:
                continue
            path = output_dirs["figures"] / f"{experiment_group}_metric_panel.svg"
            _plot_comparison_metric_panel(sub, meta["title"], path)
            created_files.append(path)

    if not aggregated_family_stats.empty:
        for experiment_group, meta in EXPERIMENT_GROUPS.items():
            sub = aggregated_family_stats[aggregated_family_stats["experiment_group"] == experiment_group]
            if sub.empty:
                continue
            path = output_dirs["figures"] / f"{experiment_group}_family_panel.svg"
            _plot_comparison_family_panel(sub, meta["title"], path)
            created_files.append(path)

    if not aggregated_run_stats.empty:
        for experiment_group, meta in EXPERIMENT_GROUPS.items():
            sub = aggregated_run_stats[aggregated_run_stats["experiment_group"] == experiment_group]
            if sub.empty:
                continue
            path = output_dirs["figures"] / f"{experiment_group}_compact_summary.svg"
            _plot_comparison_compact_panel(sub, meta["title"], path)
            created_files.append(path)

    findings_path = output_dirs["text"] / "findings_inventory.md"
    _write_findings_inventory(findings_path, top5_df, aggregated_run_stats, missing_items, covered_panels)
    created_files.append(findings_path)

    manifest = _build_manifest(
        config.output_root,
        created_files,
        config,
        extra_sources=[config.notebook_path, *config.reference_style_paths, *config.result_roots.values()],
    )
    manifest_path = output_dirs["root"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    created_files.append(manifest_path)

    return {
        "output_root": str(config.output_root),
        "manifest_path": str(manifest_path),
        "created_files": [str(p) for p in created_files],
        "reference_summary": str(reference["summary_path"]),
        "metric_count": len(_flatten_metric_names()),
        "corruption_score_count": len(score_order),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the benchmark paper asset package from the active neuro notebook and Desktop result trees.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Destination folder for the paper package.")
    parser.add_argument("--config", type=Path, help="Optional JSON config override.")
    parser.add_argument("--smoke", action="store_true", help="Reserved for reduced runs; currently records smoke mode in the manifest.")
    args = parser.parse_args(argv)
    result = generate_benchmark_paper_assets(args.output_root, config_path=args.config, smoke=args.smoke)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
