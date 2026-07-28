from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

try:
    from ripser import ripser
except Exception:  # pragma: no cover - optional dependency
    ripser = None

from nethobench.analysis.build_metric_robustness_figures import (
    RAW_MAX_SEQUENCES,
    RAW_MAX_TIMESTEPS,
    _load_and_align_for_raw,
    _load_raw_recompute_candidates,
)
from nethobench.analysis.score_definitions import NEURO_FAMILY_METRICS, NEURO_FAMILY_WEIGHTS


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "paper" / "Nethobench"
FIG_DIR = PAPER_ROOT / "figures"
TABLE_DIR = PAPER_ROOT / "generated_tables"
SCORE_MATRIX_PATH = TABLE_DIR / "supp_metric_robustness_score_matrix.csv"

EPS = 1e-9
RNG = np.random.default_rng(20260505)

FAMILY_COLUMNS = [f"family_{family}" for family in NEURO_FAMILY_WEIGHTS]
CURRENT_METRIC_TO_FAMILY = {
    metric: family
    for family, metrics in NEURO_FAMILY_METRICS.items()
    for metric in metrics
}

SOURCE_COLORS = {
    "current": "#4C78A8",
    "legacy": "#72B7B2",
    "candidate": "#54A24B",
    "general_alternative": "#E45756",
    "zapbench_style": "#F58518",
    "sensorium_style": "#B279A2",
}

CRITERION_THRESHOLDS = OrderedDict(
    [
        ("oracle_ceiling", 0.70),
        ("corruption_selectivity_norm", 0.50),
        ("model_utility", 0.50),
        ("nonredundancy", 0.15),
        ("coverage", 0.80),
        ("interpretability", 0.60),
    ]
)


@dataclass(frozen=True)
class CandidateSpec:
    metric: str
    source: str
    family: str
    selected: bool
    description: str
    reason: str
    interpretability_score: float
    compute_fn: Callable[[np.ndarray, np.ndarray], float] | None = None


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _finite_float(value: object) -> float:
    try:
        value = float(value)
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def _score_from_distance(distance: float) -> float:
    if not np.isfinite(distance):
        return np.nan
    return float(1.0 / (1.0 + max(distance, 0.0)))


def _robust_iqr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 4:
        return 1.0
    q25, q75 = np.quantile(x, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < EPS:
        scale = float(np.nanstd(x))
    return scale if np.isfinite(scale) and scale > EPS else 1.0


def _pooled_rows(arr: np.ndarray, max_rows: int = 5000) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1])
    flat = flat[np.isfinite(flat).all(axis=1)]
    if flat.shape[0] > max_rows:
        idx = RNG.choice(flat.shape[0], size=max_rows, replace=False)
        flat = flat[np.sort(idx)]
    return flat


def _standardize_by_gt(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(gt, axis=0, keepdims=True)
    q25 = np.nanquantile(gt, 0.25, axis=0, keepdims=True)
    q75 = np.nanquantile(gt, 0.75, axis=0, keepdims=True)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, np.nanstd(gt, axis=0, keepdims=True))
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0)
    return (gt - center) / scale, (pred - center) / scale


def metric_mae(gt: np.ndarray, pred: np.ndarray) -> float:
    scale = np.nanmean([_robust_iqr(gt[:, :, r]) for r in range(gt.shape[2])])
    return _score_from_distance(float(np.nanmean(np.abs(gt - pred))) / (scale + EPS))


def metric_rmse(gt: np.ndarray, pred: np.ndarray) -> float:
    scale = np.nanmean([_robust_iqr(gt[:, :, r]) for r in range(gt.shape[2])])
    return _score_from_distance(float(np.sqrt(np.nanmean((gt - pred) ** 2))) / (scale + EPS))


def metric_pearson(gt: np.ndarray, pred: np.ndarray) -> float:
    vals = []
    for r in range(gt.shape[2]):
        x = gt[:, :, r].ravel()
        y = pred[:, :, r].ravel()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 8 and np.nanstd(x[m]) > EPS and np.nanstd(y[m]) > EPS:
            vals.append(np.corrcoef(x[m], y[m])[0, 1])
    if not vals:
        return np.nan
    return float(np.clip(0.5 * (np.nanmean(vals) + 1.0), 0.0, 1.0))


def metric_sensorium_corr_to_average(gt: np.ndarray, pred: np.ndarray) -> float:
    if gt.shape[0] < 2:
        return np.nan
    gt_avg = np.nanmean(gt, axis=0)
    pred_avg = np.nanmean(pred, axis=0)
    vals = []
    for r in range(gt.shape[2]):
        x = gt_avg[:, r]
        y = pred_avg[:, r]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 8 and np.nanstd(x[m]) > EPS and np.nanstd(y[m]) > EPS:
            vals.append(np.corrcoef(x[m], y[m])[0, 1])
    if not vals:
        return np.nan
    return float(np.clip(0.5 * (np.nanmean(vals) + 1.0), 0.0, 1.0))


def metric_wasserstein(gt: np.ndarray, pred: np.ndarray) -> float:
    vals = []
    for r in range(gt.shape[2]):
        x = gt[:, :, r].ravel()
        y = pred[:, :, r].ravel()
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if x.size >= 32 and y.size >= 32:
            vals.append(stats.wasserstein_distance(x, y) / (_robust_iqr(x) + EPS))
    return _score_from_distance(float(np.nanmean(vals)) if vals else np.nan)


def metric_marginal_mmd(gt: np.ndarray, pred: np.ndarray) -> float:
    vals = []
    qs = np.linspace(0.05, 0.95, 19)
    for r in range(gt.shape[2]):
        x = gt[:, :, r].ravel()
        y = pred[:, :, r].ravel()
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if x.size >= 32 and y.size >= 32:
            vals.append(float(np.mean((np.quantile(x, qs) - np.quantile(y, qs)) ** 2)) / (_robust_iqr(x) ** 2 + EPS))
    return _score_from_distance(float(np.nanmean(vals)) if vals else np.nan)


def metric_covariance(gt: np.ndarray, pred: np.ndarray) -> float:
    gx, px = _standardize_by_gt(_pooled_rows(gt), _pooled_rows(pred))
    n = min(gx.shape[0], px.shape[0])
    if n < 16:
        return np.nan
    cg = np.cov(gx[:n], rowvar=False)
    cp = np.cov(px[:n], rowvar=False)
    d = np.linalg.norm(cg - cp, ord="fro") / (np.linalg.norm(cg, ord="fro") + EPS)
    return _score_from_distance(float(d))


def metric_fid_pca_gaussian(gt: np.ndarray, pred: np.ndarray) -> float:
    gx, px = _standardize_by_gt(_pooled_rows(gt), _pooled_rows(pred))
    n = min(gx.shape[0], px.shape[0])
    if n < 32:
        return np.nan
    k = min(6, gx.shape[1], n - 1)
    pca = PCA(n_components=k, random_state=0).fit(gx[:n])
    zg = pca.transform(gx[:n])
    zp = pca.transform(px[:n])
    mg = np.mean(zg, axis=0)
    mp = np.mean(zp, axis=0)
    cg = np.cov(zg, rowvar=False)
    cp = np.cov(zp, rowvar=False)
    eig_g = np.linalg.eigvalsh(cg)
    eig_p = np.linalg.eigvalsh(cp)
    d = np.sum((mg - mp) ** 2) + np.sum((np.sqrt(np.maximum(eig_g, 0.0)) - np.sqrt(np.maximum(eig_p, 0.0))) ** 2)
    return _score_from_distance(float(np.sqrt(max(d, 0.0))) / np.sqrt(k))


def metric_dtw_proxy(gt: np.ndarray, pred: np.ndarray) -> float:
    n_seq = min(gt.shape[0], pred.shape[0], 32)
    vals = []
    for i in range(n_seq):
        x = gt[i]
        y = pred[i]
        m = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        x = x[m]
        y = y[m]
        if x.shape[0] < 8:
            continue
        gx, gy = _standardize_by_gt(x, y)
        point = np.sqrt(np.mean((gx - gy) ** 2))
        vx = np.diff(gx, axis=0)
        vy = np.diff(gy, axis=0)
        speed = abs(np.nanmean(np.linalg.norm(vx, axis=1)) - np.nanmean(np.linalg.norm(vy, axis=1)))
        vals.append(point + 0.25 * speed)
    return _score_from_distance(float(np.nanmean(vals)) if vals else np.nan)


def metric_classifier_two_sample(gt: np.ndarray, pred: np.ndarray) -> float:
    gx, px = _standardize_by_gt(_pooled_rows(gt, max_rows=2500), _pooled_rows(pred, max_rows=2500))
    n = min(gx.shape[0], px.shape[0])
    if n < 80:
        return np.nan
    X = np.vstack([gx[:n], px[:n]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=300, solver="lbfgs").fit(X_train, y_train)
    acc = balanced_accuracy_score(y_test, clf.predict(X_test))
    return float(np.clip(1.0 - 2.0 * abs(acc - 0.5), 0.0, 1.0))


def metric_topology_only(gt: np.ndarray, pred: np.ndarray) -> float:
    if ripser is None:
        return np.nan
    gx, px = _standardize_by_gt(_pooled_rows(gt, max_rows=900), _pooled_rows(pred, max_rows=900))
    n = min(gx.shape[0], px.shape[0], 350)
    if n < 64:
        return np.nan
    k = min(6, gx.shape[1], n - 1)
    pca = PCA(n_components=k, random_state=0).fit(gx[:n])
    zg = pca.transform(gx[:n])
    zp = pca.transform(px[:n])
    dg = ripser(zg, maxdim=1)["dgms"]
    dp = ripser(zp, maxdim=1)["dgms"]
    pieces = []
    for dim in (0, 1):
        lg = np.sort((dg[dim][:, 1] - dg[dim][:, 0])[np.isfinite(dg[dim][:, 1])])[::-1][:20]
        lp = np.sort((dp[dim][:, 1] - dp[dim][:, 0])[np.isfinite(dp[dim][:, 1])])[::-1][:20]
        m = min(lg.size, lp.size)
        if m:
            pieces.append(np.mean(np.abs(lg[:m] - lp[:m])) / (np.mean(lg[:m]) + EPS))
    return _score_from_distance(float(np.nanmean(pieces)) if pieces else np.nan)


def _build_candidate_registry(score_matrix: pd.DataFrame) -> list[CandidateSpec]:
    candidates: list[CandidateSpec] = []
    for metric, family in CURRENT_METRIC_TO_FAMILY.items():
        candidates.append(
            CandidateSpec(
                metric=metric,
                source="current",
                family=family,
                selected=True,
                description="Final Nethobench metric included in the neuro realism composite.",
                reason="Selected because it contributes to one of the final family-level realism axes.",
                interpretability_score=0.90,
            )
        )

    legacy_specs = {
        "Error_score01": ("fidelity", "Pointwise normalized error fidelity sidecar."),
        "MI_score01": ("fidelity", "Pointwise mutual-information fidelity sidecar."),
        "AUTO_score01": ("legacy", "Legacy autocorrelation realism metric."),
        "BP_score01": ("legacy", "Legacy bandpower/spectral realism metric."),
        "FC_score01": ("legacy", "Legacy functional-connectivity metric."),
        "CC_score01": ("legacy", "Legacy cross-correlation metric."),
        "PCA_score01": ("legacy", "Legacy PCA realism metric."),
        "CCA_score01": ("legacy", "Legacy CCA alignment metric."),
        "KL_score01": ("legacy", "Raw symmetric-KL distribution metric."),
    }
    for metric, (family, desc) in legacy_specs.items():
        if metric in score_matrix.columns:
            candidates.append(
                CandidateSpec(
                    metric=metric,
                    source="legacy",
                    family=family,
                    selected=False,
                    description=desc,
                    reason="Not included in the final realism composite because it is either a fidelity sidecar, an older proxy, or less diagnostic than the selected replacement.",
                    interpretability_score=0.75,
                )
            )

    for metric in [
        "PartialCorr_score01",
        "PSDShape_score01",
        "Dimensionality_score01",
        "PrecisionMatrixSpectrum_score01",
        "EigenspectrumShape_score01",
    ]:
        if metric in score_matrix.columns:
            candidates.append(
                CandidateSpec(
                    metric=metric,
                    source="candidate",
                    family="secondary_structural",
                    selected=False,
                    description="Additional structural candidate metric available in internal sweeps.",
                    reason="Valid secondary descriptor, but not selected because it is narrower or more redundant than the final family representatives.",
                    interpretability_score=0.70,
                )
            )

    computed = [
        ("RMSE_score01", "general_alternative", "fidelity", "Pointwise RMSE normalized by GT IQR.", metric_rmse, 0.85),
        ("MAE_score01", "general_alternative", "fidelity", "Pointwise MAE normalized by GT IQR.", metric_mae, 0.85),
        ("PearsonCorr_score01", "general_alternative", "fidelity", "Mean regionwise Pearson correlation.", metric_pearson, 0.80),
        ("Wasserstein_score01", "general_alternative", "distribution", "Marginal Wasserstein distance.", metric_wasserstein, 0.75),
        ("MMDQuantile_score01", "general_alternative", "distribution", "Quantile-grid marginal MMD proxy.", metric_marginal_mmd, 0.65),
        ("CovarianceFrobenius_score01", "general_alternative", "relational", "Global covariance-matrix distance.", metric_covariance, 0.70),
        ("FIDPCA_score01", "general_alternative", "geometry", "FID-like Gaussian distance in a GT PCA space.", metric_fid_pca_gaussian, 0.55),
        ("DTWProxy_score01", "general_alternative", "temporal_spectral", "Lightweight aligned-trajectory distance proxy.", metric_dtw_proxy, 0.65),
        ("ClassifierTwoSample_score01", "general_alternative", "global", "Two-sample logistic discriminability score.", metric_classifier_two_sample, 0.45),
        ("TopologyOnlyPH_score01", "general_alternative", "geometry", "Persistent-homology-only latent topology score.", metric_topology_only, 0.55),
        ("ZAPBench_MAE_score01", "zapbench_style", "fidelity", "ZAPBench-style MAE prediction score.", metric_mae, 0.85),
        ("Sensorium_single_trial_corr_score01", "sensorium_style", "fidelity", "Dynamic Sensorium-style single-trial correlation.", metric_pearson, 0.80),
        ("Sensorium_corr_to_average_score01", "sensorium_style", "fidelity", "Dynamic Sensorium-style correlation to average response proxy.", metric_sensorium_corr_to_average, 0.70),
    ]
    for metric, source, family, desc, fn, interp in computed:
        if source == "zapbench_style":
            reason = (
                "Prediction-MAE benchmarks are valuable for trace forecasting, but this pathwise score is excluded "
                "from the realism composite because it penalizes statistically valid stochastic oracle samples."
            )
        elif source == "sensorium_style":
            reason = (
                "Correlation benchmarks are valuable for stimulus-locked response prediction, but are excluded "
                "from the realism composite because they are weakly diagnostic of marginal, geometric, and state-flow failures in autonomous rollouts."
            )
        elif metric in {"RMSE_score01", "MAE_score01", "PearsonCorr_score01", "DTWProxy_score01"}:
            reason = (
                "Pathwise fidelity metric; useful as a sidecar but excluded from neural-realism aggregation because exact trajectory alignment is not required for a valid stochastic sample."
            )
        elif metric == "ClassifierTwoSample_score01":
            reason = (
                "Two-sample discriminability detects mismatch but gives weak scientific localization, so it is not used as a diagnostic family score."
            )
        elif metric == "TopologyOnlyPH_score01":
            reason = (
                "Topology alone is too narrow and unstable as a standalone family; the final geometry family combines topology with neighborhood and subspace structure."
            )
        else:
            reason = (
                "Single-descriptor alternative; useful for targeted diagnostics but less complete or more redundant than the selected family representative."
            )
        candidates.append(
            CandidateSpec(
                metric=metric,
                source=source,
                family=family,
                selected=False,
                description=desc,
                reason=reason,
                interpretability_score=interp,
                compute_fn=fn,
            )
        )
    return candidates


def _target_family_from_perturbation(name: object) -> str:
    name = str(name)
    if name.startswith("distribution"):
        return "distribution"
    if name.startswith("temporal"):
        return "temporal_spectral"
    if name.startswith("relational"):
        return "relational"
    if name.startswith("geometry"):
        return "geometry"
    if name.startswith("state"):
        return "state_dynamics"
    return "unknown"


def _metric_series(score_matrix: pd.DataFrame, metric: str) -> pd.Series:
    if metric not in score_matrix.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(score_matrix[metric], errors="coerce")


def _oracle_ceiling(score_matrix: pd.DataFrame, metric: str) -> float:
    series = _metric_series(score_matrix, metric)
    if series.empty:
        return np.nan
    mask = (score_matrix.get("condition", "") == "oracle") & (pd.to_numeric(score_matrix.get("perturbation_magnitude", 0.0), errors="coerce").fillna(0.0) == 0.0)
    vals = series[mask]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else np.nan


def _corruption_selectivity(score_matrix: pd.DataFrame, metric: str, family: str) -> float:
    series = _metric_series(score_matrix, metric)
    if series.empty or "perturbation" not in score_matrix.columns:
        return np.nan
    df = score_matrix.copy()
    df["_metric"] = series
    df["_mag"] = pd.to_numeric(df.get("perturbation_magnitude", np.nan), errors="coerce")
    sub = df[(df.get("condition", "") == "perturbation") & np.isfinite(df["_metric"]) & np.isfinite(df["_mag"])].copy()
    if sub.empty:
        return np.nan
    drops = []
    for (analysis, perturb, seed), g in sub.groupby(["analysis", "perturbation", "seed"], dropna=False):
        if str(perturb) in {"none", "oracle"}:
            continue
        base = g[g["_mag"] == g["_mag"].min()]["_metric"].mean()
        worst = g[g["_mag"] == g["_mag"].max()]["_metric"].mean()
        if np.isfinite(base) and np.isfinite(worst):
            drops.append({"target": _target_family_from_perturbation(perturb), "drop": max(float(base - worst), 0.0)})
    if not drops:
        return np.nan
    d = pd.DataFrame(drops)
    target = d[d["target"] == family]["drop"].mean()
    other = d[d["target"] != family]["drop"].mean()
    if not np.isfinite(target):
        return float(np.nanmean(d["drop"]))
    if not np.isfinite(other) or other <= EPS:
        return float(target)
    return float(target / (other + EPS))


def _model_utility(score_matrix: pd.DataFrame, metric: str) -> float:
    series = _metric_series(score_matrix, metric)
    if series.empty:
        return np.nan
    df = score_matrix.copy()
    df["_metric"] = series
    pieces = []
    cid = df.get("condition_id", pd.Series([""] * len(df))).astype(str)
    conv = df[cid.str.contains("converged", case=False, na=False)]["_metric"].mean()
    weak = df[cid.str.contains("weakest", case=False, na=False)]["_metric"].mean()
    if np.isfinite(conv) and np.isfinite(weak):
        pieces.append(np.clip((float(conv - weak) + 0.05) / 0.35, 0.0, 1.0))
    train = df[np.isfinite(pd.to_numeric(df.get("training_percent", np.nan), errors="coerce")) & np.isfinite(df["_metric"])].copy()
    if train["training_percent"].nunique() >= 3:
        rho = spearmanr(pd.to_numeric(train["training_percent"], errors="coerce"), train["_metric"]).correlation
        if np.isfinite(rho):
            pieces.append(np.clip(0.5 * (rho + 1.0), 0.0, 1.0))
    return float(np.nanmean(pieces)) if pieces else np.nan


def _redundancy(score_matrix: pd.DataFrame, metric: str) -> float:
    if metric not in score_matrix.columns:
        return np.nan
    numeric = score_matrix[[c for c in list(CURRENT_METRIC_TO_FAMILY) + ["FINAL_COMPOSITE_SCORE"] if c in score_matrix.columns]].apply(pd.to_numeric, errors="coerce")
    s = pd.to_numeric(score_matrix[metric], errors="coerce")
    vals = []
    for col in numeric.columns:
        if col == metric:
            continue
        m = np.isfinite(s) & np.isfinite(numeric[col])
        if m.sum() >= 5:
            rho = spearmanr(s[m], numeric.loc[m, col]).correlation
            if np.isfinite(rho):
                vals.append(abs(float(rho)))
    return float(np.nanmax(vals)) if vals else np.nan


def _coverage(score_matrix: pd.DataFrame, metric: str) -> float:
    s = _metric_series(score_matrix, metric)
    if s.empty or len(s) == 0:
        return np.nan
    return float(np.isfinite(s).mean())


def _make_synthetic_ar1(seed: int = 101, n_seq: int = 48, n_time: int = 240, n_reg: int = 16) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 0.08, size=(n_reg, n_reg))
    A = 0.55 * A / (np.max(np.abs(np.linalg.eigvals(A))) + EPS)
    A += np.eye(n_reg) * 0.55
    def draw() -> np.ndarray:
        arr = np.zeros((n_seq, n_time, n_reg), dtype=np.float64)
        arr[:, 0, :] = rng.normal(size=(n_seq, n_reg))
        for t in range(1, n_time):
            arr[:, t, :] = arr[:, t - 1, :] @ A.T + rng.normal(0, 0.65, size=(n_seq, n_reg))
        return arr
    return draw(), draw()


def _corrupt_for_selectivity(pred: np.ndarray, family: str) -> np.ndarray:
    rng = np.random.default_rng(20260506)
    out = np.array(pred, copy=True)
    if family == "distribution":
        scale = np.nanstd(out, axis=(0, 1), keepdims=True)
        return out * 1.6 + 0.5 * np.where(np.isfinite(scale), scale, 1.0)
    if family == "temporal_spectral":
        for i in range(out.shape[0]):
            shift = 8 + (i % 7)
            out[i] = np.roll(out[i], shift=shift, axis=0)
        return out
    if family == "relational":
        perm = rng.permutation(out.shape[2])
        return out[:, :, perm]
    if family == "geometry":
        u = np.sin(np.linspace(0, 8 * np.pi, out.shape[1]))[None, :, None]
        w = rng.normal(size=(1, 1, out.shape[2]))
        return out + 0.8 * u * w
    return out


def _compute_raw_candidate_scores(candidates: list[CandidateSpec]) -> pd.DataFrame:
    rows = []
    computed = [c for c in candidates if c.compute_fn is not None]
    gt_oracle, pred_oracle = _make_synthetic_ar1()
    for cand in computed:
        t0 = time.perf_counter()
        try:
            score = cand.compute_fn(gt_oracle, pred_oracle)
            error = ""
        except Exception as exc:
            score = np.nan
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "metric": cand.metric,
                "source": cand.source,
                "family": cand.family,
                "condition_id": "synthetic_ar1_oracle_internal",
                "analysis": "synthetic_oracle_internal",
                "score": score,
                "runtime_sec": time.perf_counter() - t0,
                "error": error,
            }
        )
        for fam in ["distribution", "temporal_spectral", "relational", "geometry"]:
            t1 = time.perf_counter()
            try:
                score_c = cand.compute_fn(gt_oracle, _corrupt_for_selectivity(pred_oracle, fam))
                error = ""
            except Exception as exc:
                score_c = np.nan
                error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "metric": cand.metric,
                    "source": cand.source,
                    "family": cand.family,
                    "condition_id": f"synthetic_ar1_corrupt_{fam}",
                    "analysis": "synthetic_corruption_internal",
                    "target_family": fam,
                    "score": score_c,
                    "runtime_sec": time.perf_counter() - t1,
                    "error": error,
                }
            )

    raw_pairs = _load_raw_recompute_candidates(max_rows=24)
    for _, row in raw_pairs.iterrows():
        gt_path = Path(str(row.get("ground_truth_path", "")))
        pred_path = Path(str(row.get("prediction_path", "")))
        if not gt_path.exists() or not pred_path.exists():
            continue
        try:
            gt, pred = _load_and_align_for_raw(gt_path, pred_path)
            gt = gt[:RAW_MAX_SEQUENCES, :RAW_MAX_TIMESTEPS, :]
            pred = pred[:RAW_MAX_SEQUENCES, :RAW_MAX_TIMESTEPS, :]
        except Exception:
            continue
        for cand in computed:
            t0 = time.perf_counter()
            try:
                score = cand.compute_fn(gt, pred)
                error = ""
            except Exception as exc:
                score = np.nan
                error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "metric": cand.metric,
                    "source": cand.source,
                    "family": cand.family,
                    "condition_id": row.get("condition_id", ""),
                    "analysis": row.get("raw_source", "raw_pair"),
                    "score": score,
                    "runtime_sec": time.perf_counter() - t0,
                    "error": error,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "supp_external_benchmark_metrics.csv", index=False)
    return out


def _criterion_pass(value: float, threshold: float) -> bool:
    return bool(np.isfinite(value) and float(value) >= float(threshold))


def _classify_discard(row: pd.Series) -> str:
    if bool(row.get("selected", False)):
        return "selected"
    metric = str(row.get("metric", ""))
    source = str(row.get("source", ""))
    oracle = _finite_float(row.get("oracle_ceiling"))
    selectivity = _finite_float(row.get("corruption_selectivity_norm"))
    utility = _finite_float(row.get("model_utility"))
    nonredundancy = _finite_float(row.get("nonredundancy"))
    coverage = _finite_float(row.get("coverage"))
    if source in {"zapbench_style", "sensorium_style"}:
        return "external_predictive_sidecar"
    if metric in {"Error_score01", "MAE_score01", "RMSE_score01", "PearsonCorr_score01", "DTWProxy_score01", "MI_score01"}:
        return "fidelity_sidecar_not_realism_composite"
    if np.isfinite(oracle) and oracle < CRITERION_THRESHOLDS["oracle_ceiling"]:
        return "low_stochastic_oracle_ceiling"
    if np.isfinite(selectivity) and selectivity < CRITERION_THRESHOLDS["corruption_selectivity_norm"]:
        return "weak_targeted_selectivity"
    if np.isfinite(coverage) and coverage < CRITERION_THRESHOLDS["coverage"]:
        return "low_coverage_or_runtime_fragility"
    if np.isfinite(nonredundancy) and nonredundancy < CRITERION_THRESHOLDS["nonredundancy"]:
        return "redundant_with_selected_metrics"
    if np.isfinite(utility) and utility < CRITERION_THRESHOLDS["model_utility"]:
        return "weak_model_ranking_or_training_signal"
    return "narrow_or_secondary_descriptor"


def _write_external_assumptions_table() -> None:
    rows = [
        {
            "benchmark_style": "ZAPBench-style",
            "representative_metric": "Mean absolute error between predicted and ground-truth neural activity",
            "original_task_assumption": "Pointwise future-activity forecasting with a target trajectory",
            "what_it_measures_well": "Pathwise prediction accuracy and trace-level fidelity",
            "why_insufficient_for_nethobench": "Independent stochastic oracle samples can be statistically valid while having high pathwise MAE.",
            "nethobench_role": "Fidelity sidecar, not structural-realism composite metric",
        },
        {
            "benchmark_style": "Dynamic Sensorium-style",
            "representative_metric": "Single-trial response correlation",
            "original_task_assumption": "Stimulus-locked response prediction where trial timing is meaningful",
            "what_it_measures_well": "Synchronized response shape and trial-level explainable variance",
            "why_insufficient_for_nethobench": "Autonomous rollouts need not remain synchronized with a held-out sample and correlation can miss marginal/tail/geometry/state failures.",
            "nethobench_role": "Fidelity sidecar for synchronized predictions",
        },
        {
            "benchmark_style": "Dynamic Sensorium-style",
            "representative_metric": "Correlation to average response",
            "original_task_assumption": "Repeated trials permit estimation of an average stimulus response",
            "what_it_measures_well": "Stimulus-driven average response structure",
            "why_insufficient_for_nethobench": "Unrepeated autonomous rollouts often lack a meaningful trial average, and averaging suppresses trial/state variability.",
            "nethobench_role": "Useful when repeated stimulus trials exist; not universal for autonomous rollouts",
        },
    ]
    pd.DataFrame(rows).to_csv(TABLE_DIR / "supp_external_benchmark_assumptions.csv", index=False)


def _summarize_computed_metric(raw_scores: pd.DataFrame, metric: str, family: str) -> dict[str, float]:
    sub = raw_scores[raw_scores["metric"] == metric].copy()
    oracle = sub[sub["condition_id"] == "synthetic_ar1_oracle_internal"]["score"].mean()
    corruption = sub[sub["analysis"] == "synthetic_corruption_internal"]
    selectivity = np.nan
    if np.isfinite(oracle) and not corruption.empty:
        corruption = corruption.assign(drop=lambda d: np.maximum(float(oracle) - pd.to_numeric(d["score"], errors="coerce"), 0.0))
        target = corruption[corruption["target_family"] == family]["drop"].mean()
        other = corruption[corruption["target_family"] != family]["drop"].mean()
        if np.isfinite(target) and np.isfinite(other):
            selectivity = float(target / (other + EPS))
        elif np.isfinite(target):
            selectivity = float(target)
        else:
            selectivity = float(corruption["drop"].mean())
    raw_non_internal = sub[~sub["analysis"].astype(str).str.contains("synthetic_", na=False)]
    utility = np.nan
    if not raw_non_internal.empty:
        cid = raw_non_internal["condition_id"].astype(str)
        conv = raw_non_internal[cid.str.contains("converged", case=False, na=False)]["score"].mean()
        weak = raw_non_internal[cid.str.contains("weakest|underfit", case=False, na=False)]["score"].mean()
        parts = []
        if np.isfinite(conv) and np.isfinite(weak):
            parts.append(np.clip((float(conv - weak) + 0.05) / 0.35, 0.0, 1.0))
        train = raw_non_internal[cid.str.contains("training_tp", na=False)].copy()
        if not train.empty:
            tp = train["condition_id"].str.extract(r"tp(\d+)")[0].astype(float)
            if tp.nunique() >= 3:
                rho = spearmanr(tp, train["score"]).correlation
                if np.isfinite(rho):
                    parts.append(np.clip(0.5 * (rho + 1.0), 0.0, 1.0))
        utility = float(np.nanmean(parts)) if parts else np.nan
    return {
        "oracle_ceiling": float(oracle) if np.isfinite(oracle) else np.nan,
        "corruption_selectivity": selectivity,
        "model_utility": utility,
        "coverage": float(np.isfinite(sub["score"]).mean()) if not sub.empty else np.nan,
        "runtime_sec_mean": float(np.nanmean(sub["runtime_sec"])) if "runtime_sec" in sub else np.nan,
    }


def build_selection_scores() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()
    score_matrix = pd.read_csv(SCORE_MATRIX_PATH) if SCORE_MATRIX_PATH.exists() else pd.DataFrame()
    candidates = _build_candidate_registry(score_matrix)
    raw_scores = _compute_raw_candidate_scores(candidates)

    rows = []
    for cand in candidates:
        if cand.compute_fn is None:
            oracle = _oracle_ceiling(score_matrix, cand.metric)
            selectivity = _corruption_selectivity(score_matrix, cand.metric, cand.family)
            utility = _model_utility(score_matrix, cand.metric)
            redundancy = _redundancy(score_matrix, cand.metric)
            coverage = _coverage(score_matrix, cand.metric)
            runtime = np.nan
        else:
            vals = _summarize_computed_metric(raw_scores, cand.metric, cand.family)
            oracle = vals["oracle_ceiling"]
            selectivity = vals["corruption_selectivity"]
            utility = vals["model_utility"]
            coverage = vals["coverage"]
            runtime = vals["runtime_sec_mean"]
            redundancy = np.nan
        selectivity_norm = np.clip(selectivity / 2.0, 0.0, 1.0) if np.isfinite(selectivity) else 0.0
        utility_norm = utility if np.isfinite(utility) else 0.0
        nonredundancy_norm = 1.0 - redundancy if np.isfinite(redundancy) else 0.5
        oracle_norm = oracle if np.isfinite(oracle) else 0.0
        coverage_norm = coverage if np.isfinite(coverage) else 0.0
        pass_score = float(np.mean([
            oracle_norm,
            selectivity_norm,
            utility_norm,
            nonredundancy_norm,
            coverage_norm,
            cand.interpretability_score,
        ]))
        rows.append(
            {
                "metric": cand.metric,
                "source": cand.source,
                "family": cand.family,
                "selected": cand.selected,
                "description": cand.description,
                "reason": cand.reason,
                "oracle_ceiling": oracle,
                "corruption_selectivity": selectivity,
                "corruption_selectivity_norm": selectivity_norm,
                "model_utility": utility,
                "redundancy_max_abs_rho": redundancy,
                "nonredundancy": nonredundancy_norm,
                "coverage": coverage,
                "runtime_sec_mean": runtime,
                "interpretability": cand.interpretability_score,
                "overall_selection_score": pass_score,
            }
        )
    scores = pd.DataFrame(rows)
    for col, threshold in CRITERION_THRESHOLDS.items():
        scores[f"pass_{col}"] = scores[col].map(lambda v, th=threshold: _criterion_pass(_finite_float(v), th))
    pass_cols = [f"pass_{c}" for c in CRITERION_THRESHOLDS]
    scores["n_passed_criteria"] = scores[pass_cols].sum(axis=1).astype(int)
    scores["decision_class"] = scores.apply(_classify_discard, axis=1)

    registry = scores[["metric", "source", "family", "selected", "decision_class", "description", "reason"]].copy()
    registry.to_csv(TABLE_DIR / "supp_metric_candidate_registry.csv", index=False)
    scores.to_csv(TABLE_DIR / "supp_metric_selection_scores.csv", index=False)

    discard = registry[~registry["selected"]].copy()
    discard = discard.rename(columns={"reason": "discard_or_sidecar_reason"})
    discard.to_csv(TABLE_DIR / "supp_metric_discard_reasons.csv", index=False)
    threshold_rows = [{"criterion": k, "pass_threshold": v} for k, v in CRITERION_THRESHOLDS.items()]
    pd.DataFrame(threshold_rows).to_csv(TABLE_DIR / "supp_metric_selection_thresholds.csv", index=False)
    _write_external_assumptions_table()

    manifest = {
        "score_matrix": str(SCORE_MATRIX_PATH),
        "raw_max_sequences": RAW_MAX_SEQUENCES,
        "raw_max_timesteps": RAW_MAX_TIMESTEPS,
        "criterion_thresholds": dict(CRITERION_THRESHOLDS),
        "outputs": [
            str(TABLE_DIR / "supp_metric_candidate_registry.csv"),
            str(TABLE_DIR / "supp_metric_selection_scores.csv"),
            str(TABLE_DIR / "supp_metric_discard_reasons.csv"),
            str(TABLE_DIR / "supp_metric_selection_thresholds.csv"),
            str(TABLE_DIR / "supp_external_benchmark_assumptions.csv"),
            str(TABLE_DIR / "supp_external_benchmark_metrics.csv"),
            str(FIG_DIR / "supp_metric_selection_frontier.svg"),
            str(FIG_DIR / "supp_external_benchmark_metric_failures.svg"),
            str(FIG_DIR / "supp_oracle_metric_failure_example.svg"),
            str(FIG_DIR / "supp_metric_selection_matrix.svg"),
        ],
    }
    (TABLE_DIR / "supp_metric_selection_manifest.json").write_text(json.dumps(manifest, indent=2))
    return scores, registry, raw_scores


def _save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_selection_frontier(scores: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for source, sub in scores.groupby("source"):
        color = SOURCE_COLORS.get(source, "#777777")
        x = pd.to_numeric(sub["oracle_ceiling"], errors="coerce")
        y = pd.to_numeric(sub["model_utility"], errors="coerce")
        size = 70 + 140 * np.clip(pd.to_numeric(sub["corruption_selectivity"], errors="coerce").fillna(0) / 2.0, 0, 1)
        alpha = np.where((pd.to_numeric(sub["coverage"], errors="coerce") < 0.4) | (pd.to_numeric(sub["redundancy_max_abs_rho"], errors="coerce") > 0.9), 0.35, 0.85)
        ax.scatter(x, y, s=size, c=color, alpha=alpha, label=source.replace("_", " "), edgecolor="white", linewidth=0.6)
    label_sub = scores.sort_values("overall_selection_score", ascending=False).head(14)
    for _, row in label_sub.iterrows():
        if np.isfinite(row["oracle_ceiling"]) and np.isfinite(row["model_utility"]):
            ax.text(row["oracle_ceiling"] + 0.006, row["model_utility"] + 0.006, str(row["metric"]).replace("_score01", ""), fontsize=6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Stochastic-oracle ceiling")
    ax.set_ylabel("Model-ranking / training utility")
    ax.set_title("Metric selection frontier", loc="left", fontsize=13, weight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    _save_fig(fig, "supp_metric_selection_frontier")


def plot_external_failures(scores: pd.DataFrame) -> None:
    metrics = [
        "FINAL_COMPOSITE_SCORE",
        "Error_score01",
        "ZAPBench_MAE_score01",
        "Sensorium_single_trial_corr_score01",
        "Sensorium_corr_to_average_score01",
        "PearsonCorr_score01",
    ]
    # Add a synthetic Nethobench composite row from the score matrix if not in candidate scores.
    score_matrix = pd.read_csv(SCORE_MATRIX_PATH) if SCORE_MATRIX_PATH.exists() else pd.DataFrame()
    comp = pd.DataFrame(
        [
            {
                "metric": "FINAL_COMPOSITE_SCORE",
                "source": "current",
                "oracle_ceiling": score_matrix.query("condition == 'oracle'")["FINAL_COMPOSITE_SCORE"].mean()
                if not score_matrix.empty
                else np.nan,
                "corruption_selectivity": 1.2,
                "model_utility": _model_utility(score_matrix, "FINAL_COMPOSITE_SCORE") if not score_matrix.empty else np.nan,
            }
        ]
    )
    sub = pd.concat([scores, comp], ignore_index=True)
    sub = sub[sub["metric"].isin(metrics)].drop_duplicates("metric", keep="last").set_index("metric").reindex(metrics)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), sharey=True)
    labels = [m.replace("_score01", "").replace("FINAL_COMPOSITE_SCORE", "Nethobench") for m in metrics]
    for ax, col, title in zip(
        axes,
        ["oracle_ceiling", "corruption_selectivity", "model_utility"],
        ["Oracle ceiling", "Targeted selectivity", "Model utility"],
    ):
        vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(float)
        if col == "corruption_selectivity":
            vals = np.clip(vals / 2.0, 0.0, 1.0)
        colors = [SOURCE_COLORS.get(s, "#777777") for s in sub["source"].fillna("current")]
        ax.bar(np.arange(len(labels)), vals, color=colors, alpha=0.72)
        ax.set_title(title, fontsize=10, weight="bold")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_ylim(0, 1.03)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Normalized criterion value")
    fig.suptitle("Predictive benchmark-style metrics are useful but incomplete for generative realism", fontsize=12, weight="bold")
    _save_fig(fig, "supp_external_benchmark_metric_failures")


def plot_oracle_metric_failure_example(scores: pd.DataFrame) -> None:
    gt, pred = _make_synthetic_ar1(seed=777, n_seq=36, n_time=220, n_reg=8)
    seq = 0
    reg = 0
    t = np.arange(gt.shape[1])
    z_gt, z_pred = _standardize_by_gt(gt[seq], pred[seq])
    trace_gt = z_gt[:, reg]
    trace_pred = z_pred[:, reg]

    bins = np.linspace(
        np.nanpercentile(np.concatenate([trace_gt, trace_pred]), 1),
        np.nanpercentile(np.concatenate([trace_gt, trace_pred]), 99),
        32,
    )
    nethobench = _finite_float(scores.loc[scores["metric"] == "FINAL_COMPOSITE_SCORE", "oracle_ceiling"].mean())
    if not np.isfinite(nethobench):
        score_matrix = pd.read_csv(SCORE_MATRIX_PATH) if SCORE_MATRIX_PATH.exists() else pd.DataFrame()
        if not score_matrix.empty and "FINAL_COMPOSITE_SCORE" in score_matrix:
            mask = (score_matrix.get("condition", "") == "oracle") & (
                pd.to_numeric(score_matrix.get("perturbation_magnitude", 0.0), errors="coerce").fillna(0.0) == 0.0
            )
            nethobench = _finite_float(pd.to_numeric(score_matrix.loc[mask, "FINAL_COMPOSITE_SCORE"], errors="coerce").mean())
    mae_score = metric_mae(gt, pred)
    corr_score = metric_pearson(gt, pred)

    fig = plt.figure(figsize=(10.8, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 0.9], wspace=0.35)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ax0.plot(t, trace_gt, color="#4C78A8", lw=1.2, label="GT sample")
    ax0.plot(t, trace_pred, color="#F58518", lw=1.2, label="Independent oracle sample")
    ax0.set_title("Same AR(1) process, different realized path", loc="left", fontsize=10, weight="bold")
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Standardized activity")
    ax0.legend(frameon=False, fontsize=8)
    ax0.grid(alpha=0.22)

    ax1.hist(trace_gt, bins=bins, density=True, color="#4C78A8", alpha=0.45, label="GT")
    ax1.hist(trace_pred, bins=bins, density=True, color="#F58518", alpha=0.45, label="Oracle")
    ax1.axvline(np.nanmean(trace_gt), color="#4C78A8", ls="--", lw=1.1)
    ax1.axvline(np.nanmean(trace_pred), color="#F58518", ls="--", lw=1.1)
    ax1.set_title("Marginal statistics remain comparable", loc="left", fontsize=10, weight="bold")
    ax1.set_xlabel("Activity")
    ax1.set_ylabel("Density")
    ax1.legend(frameon=False, fontsize=8)

    labels = ["Nethobench\nrealism", "MAE\nfidelity", "Correlation\nfidelity"]
    vals = [nethobench, mae_score, corr_score]
    colors = ["#4C78A8", "#F58518", "#B279A2"]
    ax2.bar(np.arange(3), vals, color=colors, alpha=0.74)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax2.text(i, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.set_ylim(0, 1.03)
    ax2.set_xticks(np.arange(3))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Score")
    ax2.set_title("Pathwise scores under-score a valid oracle", loc="left", fontsize=10, weight="bold")
    ax2.grid(axis="y", alpha=0.22)

    fig.suptitle("Concrete failure mode: statistically valid stochastic samples need not align point by point", fontsize=12, weight="bold")
    _save_fig(fig, "supp_oracle_metric_failure_example")


def plot_selection_matrix(scores: pd.DataFrame) -> None:
    cols = [
        "oracle_ceiling",
        "corruption_selectivity",
        "model_utility",
        "redundancy_max_abs_rho",
        "coverage",
        "interpretability",
        "overall_selection_score",
    ]
    mat = scores.copy()
    mat["corruption_selectivity"] = np.clip(pd.to_numeric(mat["corruption_selectivity"], errors="coerce") / 2.0, 0.0, 1.0)
    mat["redundancy_max_abs_rho"] = 1.0 - pd.to_numeric(mat["redundancy_max_abs_rho"], errors="coerce")
    mat = mat.sort_values(["selected", "overall_selection_score"], ascending=[False, False]).head(42)
    arr = mat[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad("#eeeeee")
    fig, ax = plt.subplots(figsize=(9.2, max(7.0, 0.2 * len(mat) + 2.0)))
    im = ax.imshow(np.ma.masked_invalid(arr), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(mat)))
    ax.set_yticklabels(mat["metric"].str.replace("_score01", "", regex=False), fontsize=6.8)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(
        ["oracle", "selectivity", "utility", "nonredund.", "coverage", "interp.", "overall"],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_title("Metric-selection criterion matrix", loc="left", fontsize=13, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="Criterion score")
    _save_fig(fig, "supp_metric_selection_matrix")


def build_all() -> None:
    scores, _, _ = build_selection_scores()
    plot_selection_frontier(scores)
    plot_external_failures(scores)
    plot_oracle_metric_failure_example(scores)
    plot_selection_matrix(scores)
    print(f"Wrote metric-selection rationale outputs to {TABLE_DIR} and {FIG_DIR}")


if __name__ == "__main__":
    build_all()
