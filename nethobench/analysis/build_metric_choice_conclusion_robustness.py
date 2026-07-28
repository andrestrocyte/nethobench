from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nethobench.analysis.score_definitions import NEURO_FAMILY_METRICS, NEURO_FAMILY_WEIGHTS, weighted_mean_available


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "paper" / "Nethobench"
FIG_DIR = PAPER_ROOT / "figures"
TABLE_DIR = PAPER_ROOT / "generated_tables"
TEXT_DIR = PAPER_ROOT / "generated_text"

FIG2_SCORES = TABLE_DIR / "fig2_multiseed_scores_long.csv"
SCORE_MATRIX = TABLE_DIR / "supp_metric_robustness_score_matrix.csv"
VARIANTS_LONG = TABLE_DIR / "supp_metric_robustness_variants_long.csv"
VARIANT_SUMMARY = TABLE_DIR / "supp_metric_robustness_rank_summary.csv"
RAW_VARIANTS = TABLE_DIR / "supp_metric_raw_hyperparameter_variants.csv"

FAMILY_COLS = [f"family_{k}" for k in NEURO_FAMILY_WEIGHTS]
FAMILY_LABELS = OrderedDict(
    [
        ("family_distribution", "Distribution"),
        ("family_temporal_spectral", "Temporal"),
        ("family_relational", "Relational"),
        ("family_geometry", "Geometry"),
        ("family_state_dynamics", "State dynamics"),
    ]
)
DEFAULT_FAMILY_WEIGHTS = OrderedDict((f"family_{k}", float(v)) for k, v in NEURO_FAMILY_WEIGHTS.items())

COLORS = {
    "family_weights": "#4C78A8",
    "submetric_weights": "#F58518",
    "distribution_divergence": "#54A24B",
    "topology": "#B279A2",
    "state_hyperparameters": "#E45756",
    "raw_hyperparameters": "#72B7B2",
    "score_transform": "#8E6C8A",
    "random_ensemble": "#2F4B7C",
    "default": "#606060",
}


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _finite_float(x: object) -> float:
    try:
        y = float(x)
    except Exception:
        return np.nan
    return y if np.isfinite(y) else np.nan


def _renorm(weights: OrderedDict[str, float]) -> OrderedDict[str, float]:
    vals = OrderedDict((k, max(float(v), 0.0)) for k, v in weights.items())
    s = float(sum(vals.values()))
    if s <= 0:
        return OrderedDict((k, 1.0 / len(vals)) for k in vals)
    return OrderedDict((k, v / s) for k, v in vals.items())


def _family_scores_from_metrics(metrics: dict[str, float], family_metric_weights: dict[str, OrderedDict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for fam, weights in family_metric_weights.items():
        values = {m: _finite_float(metrics.get(m, np.nan)) for m in weights}
        out[f"family_{fam}"] = weighted_mean_available(values, weights)
    return out


def _default_families(row: pd.Series) -> dict[str, float]:
    out = {}
    metrics = {c: _finite_float(row.get(c, np.nan)) for c in row.index}
    for fam, weights in NEURO_FAMILY_METRICS.items():
        vals = {m: metrics.get(m, np.nan) for m in weights}
        out[f"family_{fam}"] = weighted_mean_available(vals, weights)
        if not np.isfinite(out[f"family_{fam}"]):
            out[f"family_{fam}"] = _finite_float(row.get(f"family_{fam}", np.nan))
    return out


def _composite_from_families(families: dict[str, float], weights: OrderedDict[str, float]) -> float:
    return weighted_mean_available({k: families.get(k, np.nan) for k in weights}, weights)


def _profile(families: dict[str, float]) -> np.ndarray:
    x = np.array([families.get(c, np.nan) for c in FAMILY_COLS], dtype=float)
    if not np.all(np.isfinite(x)) or np.nansum(x) <= 0:
        return np.full(len(FAMILY_COLS), np.nan)
    return x / np.nansum(x)


def _profile_corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    if np.nanstd(a[m]) == 0 or np.nanstd(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _rank_corr(default_scores: pd.Series, variant_scores: pd.Series) -> float:
    common = default_scores.index.intersection(variant_scores.index)
    if len(common) < 3:
        return np.nan
    a = default_scores.loc[common].rank(ascending=False, method="average")
    b = variant_scores.loc[common].rank(ascending=False, method="average")
    rho = spearmanr(a, b, nan_policy="omit").correlation
    return float(rho) if np.isfinite(rho) else np.nan


def _make_condition_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "condition_id" in df.columns and df["condition_id"].notna().any():
        return df
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
    return df


def _transform_scores(values: dict[str, float], mode: str, percentiles: dict[str, dict[float, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in values.items():
        x = _finite_float(v)
        if not np.isfinite(x):
            out[k] = np.nan
        elif mode == "identity":
            out[k] = float(np.clip(x, 0, 1))
        elif mode == "sqrt":
            out[k] = float(np.sqrt(np.clip(x, 0, 1)))
        elif mode == "square":
            out[k] = float(np.clip(x, 0, 1) ** 2)
        elif mode == "sigmoid":
            out[k] = float(1.0 / (1.0 + np.exp(-6.0 * (np.clip(x, 0, 1) - 0.5))))
        elif mode == "percentile":
            out[k] = float(percentiles.get(k, {}).get(x, np.nan))
        else:
            out[k] = float(np.clip(x, 0, 1))
    return out


def _build_percentile_maps(df: pd.DataFrame, metric_cols: list[str]) -> dict[str, dict[float, float]]:
    maps: dict[str, dict[float, float]] = {}
    for col in metric_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        order = pd.Series(vals).rank(pct=True, method="average").to_numpy(dtype=float)
        maps[col] = {float(v): float(r) for v, r in zip(vals, order)}
    return maps


def _sample_family_metric_weights(rng: np.random.Generator) -> dict[str, OrderedDict[str, float]]:
    sampled: dict[str, OrderedDict[str, float]] = {}
    for fam, weights in NEURO_FAMILY_METRICS.items():
        keys = list(weights.keys())
        base = np.array([float(weights[k]) for k in keys], dtype=float)
        base = base / base.sum()
        alpha = np.maximum(base * 35.0, 0.2)
        vals = rng.dirichlet(alpha)
        sampled[fam] = OrderedDict((k, float(v)) for k, v in zip(keys, vals))
    return sampled


def _apply_variant_family_logic(
    row: pd.Series,
    rng: np.random.Generator,
    percentile_maps: dict[str, dict[float, float]],
) -> tuple[dict[str, float], OrderedDict[str, float], dict[str, str]]:
    metric_values = {m: _finite_float(row.get(m, np.nan)) for fam in NEURO_FAMILY_METRICS.values() for m in fam}

    distribution_choice = rng.choice(["default", "symmetric_kl"], p=[0.7, 0.3])
    if distribution_choice == "symmetric_kl" and np.isfinite(_finite_float(row.get("KL_score01", np.nan))):
        metric_values["KL_or_JSD_score01"] = _finite_float(row.get("KL_score01", np.nan))

    transform_choice = rng.choice(["identity", "sqrt", "square", "sigmoid", "percentile"], p=[0.45, 0.15, 0.15, 0.15, 0.10])
    metric_values = _transform_scores(metric_values, transform_choice, percentile_maps)

    fam_metric_weights = _sample_family_metric_weights(rng)

    topology_choice = rng.choice(["default", "topology_excluded", "topology_only"], p=[0.65, 0.20, 0.15])
    if topology_choice == "topology_excluded":
        fam_metric_weights["geometry"] = OrderedDict([("SubspaceAngle_score01", 1.0)])
    elif topology_choice == "topology_only":
        fam_metric_weights["geometry"] = OrderedDict([("MANI_score01", 1.0)])

    state_choice = rng.choice(["all", "occupancy_only", "transition_only", "k11_only", "k12_only", "lag1_only"], p=[0.45, 0.15, 0.15, 0.10, 0.10, 0.05])
    if state_choice == "occupancy_only":
        fam_metric_weights["state_dynamics"] = OrderedDict(
            [("LatentStateOccupancyK11_score01", 0.5), ("LatentStateOccupancyK12_score01", 0.5)]
        )
    elif state_choice == "transition_only":
        fam_metric_weights["state_dynamics"] = OrderedDict(
            [
                ("LatentStateTransitionLag1K11_score01", 1 / 3),
                ("LatentStateTransitionLag2K11_score01", 1 / 3),
                ("LatentStateTransitionLag3K11_score01", 1 / 3),
            ]
        )
    elif state_choice == "k11_only":
        fam_metric_weights["state_dynamics"] = OrderedDict(
            [
                ("LatentStateOccupancyK11_score01", 0.25),
                ("LatentStateTransitionLag1K11_score01", 0.25),
                ("LatentStateTransitionLag2K11_score01", 0.25),
                ("LatentStateTransitionLag3K11_score01", 0.25),
            ]
        )
    elif state_choice == "k12_only":
        fam_metric_weights["state_dynamics"] = OrderedDict([("LatentStateOccupancyK12_score01", 1.0)])
    elif state_choice == "lag1_only":
        fam_metric_weights["state_dynamics"] = OrderedDict([("LatentStateTransitionLag1K11_score01", 1.0)])

    families = _family_scores_from_metrics(metric_values, fam_metric_weights)
    # Some trained-model summary rows only store family-level scores, not all
    # submetrics. Keep those rows in the ensemble by stressing family weights and
    # monotone transforms while leaving unavailable submetric choices inactive.
    family_fallback = {c: _finite_float(row.get(c, np.nan)) for c in FAMILY_COLS}
    family_fallback = _transform_scores(family_fallback, transform_choice, {k: {} for k in FAMILY_COLS})
    for c in FAMILY_COLS:
        if not np.isfinite(families.get(c, np.nan)):
            families[c] = family_fallback.get(c, np.nan)

    family_weight_mode = rng.choice(["dirichlet_default", "equal", "leave_one", "double_one"], p=[0.70, 0.10, 0.10, 0.10])
    if family_weight_mode == "dirichlet_default":
        base = np.array([DEFAULT_FAMILY_WEIGHTS[c] for c in FAMILY_COLS], dtype=float)
        weights_v = rng.dirichlet(np.maximum(base * 80.0, 0.5))
        fam_weights = OrderedDict((c, float(v)) for c, v in zip(FAMILY_COLS, weights_v))
    elif family_weight_mode == "equal":
        fam_weights = OrderedDict((c, 1.0 / len(FAMILY_COLS)) for c in FAMILY_COLS)
    elif family_weight_mode == "leave_one":
        fam_weights = OrderedDict(DEFAULT_FAMILY_WEIGHTS)
        fam_weights[rng.choice(FAMILY_COLS)] = 0.0
        fam_weights = _renorm(fam_weights)
    else:
        fam_weights = OrderedDict(DEFAULT_FAMILY_WEIGHTS)
        fam_weights[rng.choice(FAMILY_COLS)] *= 2.0
        fam_weights = _renorm(fam_weights)

    meta = {
        "distribution": str(distribution_choice),
        "transform": str(transform_choice),
        "topology": str(topology_choice),
        "state": str(state_choice),
        "family_weights": str(family_weight_mode),
    }
    return families, fam_weights, meta


def build_random_ensemble(df: pd.DataFrame, n_samples: int = 5000, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    df = _make_condition_ids(df)
    metric_cols = sorted({m for fam in NEURO_FAMILY_METRICS.values() for m in fam} | {"KL_score01"})
    for col in metric_cols:
        if col not in df.columns:
            df[col] = np.nan
    percentile_maps = _build_percentile_maps(df, [c for c in metric_cols if c in df.columns])

    default_rows = []
    for _, row in df.iterrows():
        fam = _default_families(row)
        default_rows.append(
            {
                "condition_id": row["condition_id"],
                "analysis": row.get("analysis", ""),
                "condition": row.get("condition", ""),
                "model": row.get("model", ""),
                "seed": row.get("seed", np.nan),
                "default_score": _composite_from_families(fam, DEFAULT_FAMILY_WEIGHTS),
                **fam,
            }
        )
    default_df = pd.DataFrame(default_rows)
    default_scores = default_df.set_index("condition_id")["default_score"]
    default_profiles = {
        r["condition_id"]: _profile({c: r[c] for c in FAMILY_COLS})
        for _, r in default_df.iterrows()
    }

    score_rows = []
    variant_rows = []
    profile_rows = []
    for s in range(n_samples):
        scores: dict[str, float] = {}
        profiles = []
        meta_ref = None
        for _, row in df.iterrows():
            fam, fam_weights, meta = _apply_variant_family_logic(row, rng, percentile_maps)
            score = _composite_from_families(fam, fam_weights)
            cid = row["condition_id"]
            scores[cid] = score
            score_rows.append(
                {
                    "sample": s,
                    "condition_id": cid,
                    "analysis": row.get("analysis", ""),
                    "condition": row.get("condition", ""),
                    "model": row.get("model", ""),
                    "seed": row.get("seed", np.nan),
                    "score": score,
                    **{f"variant_{c}": fam.get(c, np.nan) for c in FAMILY_COLS},
                }
            )
            profiles.append(_profile_corr(default_profiles[cid], _profile(fam)))
            meta_ref = meta
        variant_score = pd.Series(scores)
        rho = _rank_corr(default_scores, variant_score)
        variant_rows.append(
            {
                "sample": s,
                "spearman_to_default": rho,
                "mean_profile_similarity": float(np.nanmedian(profiles)),
                **(meta_ref or {}),
            }
        )
        profile_rows.append({"sample": s, "profile_similarity": float(np.nanmedian(profiles))})

    return pd.DataFrame(score_rows), pd.DataFrame(variant_rows), default_df


def _conclusion_probabilities(ensemble: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for analysis in ["biophysical_model_comparison", "transfer_calciumgan"]:
        sub = ensemble[ensemble["analysis"] == analysis]
        if sub.empty:
            continue
        piv = sub.groupby(["sample", "condition"])["score"].mean().unstack()
        if {"converged", "weakest"}.issubset(piv.columns):
            diff = piv["converged"] - piv["weakest"]
            rows.append(
                {
                    "conclusion": f"{analysis}: converged > weakest",
                    "probability": float((diff > 0).mean()),
                    "median_margin": float(diff.median()),
                    "q05_margin": float(diff.quantile(0.05)),
                    "q95_margin": float(diff.quantile(0.95)),
                    "n_variants": int(diff.notna().sum()),
                }
            )

    sub = ensemble[ensemble["analysis"] == "training_progress"]
    if not sub.empty:
        piv = sub.groupby(["sample", "condition"])["score"].mean().unstack()
        needed = ["10pct", "30pct", "100pct"]
        if set(needed).issubset(piv.columns):
            margin = np.minimum(piv["30pct"] - piv["10pct"], piv["100pct"] - piv["30pct"])
            rows.append(
                {
                    "conclusion": "training progress: 10 < 30 < 100",
                    "probability": float((margin > 0).mean()),
                    "median_margin": float(pd.Series(margin).median()),
                    "q05_margin": float(pd.Series(margin).quantile(0.05)),
                    "q95_margin": float(pd.Series(margin).quantile(0.95)),
                    "n_variants": int(pd.Series(margin).notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def _rank_distribution(ensemble: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for analysis in ["biophysical_model_comparison", "transfer_calciumgan", "training_progress"]:
        sub = ensemble[ensemble["analysis"] == analysis]
        if sub.empty:
            continue
        label_col = "condition"
        piv = sub.groupby(["sample", label_col])["score"].mean().unstack()
        ranks = piv.rank(axis=1, ascending=False, method="average")
        for label in ranks.columns:
            vals = ranks[label].dropna()
            rows.append(
                {
                    "analysis": analysis,
                    "condition": label,
                    "mean_rank": float(vals.mean()),
                    "rank_q05": float(vals.quantile(0.05)),
                    "rank_q95": float(vals.quantile(0.95)),
                    "top_probability": float((vals == 1).mean()),
                    "n_variants": int(vals.size),
                }
            )
    return pd.DataFrame(rows)


def _pareto_table(df: pd.DataFrame) -> pd.DataFrame:
    use = df[df["analysis"].isin(["biophysical_model_comparison", "transfer_calciumgan", "training_progress"])].copy()
    if use.empty:
        return pd.DataFrame()
    group_cols = ["analysis", "condition"]
    fam = use.groupby(group_cols)[FAMILY_COLS].mean(numeric_only=True).reset_index()
    fam["label"] = fam["analysis"].astype(str).str.replace("_", " ") + ": " + fam["condition"].astype(str)
    X = fam[FAMILY_COLS].to_numpy(dtype=float)
    dominated = []
    for i in range(len(fam)):
        xi = X[i]
        is_dom = False
        for j in range(len(fam)):
            if i == j:
                continue
            xj = X[j]
            if np.all(xj >= xi - 1e-12) and np.any(xj > xi + 1e-12):
                is_dom = True
                break
        dominated.append(is_dom)
    fam["pareto_status"] = np.where(dominated, "dominated", "pareto_front")
    Xc = X - np.nanmean(X, axis=0)
    _, _, vt = np.linalg.svd(np.nan_to_num(Xc), full_matrices=False)
    coords = Xc @ vt[:2].T
    fam["pc1"] = coords[:, 0]
    fam["pc2"] = coords[:, 1]
    return fam


def _plot_box(ax, data: list[np.ndarray], labels: list[str], color: str) -> None:
    clean = [np.asarray(d, dtype=float)[np.isfinite(d)] for d in data]
    bp = ax.boxplot(clean, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor(color)
    for key in ["medians", "whiskers", "caps"]:
        for line in bp[key]:
            line.set_color(color)
            line.set_linewidth(1.2)
    for i, arr in enumerate(clean, start=1):
        if arr.size:
            rng = np.random.default_rng(i)
            xs = i + rng.normal(0, 0.035, size=min(arr.size, 250))
            ys = rng.choice(arr, size=min(arr.size, 250), replace=False)
            ax.scatter(xs, ys, s=5, color=color, alpha=0.18, linewidths=0)


def build_figure() -> dict[str, object]:
    _ensure_dirs()
    fig2 = _make_condition_ids(_read_csv(FIG2_SCORES))
    if fig2.empty:
        raise FileNotFoundError(FIG2_SCORES)
    score_matrix = _make_condition_ids(_read_csv(SCORE_MATRIX))
    variants = _read_csv(VARIANTS_LONG)
    variant_summary = _read_csv(VARIANT_SUMMARY)
    raw_variants = _read_csv(RAW_VARIANTS)

    ensemble_input = fig2[fig2["analysis"].isin(["biophysical_model_comparison", "transfer_calciumgan", "training_progress"])].copy()
    if ensemble_input.empty:
        ensemble_input = fig2.copy()
    ensemble, ensemble_summary, default_df = build_random_ensemble(ensemble_input, n_samples=5000, seed=19)
    default_group = default_df.copy()
    default_group["group"] = default_group["analysis"].astype(str) + "|" + default_group["condition"].astype(str)
    default_group_scores = default_group.groupby("group")["default_score"].mean()
    group_rhos = []
    for sample, sub in ensemble.groupby("sample"):
        tmp = sub.copy()
        tmp["group"] = tmp["analysis"].astype(str) + "|" + tmp["condition"].astype(str)
        scores = tmp.groupby("group")["score"].mean()
        group_rhos.append({"sample": int(sample), "condition_mean_spearman_to_default": _rank_corr(default_group_scores, scores)})
    ensemble_summary = ensemble_summary.merge(pd.DataFrame(group_rhos), on="sample", how="left")
    conclusions = _conclusion_probabilities(ensemble)
    rank_dist = _rank_distribution(ensemble)
    pareto = _pareto_table(default_df)

    ensemble.to_csv(TABLE_DIR / "supp_random_benchmark_ensemble_scores.csv", index=False)
    ensemble_summary.to_csv(TABLE_DIR / "supp_random_benchmark_ensemble_summary.csv", index=False)
    conclusions.to_csv(TABLE_DIR / "supp_random_benchmark_conclusion_probabilities.csv", index=False)
    rank_dist.to_csv(TABLE_DIR / "supp_random_benchmark_rank_distribution.csv", index=False)
    pareto.to_csv(TABLE_DIR / "supp_pareto_family_frontier.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D0D0",
            "axes.linewidth": 0.8,
        }
    )

    fig = plt.figure(figsize=(18, 17), constrained_layout=True)
    gs = fig.add_gridspec(4, 3, height_ratios=[1.1, 1.1, 1.15, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    if not variant_summary.empty:
        groups = variant_summary.groupby("variant_group")["spearman_to_default"].apply(lambda x: x.dropna().to_numpy())
        labels = [g.replace("_", "\n") for g in groups.index]
        _plot_box(ax, list(groups), labels, "#4C78A8")
    ax.axhline(0.8, color="#999999", lw=0.8, ls="--")
    ax.set_ylim(0, 1.03)
    ax.set_title("A. Rank stability across fixed metric variants", loc="left", fontweight="bold")
    ax.set_ylabel("Spearman vs default rank")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", color="#E8E8E8")

    ax = fig.add_subplot(gs[0, 1])
    if not variant_summary.empty:
        g = variant_summary.groupby("variant_group")[["mean_abs_score_delta", "max_abs_score_delta"]].mean().sort_values("mean_abs_score_delta")
        y = np.arange(len(g))
        ax.barh(y - 0.18, g["mean_abs_score_delta"], height=0.34, color="#72B7B2", label="mean |Δ score|")
        ax.barh(y + 0.18, g["max_abs_score_delta"], height=0.34, color="#E45756", alpha=0.75, label="max |Δ score|")
        ax.set_yticks(y)
        ax.set_yticklabels([x.replace("_", " ") for x in g.index])
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(0, None)
    ax.set_title("B. Scores shift, but usually modestly", loc="left", fontweight="bold")
    ax.set_xlabel("Absolute composite-score change")
    ax.grid(axis="x", color="#E8E8E8")

    ax = fig.add_subplot(gs[0, 2])
    data = [ensemble_summary["condition_mean_spearman_to_default"].to_numpy(), ensemble_summary["mean_profile_similarity"].to_numpy()]
    _plot_box(ax, data, ["condition-rank\nstability", "family-profile\nstability"], "#2F4B7C")
    ax.axhline(0.8, color="#999999", lw=0.8, ls="--")
    ax.set_ylim(0, 1.03)
    ax.set_title("C. Random benchmark ensemble", loc="left", fontweight="bold")
    ax.set_ylabel("Correlation to default")
    ax.grid(axis="y", color="#E8E8E8")

    ax = fig.add_subplot(gs[1, 0])
    if not rank_dist.empty:
        sub = rank_dist[rank_dist["analysis"].isin(["biophysical_model_comparison", "transfer_calciumgan"])]
        labels = [f"{r.analysis.replace('_', ' ')}\n{r.condition}" for r in sub.itertuples()]
        x = np.arange(len(sub))
        ax.errorbar(
            x,
            sub["mean_rank"],
            yerr=[sub["mean_rank"] - sub["rank_q05"], sub["rank_q95"] - sub["mean_rank"]],
            fmt="o",
            color="#4C78A8",
            ecolor="#B7C9E2",
            capsize=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.invert_yaxis()
    ax.set_title("D. Rank distribution over random variants", loc="left", fontweight="bold")
    ax.set_ylabel("Rank (lower is better)")
    ax.grid(axis="y", color="#E8E8E8")

    ax = fig.add_subplot(gs[1, 1])
    if not conclusions.empty:
        y = np.arange(len(conclusions))
        colors = np.where(conclusions["probability"] >= 0.8, "#54A24B", np.where(conclusions["probability"] >= 0.6, "#F2BE42", "#E45756"))
        ax.barh(y, conclusions["probability"], color=colors, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(conclusions["conclusion"].str.replace("_", " "), fontsize=8)
        for yi, p in zip(y, conclusions["probability"]):
            ax.text(min(p + 0.02, 0.98), yi, f"{p:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_title("E. Probability key conclusions hold", loc="left", fontweight="bold")
    ax.set_xlabel("Fraction of random variants")
    ax.grid(axis="x", color="#E8E8E8")

    ax = fig.add_subplot(gs[1, 2])
    if not raw_variants.empty:
        raw = raw_variants.drop_duplicates(["variant", "variant_group"])[["variant", "variant_group", "spearman_to_default"]].dropna()
        raw = raw[raw["variant"] != "default_weights"].copy()
        raw["label"] = raw["variant"].str.replace("_", "\n")
        x = np.arange(len(raw))
        colors = [COLORS.get(g, "#777777") for g in raw["variant_group"]]
        ax.scatter(x, raw["spearman_to_default"], s=50, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(raw["label"], rotation=45, ha="right", fontsize=7)
    ax.axhline(0.8, color="#999999", lw=0.8, ls="--")
    ax.set_ylim(0, 1.03)
    ax.set_title("F. Raw hyperparameter response", loc="left", fontweight="bold")
    ax.set_ylabel("Spearman vs default rank")
    ax.grid(axis="y", color="#E8E8E8")

    ax = fig.add_subplot(gs[2, 0])
    if not variants.empty:
        wide = variants.pivot_table(index="variant", columns="condition_id", values="rank", aggfunc="mean")
        summary = variant_summary.set_index("variant")["spearman_to_default"] if not variant_summary.empty else pd.Series(dtype=float)
        keep = [v for v in summary.sort_values().index if v in wide.index][:20]
        if keep:
            mat = wide.loc[keep]
            im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="viridis_r")
            ax.set_yticks(np.arange(len(keep)))
            ax.set_yticklabels([k.replace("_", " ") for k in keep], fontsize=7)
            ax.set_xticks([])
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="rank")
    ax.set_title("G. Most stressful fixed variants still preserve structure", loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[2, 1])
    if not pareto.empty:
        for status, color in [("pareto_front", "#54A24B"), ("dominated", "#B8B8B8")]:
            sub = pareto[pareto["pareto_status"] == status]
            ax.scatter(sub["pc1"], sub["pc2"], s=90 if status == "pareto_front" else 60, color=color, edgecolor="white", lw=0.8, label=status.replace("_", " "))
            for _, r in sub.iterrows():
                ax.text(r["pc1"], r["pc2"], str(r["label"]).replace("biophysical model comparison", "bio").replace("transfer calciumgan", "transfer").replace("training progress", "train"), fontsize=7, ha="left", va="bottom")
        ax.legend(frameon=False, fontsize=8)
    ax.set_title("H. Pareto view without a scalar composite", loc="left", fontweight="bold")
    ax.set_xlabel("family-score PC1")
    ax.set_ylabel("family-score PC2")
    ax.grid(color="#E8E8E8")

    ax = fig.add_subplot(gs[2, 2])
    if not pareto.empty:
        fam_mean = pareto.set_index("label")[FAMILY_COLS]
        im = ax.imshow(fam_mean.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
        ax.set_yticks(np.arange(len(fam_mean)))
        ax.set_yticklabels([x.replace("biophysical_model_comparison", "bio").replace("transfer_calciumgan", "transfer") for x in fam_mean.index], fontsize=7)
        ax.set_xticks(np.arange(len(FAMILY_COLS)))
        ax.set_xticklabels([FAMILY_LABELS[c] for c in FAMILY_COLS], rotation=35, ha="right")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="family score")
    ax.set_title("I. Family profiles behind Pareto status", loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    median_row_rho = float(np.nanmedian(ensemble_summary["spearman_to_default"]))
    q05_row_rho = float(np.nanquantile(ensemble_summary["spearman_to_default"], 0.05))
    median_rho = float(np.nanmedian(ensemble_summary["condition_mean_spearman_to_default"]))
    q05_rho = float(np.nanquantile(ensemble_summary["condition_mean_spearman_to_default"], 0.05))
    median_prof = float(np.nanmedian(ensemble_summary["mean_profile_similarity"]))
    q05_prof = float(np.nanquantile(ensemble_summary["mean_profile_similarity"], 0.05))
    text = (
        "Interpretation. This analysis converts metric arbitrariness into an explicit uncertainty analysis. "
        f"Across 5,000 random reasonable benchmark definitions, median condition-level rank correlation to the default composite is {median_rho:.3f} "
        f"(5th percentile {q05_rho:.3f}), and median family-profile similarity is {median_prof:.3f} "
        f"(5th percentile {q05_prof:.3f}). "
        "Thus, scalar scores move under alternative definitions, but model rankings and diagnostic family profiles are not driven by one hand-picked metric choice. "
        "The Pareto panel additionally shows which comparisons hold without collapsing the five family axes into a single weighted score."
    )
    ax.text(0.01, 0.68, text, ha="left", va="top", fontsize=12, wrap=True)

    svg = FIG_DIR / "supp_benchmark_choice_conclusion_robustness.svg"
    pdf = FIG_DIR / "supp_benchmark_choice_conclusion_robustness.pdf"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_random_variants": int(ensemble_summary["sample"].nunique()),
        "n_conditions": int(default_df["condition_id"].nunique()),
        "median_random_rank_spearman": median_rho,
        "q05_random_rank_spearman": q05_rho,
        "median_random_row_rank_spearman": median_row_rho,
        "q05_random_row_rank_spearman": q05_row_rho,
        "median_family_profile_similarity": median_prof,
        "q05_family_profile_similarity": q05_prof,
        "conclusion_probabilities": conclusions.to_dict(orient="records"),
        "pareto_front": pareto[pareto["pareto_status"] == "pareto_front"]["label"].tolist() if not pareto.empty else [],
        "figure_svg": str(svg),
        "figure_pdf": str(pdf),
    }
    (TABLE_DIR / "supp_benchmark_choice_conclusion_robustness_summary.json").write_text(json.dumps(summary, indent=2))

    text_path = TEXT_DIR / "supp_benchmark_choice_conclusion_robustness_caption_methods.md"
    text_path.write_text(_caption_methods_text(summary))
    return summary


def _caption_methods_text(summary: dict[str, object]) -> str:
    probs = summary.get("conclusion_probabilities", [])
    prob_lines = "\n".join(
        [
            f"- {p['conclusion']}: probability {p['probability']:.3f}, median margin {p['median_margin']:.3f}."
            for p in probs
        ]
    )
    pareto = ", ".join(summary.get("pareto_front", []))
    return f"""# Supplementary Figure: Benchmark Design Choices Do Not Drive the Conclusions

**Caption.** Robustness of Nethobench conclusions to alternative metric definitions and aggregation choices. **(A)** Rank stability across fixed benchmark variants, including family-weight changes, submetric-weight changes, distribution-divergence substitutions, raw hyperparameter recomputation, state-dynamics variants, and topology inclusion/exclusion. **(B)** Mean and maximum absolute composite-score shifts under these variant classes. **(C)** Random benchmark ensemble in which each variant samples family weights, within-family submetric weights, distribution-score choice, topology contribution, state/transition subset, and monotone score transform. The panel reports both global rank stability and normalized family-profile stability relative to the default benchmark. **(D)** Rank distributions for converged/weak models across random variants. **(E)** Probability that the key qualitative conclusions hold across the random ensemble. **(F)** Raw hyperparameter response for variants recomputed from aligned GT/prediction arrays, including PCA dimension, state count, transition lag set, and distribution divergence. **(G)** Rank matrix for the most stressful fixed variants. **(H)** Pareto-front analysis in five-dimensional family-score space, showing which conditions remain non-dominated without imposing any scalar composite. **(I)** Family-score profiles underlying the Pareto analysis. Together, the panels show that exact scalar scores are configurable, but the paper's main model-comparison and family-profile conclusions are not artifacts of one arbitrary metric or weighting choice.

**Methods.** We constructed a benchmark-design robustness analysis using the existing Nethobench score matrix and multiseed Fig. 2 model-validation outputs. Fixed variants were taken from the benchmark robustness table and included equal family weights, leave-one-family-out weights, doubled-family weights, equal submetric weights, leave-one-submetric-out composites, JSD versus symmetric-KL distribution scoring, raw recomputation at alternative PCA dimensions, alternative state-cluster counts, alternative transition-lag sets, and topology inclusion/exclusion. For each fixed variant, we computed composite scores, ranks, Spearman rank correlation with the default benchmark, and absolute score shifts.

For the random benchmark ensemble, we generated {summary['n_random_variants']} plausible benchmark definitions. Each random variant sampled family weights from a Dirichlet distribution centered on the default weights, sampled within-family submetric weights, selected the default or symmetric-KL distribution score when available, selected default/topology-excluded/topology-only geometry, selected alternative state-dynamics subsets, and applied a monotone score transform chosen from identity, square-root, square, sigmoid, or percentile normalization. Each variant was evaluated on {summary['n_conditions']} model or validation conditions. We then computed the rank correlation with the default benchmark and the correlation between normalized family profiles under the default and variant definitions.

Family-profile stability was computed by normalizing each condition's five family scores to sum to one and correlating this vector with the corresponding vector under the default benchmark. This tests whether diagnostic interpretation is stable even when the scalar composite changes. Pareto dominance was computed in the five-dimensional family-score space. A condition was considered dominated if another condition scored at least as high in all families and strictly higher in at least one family. This analysis removes the need for any scalar weighting and tests whether model trade-offs remain visible in the family-score space.

**Quantitative summary.** Across the random benchmark ensemble, the median Spearman rank correlation with the default benchmark was {summary['median_random_rank_spearman']:.3f}, with 5th percentile {summary['q05_random_rank_spearman']:.3f}. Median family-profile similarity was {summary['median_family_profile_similarity']:.3f}, with 5th percentile {summary['q05_family_profile_similarity']:.3f}. Key conclusion probabilities were:

{prob_lines}

The Pareto-front conditions were: {pareto}.
"""


def main() -> None:
    summary = build_figure()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
