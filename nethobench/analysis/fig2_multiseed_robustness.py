from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from nethobench.analysis.score_definitions import NEURO_FAMILY_WEIGHTS


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "paper" / "Nethobench"
FIG_DIR = PAPER_ROOT / "figures"
TABLE_DIR = PAPER_ROOT / "generated_tables"

SYNTHETIC_SCORE_CANDIDATES = [
    Path.home() / "Desktop" / "nethobench_synthetic_validation_fig2_20seed" / "tables" / "synthetic_score_runs.csv",
    Path.home() / "Desktop" / "nethobench_synthetic_validation_final" / "tables" / "synthetic_score_runs.csv",
]
SYNTHETIC_SCORES = next((p for p in SYNTHETIC_SCORE_CANDIDATES if p.exists()), SYNTHETIC_SCORE_CANDIDATES[0])
BIOPHYSICAL_SCORE_CANDIDATES = [
    Path.home() / "Desktop" / "nethobench_biophysical_validation_fig2_20seed" / "tables" / "biophysical_scores.csv",
    Path.home() / "Desktop" / "nethobench_biophysical_validation_final" / "tables" / "biophysical_scores.csv",
]
BIOPHYSICAL_SCORES = next((p for p in BIOPHYSICAL_SCORE_CANDIDATES if p.exists()), BIOPHYSICAL_SCORE_CANDIDATES[0])
SCALING_WIDE = Path.home() / "Desktop" / "netho-seq-scaling-rerun" / "tables" / "nethobench_neurobench_scores_scores_wide.csv"
MODEL_FAMILY_SCORES = Path.home() / "Desktop" / "final_pruned_scores_360_1260.csv"
TRANSFER_ROOT = Path.home() / "Desktop" / "nethobench-calciumgan-biophysical" / "results"
BIOPHYS_ROOT = Path.home() / "Desktop" / "nethobench-sequifier-convergence-biophysical" / "results"
TRAINED_SEED_FULL_ROOT = Path.home() / "Desktop" / "fig2_model_multiseed_full"
TRAINED_SEED_FAST_ROOT = Path.home() / "Desktop" / "fig2_model_multiseed_fast"
TRAINED_SEED_ROOTS = [
    (TRAINED_SEED_FULL_ROOT, "trained_seed_full", 1024, 1024),
    (TRAINED_SEED_FAST_ROOT, "trained_seed_fast", 256, 64),
]
TRAINED_SEED_MIN_N = 5

FAMILY_COLS = [
    "family_distribution",
    "family_temporal_spectral",
    "family_relational",
    "family_geometry",
    "family_state_dynamics",
]
FAMILY_SHORT = {
    "family_distribution": "Distribution",
    "family_temporal_spectral": "Temporal",
    "family_relational": "Relational",
    "family_geometry": "Geometry",
    "family_state_dynamics": "State dynamics",
}
DISPLAY_FAMILY_TO_COL = {
    "distribution": "family_distribution",
    "temporal": "family_temporal_spectral",
    "relational": "family_relational",
    "geometry": "family_geometry",
    "state_dynamics": "family_state_dynamics",
}
DEFAULT_WEIGHTS = np.array([NEURO_FAMILY_WEIGHTS[k.replace("family_", "")] for k in FAMILY_COLS], dtype=float)
DEFAULT_WEIGHTS = DEFAULT_WEIGHTS / DEFAULT_WEIGHTS.sum()
COLORS = {
    "Distribution": "#4C78A8",
    "Temporal": "#9D755D",
    "Relational": "#E45756",
    "Geometry": "#54A24B",
    "State dynamics": "#7F6AAD",
    "Composite": "#2F5AA8",
    "Fidelity": "#F28E2B",
    "Converged": "#4C78A8",
    "Weakest": "#DD8452",
}


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _summary(vals: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(float)
    if x.size == 0:
        return {"n": 0, "mean": np.nan, "sem": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    sem = float(np.std(x, ddof=1) / math.sqrt(x.size)) if x.size > 1 else np.nan
    if x.size >= 5:
        rng = np.random.default_rng(20260501)
        boots = np.array([np.mean(rng.choice(x, size=x.size, replace=True)) for _ in range(4000)])
        ci_low, ci_high = np.quantile(boots, [0.025, 0.975])
    else:
        ci_low = ci_high = np.nan
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "sem": sem,
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
        "ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
    }


def _save_fig(fig: plt.Figure, stem: str) -> tuple[str, str]:
    svg = FIG_DIR / f"{stem}.svg"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(svg), str(pdf)


def _weighted_composite(row: pd.Series, weights: np.ndarray) -> float:
    vals = np.asarray([row.get(col, np.nan) for col in FAMILY_COLS], dtype=float)
    mask = np.isfinite(vals)
    if not np.any(mask):
        return np.nan
    w = weights[mask]
    return float(np.sum(vals[mask] * w) / np.sum(w))


def _parse_seed(path: Path) -> int:
    for part in path.parts[::-1]:
        if part.startswith("biophysical_seed_"):
            return int(part.rsplit("_", 1)[-1])
        if part.startswith("seed_"):
            bits = part.split("_")
            for bit in bits:
                if bit.isdigit():
                    return int(bit)
    return -1


def _result_row_from_dir(
    *,
    analysis: str,
    model: str,
    seed: int,
    rollout_horizon: int,
    result_dir: Path,
    replicate_type: str,
) -> dict[str, object] | None:
    fam_path = result_dir / "family_comparison_rollout.csv"
    fid_path = result_dir / "fidelity_comparison_rollout.csv"
    summary_path = result_dir / "summary_rollout.json"
    fam = _read_csv(fam_path)
    fid = _read_csv(fid_path)
    if fam.empty or not summary_path.exists():
        return None
    row: dict[str, object] = {
        "analysis": analysis,
        "condition": model,
        "model": model,
        "seed": seed,
        "rollout_horizon": rollout_horizon,
        "perturbation": "none",
        "perturbation_magnitude": 0.0,
        "replicate_type": replicate_type,
        "result_dir": str(result_dir),
    }
    for _, fr in fam.iterrows():
        family = str(fr.get("family", ""))
        key = DISPLAY_FAMILY_TO_COL.get(family, family if family.startswith("family_") else f"family_{family}")
        row[key] = float(fr.get("model_score", np.nan))
        row[f"oracle_{key}"] = float(fr.get("oracle_score", np.nan))
    row["FINAL_COMPOSITE_SCORE"] = _weighted_composite(pd.Series(row), DEFAULT_WEIGHTS)
    if not fid.empty:
        for _, frow in fid.iterrows():
            score_name = str(frow.get("score", ""))
            value = float(frow.get("model_score", np.nan))
            oracle_value = float(frow.get("oracle_score", np.nan))
            if score_name:
                row[score_name] = value
                row[f"oracle_{score_name}"] = oracle_value
        vals = pd.to_numeric(fid.get("model_score", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(vals):
            row.setdefault("FIDELITY_SCORE", float(vals.mean()))
        if "family_fidelity" in row:
            row["FIDELITY_SCORE"] = row["family_fidelity"]
    return row


def _load_trained_seed_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trained_root, replicate_type, bio_horizon, transfer_horizon in TRAINED_SEED_ROOTS:
        if not trained_root.exists():
            continue

        for seed_dir in sorted(trained_root.glob("biophysical_seed_*")):
            seed = _parse_seed(seed_dir)
            for model in ["converged", "weakest"]:
                row = _result_row_from_dir(
                    analysis="biophysical_model_comparison",
                    model=model,
                    seed=seed,
                    rollout_horizon=bio_horizon,
                    result_dir=seed_dir / "results" / model,
                    replicate_type=replicate_type,
                )
                if row is not None:
                    rows.append(row)

        transfer_root = trained_root / "calciumgan_transfer_seed_results"
        for result_dir in sorted(transfer_root.glob("seed_*_*")):
            name = result_dir.name
            if name.endswith("_converged"):
                model = "converged"
            elif name.endswith("_weakest"):
                model = "weakest"
            else:
                continue
            seed = _parse_seed(result_dir)
            row = _result_row_from_dir(
                analysis="transfer_calciumgan",
                model=model,
                seed=seed,
                rollout_horizon=transfer_horizon,
                result_dir=result_dir,
                replicate_type=replicate_type,
            )
            if row is not None:
                rows.append(row)
    return rows


def _has_min_trained_seed_coverage(rows: list[dict[str, object]], analysis: str) -> bool:
    sub = pd.DataFrame([r for r in rows if r.get("analysis") == analysis])
    if sub.empty:
        return False
    for replicate_type in ["trained_seed_full", "trained_seed_fast"]:
        rep = sub[sub["replicate_type"] == replicate_type]
        counts = rep.groupby("condition")["seed"].nunique()
        if all(counts.get(model, 0) >= TRAINED_SEED_MIN_N for model in ["converged", "weakest"]):
            return True
    return False


def build_multiseed_scores_long() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    synth = _read_csv(SYNTHETIC_SCORES)
    if not synth.empty:
        for _, r in synth.iterrows():
            base = {
                "analysis": "synthetic_ar1",
                "condition": str(r.get("comparison_kind", "")),
                "model": str(r.get("perturbation_name", "oracle")),
                "seed": int(r.get("sample_seed", -1)),
                "rollout_horizon": np.nan,
                "perturbation": str(r.get("perturbation_name", "oracle")),
                "perturbation_magnitude": float(r.get("perturbation_level", 0.0)),
                "replicate_type": "synthetic_seed" if r.get("comparison_kind") == "oracle" else "fixed_perturbation_seed",
            }
            score_cols = [c for c in synth.columns if c.endswith("_score01") or c in FAMILY_COLS or c in {"FINAL_COMPOSITE_SCORE", "FIDELITY_SCORE"}]
            rows.append({**base, **{c: r.get(c, np.nan) for c in score_cols}})

    bio = _read_csv(BIOPHYSICAL_SCORES)
    if not bio.empty:
        for _, r in bio.iterrows():
            base = {
                "analysis": "biophysical_oracle",
                "condition": str(r.get("comparison_kind", "")),
                "model": str(r.get("perturbation_name", "oracle")),
                "seed": int(r.get("sample_seed", -1)),
                "rollout_horizon": 1024,
                "perturbation": str(r.get("perturbation_name", "oracle")),
                "perturbation_magnitude": float(r.get("perturbation_level", 0.0)),
                "replicate_type": "biophysical_seed" if r.get("comparison_kind") == "oracle" else "fixed_perturbation_seed",
            }
            score_cols = [c for c in bio.columns if c.endswith("_score01") or c in FAMILY_COLS or c in {"FINAL_COMPOSITE_SCORE", "FIDELITY_SCORE"}]
            rows.append({**base, **{c: r.get(c, np.nan) for c in score_cols}})

    scaling = _read_csv(SCALING_WIDE)
    if not scaling.empty:
        sub = scaling[
            (scaling["sequence_length"] == 90)
            & (scaling["brain_areas"] == 16)
            & (scaling["data_share"] == 100)
            & (scaling["training_percent"].isin([10.0, 30.0, 100.0]))
        ].copy()
        for _, r in sub.iterrows():
            base = {
                "analysis": "training_progress",
                "condition": f"{int(r['training_percent'])}pct",
                "model": str(r.get("model_family", "netho-seq")),
                "seed": int(r.get("seed", -1)),
                "rollout_horizon": np.nan,
                "perturbation": "none",
                "perturbation_magnitude": 0.0,
                "replicate_type": "training_seed",
            }
            score_cols = [c for c in sub.columns if c.endswith("_score01") or c in FAMILY_COLS or c in {"FINAL_COMPOSITE_SCORE", "FIDELITY_SCORE", "family_fidelity"}]
            rows.append({**base, **{c: r.get(c, np.nan) for c in score_cols}})

    trained_seed_rows = _load_trained_seed_rows()
    use_trained_biophysical = _has_min_trained_seed_coverage(trained_seed_rows, "biophysical_model_comparison")
    use_trained_transfer = _has_min_trained_seed_coverage(trained_seed_rows, "transfer_calciumgan")
    rows.extend(trained_seed_rows)

    fixed_runs = [
        ("biophysical_model_comparison", "converged", BIOPHYS_ROOT / "converged" / "family_comparison_rollout.csv", BIOPHYS_ROOT / "converged" / "fidelity_comparison_rollout.csv"),
        ("biophysical_model_comparison", "weakest", BIOPHYS_ROOT / "weakest" / "family_comparison_rollout.csv", BIOPHYS_ROOT / "weakest" / "fidelity_comparison_rollout.csv"),
        ("transfer_calciumgan", "converged", TRANSFER_ROOT / "transfer_calciumgan_converged" / "family_comparison_rollout.csv", TRANSFER_ROOT / "transfer_calciumgan_converged" / "fidelity_comparison_rollout.csv"),
        ("transfer_calciumgan", "weakest", TRANSFER_ROOT / "transfer_calciumgan_weakest" / "family_comparison_rollout.csv", TRANSFER_ROOT / "transfer_calciumgan_weakest" / "fidelity_comparison_rollout.csv"),
    ]
    for analysis, model, fam_path, fid_path in fixed_runs:
        if analysis == "biophysical_model_comparison" and use_trained_biophysical:
            continue
        if analysis == "transfer_calciumgan" and use_trained_transfer:
            continue
        fam = _read_csv(fam_path)
        fid = _read_csv(fid_path)
        if fam.empty:
            continue
        row = {
            "analysis": analysis,
            "condition": model,
            "model": model,
            "seed": 0,
            "rollout_horizon": 1024,
            "perturbation": "none",
            "perturbation_magnitude": 0.0,
            "replicate_type": "fixed_run",
        }
        for _, fr in fam.iterrows():
            key = DISPLAY_FAMILY_TO_COL.get(str(fr.get("family", "")), f"family_{fr.get('family', '')}")
            row[key] = float(fr.get("model_score", np.nan))
        row["FINAL_COMPOSITE_SCORE"] = _weighted_composite(pd.Series(row), DEFAULT_WEIGHTS)
        if not fid.empty:
            vals = pd.to_numeric(fid.get("model_score", pd.Series(dtype=float)), errors="coerce").dropna()
            row["FIDELITY_SCORE"] = float(vals.mean()) if len(vals) else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "fig2_multiseed_scores_long.csv", index=False)
    return out


def build_weight_ablation() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _read_csv(MODEL_FAMILY_SCORES)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = df.rename(columns={"temporal_trjdist_only": "family_temporal_spectral", "state_dynamics": "family_state_dynamics"})
    df["family_distribution"] = df["distribution"]
    df["family_relational"] = df["relational"]
    df["family_geometry"] = df["geometry"]
    df["condition_id"] = df["model"].astype(str) + "_h" + df["horizon"].astype(str)
    df["default_composite"] = df.apply(lambda r: _weighted_composite(r, DEFAULT_WEIGHTS), axis=1)
    default_rank = df.set_index("condition_id")["default_composite"].rank(ascending=False, method="min")

    rng = np.random.default_rng(20260501)
    concentration = 80.0
    random_weights = rng.dirichlet(DEFAULT_WEIGHTS * concentration, size=10000)
    deterministic = [DEFAULT_WEIGHTS, np.ones_like(DEFAULT_WEIGHTS) / len(DEFAULT_WEIGHTS)]
    for i in range(len(DEFAULT_WEIGHTS)):
        w = DEFAULT_WEIGHTS.copy()
        w[i] = 0.0
        w = w / w.sum()
        deterministic.append(w)
        w = DEFAULT_WEIGHTS.copy()
        w[i] *= 2.0
        w = w / w.sum()
        deterministic.append(w)
        for delta in (-0.25, 0.25):
            w = DEFAULT_WEIGHTS.copy()
            w[i] *= 1.0 + delta
            w = w / w.sum()
            deterministic.append(w)
    weights = np.vstack([random_weights, np.asarray(deterministic)])

    rows = []
    for wi, w in enumerate(weights):
        scores = df[FAMILY_COLS].to_numpy(float) @ w
        ranks = pd.Series(scores, index=df["condition_id"]).rank(ascending=False, method="min")
        rho = spearmanr(default_rank.loc[ranks.index], ranks).correlation
        top = df.loc[int(np.nanargmax(scores)), "condition_id"]
        for cid, score, rank in zip(df["condition_id"], scores, ranks):
            rows.append(
                {
                    "weight_sample": wi,
                    "weight_type": "random_dirichlet" if wi < len(random_weights) else "deterministic_stress",
                    "condition_id": cid,
                    "model": df.loc[df["condition_id"] == cid, "model"].iloc[0],
                    "horizon": int(df.loc[df["condition_id"] == cid, "horizon"].iloc[0]),
                    "score": float(score),
                    "rank": float(rank),
                    "default_rank": float(default_rank[cid]),
                    "spearman_to_default": float(rho) if np.isfinite(rho) else np.nan,
                    "is_top": cid == top,
                    **{f"w_{FAMILY_SHORT[c]}": float(v) for c, v in zip(FAMILY_COLS, w)},
                }
            )
    ablation = pd.DataFrame(rows)
    rank_summary = (
        ablation.groupby(["condition_id", "model", "horizon"], as_index=False)
        .agg(
            mean_rank=("rank", "mean"),
            rank_q05=("rank", lambda x: float(np.quantile(x, 0.05))),
            rank_q95=("rank", lambda x: float(np.quantile(x, 0.95))),
            top_probability=("is_top", "mean"),
            default_rank=("default_rank", "first"),
        )
        .sort_values(["horizon", "mean_rank"])
    )
    ablation.to_csv(TABLE_DIR / "fig2_weight_ablation_results.csv", index=False)
    rank_summary.to_csv(TABLE_DIR / "fig2_weight_ablation_rank_summary.csv", index=False)
    return ablation, rank_summary


def build_correlation_and_redundancy(master: pd.DataFrame, ablation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        c
        for c in master.columns
        if (c.endswith("_score01") or c in FAMILY_COLS or c in {"FINAL_COMPOSITE_SCORE", "FIDELITY_SCORE", "family_fidelity"})
        and pd.to_numeric(master[c], errors="coerce").notna().sum() >= 3
    ]
    data = master[metric_cols].apply(pd.to_numeric, errors="coerce")
    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")
    pearson.to_csv(TABLE_DIR / "fig2_family_correlation_results.csv")
    spearman.to_csv(TABLE_DIR / "fig2_metric_spearman_correlation_results.csv")

    rows = []
    # Leave-one-family-out rank shifts from model-family ablation source.
    src = _read_csv(MODEL_FAMILY_SCORES)
    if not src.empty:
        src = src.rename(columns={"temporal_trjdist_only": "family_temporal_spectral", "state_dynamics": "family_state_dynamics"})
        src["family_distribution"] = src["distribution"]
        src["family_relational"] = src["relational"]
        src["family_geometry"] = src["geometry"]
        src["condition_id"] = src["model"].astype(str) + "_h" + src["horizon"].astype(str)
        default_scores = src.apply(lambda r: _weighted_composite(r, DEFAULT_WEIGHTS), axis=1)
        default_ranks = pd.Series(default_scores.to_numpy(), index=src["condition_id"]).rank(ascending=False, method="min")
        for omit_idx, omit_col in enumerate(FAMILY_COLS):
            w = DEFAULT_WEIGHTS.copy()
            w[omit_idx] = 0.0
            w = w / w.sum()
            scores = src[FAMILY_COLS].to_numpy(float) @ w
            ranks = pd.Series(scores, index=src["condition_id"]).rank(ascending=False, method="min")
            for cid in src["condition_id"]:
                rows.append(
                    {
                        "analysis": "leave_one_family_out",
                        "condition_id": cid,
                        "model": src.loc[src["condition_id"] == cid, "model"].iloc[0],
                        "horizon": int(src.loc[src["condition_id"] == cid, "horizon"].iloc[0]),
                        "omitted_family": FAMILY_SHORT[omit_col],
                        "default_rank": float(default_ranks[cid]),
                        "ablated_rank": float(ranks[cid]),
                        "rank_shift_abs": float(abs(ranks[cid] - default_ranks[cid])),
                    }
                )

    # Perturbation selectivity from synthetic and biophysical perturbation tables.
    for source_name, df in [("synthetic_ar1", _read_csv(SYNTHETIC_SCORES)), ("biophysical_oracle", _read_csv(BIOPHYSICAL_SCORES))]:
        if df.empty:
            continue
        base = df[(df["comparison_kind"] == "oracle") | ((df["comparison_kind"] == "perturbation") & (df["perturbation_level"] == 0.0))]
        for perturb, sub in df[df["comparison_kind"] == "perturbation"].groupby("perturbation_name"):
            strongest = sub.sort_values("perturbation_level").tail(1)
            if strongest.empty:
                continue
            b = base.iloc[0]
            s = strongest.iloc[0]
            drops = {}
            for col in FAMILY_COLS:
                if col in df.columns:
                    drops[col] = max(float(b.get(col, np.nan)) - float(s.get(col, np.nan)), 0.0)
            if not drops:
                continue
            max_family = max(drops, key=drops.get)
            other = [v for k, v in drops.items() if k != max_family and np.isfinite(v)]
            selectivity = drops[max_family] / (float(np.mean(other)) + 1e-12) if other else np.nan
            rows.append(
                {
                    "analysis": "perturbation_selectivity",
                    "source": source_name,
                    "perturbation": perturb,
                    "target_family": str(s.get("target_family", "")),
                    "max_drop_family": FAMILY_SHORT.get(max_family, max_family),
                    "max_drop_abs": float(drops[max_family]),
                    "selectivity_index": float(selectivity),
                    **{f"drop_{FAMILY_SHORT[k]}": float(v) for k, v in drops.items()},
                }
            )

    redundancy = pd.DataFrame(rows)
    redundancy.to_csv(TABLE_DIR / "fig2_metric_redundancy_results.csv", index=False)
    return pearson, redundancy


def plot_weight_robustness(rank_summary: pd.DataFrame, ablation: pd.DataFrame) -> None:
    if rank_summary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ax = axes[0]
    sub = rank_summary.sort_values(["horizon", "mean_rank"])
    y = np.arange(len(sub))
    xerr_low = np.maximum(sub["mean_rank"].to_numpy(float) - sub["rank_q05"].to_numpy(float), 0.0)
    xerr_high = np.maximum(sub["rank_q95"].to_numpy(float) - sub["mean_rank"].to_numpy(float), 0.0)
    ax.errorbar(sub["mean_rank"], y, xerr=[xerr_low, xerr_high], fmt="o", color="#355C9A")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["condition_id"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Rank under weight perturbations")
    ax.set_title("Composite rank stability")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    top = rank_summary.sort_values("top_probability", ascending=False).head(10)
    ax.barh(top["condition_id"], top["top_probability"], color="#6BAA75")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability of top rank")
    ax.set_title("Top-model probability")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save_fig(fig, "supp_fig_composite_weight_robustness")


def plot_correlations(corr: pd.DataFrame, master: pd.DataFrame) -> None:
    if corr.empty:
        return
    family_plus = [c for c in FAMILY_COLS + ["FINAL_COMPOSITE_SCORE", "FIDELITY_SCORE", "family_fidelity"] if c in corr.columns]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    mat = corr.loc[family_plus, family_plus]
    im = axes[0].imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    axes[0].set_xticks(range(len(family_plus)))
    axes[0].set_yticks(range(len(family_plus)))
    axes[0].set_xticklabels([FAMILY_SHORT.get(x, x) for x in family_plus], rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels([FAMILY_SHORT.get(x, x) for x in family_plus], fontsize=8)
    axes[0].set_title("Family/fidelity correlations")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    numeric = master[[c for c in master.columns if c.endswith("_score01") and pd.to_numeric(master[c], errors="coerce").notna().sum() >= 3]].apply(pd.to_numeric, errors="coerce")
    metric_corr = numeric.corr(method="spearman").fillna(0.0)
    labels = list(metric_corr.columns)
    if len(labels) >= 2:
        dist = 1.0 - np.abs(metric_corr.to_numpy())
        np.fill_diagonal(dist, 0.0)
        link = linkage(squareform(dist, checks=False), method="average")
        dendrogram(link, labels=labels, leaf_rotation=90, leaf_font_size=6, ax=axes[1], color_threshold=0.7)
        axes[1].set_title("Metric clustering by Spearman |rho|")
    else:
        axes[1].axis("off")
    fig.tight_layout()
    _save_fig(fig, "supp_fig_metric_family_correlations")


def plot_nonredundancy(redundancy: pd.DataFrame) -> None:
    if redundancy.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    loo = redundancy[redundancy["analysis"] == "leave_one_family_out"]
    if not loo.empty:
        piv = loo.pivot_table(index="condition_id", columns="omitted_family", values="rank_shift_abs", aggfunc="mean").fillna(0.0)
        im = axes[0].imshow(piv, cmap="YlOrRd")
        axes[0].set_xticks(range(piv.shape[1]))
        axes[0].set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8)
        axes[0].set_yticks(range(piv.shape[0]))
        axes[0].set_yticklabels(piv.index, fontsize=7)
        axes[0].set_title("Leave-one-family-out rank shifts")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    else:
        axes[0].axis("off")

    sel = redundancy[redundancy["analysis"] == "perturbation_selectivity"]
    if not sel.empty:
        drop_cols = [c for c in sel.columns if c.startswith("drop_")]
        piv = sel.set_index("perturbation")[drop_cols].fillna(0.0)
        im = axes[1].imshow(piv, cmap="Blues")
        axes[1].set_xticks(range(piv.shape[1]))
        axes[1].set_xticklabels([c.replace("drop_", "") for c in piv.columns], rotation=45, ha="right", fontsize=8)
        axes[1].set_yticks(range(piv.shape[0]))
        axes[1].set_yticklabels(piv.index, fontsize=6)
        axes[1].set_title("Perturbation selectivity")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].axis("off")
    fig.tight_layout()
    _save_fig(fig, "supp_fig_nonredundancy_ablations")


def _preferred_model_rows(master: pd.DataFrame, analysis: str) -> pd.DataFrame:
    sub = master[master["analysis"] == analysis].copy()
    if sub.empty:
        return sub
    for replicate_type in ["trained_seed_full", "trained_seed_fast"]:
        trained = sub[sub["replicate_type"] == replicate_type].copy()
        if trained.empty:
            continue
        counts = trained.groupby("condition")["seed"].nunique()
        if all(counts.get(model, 0) >= TRAINED_SEED_MIN_N for model in ["converged", "weakest"]):
            return trained
    return sub[sub["replicate_type"] == "fixed_run"].copy()


def _model_sem_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_cols: list[str],
    title: str,
    ylabel: str,
    show_legend: bool = False,
) -> None:
    if df.empty:
        ax.axis("off")
        return
    conditions = ["converged", "weakest"]
    x = np.arange(len(value_cols), dtype=float)
    offsets = {"converged": -0.18, "weakest": 0.18}
    rng = np.random.default_rng(20260505)
    for cond in conditions:
        sub = df[df["condition"] == cond].copy()
        color = COLORS["Converged"] if cond == "converged" else COLORS["Weakest"]
        means = []
        sems = []
        for col in value_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(float) if col in sub else np.array([])
            means.append(float(np.mean(vals)) if vals.size else np.nan)
            sems.append(float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0)
            if vals.size:
                jitter = rng.normal(0.0, 0.025, size=vals.size)
                ax.scatter(
                    np.full(vals.size, x[value_cols.index(col)] + offsets[cond]) + jitter,
                    vals,
                    s=18,
                    color=color,
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.78,
                    zorder=3,
                )
        ax.errorbar(
            x + offsets[cond],
            means,
            yerr=sems,
            fmt="o",
            markersize=5,
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=3,
            label=cond.capitalize(),
            zorder=4,
        )

    # Pair seeds within each metric where both model conditions exist.
    for col_idx, col in enumerate(value_cols):
        wide = df.pivot_table(index="seed", columns="condition", values=col, aggfunc="mean")
        if {"converged", "weakest"}.issubset(wide.columns):
            for _, row in wide.dropna(subset=["converged", "weakest"]).iterrows():
                ax.plot(
                    [x[col_idx] + offsets["converged"], x[col_idx] + offsets["weakest"]],
                    [row["converged"], row["weakest"]],
                    color="#B8B8B8",
                    linewidth=0.7,
                    alpha=0.35,
                    zorder=1,
                )

    labels = [
        FAMILY_SHORT.get(col, col)
        .replace("family_", "")
        .replace("FINAL_COMPOSITE_SCORE", "Composite")
        .replace("FIDELITY_SCORE", "Fidelity")
        .replace("Error_score01", "Error")
        .replace("MI_score01", "MI")
        for col in value_cols
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.03)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    if show_legend:
        ax.legend(frameon=False, fontsize=8, loc="upper right")


def plot_trained_seed_model_validation(master: pd.DataFrame) -> None:
    bio = _preferred_model_rows(master, "biophysical_model_comparison")
    transfer = _preferred_model_rows(master, "transfer_calciumgan")
    if bio.empty and transfer.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), sharey=True)
    family_values = FAMILY_COLS + ["FINAL_COMPOSITE_SCORE"]
    fidelity_values = [c for c in ["FIDELITY_SCORE", "Error_score01", "MI_score01"] if c in bio.columns]
    if not fidelity_values:
        fidelity_values = ["FIDELITY_SCORE"]

    def _replicate_label(df: pd.DataFrame) -> str:
        if df.empty:
            return "fixed-run"
        reps = set(df["replicate_type"].dropna().astype(str))
        if "trained_seed_full" in reps:
            return "trained-seed full"
        if "trained_seed_fast" in reps:
            return "trained-seed fast"
        return "fixed-run"

    bio_type = _replicate_label(bio)
    transfer_type = _replicate_label(transfer)
    bio_n = bio.groupby("condition")["seed"].nunique().to_dict() if not bio.empty else {}
    transfer_n = transfer.groupby("condition")["seed"].nunique().to_dict() if not transfer.empty else {}

    _model_sem_panel(
        axes[0],
        bio,
        family_values,
        f"Biophysical rollout ({bio_type}; n={min(bio_n.values()) if bio_n else 0})",
        "Model / oracle score",
        show_legend=True,
    )
    _model_sem_panel(
        axes[1],
        bio,
        fidelity_values,
        f"Fidelity sidecar ({bio_type})",
        "Score",
        show_legend=False,
    )
    _model_sem_panel(
        axes[2],
        transfer,
        family_values,
        f"CalciumGAN transfer ({transfer_type}; n={min(transfer_n.values()) if transfer_n else 0})",
        "Model / oracle score",
        show_legend=False,
    )
    title_prefix = "Full" if any("full" in label for label in [bio_type, transfer_type]) else "Fast"
    fig.suptitle(f"{title_prefix} multiseed trained-seed validation for model quality and transfer", fontsize=12, weight="bold")
    fig.tight_layout()
    _save_fig(fig, "fig2_model_multiseed_validation")


def write_manifest(master: pd.DataFrame, rank_summary: pd.DataFrame) -> None:
    manifest = {
        "generated_tables": [
            str(TABLE_DIR / "fig2_multiseed_scores_long.csv"),
            str(TABLE_DIR / "fig2_weight_ablation_results.csv"),
            str(TABLE_DIR / "fig2_family_correlation_results.csv"),
            str(TABLE_DIR / "fig2_metric_redundancy_results.csv"),
        ],
        "generated_figures": [
            str(FIG_DIR / "fig2_model_multiseed_validation.pdf"),
            str(FIG_DIR / "supp_fig_composite_weight_robustness.pdf"),
            str(FIG_DIR / "supp_fig_metric_family_correlations.pdf"),
            str(FIG_DIR / "supp_fig_nonredundancy_ablations.pdf"),
        ],
        "source_tables": {
            "synthetic_scores": str(SYNTHETIC_SCORES),
            "synthetic_score_candidates": [str(p) for p in SYNTHETIC_SCORE_CANDIDATES],
            "biophysical_scores": str(BIOPHYSICAL_SCORES),
            "biophysical_score_candidates": [str(p) for p in BIOPHYSICAL_SCORE_CANDIDATES],
            "training_progress": str(SCALING_WIDE),
            "model_family_scores": str(MODEL_FAMILY_SCORES),
            "biophysical_fixed_runs": str(BIOPHYS_ROOT),
            "transfer_fixed_runs": str(TRANSFER_ROOT),
            "trained_seed_full_root": str(TRAINED_SEED_FULL_ROOT),
            "trained_seed_fast_root": str(TRAINED_SEED_FAST_ROOT),
        },
        "replicate_counts": master.groupby(["analysis", "replicate_type"], dropna=False)["seed"].nunique().reset_index().to_dict("records")
        if not master.empty
        else [],
        "weight_ablation": {
            "n_weight_samples": int(rank_summary.shape[0]) if not rank_summary.empty else 0,
            "default_weights": {FAMILY_SHORT[c]: float(w) for c, w in zip(FAMILY_COLS, DEFAULT_WEIGHTS)},
        },
        "note": (
            "Biophysical model and CalciumGAN transfer comparisons prefer trained_seed_full rows, then trained_seed_fast rows, "
            f"when both converged and weakest have at least {TRAINED_SEED_MIN_N} completed seeds; otherwise they fall back to fixed-run rows."
        ),
    }
    (TABLE_DIR / "fig2_robustness_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> int:
    _ensure_dirs()
    master = build_multiseed_scores_long()
    ablation, rank_summary = build_weight_ablation()
    corr, redundancy = build_correlation_and_redundancy(master, ablation)
    plot_weight_robustness(rank_summary, ablation)
    plot_correlations(corr, master)
    plot_nonredundancy(redundancy)
    plot_trained_seed_model_validation(master)
    write_manifest(master, rank_summary)
    print(f"Wrote Figure 2 robustness tables to {TABLE_DIR}")
    print(f"Wrote supplemental figures to {FIG_DIR}")
    if not master.empty:
        print(master.groupby(['analysis', 'replicate_type'])['seed'].nunique().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
