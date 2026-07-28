from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "paper" / "Nethobench"
TABLE_DIR = PAPER_ROOT / "generated_tables"
FIG_DIR = PAPER_ROOT / "figures"
TEXT_DIR = PAPER_ROOT / "generated_text"

FAMILY_COLS = [
    "family_distribution",
    "family_temporal_spectral",
    "family_relational",
    "family_geometry",
    "family_state_dynamics",
]

FAMILY_LABELS = {
    "family_distribution": "Distribution",
    "family_temporal_spectral": "Temporal",
    "family_relational": "Relational",
    "family_geometry": "Geometry",
    "family_state_dynamics": "State dynamics",
}

SOURCE_COLORS = {
    "current": "#4C78A8",
    "legacy": "#72B7B2",
    "candidate": "#54A24B",
    "general_alternative": "#E45756",
    "zapbench_style": "#F58518",
    "sensorium_style": "#B279A2",
}

DECISION_COLORS = {
    "selected": "#4C78A8",
    "external_predictive_sidecar": "#F58518",
    "fidelity_sidecar_not_realism_composite": "#B279A2",
    "narrow_or_secondary_descriptor": "#9D755D",
    "weak_targeted_selectivity": "#E45756",
}

VARIANT_GROUP_LABELS = {
    "family_weights": "Family weights",
    "submetric_weights": "Submetric weights",
    "distribution_divergence": "Distribution divergence",
    "raw_hyperparameters": "PCA dimensions",
    "state_hyperparameters": "State dynamics hyperparameters",
    "topology": "Topology inclusion",
    "raw_default": "Default",
}


def _read(name: str, **kwargs) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    svg = FIG_DIR / f"{stem}.svg"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return svg, pdf


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.08, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def _short_metric(name: str, max_len: int = 30) -> str:
    name = str(name).replace("_score01", "").replace("LatentState", "LS").replace("Transition", "Trans")
    name = name.replace("CrossRegion", "CrossReg").replace("LaggedCovariance", "LagCov")
    name = name.replace("ImpulseResponse", "Impulse").replace("SubspaceAngle", "Subspace")
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.18)


def _criteria_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "oracle_ceiling",
        "corruption_selectivity_norm",
        "model_utility",
        "nonredundancy",
        "coverage",
        "interpretability",
    ]
    return [c for c in cols if c in df.columns]


def build_metric_selection_story() -> dict[str, object]:
    scores = _read("supp_metric_selection_scores.csv")
    registry = _read("supp_metric_candidate_registry.csv")
    external = _read("supp_external_benchmark_metrics.csv")
    discard = _read("supp_metric_discard_reasons.csv")
    if scores.empty:
        raise FileNotFoundError("supp_metric_selection_scores.csv is required")

    criteria = _criteria_columns(scores)
    scores = scores.copy()
    scores["selected"] = scores["selected"].astype(bool)
    scores["selection_order"] = np.where(scores["selected"], 0, 1)
    scores = scores.sort_values(["selection_order", "source", "overall_selection_score"], ascending=[True, True, False])

    fig = plt.figure(figsize=(18.2, 14.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.92, 1.22, 1.05], width_ratios=[1.0, 1.25, 1.0], hspace=0.42, wspace=0.34)

    # A. Candidate inventory.
    ax = fig.add_subplot(gs[0, 0])
    decision_counts = registry["decision_class"].value_counts().reindex(DECISION_COLORS.keys()).dropna()
    colors = [DECISION_COLORS.get(k, "#999999") for k in decision_counts.index]
    ax.barh([str(x).replace("_", "\n") for x in decision_counts.index], decision_counts.values, color=colors, alpha=0.88)
    ax.set_xlabel("Number of candidate metrics")
    ax.set_title("Candidate inventory and final decisions", fontsize=11, weight="bold")
    _panel_label(ax, "a")
    _clean_axis(ax)

    # B. Selection criteria matrix.
    ax = fig.add_subplot(gs[:, 1])
    mat_df = scores[["metric", "source", "selected", "decision_class", *criteria]].copy()
    mat = mat_df[criteria].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(criteria)))
    ax.set_xticklabels(
        ["Oracle\nceiling", "Targeted\nselectivity", "Model\nutility", "Non-\nredundancy", "Coverage", "Interpret-\nability"][: len(criteria)],
        rotation=0,
        fontsize=9,
    )
    ylabels = [
        ("✓ " if sel else "  ") + _short_metric(metric, 34)
        for metric, sel in zip(mat_df["metric"], mat_df["selected"])
    ]
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=6.7)
    ax.set_title("Each metric was scored against explicit retention criteria", fontsize=11, weight="bold")
    ax.axhline(mat_df["selected"].sum() - 0.5, color="white", lw=2.0)
    for i, source in enumerate(mat_df["source"]):
        ax.add_patch(plt.Rectangle((-0.95, i - 0.48), 0.25, 0.96, color=SOURCE_COLORS.get(source, "#999999"), clip_on=False))
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cbar.set_label("Criterion score", fontsize=8)
    _panel_label(ax, "b")

    # C. Frontier.
    ax = fig.add_subplot(gs[0, 2])
    for source, sub in scores.groupby("source"):
        size = 40 + 130 * pd.to_numeric(sub["corruption_selectivity_norm"], errors="coerce").fillna(0.0)
        alpha = np.where(sub["selected"], 0.96, 0.46)
        ax.scatter(
            sub["oracle_ceiling"],
            sub["model_utility"],
            s=size,
            c=SOURCE_COLORS.get(source, "#999999"),
            alpha=alpha,
            edgecolor="white",
            linewidth=0.7,
            label=source.replace("_", " "),
        )
    label_metrics = scores[scores["selected"] | scores["source"].isin(["zapbench_style", "sensorium_style"])].copy()
    for _, r in label_metrics.iterrows():
        ax.text(r["oracle_ceiling"] + 0.01, r["model_utility"] + 0.006, _short_metric(r["metric"], 16), fontsize=6.5)
    ax.axvline(0.70, color="#888888", ls="--", lw=1)
    ax.axhline(0.50, color="#888888", ls="--", lw=1)
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Stochastic-oracle ceiling")
    ax.set_ylabel("Model-utility score")
    ax.set_title("Metric-selection frontier", fontsize=11, weight="bold")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _panel_label(ax, "c")
    _clean_axis(ax)

    # D. Mean criterion profiles by source.
    ax = fig.add_subplot(gs[1, 0])
    profile = scores.groupby("source")[criteria].mean(numeric_only=True)
    source_order = ["current", "legacy", "candidate", "general_alternative", "zapbench_style", "sensorium_style"]
    profile = profile.reindex([s for s in source_order if s in profile.index])
    im = ax.imshow(profile.to_numpy(float), aspect="auto", cmap="mako" if "mako" in plt.colormaps() else "YlGnBu", vmin=0, vmax=1)
    ax.set_yticks(np.arange(profile.shape[0]))
    ax.set_yticklabels([s.replace("_", " ") for s in profile.index], fontsize=8)
    ax.set_xticks(np.arange(len(criteria)))
    ax.set_xticklabels([c.replace("_", "\n") for c in criteria], fontsize=7, rotation=0)
    ax.set_title("Current metrics outperform single-axis alternatives across criteria", fontsize=11, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _panel_label(ax, "d")

    # E. External benchmark styles are useful sidecars but weak realism composites.
    ax = fig.add_subplot(gs[1, 2])
    ext_scores = scores[scores["source"].isin(["current", "zapbench_style", "sensorium_style", "general_alternative"])].copy()
    ext_scores["source_label"] = ext_scores["source"].replace(
        {
            "current": "Nethobench\nselected",
            "zapbench_style": "ZAPBench-\nstyle MAE",
            "sensorium_style": "Sensorium-\nstyle corr.",
            "general_alternative": "Generic\nalternatives",
        }
    )
    ext_profile = ext_scores.groupby("source_label")[["oracle_ceiling", "corruption_selectivity_norm", "model_utility"]].mean()
    ext_profile = ext_profile.reindex(["Nethobench\nselected", "ZAPBench-\nstyle MAE", "Sensorium-\nstyle corr.", "Generic\nalternatives"]).dropna(how="all")
    x = np.arange(ext_profile.shape[0])
    width = 0.24
    for j, col in enumerate(ext_profile.columns):
        ax.bar(x + (j - 1) * width, ext_profile[col], width=width, label=col.replace("_", " "), alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(ext_profile.index, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean criterion score")
    ax.set_title("Predictive metrics are retained as sidecars, not realism composites", fontsize=11, weight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    _panel_label(ax, "e")
    _clean_axis(ax)

    # F. Pass-rate by source.
    ax = fig.add_subplot(gs[2, 0])
    pass_cols = [c for c in scores.columns if c.startswith("pass_")]
    pass_rate = scores.groupby("source")[pass_cols].mean(numeric_only=True)
    pass_rate["mean_pass_rate"] = pass_rate.mean(axis=1)
    pass_rate = pass_rate.reindex([s for s in source_order if s in pass_rate.index]).dropna(how="all")
    ax.barh(
        [s.replace("_", " ") for s in pass_rate.index],
        pass_rate["mean_pass_rate"],
        color=[SOURCE_COLORS.get(s, "#999999") for s in pass_rate.index],
        alpha=0.88,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean fraction of criteria passed")
    ax.set_title("Final set is not arbitrary: it passes more retention tests", fontsize=11, weight="bold")
    _panel_label(ax, "f")
    _clean_axis(ax)

    # G. Redundancy and overall selection score.
    ax = fig.add_subplot(gs[2, 2])
    x = scores["redundancy_max_abs_rho"].astype(float)
    y = scores["overall_selection_score"].astype(float)
    for source, sub in scores.groupby("source"):
        ax.scatter(
            sub["redundancy_max_abs_rho"],
            sub["overall_selection_score"],
            s=70,
            c=SOURCE_COLORS.get(source, "#999999"),
            alpha=np.where(sub["selected"], 0.95, 0.48),
            edgecolor="white",
            linewidth=0.7,
        )
    for _, r in scores[scores["selected"]].iterrows():
        ax.text(r["redundancy_max_abs_rho"] + 0.006, r["overall_selection_score"] + 0.004, _short_metric(r["metric"], 14), fontsize=6.4)
    ax.set_xlabel("Maximum absolute correlation with another metric")
    ax.set_ylabel("Overall retention score")
    ax.set_title("Selected metrics balance utility with partial nonredundancy", fontsize=11, weight="bold")
    _panel_label(ax, "g")
    _clean_axis(ax)

    fig.suptitle("Why these metrics? Empirical metric-selection rationale", fontsize=17, weight="bold", y=0.995)
    svg, pdf = _save(fig, "supp_big_metric_selection_rationale")

    summary = {
        "n_candidates": int(len(scores)),
        "n_selected": int(scores["selected"].sum()),
        "selected_metrics": scores.loc[scores["selected"], "metric"].tolist(),
        "mean_current_criteria": scores[scores["source"] == "current"][criteria].mean(numeric_only=True).to_dict(),
        "external_sources": sorted(external["source"].dropna().unique().tolist()) if not external.empty else [],
        "figure_svg": str(svg),
        "figure_pdf": str(pdf),
    }
    (TABLE_DIR / "supp_big_metric_selection_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_metric_robustness_story() -> dict[str, object]:
    rank = _read("supp_metric_robustness_rank_summary.csv")
    variants = _read("supp_metric_robustness_variants_long.csv")
    raw = _read("supp_metric_raw_hyperparameter_variants.csv")
    family_corr = _read("supp_family_correlation_matrix.csv", index_col=0)
    metric_corr = _read("supp_metric_correlation_matrix.csv", index_col=0)
    redundancy = _read("supp_metric_redundancy_summary.csv")
    fig2_red = _read("fig2_metric_redundancy_results.csv")
    if rank.empty or variants.empty:
        raise FileNotFoundError("robustness rank/variant tables are required")

    fig = plt.figure(figsize=(18.4, 14.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.9, 1.06, 1.02], width_ratios=[1.0, 1.0, 1.12], hspace=0.42, wspace=0.34)

    # A. Rank correlation by variant group.
    ax = fig.add_subplot(gs[0, 0])
    plot_rank = rank[rank["variant"] != "default_weights"].copy()
    group_order = [
        "family_weights",
        "submetric_weights",
        "distribution_divergence",
        "raw_hyperparameters",
        "state_hyperparameters",
        "topology",
    ]
    box_data = [plot_rank.loc[plot_rank["variant_group"] == g, "spearman_to_default"].dropna().to_numpy(float) for g in group_order]
    labels = [VARIANT_GROUP_LABELS[g] for g, vals in zip(group_order, box_data) if len(vals)]
    box_data = [vals for vals in box_data if len(vals)]
    ax.boxplot(box_data, vert=False, patch_artist=True, boxprops={"facecolor": "#D7E5F0", "edgecolor": "#4C78A8"}, medianprops={"color": "#17324D", "lw": 1.6})
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Spearman rank correlation vs default")
    ax.set_title("Model rankings are stable under benchmark variants", fontsize=11, weight="bold")
    _panel_label(ax, "a")
    _clean_axis(ax)

    # B. Variant-level score shifts.
    ax = fig.add_subplot(gs[0, 1])
    agg = plot_rank.groupby("variant_group").agg(
        mean_delta=("mean_abs_score_delta", "mean"),
        max_delta=("max_abs_score_delta", "max"),
        mean_rho=("spearman_to_default", "mean"),
    ).reindex(group_order).dropna(how="all")
    y = np.arange(len(agg))
    ax.barh(y - 0.17, agg["mean_delta"], height=0.32, label="Mean |score shift|", color="#9ECAE1")
    ax.barh(y + 0.17, agg["max_delta"], height=0.32, label="Max |score shift|", color="#FB9A99")
    ax.set_yticks(y)
    ax.set_yticklabels([VARIANT_GROUP_LABELS.get(g, g) for g in agg.index], fontsize=8)
    ax.set_xlabel("Absolute composite-score change")
    ax.set_title("Scores move, but rank order usually persists", fontsize=11, weight="bold")
    ax.legend(frameon=False, fontsize=7)
    _panel_label(ax, "b")
    _clean_axis(ax)

    # C. Raw hyperparameter variants.
    ax = fig.add_subplot(gs[0, 2])
    raw_rank = raw[raw["variant"] != "default_weights"].groupby(["variant_group", "variant"], as_index=False).agg(
        rho=("spearman_to_default", "mean"),
        mean_delta=("score", lambda x: np.nan),
    )
    # Use the robust summary table when available for score deltas.
    rr = rank[rank["variant"].isin(raw_rank["variant"])]
    raw_rank = raw_rank.merge(rr[["variant", "mean_abs_score_delta"]], on="variant", how="left")
    raw_rank = raw_rank.sort_values(["variant_group", "rho"], ascending=[True, False])
    colors = [SOURCE_COLORS.get("current", "#4C78A8") if "pca" in v else "#54A24B" if "state" in v or "transition" in v else "#F58518" for v in raw_rank["variant"]]
    ax.scatter(raw_rank["rho"], raw_rank["mean_abs_score_delta"], s=80, c=colors, alpha=0.86, edgecolor="white", linewidth=0.7)
    for _, r in raw_rank.iterrows():
        ax.text(r["rho"] + 0.004, r["mean_abs_score_delta"] + 0.001, r["variant"].replace("_", " "), fontsize=6.5)
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("Rank correlation vs default")
    ax.set_ylabel("Mean |score shift|")
    ax.set_title("Raw hyperparameters change scores more than conclusions", fontsize=11, weight="bold")
    _panel_label(ax, "c")
    _clean_axis(ax)

    # D. Family correlation heatmap.
    ax = fig.add_subplot(gs[1, 0])
    fam_cols = [c for c in FAMILY_COLS + ["FINAL_COMPOSITE_SCORE", "FIDELITY_SCORE"] if c in family_corr.index and c in family_corr.columns]
    fam_mat = family_corr.loc[fam_cols, fam_cols].to_numpy(float)
    im = ax.imshow(fam_mat, vmin=-1, vmax=1, cmap="coolwarm")
    fam_labels = [FAMILY_LABELS.get(c, c.replace("_", " ")) for c in fam_cols]
    ax.set_xticks(np.arange(len(fam_labels)))
    ax.set_yticks(np.arange(len(fam_labels)))
    ax.set_xticklabels(fam_labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(fam_labels, fontsize=7.5)
    ax.set_title("Families covary but fidelity is not a sufficient proxy", fontsize=11, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _panel_label(ax, "d")

    # E. Metric clustering.
    ax = fig.add_subplot(gs[1, 1:])
    selected_cols = [c for c in metric_corr.columns if c.endswith("_score01") and c in metric_corr.index]
    # Prefer a readable subset: final selected metrics and sidecars.
    preferred = [
        "KL_or_JSD_score01",
        "QNT_score01",
        "MOM_score01",
        "Mean_score01",
        "TRJDIST_score01",
        "GRAPH_score01",
        "CrossRegionMI_score01",
        "LaggedCovariance_score01",
        "ImpulseResponse_score01",
        "MANI_score01",
        "SubspaceAngle_score01",
        "LatentStateOccupancyK11_score01",
        "LatentStateTransitionLag1K11_score01",
        "Error_score01",
        "MI_score01",
    ]
    selected_cols = [c for c in preferred if c in selected_cols]
    if len(selected_cols) > 2:
        mc = metric_corr.loc[selected_cols, selected_cols].fillna(0.0).clip(-1, 1)
        dist = 1.0 - np.abs(mc.to_numpy(float))
        np.fill_diagonal(dist, 0.0)
        link = linkage(squareform(dist, checks=False), method="average")
        dendrogram(link, labels=[_short_metric(c, 22) for c in selected_cols], leaf_rotation=65, leaf_font_size=8, ax=ax, color_threshold=0.65)
        ax.set_ylabel("1 - |Spearman rho|")
    else:
        ax.axis("off")
    ax.set_title("Individual metrics are related but do not collapse into one axis", fontsize=11, weight="bold")
    _panel_label(ax, "e")

    # F. Redundancy regression.
    ax = fig.add_subplot(gs[2, 0])
    reg = redundancy[redundancy["analysis"] == "family_redundancy_regression"].copy()
    if not reg.empty:
        reg["family_label"] = reg["target_family"].map(FAMILY_LABELS).fillna(reg["target_family"])
        reg = reg.sort_values("cv_r2_mean")
        ax.barh(reg["family_label"], reg["cv_r2_mean"], xerr=reg["cv_r2_std"], color="#B7C9E2", edgecolor="#4C78A8")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Cross-validated $R^2$ from other families")
    else:
        ax.axis("off")
    ax.set_title("Family scores retain partially unique information", fontsize=11, weight="bold")
    _panel_label(ax, "f")
    _clean_axis(ax)

    # G. Leave-one-family-out rank shifts.
    ax = fig.add_subplot(gs[2, 1])
    loo = redundancy[redundancy["analysis"] == "leave_one_family_out_rank_shift"].copy()
    if not loo.empty:
        piv = loo.pivot_table(index="condition_id", columns="variant", values="rank_shift_abs", aggfunc="mean").fillna(0.0)
        # Show conditions with largest sensitivity to omissions for readability.
        keep = piv.max(axis=1).sort_values(ascending=False).head(14).index
        piv = piv.loc[keep]
        piv.columns = [c.replace("leave_family_", "").replace("_", " ").title() for c in piv.columns]
        im = ax.imshow(piv.to_numpy(float), aspect="auto", cmap="YlOrRd")
        ax.set_yticks(np.arange(piv.shape[0]))
        ax.set_yticklabels([_short_metric(x, 28) for x in piv.index], fontsize=6.7)
        ax.set_xticks(np.arange(piv.shape[1]))
        ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.axis("off")
    ax.set_title("Family omissions reveal which comparisons depend on which axis", fontsize=11, weight="bold")
    _panel_label(ax, "g")

    # H. Perturbation selectivity matrix.
    ax = fig.add_subplot(gs[2, 2])
    sel = fig2_red[fig2_red["analysis"] == "perturbation_selectivity"].copy()
    drop_cols = [c for c in sel.columns if c.startswith("drop_")]
    if not sel.empty and drop_cols:
        sel["label"] = sel["source"].str.replace("_", " ") + " | " + sel["perturbation"].astype(str)
        piv = sel.set_index("label")[drop_cols].fillna(0.0)
        piv = piv.loc[piv.max(axis=1).sort_values(ascending=False).index]
        im = ax.imshow(piv.to_numpy(float), aspect="auto", cmap="Blues")
        ax.set_yticks(np.arange(piv.shape[0]))
        ax.set_yticklabels([_short_metric(x, 32) for x in piv.index], fontsize=6.5)
        ax.set_xticks(np.arange(piv.shape[1]))
        ax.set_xticklabels([c.replace("drop_", "") for c in piv.columns], rotation=45, ha="right", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Absolute score drop")
    else:
        ax.axis("off")
    ax.set_title("Targeted corruptions produce family-specific degradation", fontsize=11, weight="bold")
    _panel_label(ax, "h")

    fig.suptitle("Robustness of the Nethobench composite and family decomposition", fontsize=17, weight="bold", y=0.995)
    svg, pdf = _save(fig, "supp_big_metric_robustness_and_sensitivity")

    rank_no_default = rank[rank["variant"] != "default_weights"]
    summary = {
        "n_variants": int(rank_no_default["variant"].nunique()),
        "median_spearman_to_default": float(rank_no_default["spearman_to_default"].median()),
        "q05_spearman_to_default": float(rank_no_default["spearman_to_default"].quantile(0.05)),
        "min_spearman_to_default": float(rank_no_default["spearman_to_default"].min()),
        "mean_abs_score_delta": float(rank_no_default["mean_abs_score_delta"].mean()),
        "figure_svg": str(svg),
        "figure_pdf": str(pdf),
    }
    (TABLE_DIR / "supp_big_metric_robustness_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def write_caption_and_methods(selection_summary: dict[str, object], robustness_summary: dict[str, object]) -> Path:
    text = f"""
# Supplementary Figure: Metric Selection Rationale

**Caption.** Empirical rationale for the final Nethobench metric set. **(a)** Candidate metrics were grouped into selected final metrics, fidelity sidecars, external predictive sidecars, narrow secondary descriptors, and weak-selectivity candidates. This panel documents that the final benchmark is the outcome of an explicit retention procedure rather than an unconstrained metric collection. **(b)** Each candidate metric was scored against six criteria: stochastic-oracle ceiling, targeted corruption selectivity, model-ranking/training utility, nonredundancy, coverage, and interpretability. Check marks denote metrics retained in the final structural-realism composite. **(c)** Metric-selection frontier, plotting oracle ceiling against model utility. Marker size indicates targeted-corruption selectivity and colors indicate source class. Selected Nethobench metrics occupy the region with higher utility and selectivity than pointwise or single-descriptor alternatives. **(d)** Average criterion profiles by metric source show that current Nethobench metrics perform well across multiple criteria, whereas legacy, generic, and external predictive metrics tend to be strong on a narrower subset of axes. **(e)** ZAPBench-style MAE and Dynamic Sensorium-style correlation metrics are useful for their original predictive tasks, but under autonomous stochastic rollout they are less suitable as structural-realism composites because they emphasize pathwise synchronization. **(f)** Mean pass rate across retention criteria by source. **(g)** Selected metrics balance overall selection score with partial nonredundancy; high correlations are tolerated when metrics probe established neuroscience quantities, but redundant or weakly diagnostic candidates are not promoted to the final composite.

# Supplementary Figure: Composite Robustness and Sensitivity

**Caption.** Robustness analyses for the final composite and family decomposition. **(a)** Rank correlation between the default benchmark and alternative benchmark variants, grouped by the kind of design choice being perturbed. Variants include equal family weights, doubled or omitted families, equalized or omitted submetrics, alternative distribution divergences, PCA dimensions, state counts, transition lags, and topology inclusion/exclusion. **(b)** Mean and maximum composite-score shifts under the same variant groups. The exact scalar value changes, as expected, but rankings generally remain close to the default. **(c)** Raw hyperparameter recomputation for distribution divergence, PCA dimension, number of latent states, and transition lags shows that reasonable hyperparameter changes alter score calibration more than qualitative ordering. **(d)** Family correlation matrix. The family scores covary because they measure related properties of the same neural process, but fidelity is not a sufficient proxy for the structural-realism families. **(e)** Hierarchical clustering of individual metrics by absolute Spearman correlation shows interpretable clusters without collapse into one single metric axis. **(f)** Cross-validated redundancy regression estimates how well each family can be predicted from the others; imperfect prediction supports family-specific information. **(g)** Leave-one-family-out rank shifts show which model comparisons depend on each family. **(h)** Perturbation selectivity matrix shows that targeted corruptions produce family-specific score drops, supporting the claim that the benchmark families diagnose distinct failure modes.

# Methods: Metric Selection and Robustness Analyses

We evaluated candidate metrics using existing Nethobench validation outputs and lightweight raw recomputation where aligned ground-truth and prediction arrays were available. Candidate metrics included final Nethobench metrics, discarded internal candidates, generic alternatives such as RMSE/MAE, Pearson correlation, Wasserstein/MMD-style marginal distances, covariance and PCA-Gaussian distances, topology-only persistent homology, classifier two-sample discriminability, and external benchmark-style predictive metrics inspired by ZAPBench and Dynamic Sensorium. For each candidate, we computed six retention criteria. The stochastic-oracle ceiling measured whether a metric assigned a high score to independent samples from the same generative process. Targeted-corruption selectivity measured whether the metric degraded under corruptions designed to affect its intended family. Model utility measured whether the metric contributed to distinguishing stronger from weaker models and tracking training progress. Nonredundancy was defined as one minus the largest absolute Spearman correlation with another candidate metric. Coverage measured the fraction of valid evaluations. Interpretability was a fixed rubric reflecting whether the metric has a direct neuroscience readout and localizable failure mode.

Metrics were retained in the structural-realism composite when they jointly provided high oracle ceiling, targeted sensitivity, empirical utility, adequate coverage, and interpretability. Metrics that were useful but measured pathwise synchronization, such as error and single-trial correlation, were retained as fidelity sidecars rather than included in the realism composite. External benchmark-style metrics were treated as task-matched baselines: ZAPBench-style MAE captures future-activity prediction accuracy, and Dynamic Sensorium-style correlations capture stimulus-locked response prediction. These are not invalid metrics; they answer a different question from autonomous generative neural realism.

Composite robustness was evaluated by recomputing scores and ranks under multiple benchmark variants. Family-weight variants included equal weights, leave-one-family-out weights, doubled-family weights, and random Dirichlet perturbations centered on the default weights. Submetric variants included equalized submetric weights and leave-one-submetric-out composites. Raw hyperparameter variants changed the distribution divergence (JSD versus symmetric KL), PCA dimension, number of latent states, and transition-lag set. Topology variants included excluding topology metrics or isolating topology-only contributions where available. For each variant, we computed composite scores, ranks, Spearman rank correlation with the default benchmark, mean absolute score shift, and maximum absolute score shift.

Redundancy analyses pooled model outputs, perturbation outputs, training checkpoints, synthetic seeds, and transfer evaluations. We computed Spearman correlations among individual metrics and family composites, hierarchical clustering based on one minus absolute Spearman correlation, leave-one-family-out rank shifts, and cross-validated regressions predicting each family from the remaining families. These analyses test the reviewer concern that Nethobench may be a metric zoo: the desired outcome is not zero correlation, because neural properties are coupled, but partial dissociation and stable qualitative conclusions under reasonable design changes.

# Quantitative Summary

Metric-selection analysis considered {selection_summary.get('n_candidates')} candidate metrics and retained {selection_summary.get('n_selected')} final structural-realism metrics. Robustness analysis evaluated {robustness_summary.get('n_variants')} non-default benchmark variants. Across variants, the median Spearman rank correlation with the default benchmark was {robustness_summary.get('median_spearman_to_default'):.3f}, the 5th percentile was {robustness_summary.get('q05_spearman_to_default'):.3f}, and the minimum was {robustness_summary.get('min_spearman_to_default'):.3f}. These results support the interpretation that exact scores are configurable and should be reported with family profiles, but the main qualitative conclusions are not artifacts of one arbitrary weighting or hyperparameter setting.
"""
    out = TEXT_DIR / "supp_metric_selection_and_robustness_caption_methods.md"
    out.write_text(textwrap.dedent(text).strip() + "\n")
    return out


def main() -> int:
    _ensure_dirs()
    selection_summary = build_metric_selection_story()
    robustness_summary = build_metric_robustness_story()
    text_path = write_caption_and_methods(selection_summary, robustness_summary)
    print("Generated:")
    print(selection_summary["figure_svg"])
    print(selection_summary["figure_pdf"])
    print(robustness_summary["figure_svg"])
    print(robustness_summary["figure_pdf"])
    print(text_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
