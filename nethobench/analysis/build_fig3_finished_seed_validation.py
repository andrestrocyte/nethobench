from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.stats import ttest_rel, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "paper" / "Nethobench"
FIG_DIR = PAPER_ROOT / "figures"
TABLE_DIR = PAPER_ROOT / "generated_tables"
TEXT_DIR = PAPER_ROOT / "generated_text"

FULL_ROOT = Path.home() / "Desktop" / "fig2_model_multiseed_full"
TRANSFER_ROOT = FULL_ROOT / "calciumgan_transfer_seed_results"
SCALING_WIDE = Path.home() / "Desktop" / "netho-seq-scaling-rerun" / "tables" / "nethobench_neurobench_scores_scores_wide.csv"

FAMILY_ROWS = [
    ("family_distribution", "Distribution"),
    ("family_temporal_spectral", "Temporal"),
    ("family_relational", "Relational"),
    ("family_geometry", "Geometry"),
    ("family_state_dynamics", "State dynamics"),
    ("FINAL_COMPOSITE_SCORE", "Composite"),
]
COLORS = {"converged": "#4C78A8", "weakest": "#E68653", "oracle": "#8ED0E6", "weak_oracle": "#F3B08F"}


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)


def _sem(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size <= 1:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(vals.size))


def _p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n.s."
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def _paired_stats(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in value_cols:
        if col not in df.columns:
            continue
        wide = df.pivot_table(index="seed", columns="condition", values=col, aggfunc="mean")
        if not {"converged", "weakest"}.issubset(wide.columns):
            continue
        paired = wide[["converged", "weakest"]].dropna()
        if len(paired) < 2:
            continue
        diff = paired["converged"] - paired["weakest"]
        p_t = float(ttest_rel(paired["converged"], paired["weakest"]).pvalue)
        try:
            p_w = float(wilcoxon(paired["converged"], paired["weakest"], zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            p_w = np.nan
        rows.append(
            {
                "metric": col,
                "n": int(len(paired)),
                "mean_converged": float(paired["converged"].mean()),
                "mean_weakest": float(paired["weakest"].mean()),
                "mean_diff": float(diff.mean()),
                "sem_diff": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                "p_paired_t": p_t,
                "p_wilcoxon": p_w,
                "stars": _p_to_stars(p_t),
            }
        )
    return pd.DataFrame(rows)


def _add_sig_bracket(ax: plt.Axes, x1: float, x2: float, y: float, stars: str, h: float = 0.018) -> None:
    if not stars or stars == "n.s.":
        return
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#222222", lw=0.9, clip_on=False)
    ax.text((x1 + x2) / 2, y + h + 0.004, stars, ha="center", va="bottom", fontsize=11, fontweight="bold")


def _read_family_result(result_dir: Path, analysis: str, condition: str, seed: int) -> dict[str, object] | None:
    fam_path = result_dir / "family_comparison_rollout.csv"
    fid_path = result_dir / "fidelity_comparison_rollout.csv"
    if not fam_path.exists():
        return None
    fam = pd.read_csv(fam_path)
    row: dict[str, object] = {"analysis": analysis, "condition": condition, "seed": seed, "result_dir": str(result_dir)}
    for _, r in fam.iterrows():
        key = str(r["family"])
        row[key] = float(r["model_score"])
        row[f"oracle_{key}"] = float(r["oracle_score"])
    if fid_path.exists():
        fid = pd.read_csv(fid_path)
        for _, r in fid.iterrows():
            key = str(r["score"])
            row[key] = float(r["model_score"])
            row[f"oracle_{key}"] = float(r["oracle_score"])
    return row


def load_completed_full_seed_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in range(2001, 2006):
        seed_dir = FULL_ROOT / f"biophysical_seed_{seed}"
        for condition in ["converged", "weakest"]:
            row = _read_family_result(
                seed_dir / "results" / condition,
                analysis="biophysical_model_comparison",
                condition=condition,
                seed=seed,
            )
            if row is not None:
                rows.append(row)

        for condition in ["converged", "weakest"]:
            row = _read_family_result(
                TRANSFER_ROOT / f"seed_{seed}_{condition}",
                analysis="transfer_calciumgan",
                condition=condition,
                seed=seed,
            )
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def load_training_progress() -> pd.DataFrame:
    if not SCALING_WIDE.exists():
        return pd.DataFrame()
    df = pd.read_csv(SCALING_WIDE)
    sub = df[
        (df["sequence_length"] == 90)
        & (df["brain_areas"] == 16)
        & (df["data_share"] == 100)
        & (df["training_percent"].isin([10.0, 30.0, 100.0]))
    ].copy()
    sub["training_percent"] = sub["training_percent"].astype(int)
    return sub


def _panel_grouped_bars(ax: plt.Axes, df: pd.DataFrame, title: str, ylabel: str, show_legend: bool = True) -> pd.DataFrame:
    labels = [label for _, label in FAMILY_ROWS]
    keys = [key for key, _ in FAMILY_ROWS]
    x = np.arange(len(keys))
    width = 0.34
    offsets = {"converged": -width / 2, "weakest": width / 2}
    rng = np.random.default_rng(10)

    for condition in ["converged", "weakest"]:
        sub = df[df["condition"] == condition]
        means = [pd.to_numeric(sub[k], errors="coerce").mean() for k in keys]
        sems = [_sem(sub[k]) for k in keys]
        ax.bar(x + offsets[condition], means, width, color=COLORS[condition], alpha=0.88, label=condition.capitalize(), zorder=2)
        ax.errorbar(x + offsets[condition], means, yerr=sems, fmt="none", ecolor="#333333", elinewidth=1.0, capsize=2.5, zorder=4)
        for i, k in enumerate(keys):
            vals = pd.to_numeric(sub[k], errors="coerce").dropna().to_numpy(dtype=float)
            jitter = rng.normal(0, 0.025, size=vals.size)
            ax.scatter(np.full(vals.size, x[i] + offsets[condition]) + jitter, vals, s=15, color="white", edgecolor="#333333", linewidth=0.4, alpha=0.8, zorder=5)

    stats = _paired_stats(df, keys)
    # Paired seed lines.
    for i, k in enumerate(keys):
        wide = df.pivot_table(index="seed", columns="condition", values=k, aggfunc="mean")
        if {"converged", "weakest"}.issubset(wide.columns):
            for _, r in wide.dropna(subset=["converged", "weakest"]).iterrows():
                ax.plot([x[i] + offsets["converged"], x[i] + offsets["weakest"]], [r["converged"], r["weakest"]], color="#A0A0A0", lw=0.7, alpha=0.35, zorder=1)
        stat_row = stats[stats["metric"] == k]
        if not stat_row.empty:
            max_y = pd.to_numeric(df[k], errors="coerce").max()
            if np.isfinite(max_y):
                _add_sig_bracket(
                    ax,
                    x[i] + offsets["converged"],
                    x[i] + offsets["weakest"],
                    min(0.965, max_y + 0.04),
                    str(stat_row["stars"].iloc[0]),
                )

    n = df.groupby("condition")["seed"].nunique().to_dict()
    ax.set_title(f"{title} (n={min(n.values()) if n else 0} paired seeds)", loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.03)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", color="#E5E5E5", zorder=0)
    if show_legend:
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    return stats


def _panel_fidelity(ax: plt.Axes, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in ["converged", "weakest"]:
        sub = df[df["condition"] == condition]
        for kind, key in [("model", "FIDELITY_SCORE"), ("oracle", "oracle_FIDELITY_SCORE")]:
            rows.append(
                {
                    "label": f"{condition}\n{kind}",
                    "condition": condition,
                    "kind": kind,
                    "mean": pd.to_numeric(sub[key], errors="coerce").mean(),
                    "sem": _sem(sub[key]),
                    "values": pd.to_numeric(sub[key], errors="coerce").dropna().to_numpy(dtype=float),
                }
            )
    x = np.arange(len(rows))
    color_map = {
        ("converged", "model"): COLORS["converged"],
        ("weakest", "model"): "#F58518",
        ("converged", "oracle"): COLORS["oracle"],
        ("weakest", "oracle"): COLORS["weak_oracle"],
    }
    rng = np.random.default_rng(12)
    for i, r in enumerate(rows):
        color = color_map[(r["condition"], r["kind"])]
        ax.bar(i, r["mean"], color=color, alpha=0.88, width=0.72, zorder=2)
        ax.errorbar(i, r["mean"], yerr=r["sem"], fmt="none", ecolor="#333333", elinewidth=1.0, capsize=3, zorder=3)
        vals = r["values"]
        ax.scatter(np.full(vals.size, i) + rng.normal(0, 0.035, vals.size), vals, s=16, color="white", edgecolor="#333333", linewidth=0.4, zorder=4)
        ax.text(i, r["mean"] + r["sem"] + 0.025, f"{r['mean']:.3f}", ha="center", va="bottom", fontsize=8)
    stats = _paired_stats(df, ["FIDELITY_SCORE", "oracle_FIDELITY_SCORE"])
    model_row = stats[stats["metric"] == "FIDELITY_SCORE"] if not stats.empty else pd.DataFrame()
    oracle_row = stats[stats["metric"] == "oracle_FIDELITY_SCORE"] if not stats.empty else pd.DataFrame()
    if not model_row.empty:
        y = min(0.94, max(rows[0]["mean"] + rows[0]["sem"], rows[1]["mean"] + rows[1]["sem"]) + 0.055)
        _add_sig_bracket(ax, 0, 1, y, str(model_row["stars"].iloc[0]), h=0.02)
    if not oracle_row.empty:
        y = min(0.94, max(rows[2]["mean"] + rows[2]["sem"], rows[3]["mean"] + rows[3]["sem"]) + 0.055)
        _add_sig_bracket(ax, 2, 3, y, str(oracle_row["stars"].iloc[0]), h=0.02)
    n = df.groupby("condition")["seed"].nunique().to_dict()
    ax.set_title(f"c  Fidelity does not drive separation (n={min(n.values()) if n else 0})", loc="left", fontweight="bold")
    ax.set_ylabel("Fidelity score")
    ax.set_ylim(0, 1.03)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows], fontsize=8)
    ax.grid(axis="y", color="#E5E5E5", zorder=0)
    return stats


def _panel_training(ax: plt.Axes, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        ax.axis("off")
        return pd.DataFrame()
    g = df.groupby("training_percent")["FINAL_COMPOSITE_SCORE"].agg(["mean", _sem]).reset_index()
    g = g.sort_values("training_percent")
    ax.plot(g["training_percent"], g["mean"], color=COLORS["converged"], marker="o", lw=1.8, zorder=3)
    ax.errorbar(g["training_percent"], g["mean"], yerr=g["_sem"], fmt="none", ecolor=COLORS["converged"], capsize=3, zorder=4)
    rng = np.random.default_rng(13)
    for pct, sub in df.groupby("training_percent"):
        vals = pd.to_numeric(sub["FINAL_COMPOSITE_SCORE"], errors="coerce").dropna().to_numpy(dtype=float)
        xs = np.full(vals.size, pct) + rng.normal(0, 1.1, vals.size)
        ax.scatter(xs, vals, s=18, color="white", edgecolor=COLORS["converged"], linewidth=0.7, alpha=0.9, zorder=5)
    for _, r in g.iterrows():
        ax.text(r["training_percent"], r["mean"] + r["_sem"] + 0.01, f"{r['mean']:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("e  Score tracks training progress (n=5 seeds)", loc="left", fontweight="bold")
    ax.set_ylabel("Nethobench global score")
    ax.set_xlabel("Training progress (% of schedule)")
    ax.set_ylim(max(0, g["mean"].min() - 0.16), min(1.0, g["mean"].max() + 0.13))
    ax.set_xticks([10, 30, 100])
    ax.grid(color="#E5E5E5")
    rows = []
    for a, b in [(10, 30), (30, 100), (10, 100)]:
        va = df[df["training_percent"] == a].set_index("seed")["FINAL_COMPOSITE_SCORE"]
        vb = df[df["training_percent"] == b].set_index("seed")["FINAL_COMPOSITE_SCORE"]
        common = va.index.intersection(vb.index)
        if len(common) >= 2:
            diff = vb.loc[common] - va.loc[common]
            p_t = float(ttest_rel(vb.loc[common], va.loc[common]).pvalue)
            try:
                p_w = float(wilcoxon(vb.loc[common], va.loc[common], zero_method="wilcox", alternative="two-sided").pvalue)
            except ValueError:
                p_w = np.nan
            rows.append(
                {
                    "comparison": f"{a}_vs_{b}",
                    "n": int(len(common)),
                    "mean_diff": float(diff.mean()),
                    "sem_diff": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                    "p_paired_t": p_t,
                    "p_wilcoxon": p_w,
                    "stars": _p_to_stars(p_t),
                }
            )
    stats = pd.DataFrame(rows)
    ybase = pd.to_numeric(df["FINAL_COMPOSITE_SCORE"], errors="coerce").max() + 0.025
    for idx, (a, b) in enumerate([(10, 30), (30, 100), (10, 100)]):
        row = stats[stats["comparison"] == f"{a}_vs_{b}"] if not stats.empty else pd.DataFrame()
        if not row.empty:
            _add_sig_bracket(ax, a, b, ybase + idx * 0.018, str(row["stars"].iloc[0]), h=0.005)
    return stats


def _draw_schematic(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_title("a  Practical validation of model quality and transfer", loc="left", fontweight="bold")

    def box(x, y, w, h, text, fc, ec):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.18,rounding_size=0.12", fc=fc, ec=ec, lw=1.0)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.35, text, ha="center", va="top", fontsize=9, weight="bold")
        return patch

    box(0.4, 2.4, 2.4, 1.7, "Source domain", "#F8FBFD", "#95A5A6")
    ax.text(1.6, 3.2, "Biophysical rollout", ha="center", fontsize=7)
    xs = np.linspace(0.8, 2.4, 80)
    for dy in [0.0, -0.18]:
        ax.plot(xs, 2.8 + dy + 0.08 * np.sin(8 * xs), color="#7F8C8D", lw=1.0)

    box(3.4, 3.45, 2.25, 1.25, "Weak model", "#FFF4EC", COLORS["weakest"])
    ax.text(4.52, 4.1, "low realism", ha="center", fontsize=7)
    for i, h in enumerate([0.18, 0.28, 0.24, 0.34, 0.26]):
        ax.add_patch(Rectangle((4.0 + i * 0.23, 3.6), 0.11, h, color=COLORS["weakest"]))

    box(3.4, 1.75, 2.25, 1.25, "Converged model", "#EDF5FF", COLORS["converged"])
    ax.text(4.52, 2.4, "higher realism", ha="center", fontsize=7)
    for i, h in enumerate([0.32, 0.48, 0.55, 0.68, 0.62]):
        ax.add_patch(Rectangle((4.0 + i * 0.23, 1.92), 0.11, h, color=COLORS["converged"]))

    box(7.0, 2.4, 2.4, 1.7, "Target domain", "#F2FFF6", "#5AA469")
    ax.text(8.2, 3.2, "Domain shift / transfer", ha="center", fontsize=7)
    for dy in [0.0, -0.18]:
        ax.plot(xs + 6.6, 2.8 + dy + 0.08 * np.sin(8 * xs + 0.6), color="#5AA469", lw=1.0)

    for y0, y1 in [(3.25, 4.0), (3.05, 2.35)]:
        ax.add_patch(FancyArrowPatch((2.85, y0), (3.35, y1), arrowstyle="->", mutation_scale=10, color="#6F6F6F", lw=1.0))
    ax.text(3.05, 3.55, "train", fontsize=7)
    for y0 in [4.05, 2.35]:
        ax.add_patch(FancyArrowPatch((5.72, y0), (6.95, 3.25), arrowstyle="->", mutation_scale=10, color="#6F6F6F", lw=1.0))
    ax.text(6.1, 3.6, "evaluate", fontsize=7)
    ax.text(
        0.3,
        0.55,
        "Expected validation: converged model scores higher than weak model, and ranking persists under transfer.",
        fontsize=8,
        style="italic",
    )


def build_figure() -> dict[str, object]:
    _ensure_dirs()
    full = load_completed_full_seed_rows()
    training = load_training_progress()
    full.to_csv(TABLE_DIR / "fig3_finished_full_seed_model_transfer_scores.csv", index=False)
    training.to_csv(TABLE_DIR / "fig3_training_progress_seed_scores.csv", index=False)

    bio = full[full["analysis"] == "biophysical_model_comparison"].copy()
    transfer = full[full["analysis"] == "transfer_calciumgan"].copy()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": "#CFCFCF",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(13.5, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.1])

    ax_a = fig.add_subplot(gs[0, :3])
    _draw_schematic(ax_a)

    ax_b = fig.add_subplot(gs[0, 3:5])
    bio_stats = _panel_grouped_bars(ax_b, bio, "b  Biophysical rollout: converged vs weakest", "Score", show_legend=True)

    ax_c = fig.add_subplot(gs[0, 5])
    fid_stats = _panel_fidelity(ax_c, bio)

    ax_d = fig.add_subplot(gs[1, :3])
    transfer_stats = _panel_grouped_bars(ax_d, transfer, "d  Cross-domain transfer", "Model / oracle score", show_legend=True)

    ax_e = fig.add_subplot(gs[1, 3:])
    training_stats = _panel_training(ax_e, training)

    svg = FIG_DIR / "fig3_model_validation_finished_seeds.svg"
    pdf = FIG_DIR / "fig3_model_validation_finished_seeds.pdf"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    bio_stats.assign(analysis="biophysical_model_comparison").to_csv(TABLE_DIR / "fig3_biophysical_paired_stats.csv", index=False)
    transfer_stats.assign(analysis="transfer_calciumgan").to_csv(TABLE_DIR / "fig3_transfer_paired_stats.csv", index=False)
    fid_stats.assign(analysis="biophysical_fidelity").to_csv(TABLE_DIR / "fig3_fidelity_paired_stats.csv", index=False)
    training_stats.assign(analysis="training_progress").to_csv(TABLE_DIR / "fig3_training_progress_paired_stats.csv", index=False)

    summary = {
        "bio_n_by_condition": bio.groupby("condition")["seed"].nunique().to_dict(),
        "transfer_n_by_condition": transfer.groupby("condition")["seed"].nunique().to_dict(),
        "training_n_by_percent": training.groupby("training_percent")["seed"].nunique().to_dict() if not training.empty else {},
        "figure_svg": str(svg),
        "figure_pdf": str(pdf),
        "bio_paired_stats": bio_stats.to_dict(orient="records"),
        "transfer_paired_stats": transfer_stats.to_dict(orient="records"),
        "fidelity_paired_stats": fid_stats.to_dict(orient="records"),
        "training_paired_stats": training_stats.to_dict(orient="records"),
    }
    pd.Series(summary, dtype=object).to_json(TABLE_DIR / "fig3_finished_seed_validation_summary.json", indent=2)
    (TEXT_DIR / "fig3_finished_seed_validation_caption.md").write_text(_caption(summary))
    return summary


def _caption(summary: dict[str, object]) -> str:
    def fmt_stats(rows: list[dict[str, object]], label: str) -> str:
        if not rows:
            return f"{label}: no paired statistics available."
        bits = []
        for r in rows:
            name = str(r.get("metric", r.get("comparison", "")))
            bits.append(
                f"{name}: paired t-test p={float(r.get('p_paired_t', np.nan)):.4g}, "
                f"Wilcoxon p={float(r.get('p_wilcoxon', np.nan)):.4g}, "
                f"mean paired difference={float(r.get('mean_diff', np.nan)):.3f}"
            )
        return f"{label}: " + "; ".join(bits) + "."

    return f"""# Figure 3 Caption

**Nethobench tracks model quality, transfer, and training progress.** **(a)** Practical validation schematic: a converged model should receive higher structural-realism scores than a weak model, and this ranking should remain informative under transfer to a shifted target domain. **(b)** In the completed full-protocol biophysical rollout seeds, the converged model is compared with the weakest model across Nethobench families and the final composite. Bars show mean across completed paired seeds, error bars show SEM, white points show individual seeds, and gray lines connect paired seeds. Asterisks indicate paired t-test significance only; exact p-values are reported below. Completed seed counts: {summary['bio_n_by_condition']}. **(c)** Fidelity sidecar scores for the same biophysical seeds, including model fidelity and oracle-fidelity controls. This panel tests whether structural-realism separation can be reduced to pointwise fidelity alone. **(d)** Cross-domain transfer to the CalciumGAN target domain, using completed full-protocol transfer seeds. Bars, points, paired lines, and asterisks are defined as in panel b. Completed seed counts: {summary['transfer_n_by_condition']}. **(e)** Global Nethobench score across training progress, with individual seed points and mean SEM. Completed seed counts: {summary['training_n_by_percent']}. Together, these panels test whether Nethobench tracks stronger models, remains informative under transfer, separates structural realism from fidelity, and increases with training progress.

Significance annotations use paired t-tests for the displayed asterisks: * p<0.05, ** p<0.01, *** p<0.001; non-significant comparisons are unlabeled. Exact p-values: {fmt_stats(summary.get('bio_paired_stats', []), 'Panel b')}; {fmt_stats(summary.get('fidelity_paired_stats', []), 'Panel c')}; {fmt_stats(summary.get('transfer_paired_stats', []), 'Panel d')}; {fmt_stats(summary.get('training_paired_stats', []), 'Panel e')}
"""


def main() -> None:
    summary = build_figure()
    print(summary)


if __name__ == "__main__":
    main()
