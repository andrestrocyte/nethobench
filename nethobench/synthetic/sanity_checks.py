from __future__ import annotations

import argparse
import io
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nethobench.neuro.pipeline import compute_neuro_scores

logger = logging.getLogger(__name__)

NOISE_RE = re.compile(r"\.noise_(\d+(?:\.\d+)?)\.csv$")


def _extract_noise_level(path: Path) -> float | None:
    match = NOISE_RE.search(path.name)
    if not match:
        return None
    return float(match.group(1))


def _prepare_gt_subset(
    source_csv: Path,
    out_csv: Path,
    *,
    seq_limit: int,
    max_item_position: int,
    chunksize: int = 400_000,
) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        return out_csv

    first = True
    for chunk in pd.read_csv(source_csv, chunksize=chunksize):
        filt = chunk[
            (chunk["sequenceId"] < seq_limit)
            & (chunk["itemPosition"] < max_item_position)
        ]
        if filt.empty:
            continue
        filt.to_csv(out_csv, index=False, mode="w" if first else "a", header=first)
        first = False

    if not out_csv.exists():
        raise RuntimeError(f"No rows written when preparing subset from {source_csv}")
    return out_csv


def _prepare_pred_subset(
    source_csv: Path,
    out_csv: Path,
    *,
    seq_limit: int,
) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        return out_csv
    pred = pd.read_csv(source_csv, index_col=0)
    pred = pred[pred.index < seq_limit]
    pred.to_csv(out_csv)
    return out_csv


def _corrupt_csv_values(
    source_csv: Path,
    out_csv: Path,
    *,
    sigma: float,
    seed: int,
    is_gt: bool,
    sigma_mode: str = "relative_std",
) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        return out_csv

    rng = np.random.default_rng(seed)
    if is_gt:
        df = pd.read_csv(source_csv)
        value_cols = [c for c in df.columns if c not in {"sequenceId", "itemPosition"}]
        vals = df[value_cols].to_numpy(dtype=np.float64)
        if sigma_mode == "relative_std":
            scales = np.nanstd(vals, axis=0)
            scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
            vals = vals + rng.normal(0.0, sigma, size=vals.shape) * scales[None, :]
        else:
            vals = vals + rng.normal(0.0, sigma, size=vals.shape)
        df[value_cols] = vals
        df.to_csv(out_csv, index=False)
    else:
        df = pd.read_csv(source_csv, index_col=0)
        vals = df.to_numpy(dtype=np.float64)
        if sigma_mode == "relative_std":
            scales = np.nanstd(vals, axis=0)
            scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
            vals = vals + rng.normal(0.0, sigma, size=vals.shape) * scales[None, :]
        else:
            vals = vals + rng.normal(0.0, sigma, size=vals.shape)
        out = pd.DataFrame(vals, index=df.index, columns=df.columns)
        out.to_csv(out_csv)
    return out_csv


def _score_pair(
    *,
    gt_csv: Path,
    pred_csv: Path,
    ddconfig: Path,
) -> Dict[str, float]:
    # Silence notebook-derived prints while preserving returned metrics.
    buf = io.StringIO()
    start = time.time()
    with io.StringIO() as _err, io.StringIO() as _out:
        with pd.option_context("mode.chained_assignment", None):
            with np.errstate(all="ignore"):
                # redirect via lightweight local context managers
                import contextlib

                with contextlib.redirect_stdout(_out), contextlib.redirect_stderr(_err):
                    scores = compute_neuro_scores(
                        pred_csv, gt_csv, ddconfig_path=ddconfig
                    )
                buf.write(_out.getvalue())
                buf.write(_err.getvalue())
    elapsed = time.time() - start
    out = {
        k: float(v)
        for k, v in scores.items()
        if isinstance(v, (int, float, np.floating))
    }
    out["runtime_sec"] = float(elapsed)
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _monotonicity(
    df: pd.DataFrame, metric: str, descending: bool = True
) -> Dict[str, float]:
    d = df.sort_values("noise_sigma")
    vals = d[metric].to_numpy(dtype=np.float64)
    sig = d["noise_sigma"].to_numpy(dtype=np.float64)
    if len(vals) < 2:
        return {
            "spearman": float("nan"),
            "adjacent_ok_frac": float("nan"),
            "range_delta": float("nan"),
        }
    diffs = np.diff(vals)
    ok = diffs <= 0 if descending else diffs >= 0
    return {
        "spearman": _spearman(sig, vals),
        "adjacent_ok_frac": float(np.mean(ok)),
        "range_delta": float(vals[-1] - vals[0]),
    }


def _build_report_tables(
    results: pd.DataFrame,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    tables: Dict[str, Dict[str, Dict[str, float]]] = {}
    metric_candidates = [
        "composite_score",
        "monotonic_v2_composite_score",
        "multiplicative_v3_composite_score",
        "MeanShiftZ_mean",
        "Bandpower_score_avg",
        "QNT_tail_score_avg",
        "ERR_nRMSE_score_avg",
        "MI_mean_score_avg",
        "FC_core_score_avg",
        "PCA_comp_score_avg",
        "GRAPH_core_score_avg",
    ]
    for tag in ["provided_gt_noise", "synthetic_gt_noise", "synthetic_pred_noise"]:
        chunk = results[results["scenario_group"] == tag]
        if chunk.empty:
            continue
        tables[tag] = {}
        for metric in metric_candidates:
            if metric in chunk.columns:
                tables[tag][metric] = _monotonicity(chunk, metric, descending=True)
    return tables


def _build_saturation(results: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    sat: Dict[str, Dict[str, float]] = {}
    for metric in [
        "composite_score",
        "monotonic_v2_composite_score",
        "multiplicative_v3_composite_score",
        "MeanShiftZ_mean",
        "Bandpower_score_avg",
        "QNT_tail_score_avg",
        "ERR_nRMSE_score_avg",
        "MI_mean_score_avg",
        "FC_core_score_avg",
        "PCA_comp_score_avg",
        "GRAPH_core_score_avg",
    ]:
        if metric not in results.columns:
            continue
        vals = results[metric].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        sat[metric] = {
            "frac_ge_0p90": float(np.mean(vals >= 0.90)),
            "frac_ge_0p95": float(np.mean(vals >= 0.95)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
    return sat


def _save_v2_v3_plots(results: pd.DataFrame, output_dir: Path) -> None:
    if "monotonic_v2_composite_score" not in results.columns:
        return
    if "multiplicative_v3_composite_score" not in results.columns:
        return

    baseline_rows = results[results["scenario_group"] == "baseline"]
    if baseline_rows.empty:
        return
    baseline = baseline_rows.iloc[0]
    base_v2 = float(baseline["monotonic_v2_composite_score"])
    base_v3 = float(baseline["multiplicative_v3_composite_score"])

    panel_groups = [
        ("provided_gt_noise", "Provided GT Corruption"),
        ("synthetic_gt_noise", "Synthetic GT Corruption"),
        ("synthetic_pred_noise", "Synthetic Inference Corruption"),
    ]
    present = [
        (g, t) for g, t in panel_groups if g in set(results["scenario_group"].unique())
    ]
    if not present:
        return

    fig, axes = plt.subplots(
        1, len(present), figsize=(6 * len(present), 5), sharey=True
    )
    if len(present) == 1:
        axes = [axes]

    export_rows = []
    for ax, (group_name, title) in zip(axes, present):
        d = results[results["scenario_group"] == group_name].sort_values("noise_sigma")
        if d.empty:
            continue
        ax.plot(
            d["noise_sigma"],
            d["monotonic_v2_composite_score"],
            marker="o",
            lw=1.8,
            label="Composite v2",
        )
        ax.plot(
            d["noise_sigma"],
            d["multiplicative_v3_composite_score"],
            marker="s",
            lw=2.0,
            label="Composite v3 (strict)",
        )
        ax.axhline(
            base_v2, linestyle="--", linewidth=1.1, alpha=0.7, label="Baseline v2"
        )
        ax.axhline(
            base_v3, linestyle=":", linewidth=1.4, alpha=0.8, label="Baseline v3"
        )
        ax.set_title(title)
        ax.set_xlabel("Noise sigma")
        ax.set_ylabel("Composite score")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")

        for _, row in d.iterrows():
            export_rows.append(
                {
                    "scenario_group": group_name,
                    "noise_sigma": float(row["noise_sigma"]),
                    "v2_composite": float(row["monotonic_v2_composite_score"]),
                    "v3_composite": float(row["multiplicative_v3_composite_score"]),
                }
            )

    fig.suptitle(
        "Composite Comparison: Monotonic v2 vs Strict Multiplicative v3", fontsize=13
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        output_dir / "composite_v2_vs_v3_side_by_side.png", dpi=180, bbox_inches="tight"
    )
    plt.close(fig)

    pd.DataFrame(export_rows).to_csv(
        output_dir / "composite_v2_vs_v3_side_by_side.csv", index=False
    )


def run(
    *,
    data_dir: Path,
    output_dir: Path,
    seq_limit: int,
    synthetic_sigmas: List[float],
    synthetic_sigma_mode: str,
) -> Path:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_base = data_dir / "data-clean.csv"
    ddconfig = data_dir / "data-clean-all.json"
    pred_base = (
        data_dir / "sequifier-netho-hp-search-9-run-2-best-10000-predictions.csv"
    )
    if not gt_base.is_file() or not ddconfig.is_file() or not pred_base.is_file():
        raise FileNotFoundError("Missing expected files in data_dir.")

    pred_df = pd.read_csv(pred_base, index_col=0)
    pred_time = int(pred_df.index.value_counts().iloc[0])
    subset_dir = output_dir / "prepared_subsets"
    synth_dir = output_dir / "synthetic_corruptions"
    subset_dir.mkdir(exist_ok=True, parents=True)
    synth_dir.mkdir(exist_ok=True, parents=True)

    pred_subset = _prepare_pred_subset(
        pred_base,
        subset_dir / f"{pred_base.stem}.seq{seq_limit}.csv",
        seq_limit=seq_limit,
    )
    gt_subset = _prepare_gt_subset(
        gt_base,
        subset_dir / f"{gt_base.stem}.trim{pred_time}.seq{seq_limit}.csv",
        seq_limit=seq_limit,
        max_item_position=pred_time,
    )

    scenarios = []
    scenarios.append(
        dict(
            scenario_name="baseline",
            scenario_group="baseline",
            noise_sigma=0.0,
            gt_csv=gt_subset,
            pred_csv=pred_subset,
        )
    )

    # Provided GT-corruption ladder from your directory.
    provided_noisy = sorted(
        data_dir.glob("data-clean.noise_*.csv"),
        key=lambda p: _extract_noise_level(p) or -1.0,
    )
    for p in provided_noisy:
        sigma = _extract_noise_level(p)
        if sigma is None:
            continue
        prepared = _prepare_gt_subset(
            p,
            subset_dir / f"{p.stem}.trim{pred_time}.seq{seq_limit}.csv",
            seq_limit=seq_limit,
            max_item_position=pred_time,
        )
        scenarios.append(
            dict(
                scenario_name=f"provided_gt_noise_{sigma:.2f}",
                scenario_group="provided_gt_noise",
                noise_sigma=float(sigma),
                gt_csv=prepared,
                pred_csv=pred_subset,
            )
        )

    # Synthetic GT and inference corruptions on aligned subset.
    for idx, sigma in enumerate(synthetic_sigmas):
        gt_syn = _corrupt_csv_values(
            gt_subset,
            synth_dir / f"{gt_subset.stem}.synthetic_gt_sigma_{sigma:.3f}.csv",
            sigma=sigma,
            seed=10_000 + idx,
            is_gt=True,
            sigma_mode=synthetic_sigma_mode,
        )
        scenarios.append(
            dict(
                scenario_name=f"synthetic_gt_noise_{sigma:.3f}",
                scenario_group="synthetic_gt_noise",
                noise_sigma=float(sigma),
                gt_csv=gt_syn,
                pred_csv=pred_subset,
            )
        )

    for idx, sigma in enumerate(synthetic_sigmas):
        pr_syn = _corrupt_csv_values(
            pred_subset,
            synth_dir / f"{pred_subset.stem}.synthetic_pred_sigma_{sigma:.3f}.csv",
            sigma=sigma,
            seed=20_000 + idx,
            is_gt=False,
            sigma_mode=synthetic_sigma_mode,
        )
        scenarios.append(
            dict(
                scenario_name=f"synthetic_pred_noise_{sigma:.3f}",
                scenario_group="synthetic_pred_noise",
                noise_sigma=float(sigma),
                gt_csv=gt_subset,
                pred_csv=pr_syn,
            )
        )

    rows = []
    for i, sc in enumerate(scenarios, start=1):
        scores = _score_pair(
            gt_csv=sc["gt_csv"],
            pred_csv=sc["pred_csv"],
            ddconfig=ddconfig,
        )
        row = dict(sc)
        row.update(scores)
        row["gt_csv"] = str(row["gt_csv"])
        row["pred_csv"] = str(row["pred_csv"])
        row["run_index"] = i
        rows.append(row)
        logger.info(
            f"[{i:02d}/{len(scenarios):02d}] {sc['scenario_name']}: composite={scores.get('composite_score', float('nan')):.4f} runtime={scores['runtime_sec']:.2f}s"
        )

    results = pd.DataFrame(rows).sort_values("run_index").reset_index(drop=True)
    metrics_json = output_dir / "scores_per_scenario.json"
    metrics_csv = output_dir / "scores_per_scenario.csv"
    results.to_json(metrics_json, orient="records", indent=2)
    results.to_csv(metrics_csv, index=False)
    _save_v2_v3_plots(results, output_dir)

    monotonic = _build_report_tables(results)
    saturation = _build_saturation(results)
    diagnostics = {
        "monotonicity": monotonic,
        "saturation": saturation,
        "seq_limit": seq_limit,
        "pred_time_horizon": pred_time,
        "n_scenarios": int(len(results)),
    }
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    report_lines = [
        "# Neuro Sanity-Check Report",
        "",
        f"- Date: {datetime.now().isoformat(timespec='seconds')}",
        f"- Data dir: `{data_dir}`",
        "- Mode: `full` (notebook-derived metrics via `compute_neuro_scores`)",
        f"- Sequence subset: first `{seq_limit}` sequences",
        f"- Time horizon used: `{pred_time}` steps",
        f"- Scenarios: `{len(results)}`",
        f"- Synthetic sigma mode: `{synthetic_sigma_mode}`",
        "",
        "## Key Acceptance Criteria",
        "| Criterion | Target | Result Source |",
        "|---|---|---|",
        "| Composite monotonicity under provided GT noise | Spearman <= -0.70 | `diagnostics.json` -> `monotonicity.provided_gt_noise.composite_score.spearman` |",
        "| Composite adjacent monotonicity | >= 0.80 | `diagnostics.json` -> `adjacent_ok_frac` |",
        "| Dynamic range (max noise - baseline) | <= -0.10 | `diagnostics.json` -> `range_delta` |",
        "| Saturation guard | fraction >=0.95 <= 0.20 | `diagnostics.json` -> `saturation.*.frac_ge_0p95` |",
        "",
        "## Publish Template",
        "1. Data: dataset name/version, split protocol, sequence/time coverage.",
        "2. Benchmark config: nethobench version, run mode, sequence subset/full, runtime-cap policy.",
        "3. Main table: baseline + corruption ladder scores for composite and key submetrics.",
        "4. Robustness: monotonicity statistics (Spearman + adjacent pass fraction).",
        "5. Saturation: per-metric ceiling occupancy (>=0.90, >=0.95).",
        "6. Reproducibility: random seeds, command lines, and raw output file links.",
    ]
    (output_dir / "REPORT_TEMPLATE.md").write_text("\n".join(report_lines))
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run notebook-derived neuro sanity checks on real benchmark data with provided and synthetic corruptions."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing data-clean.csv, data-clean-all.json, predictions CSV, and optional noise ladder CSVs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.cwd() / "outputs",
        help="Where to write timestamped sanity-check outputs.",
    )
    parser.add_argument(
        "--seq-limit",
        type=int,
        default=10,
        help="Use first N sequences for sanity sweep.",
    )
    parser.add_argument(
        "--sigmas",
        nargs="+",
        type=float,
        default=[0.01, 0.03, 0.06, 0.12, 0.24],
        help="Synthetic corruption sigmas for GT and predictions.",
    )
    parser.add_argument(
        "--sigma-mode",
        type=str,
        choices=["relative_std", "absolute"],
        default="relative_std",
        help="Interpret synthetic sigma as a fraction of each feature std (`relative_std`) or raw-value units (`absolute`).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outdir = args.output_root / f"neuro-sanity"
    final = run(
        data_dir=args.data_dir,
        output_dir=outdir,
        seq_limit=args.seq_limit,
        synthetic_sigmas=list(args.sigmas),
        synthetic_sigma_mode=args.sigma_mode,
    )
    logger.info(f"Saved sanity-check outputs to: {final}")


if __name__ == "__main__":
    main()
