from __future__ import annotations

import os
import argparse
import contextlib
from datetime import datetime
import io
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from nethobench import (
    BiophysicalSyntheticNeuralSpec,
    SyntheticNeuralSpec,
    compute_cross_scores,
    compute_etho_scores,
    compute_fidelity_scores,
    compute_neuro_scores,
    run_biophysical_synthetic_neuro_validation,
    run_synthetic_neuro_validation,
    run_cross_full_analysis,
    run_etho_full_analysis,
    run_neuro_full_analysis,
)


def _find_candidates(prefix: str) -> list[Path]:
    cwd = Path.cwd()
    return sorted(path for path in cwd.glob(f"{prefix}*") if path.is_file())


def _prompt_for_file(label: str, prefix: str, provided: Optional[str]) -> Path:
    if provided:
        path = Path(provided).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise FileNotFoundError(f"{label} file '{path}' does not exist.")
        return path

    candidates = _find_candidates(prefix)
    if candidates:
        print(f"Detected {label.lower()} candidates in {Path.cwd()}:")
        for idx, candidate in enumerate(candidates, start=1):
            print(f"  [{idx}] {candidate.name}")

    default_candidate = candidates[0] if len(candidates) == 1 else None

    while True:
        prompt = f"Enter {label} filename"
        if default_candidate is not None:
            prompt += f" [{default_candidate.name}]"
        prompt += ": "

        response = input(prompt).strip()
        if not response and default_candidate is not None:
            selection = default_candidate
        elif response.isdigit() and candidates:
            idx = int(response) - 1
            if 0 <= idx < len(candidates):
                selection = candidates[idx]
            else:
                print("Invalid selection number. Try again.")
                continue
        elif response:
            selection = Path(response)
        else:
            print("Please provide a filename or choose one of the listed entries.")
            continue

        selection = selection.expanduser()
        if not selection.is_absolute():
            selection = Path.cwd() / selection
        if selection.is_file():
            return selection
        print(f"{selection} does not exist. Try again.")


def _prompt_for_config(provided: Optional[str]) -> Optional[Path]:
    if provided:
        return Path(provided)
    jsons = sorted(Path.cwd().glob("*.json"))
    if len(jsons) == 1:
        return jsons[0]
    if jsons:
        print("Detected possible config JSON files:")
        for idx, candidate in enumerate(jsons, start=1):
            print(f"  [{idx}] {candidate.name}")
        response = input("Enter config filename (or leave blank to auto-infer): ").strip()
        if response.isdigit():
            idx = int(response) - 1
            if 0 <= idx < len(jsons):
                return jsons[idx]
        elif response:
            path = Path(response)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.is_file():
                return path
    return None


def _score_to_color(value: float) -> str:
    try:
        v = float(value)
    except Exception:
        return ""
    if v >= 0.8:
        return "\033[32m"
    if v >= 0.4:
        return "\033[33m"
    return "\033[31m"


def _render_score_bar(value: float, width: int = 16) -> str:
    try:
        v = float(value)
    except Exception:
        return ""
    v = max(0.0, min(1.0, v))
    arrow_idx = min(width - 1, max(0, int(round(v * (width - 1)))))
    icon = "↗" if v >= 0.8 else ("→" if v >= 0.4 else "↘")
    chars = []
    for idx in range(width):
        if idx < arrow_idx:
            chars.append("━")
        elif idx == arrow_idx:
            chars.append("▶")
        else:
            chars.append("─")
    color = _score_to_color(v)
    reset = "\033[0m"
    return f"{color}{icon} 0 {''.join(chars)} 1{reset}"


def _print_scores(label: str, scores: dict[str, float]) -> None:
    print(f"\n{label}:")
    for key, value in scores.items():
        print(f"  {key:24s}: {value:.3f} {_render_score_bar(value)}")



def _print_composite(label: str, value: float) -> None:
    if value == value:
        print(f"{label:18s}: {value:.3f} {_render_score_bar(value)}")
    else:
        print(f"{label:18s}: NaN")


def _quiet_call(func, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        return func(*args, **kwargs)


def _default_json_output(command: str, preds: Path) -> Path:
    outdir = Path.cwd() / "outputs" / f"{preds.stem}-{command}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / "scores.json"


def _default_output_dir(command: str) -> Path:
    outdir = Path.cwd() / "outputs" / f"{command}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _save_json_payload(payload: dict, *, requested: Optional[str], command: str, preds: Path) -> Path:
    out = Path(requested) if requested else _default_json_output(command, preds)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nethobench",
        description="Unified neural + behavioral + cross-modal benchmarking.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    neuro = subparsers.add_parser("neuro-scores", help="Compute neural-only scores.")
    neuro.add_argument("--gt", help="Ground-truth neural CSV (auto-detected if omitted).")
    neuro.add_argument("--preds", help="Predicted neural CSV (auto-detected if omitted).")
    neuro.add_argument("--json-out", help="Optional JSON output path.")
    neuro.set_defaults(func=_run_neuro)

    fidelity = subparsers.add_parser("fidelity-scores", help="Compute the separate fidelity-only neural family.")
    fidelity.add_argument("--gt", help="Ground-truth neural CSV (auto-detected if omitted).")
    fidelity.add_argument("--preds", help="Predicted neural CSV (auto-detected if omitted).")
    fidelity.add_argument("--json-out", help="Optional JSON output path.")
    fidelity.set_defaults(func=_run_fidelity)

    neuro_full = subparsers.add_parser(
        "neuro-analysis",
        help="Run full NeuroBench analysis and save figures.",
    )
    neuro_full.add_argument("--gt", help="Ground-truth neural CSV (auto-detected if omitted).")
    neuro_full.add_argument("--preds", help="Predicted neural CSV (auto-detected if omitted).")
    neuro_full.add_argument("--output-root", type=Path, help="Output root (default ./outputs/).")
    neuro_full.set_defaults(func=_run_neuro_full)

    etho = subparsers.add_parser("etho-scores", help="Compute behavior-only scores.")
    etho.add_argument("--gt-dir", required=True, help="Directory with GT pose parquet/CSV.")
    etho.add_argument("--inf-dir", required=True, help="Directory with inference pose parquet/CSV.")
    etho.add_argument("--json-out", help="Optional JSON output path.")
    etho.set_defaults(func=_run_etho)

    etho_full = subparsers.add_parser(
        "etho-analysis",
        help="Run full EthoBench analysis and save figures natively.",
    )
    etho_full.add_argument("--gt-dir", required=True, help="Directory with GT pose parquet/CSV.")
    etho_full.add_argument("--inf-dir", required=True, help="Directory with inference pose parquet/CSV.")
    etho_full.add_argument("--output-root", type=Path, help="Output root (default ./outputs/).")
    etho_full.set_defaults(func=_run_etho_full)

    cross = subparsers.add_parser(
        "cross-scores",
        help="Compute neuro + behavior + cross-modal scores from multimodal CSVs.",
    )
    cross.add_argument("--gt", help="Ground-truth multimodal CSV (auto-detected if omitted).")
    cross.add_argument("--preds", help="Predicted multimodal CSV (auto-detected if omitted).")
    cross.add_argument("--config", help="JSON config describing neuro/behavior columns (auto-inferred if omitted).")
    cross.add_argument("--json-out", help="Optional JSON output path.")
    cross.set_defaults(func=_run_cross)

    cross_full = subparsers.add_parser(
        "cross-analysis",
        help="Execute cross-modal notebook headlessly and save figures + executed notebook.",
    )
    cross_full.add_argument("--gt", help="Ground-truth multimodal CSV (auto-detected if omitted).")
    cross_full.add_argument("--preds", help="Predicted multimodal CSV (auto-detected if omitted).")
    cross_full.add_argument("--config", help="JSON config describing neuro/behavior columns (auto-inferred if omitted).")
    cross_full.add_argument("--output-root", type=Path, help="Output root (default ./outputs/).")
    cross_full.set_defaults(func=_run_cross_full)

    synthetic = subparsers.add_parser(
        "synthetic-validation",
        help="Generate synthetic neural datasets with known structure and audit the neuro benchmark.",
    )
    synthetic.add_argument("--output-root", type=Path, help="Output root (default ./outputs/synthetic-validation-<timestamp>/).")
    synthetic.add_argument("--n-sequences", type=int, default=12, help="Number of sequences in the synthetic dataset.")
    synthetic.add_argument("--seq-length", type=int, default=240, help="Sequence length in time bins.")
    synthetic.add_argument("--n-regions", type=int, default=16, help="Number of neural regions.")
    synthetic.add_argument("--latent-dim", type=int, default=4, help="Latent dynamical dimensionality.")
    synthetic.add_argument("--burn-in", type=int, default=64, help="Burn-in steps discarded from each sequence.")
    synthetic.add_argument("--system-seed", type=int, default=11, help="Random seed for the fixed synthetic system.")
    synthetic.add_argument("--oracle-replicates", type=int, default=3, help="Number of oracle draws scored against the reference dataset.")
    synthetic.add_argument("--no-save-datasets", action="store_true", help="Skip saving the generated CSV datasets.")
    synthetic.set_defaults(func=_run_synthetic_validation)

    synthetic_bio = subparsers.add_parser(
        "synthetic-validation-biophysical",
        help="Generate a harder spike/event-driven calcium world with known structure and audit the neuro benchmark.",
    )
    synthetic_bio.add_argument("--output-root", type=Path, help="Output root (default ./outputs/synthetic-validation-biophysical-<timestamp>/).")
    synthetic_bio.add_argument("--n-sequences", type=int, default=24, help="Number of sequences in the synthetic dataset.")
    synthetic_bio.add_argument("--seq-length", type=int, default=768, help="Sequence length in time bins.")
    synthetic_bio.add_argument("--n-regions", type=int, default=16, help="Number of neural regions.")
    synthetic_bio.add_argument("--latent-dim", type=int, default=6, help="Latent dynamical dimensionality.")
    synthetic_bio.add_argument("--burn-in", type=int, default=128, help="Burn-in steps discarded from each sequence.")
    synthetic_bio.add_argument("--system-seed", type=int, default=37, help="Random seed for the fixed synthetic system.")
    synthetic_bio.add_argument("--oracle-replicates", type=int, default=3, help="Number of oracle draws scored against the reference dataset.")
    synthetic_bio.add_argument("--state-switch-prob", type=float, default=0.018, help="Probability of switching hidden dynamical regimes at each step.")
    synthetic_bio.add_argument("--shared-event-prob", type=float, default=0.020, help="Probability of shared latent burst events.")
    synthetic_bio.add_argument("--refractory-strength", type=float, default=1.40, help="Strength of latent refractory suppression.")
    synthetic_bio.add_argument("--calcium-tau-rise", type=float, default=3.5, help="Mean calcium rise time constant.")
    synthetic_bio.add_argument("--calcium-tau-decay", type=float, default=18.0, help="Mean calcium decay time constant.")
    synthetic_bio.add_argument("--observation-noise", type=float, default=0.018, help="Additive observation noise scale.")
    synthetic_bio.add_argument("--neuropil-noise", type=float, default=0.012, help="Shared neuropil contamination scale.")
    synthetic_bio.add_argument("--baseline-drift-scale", type=float, default=0.035, help="Slow baseline drift amplitude.")
    synthetic_bio.add_argument("--no-save-datasets", action="store_true", help="Skip saving the generated CSV datasets.")
    synthetic_bio.set_defaults(func=_run_synthetic_validation_biophysical)

    return parser


def _run_neuro(args: argparse.Namespace) -> None:
    gt = _prompt_for_file("ground-truth", "gt_", args.gt)
    preds = _prompt_for_file("inference", "inference_", args.preds)
    scores = _quiet_call(compute_neuro_scores, preds, gt)
    _print_scores("Neuro scores", scores)
    out = _save_json_payload({"scores": scores}, requested=args.json_out, command="neuro-scores", preds=preds)
    print(f"Saved scores to {out}")


def _run_fidelity(args: argparse.Namespace) -> None:
    gt = _prompt_for_file("ground-truth", "gt_", args.gt)
    preds = _prompt_for_file("inference", "inference_", args.preds)
    scores = _quiet_call(compute_fidelity_scores, preds, gt)
    _print_scores("Fidelity scores", scores)
    out = _save_json_payload({"scores": scores}, requested=args.json_out, command="fidelity-scores", preds=preds)
    print(f"Saved scores to {out}")


def _run_neuro_full(args: argparse.Namespace) -> None:
    gt = _prompt_for_file("ground-truth", "gt_", args.gt)
    preds = _prompt_for_file("inference", "inference_", args.preds)
    out = _quiet_call(run_neuro_full_analysis, preds, gt, output_root=args.output_root)


def _run_etho(args: argparse.Namespace) -> None:
    scores, seq_scores, seq_means, seq_stds = _quiet_call(compute_etho_scores, Path(args.gt_dir), Path(args.inf_dir))
    _print_scores("Behavior scores", scores)
    if args.json_out:
        out = Path(args.json_out)
    else:
        out = Path(os.path.join("outputs", f"{args.inf_dir.split(os.sep)[-1].split('.')[0]}-etho-scores"))
    
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "global_scores": scores,
        "per_sequence": seq_scores,
        "per_sequence_mean": seq_means,
        "per_sequence_std": seq_stds,
    }
    with open(os.path.join(out, "scores.json"), "w") as f:
        f.write(json.dumps(payload, indent=2))

    print(f"Saved scores to {out}")


def _run_etho_full(args: argparse.Namespace) -> None:
    outdir = _quiet_call(run_etho_full_analysis, Path(args.gt_dir), Path(args.inf_dir), output_root=args.output_root)
    print(f"Behavioral full analysis executed natively. Outputs under {outdir}")


def _run_cross(args: argparse.Namespace) -> None:
    gt = _prompt_for_file("ground-truth", "gt_", args.gt)
    preds = _prompt_for_file("inference", "inference_", args.preds)
    cfg_path = _prompt_for_config(args.config)
    scores = _quiet_call(compute_cross_scores, preds, gt, cfg_path)
    _print_scores("Neuro scores", scores["neuro_scores"])
    _print_scores("Behavior scores", scores["behavior_scores"])
    _print_scores("Cross-modal scores", scores["cross_scores"])
    print("")
    _print_composite("Composite neuro", scores.get("neuro_composite", float("nan")))
    _print_composite("Composite etho", scores.get("etho_composite", float("nan")))
    _print_composite("Composite cross", scores.get("cross_composite", float("nan")))
    _print_composite("Final composite", scores.get("composite", float("nan")))

    if args.json_out is not None:
        out = Path(args.json_out)
    else:
        out = Path(os.path.join("outputs", f"{args.gt.split(os.sep)[-1].split('.')[0]}-cross-scores"))
    
    out.mkdir(parents=True, exist_ok=True)
    with open(f"{out}/scores.json", "w") as f:
        f.write(json.dumps(scores, indent=2))
    print(f"Saved scores to {out}")


def _run_cross_full(args: argparse.Namespace) -> None:
    gt = _prompt_for_file("ground-truth", "gt_", args.gt)
    preds = _prompt_for_file("inference", "inference_", args.preds)
    cfg_path = _prompt_for_config(args.config)
    outdir = _quiet_call(run_cross_full_analysis, preds, gt, cfg_path, output_root=args.output_root)
    print(f"Cross-modal full analysis executed. Outputs under {outdir}")


def _run_synthetic_validation(args: argparse.Namespace) -> None:
    spec = SyntheticNeuralSpec(
        n_sequences=args.n_sequences,
        seq_length=args.seq_length,
        n_regions=args.n_regions,
        latent_dim=args.latent_dim,
        burn_in=args.burn_in,
        system_seed=args.system_seed,
    )
    output_root = args.output_root or _default_output_dir("synthetic-validation")
    out = _quiet_call(
        run_synthetic_neuro_validation,
        output_root=output_root,
        spec=spec,
        oracle_replicates=args.oracle_replicates,
        save_datasets=not args.no_save_datasets,
    )
    oracle = out["family_oracle_summary"].set_index("score_name")["mean"].to_dict()
    metric_oracle = out["metric_oracle_summary"].set_index("score_name")["mean"].sort_values(ascending=False)
    fidelity_oracle = out["fidelity_oracle_summary"].set_index("score_name")["mean"].to_dict()
    selectivity = out["family_selectivity_df"]
    metric_selectivity = out["metric_selectivity_df"]

    print("\nSynthetic validation complete.")
    print(f"Output root: {out['output_root']}")
    print("\nOracle family means:")
    for key in [
        "family_distribution",
        "family_temporal_spectral",
        "family_relational",
        "family_geometry",
        "family_state_dynamics",
        "FINAL_COMPOSITE_SCORE",
        "ORACLE_VALIDATION_COMPOSITE_SCORE",
    ]:
        value = oracle.get(key, float("nan"))
        label = key.replace("family_", "").replace("_", " ")
        print(f"  {label:20s}: {value:.3f}")

    print("\nOracle metric means:")
    for metric_name, value in metric_oracle.items():
        print(f"  {metric_name:24s}: {value:.3f}")

    print("\nFidelity oracle means:")
    for key in ["Error_score01", "MI_score01", "family_fidelity", "FIDELITY_SCORE"]:
        value = fidelity_oracle.get(key, float("nan"))
        print(f"  {key:24s}: {value:.3f}")

    print("\nTargeted family drops at maximum perturbation:")
    for _, row in selectivity.iterrows():
        target = row["target_family"]
        drop = float(row[target]) if target in row else float("nan")
        print(f"  {row['perturbation_name']:28s}: {drop:.3f}")

    print("\nMost responsive metrics at maximum perturbation:")
    for _, row in metric_selectivity.iterrows():
        metric_row = pd.to_numeric(
            row.drop(labels=["perturbation_name", "target_family"]),
            errors="coerce",
        )
        metric_row = metric_row[np.isfinite(metric_row)].sort_values(ascending=False)
        if metric_row.empty:
            print(f"  {row['perturbation_name']:28s}: no finite metric drop")
            continue
        top_metric = metric_row.index[0]
        top_value = float(metric_row.iloc[0])
        print(f"  {row['perturbation_name']:28s}: {top_metric} ({top_value:.3f})")

    print("\nSaved artifacts:")
    print(f"  score runs            : {out['scores_path']}")
    print(f"  family oracle summary : {out['family_oracle_summary_path']}")
    print(f"  metric oracle summary : {out['metric_oracle_summary_path']}")
    print(f"  fidelity oracle summary: {out['fidelity_oracle_summary_path']}")
    print(f"  family selectivity    : {out['family_selectivity_path']}")
    print(f"  metric selectivity    : {out['metric_selectivity_path']}")
    print(f"  fidelity selectivity  : {out['fidelity_selectivity_path']}")
    print(f"  family dose response  : {out['family_dose_response_path']}")
    print(f"  metric dose response  : {out['metric_dose_response_path']}")
    print(f"  fidelity dose response: {out['fidelity_dose_response_path']}")
    print(f"  figures               : {out['ceiling_plot']}, {out['family_selectivity_plot']}, {out['metric_selectivity_plot']}, {out['dose_plot']}")


def _run_synthetic_validation_biophysical(args: argparse.Namespace) -> None:
    spec = BiophysicalSyntheticNeuralSpec(
        n_sequences=args.n_sequences,
        seq_length=args.seq_length,
        n_regions=args.n_regions,
        latent_dim=args.latent_dim,
        burn_in=args.burn_in,
        system_seed=args.system_seed,
        state_switch_prob=args.state_switch_prob,
        shared_event_prob=args.shared_event_prob,
        refractory_strength=args.refractory_strength,
        calcium_tau_rise=args.calcium_tau_rise,
        calcium_tau_decay=args.calcium_tau_decay,
        observation_noise=args.observation_noise,
        neuropil_noise=args.neuropil_noise,
        baseline_drift_scale=args.baseline_drift_scale,
    )
    output_root = args.output_root or _default_output_dir("synthetic-validation-biophysical")
    out = _quiet_call(
        run_biophysical_synthetic_neuro_validation,
        output_root=output_root,
        spec=spec,
        oracle_replicates=args.oracle_replicates,
        save_datasets=not args.no_save_datasets,
    )
    oracle = out["family_oracle_summary"].set_index("score_name")["mean"].to_dict()
    metric_oracle = out["metric_oracle_summary"].set_index("score_name")["mean"].sort_values(ascending=False)
    fidelity_oracle = out["fidelity_oracle_summary"].set_index("score_name")["mean"].to_dict()
    selectivity = out["family_selectivity_df"]
    metric_selectivity = out["metric_selectivity_df"]

    print("\nBiophysical synthetic validation complete.")
    print(f"Output root: {out['output_root']}")
    print("\nOracle family means:")
    for key in [
        "family_distribution",
        "family_temporal_spectral",
        "family_relational",
        "family_geometry",
        "family_state_dynamics",
        "FINAL_COMPOSITE_SCORE",
        "ORACLE_VALIDATION_COMPOSITE_SCORE",
    ]:
        value = oracle.get(key, float("nan"))
        label = key.replace("family_", "").replace("_", " ")
        print(f"  {label:20s}: {value:.3f}")

    print("\nOracle metric means:")
    for metric_name, value in metric_oracle.items():
        print(f"  {metric_name:24s}: {value:.3f}")

    print("\nFidelity oracle means:")
    for key in ["Error_score01", "MI_score01", "family_fidelity", "FIDELITY_SCORE"]:
        value = fidelity_oracle.get(key, float("nan"))
        print(f"  {key:24s}: {value:.3f}")

    print("\nTargeted family drops at maximum perturbation:")
    for _, row in selectivity.iterrows():
        target = row["target_family"]
        drop = float(row[target]) if target in row else float("nan")
        print(f"  {row['perturbation_name']:28s}: {drop:.3f}")

    print("\nMost responsive metrics at maximum perturbation:")
    for _, row in metric_selectivity.iterrows():
        metric_row = pd.to_numeric(
            row.drop(labels=["perturbation_name", "target_family"]),
            errors="coerce",
        )
        metric_row = metric_row[np.isfinite(metric_row)].sort_values(ascending=False)
        if metric_row.empty:
            print(f"  {row['perturbation_name']:28s}: no finite metric drop")
            continue
        top_metric = metric_row.index[0]
        top_value = float(metric_row.iloc[0])
        print(f"  {row['perturbation_name']:28s}: {top_metric} ({top_value:.3f})")

    print("\nSaved artifacts:")
    print(f"  score runs            : {out['scores_path']}")
    print(f"  family oracle summary : {out['family_oracle_summary_path']}")
    print(f"  metric oracle summary : {out['metric_oracle_summary_path']}")
    print(f"  fidelity oracle summary: {out['fidelity_oracle_summary_path']}")
    print(f"  family selectivity    : {out['family_selectivity_path']}")
    print(f"  metric selectivity    : {out['metric_selectivity_path']}")
    print(f"  fidelity selectivity  : {out['fidelity_selectivity_path']}")
    print(f"  family dose response  : {out['family_dose_response_path']}")
    print(f"  metric dose response  : {out['metric_dose_response_path']}")
    print(f"  fidelity dose response: {out['fidelity_dose_response_path']}")
    print(f"  figures               : {out['ceiling_plot']}, {out['family_selectivity_plot']}, {out['metric_selectivity_plot']}, {out['dose_plot']}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        parser.exit(1, "\nCancelled.\n")


if __name__ == "__main__":  # pragma: no cover
    main()