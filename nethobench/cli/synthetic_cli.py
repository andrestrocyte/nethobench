from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nethobench.synthetic.validation import (
    SyntheticNeuralSpec,
    run_synthetic_neuro_validation,
)
from nethobench.synthetic.biophysical import (
    BiophysicalSyntheticNeuralSpec,
    run_biophysical_synthetic_neuro_validation,
)
from nethobench.cli.utils import _quiet_call, _default_output_dir


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
    metric_oracle = (
        out["metric_oracle_summary"]
        .set_index("score_name")["mean"]
        .sort_values(ascending=False)
    )
    fidelity_oracle = (
        out["fidelity_oracle_summary"].set_index("score_name")["mean"].to_dict()
    )
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
    print(
        f"  figures               : {out['ceiling_plot']}, {out['family_selectivity_plot']}, {out['metric_selectivity_plot']}, {out['dose_plot']}"
    )


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
    output_root = args.output_root or _default_output_dir(
        "synthetic-validation-biophysical"
    )
    out = _quiet_call(
        run_biophysical_synthetic_neuro_validation,
        output_root=output_root,
        spec=spec,
        oracle_replicates=args.oracle_replicates,
        save_datasets=not args.no_save_datasets,
    )
    oracle = out["family_oracle_summary"].set_index("score_name")["mean"].to_dict()
    metric_oracle = (
        out["metric_oracle_summary"]
        .set_index("score_name")["mean"]
        .sort_values(ascending=False)
    )
    fidelity_oracle = (
        out["fidelity_oracle_summary"].set_index("score_name")["mean"].to_dict()
    )
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
    print(
        f"  figures               : {out['ceiling_plot']}, {out['family_selectivity_plot']}, {out['metric_selectivity_plot']}, {out['dose_plot']}"
    )


def add_synthetic_subparsers(subparsers) -> None:
    synthetic = subparsers.add_parser(
        "synthetic-validation",
        help="Generate synthetic neural datasets with known structure and audit the neuro benchmark.",
    )
    synthetic.add_argument(
        "--output-root",
        type=Path,
        help="Output root (default ./outputs/synthetic-validation-<timestamp>/).",
    )
    synthetic.add_argument(
        "--n-sequences",
        type=int,
        default=12,
        help="Number of sequences in the synthetic dataset.",
    )
    synthetic.add_argument(
        "--seq-length", type=int, default=240, help="Sequence length in time bins."
    )
    synthetic.add_argument(
        "--n-regions", type=int, default=16, help="Number of neural regions."
    )
    synthetic.add_argument(
        "--latent-dim", type=int, default=4, help="Latent dynamical dimensionality."
    )
    synthetic.add_argument(
        "--burn-in",
        type=int,
        default=64,
        help="Burn-in steps discarded from each sequence.",
    )
    synthetic.add_argument(
        "--system-seed",
        type=int,
        default=11,
        help="Random seed for the fixed synthetic system.",
    )
    synthetic.add_argument(
        "--oracle-replicates",
        type=int,
        default=3,
        help="Number of oracle draws scored against the reference dataset.",
    )
    synthetic.add_argument(
        "--no-save-datasets",
        action="store_true",
        help="Skip saving the generated CSV datasets.",
    )
    synthetic.set_defaults(func=_run_synthetic_validation)

    synthetic_bio = subparsers.add_parser(
        "synthetic-validation-biophysical",
        help="Generate a harder spike/event-driven calcium world with known structure and audit the neuro benchmark.",
    )
    synthetic_bio.add_argument(
        "--output-root",
        type=Path,
        help="Output root (default ./outputs/synthetic-validation-biophysical-<timestamp>/).",
    )
    synthetic_bio.add_argument(
        "--n-sequences",
        type=int,
        default=24,
        help="Number of sequences in the synthetic dataset.",
    )
    synthetic_bio.add_argument(
        "--seq-length", type=int, default=768, help="Sequence length in time bins."
    )
    synthetic_bio.add_argument(
        "--n-regions", type=int, default=16, help="Number of neural regions."
    )
    synthetic_bio.add_argument(
        "--latent-dim", type=int, default=6, help="Latent dynamical dimensionality."
    )
    synthetic_bio.add_argument(
        "--burn-in",
        type=int,
        default=128,
        help="Burn-in steps discarded from each sequence.",
    )
    synthetic_bio.add_argument(
        "--system-seed",
        type=int,
        default=37,
        help="Random seed for the fixed synthetic system.",
    )
    synthetic_bio.add_argument(
        "--oracle-replicates",
        type=int,
        default=3,
        help="Number of oracle draws scored against the reference dataset.",
    )
    synthetic_bio.add_argument(
        "--state-switch-prob",
        type=float,
        default=0.018,
        help="Probability of switching hidden dynamical regimes at each step.",
    )
    synthetic_bio.add_argument(
        "--shared-event-prob",
        type=float,
        default=0.020,
        help="Probability of shared latent burst events.",
    )
    synthetic_bio.add_argument(
        "--refractory-strength",
        type=float,
        default=1.40,
        help="Strength of latent refractory suppression.",
    )
    synthetic_bio.add_argument(
        "--calcium-tau-rise",
        type=float,
        default=3.5,
        help="Mean calcium rise time constant.",
    )
    synthetic_bio.add_argument(
        "--calcium-tau-decay",
        type=float,
        default=18.0,
        help="Mean calcium decay time constant.",
    )
    synthetic_bio.add_argument(
        "--observation-noise",
        type=float,
        default=0.018,
        help="Additive observation noise scale.",
    )
    synthetic_bio.add_argument(
        "--neuropil-noise",
        type=float,
        default=0.012,
        help="Shared neuropil contamination scale.",
    )
    synthetic_bio.add_argument(
        "--baseline-drift-scale",
        type=float,
        default=0.035,
        help="Slow baseline drift amplitude.",
    )
    synthetic_bio.add_argument(
        "--no-save-datasets",
        action="store_true",
        help="Skip saving the generated CSV datasets.",
    )
    synthetic_bio.set_defaults(func=_run_synthetic_validation_biophysical)
