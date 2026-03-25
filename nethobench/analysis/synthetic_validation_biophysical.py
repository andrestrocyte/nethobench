from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .synthetic_validation import (
    FAMILY_COLUMNS,
    FIDELITY_COLUMNS,
    PerturbationSpec,
    SyntheticDataset,
    _build_dose_response,
    _build_oracle_summary,
    _build_selectivity_table,
    _metric_score_columns,
    _oracle_validation_composite,
    _plot_dose_response,
    _plot_family_ceiling_floor,
    _plot_selectivity_heatmap,
    _quiet_fidelity_from_arrays,
    _quiet_scores_from_arrays,
    _score_row_metadata,
    dataset_to_sequence_frame,
)


@dataclass(frozen=True)
class BiophysicalSyntheticNeuralSpec:
    n_sequences: int = 24
    seq_length: int = 768
    n_regions: int = 16
    latent_dim: int = 6
    burn_in: int = 128
    system_seed: int = 37
    latent_noise_scale: float = 0.18
    coupling_strength: float = 0.20
    state_switch_prob: float = 0.018
    shared_event_prob: float = 0.020
    refractory_strength: float = 1.40
    calcium_tau_rise: float = 3.5
    calcium_tau_decay: float = 18.0
    observation_noise: float = 0.018
    neuropil_noise: float = 0.012
    baseline_drift_scale: float = 0.035
    perturbation_name: str = "oracle"
    perturbation_level: float = 0.0


DEFAULT_BIOPHYSICAL_PERTURBATIONS = (
    PerturbationSpec(
        name="distribution_rate_gain",
        target_family="family_distribution",
        description="Shift firing-rate gains and calcium baselines while preserving the latent event structure.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="temporal_kernel_switch",
        target_family="family_temporal_spectral",
        description="Distort calcium kernels and accelerate hidden-state switching to damage temporal structure.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="temporal_refractory_jitter",
        target_family="family_temporal_spectral",
        description="Weaken refractory structure and jitter burst timing across latent assemblies.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="temporal_combo",
        target_family="family_temporal_spectral",
        description="Combine kernel mismatch, state-switch acceleration, and refractory disruption.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="relational_assembly_shuffle",
        target_family="family_relational",
        description="Shuffle latent assembly-to-region assignments and disrupt cross-region dependency structure.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="relational_coupling_dropout",
        target_family="family_relational",
        description="Drop direct latent couplings and collapse coordinated shared events.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="relational_combo",
        target_family="family_relational",
        description="Combine assembly shuffling and coupling dropout for a strong relational mismatch.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="geometry_subspace_rotation",
        target_family="family_geometry",
        description="Rotate the dominant latent event subspace while preserving overall power.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="geometry_rank_collapse",
        target_family="family_geometry",
        description="Collapse latent dimensionality and compress state-specific event structure.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="geometry_combo",
        target_family="family_geometry",
        description="Combine latent rotation and rank collapse into a strong geometry distortion.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _region_names(n_regions: int) -> list[str]:
    return [f"biophysical-region-{idx:02d}" for idx in range(1, n_regions + 1)]


def _orthogonal_matrix(dim: int, rng: np.random.Generator) -> np.ndarray:
    mat = rng.normal(size=(dim, dim))
    q, _ = np.linalg.qr(mat)
    return q


def _module_assignments(n_regions: int, latent_dim: int) -> np.ndarray:
    bins = np.linspace(0, latent_dim, n_regions, endpoint=False)
    return np.floor(bins).astype(int)


def _blend_rotation(dim: int, level: float, rng: np.random.Generator) -> np.ndarray:
    mixed = ((1.0 - level) * np.eye(dim)) + (level * _orthogonal_matrix(dim, rng))
    q, _ = np.linalg.qr(mixed)
    return q


def _ar2_from_taus(tau_rise: np.ndarray, tau_decay: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rise = np.exp(-1.0 / np.maximum(tau_rise, 1.05))
    decay = np.exp(-1.0 / np.maximum(tau_decay, 1.10))
    g1 = rise + decay
    g2 = -(rise * decay)
    return g1.astype(np.float64), g2.astype(np.float64)


def _build_biophysical_system(spec: BiophysicalSyntheticNeuralSpec) -> dict[str, np.ndarray | float | int]:
    rng = np.random.default_rng(spec.system_seed)
    latent_dim = spec.latent_dim
    n_regions = spec.n_regions
    level = float(spec.perturbation_level)
    perturbation = spec.perturbation_name

    state_count = 3
    assignments = _module_assignments(n_regions, latent_dim)

    A_states = []
    for state_idx in range(state_count):
        base_scales = np.linspace(0.95 - 0.03 * state_idx, 0.70 - 0.02 * state_idx, latent_dim)
        rot = _orthogonal_matrix(latent_dim, rng)
        cross = rng.normal(scale=0.06, size=(latent_dim, latent_dim))
        cross -= np.diag(np.diag(cross))
        mat = rot @ np.diag(base_scales) @ rot.T
        mat += spec.coupling_strength * cross / np.sqrt(max(latent_dim, 1))
        eig = np.max(np.abs(np.linalg.eigvals(mat)))
        if np.isfinite(eig) and eig > 0.985:
            mat *= 0.985 / float(eig)
        A_states.append(mat.astype(np.float64))
    A_states = np.stack(A_states, axis=0)

    state_biases = rng.normal(scale=0.20, size=(state_count, latent_dim))
    state_biases[1, : min(2, latent_dim)] += 0.7
    state_biases[2, -min(2, latent_dim):] -= 0.6

    W_spike = rng.normal(scale=0.08, size=(n_regions, latent_dim))
    W_cont = rng.normal(scale=0.06, size=(n_regions, latent_dim))
    for region in range(n_regions):
        dom = int(assignments[region])
        W_spike[region, dom] += 1.4
        W_cont[region, dom] += 0.8
        W_cont[region, (dom + 1) % latent_dim] += 0.20 * (-1 if region % 2 else 1)
    W_spike /= np.linalg.norm(W_spike, axis=1, keepdims=True) + 1e-9
    W_cont /= np.linalg.norm(W_cont, axis=1, keepdims=True) + 1e-9

    region_bias = np.linspace(-2.6, -1.2, n_regions) + rng.normal(scale=0.18, size=n_regions)
    state_region_bias = rng.normal(scale=0.16, size=(state_count, n_regions))
    spike_amplitudes = rng.lognormal(mean=-1.0, sigma=0.22, size=n_regions)
    refractory_region = np.clip(
        spec.refractory_strength + rng.normal(scale=0.12, size=n_regions),
        0.6,
        2.2,
    )
    tau_rise = np.clip(spec.calcium_tau_rise + rng.normal(scale=0.45, size=n_regions), 1.5, 8.0)
    tau_decay = np.clip(spec.calcium_tau_decay + rng.normal(scale=2.2, size=n_regions), 6.0, 36.0)
    ar1, ar2 = _ar2_from_taus(tau_rise, tau_decay)

    drift_frequency = rng.uniform(0.002, 0.010, size=n_regions)
    drift_phase = rng.uniform(0.0, 2.0 * np.pi, size=n_regions)
    latent_event_bias = rng.normal(scale=0.18, size=(state_count, latent_dim))
    latent_event_bias[1] += 0.35
    shared_event_profile = np.clip(rng.uniform(0.40, 0.85, size=(state_count, latent_dim)), 0.20, 0.98)
    region_event_profile = np.clip(rng.uniform(0.15, 0.70, size=(state_count, n_regions)), 0.05, 0.95)

    latent_transform = np.eye(latent_dim)
    latent_rank_scale = np.ones(latent_dim, dtype=np.float64)
    if perturbation == "distribution_rate_gain":
        pattern = np.linspace(-0.6, 1.0, n_regions)
        spike_amplitudes = spike_amplitudes * (1.0 + 0.9 * level * pattern)
        region_bias = region_bias + (0.55 * level * pattern)
    if perturbation in {"temporal_kernel_switch", "temporal_combo"}:
        tau_decay = tau_decay * np.clip(1.0 - 0.55 * level, 0.35, 1.5)
        tau_rise = tau_rise * np.clip(1.0 + 0.70 * level, 0.60, 1.8)
        ar1, ar2 = _ar2_from_taus(tau_rise, tau_decay)
    if perturbation in {"temporal_refractory_jitter", "temporal_combo"}:
        refractory_region = np.clip(refractory_region * (1.0 - 0.55 * level), 0.15, 2.5)
        shared_event_profile = np.clip(shared_event_profile + (0.30 * level * rng.normal(size=shared_event_profile.shape)), 0.05, 0.99)
    if perturbation in {"relational_assembly_shuffle", "relational_combo"}:
        perm = rng.permutation(n_regions)
        mix = 0.85 * level
        W_spike = ((1.0 - mix) * W_spike) + (mix * W_spike[perm])
        W_cont = ((1.0 - mix) * W_cont) + (mix * W_cont[perm])
        W_spike /= np.linalg.norm(W_spike, axis=1, keepdims=True) + 1e-9
        W_cont /= np.linalg.norm(W_cont, axis=1, keepdims=True) + 1e-9
    if perturbation in {"relational_coupling_dropout", "relational_combo"}:
        for state_idx in range(state_count):
            mat = A_states[state_idx].copy()
            diag = np.diag(np.diag(mat))
            A_states[state_idx] = diag + ((1.0 - 0.9 * level) * (mat - diag))
        region_event_profile = ((1.0 - 0.75 * level) * region_event_profile) + (0.75 * level * region_event_profile[:, rng.permutation(n_regions)])
    if perturbation in {"geometry_subspace_rotation", "geometry_combo"}:
        latent_transform = _blend_rotation(latent_dim, level, rng) @ latent_transform
    if perturbation in {"geometry_rank_collapse", "geometry_combo"}:
        latent_rank_scale[1:] = np.maximum(1.0 - (level * np.linspace(0.80, 1.25, latent_dim - 1)), 0.03)
        latent_transform = np.diag(latent_rank_scale) @ latent_transform

    return {
        "A_states": A_states.astype(np.float64),
        "state_biases": state_biases.astype(np.float64),
        "W_spike": W_spike.astype(np.float64),
        "W_cont": W_cont.astype(np.float64),
        "region_bias": region_bias.astype(np.float64),
        "state_region_bias": state_region_bias.astype(np.float64),
        "spike_amplitudes": spike_amplitudes.astype(np.float64),
        "refractory_region": refractory_region.astype(np.float64),
        "tau_rise": tau_rise.astype(np.float64),
        "tau_decay": tau_decay.astype(np.float64),
        "ar1": ar1.astype(np.float64),
        "ar2": ar2.astype(np.float64),
        "drift_frequency": drift_frequency.astype(np.float64),
        "drift_phase": drift_phase.astype(np.float64),
        "latent_event_bias": latent_event_bias.astype(np.float64),
        "shared_event_profile": shared_event_profile.astype(np.float64),
        "region_event_profile": region_event_profile.astype(np.float64),
        "latent_transform": latent_transform.astype(np.float64),
        "latent_rank_scale": latent_rank_scale.astype(np.float64),
        "assignments": assignments.astype(int),
    }


def generate_biophysical_synthetic_dataset(
    spec: BiophysicalSyntheticNeuralSpec,
    *,
    sample_seed: int,
) -> SyntheticDataset:
    bundle = generate_biophysical_synthetic_bundle(spec, sample_seed=sample_seed)
    return SyntheticDataset(
        array=bundle["signals"],
        region_names=bundle["region_names"],
        summary=bundle["summary"],
    )


def generate_biophysical_synthetic_bundle(
    spec: BiophysicalSyntheticNeuralSpec,
    *,
    sample_seed: int,
) -> dict[str, object]:
    system = _build_biophysical_system(spec)
    rng = np.random.default_rng(sample_seed)
    n_seq = spec.n_sequences
    seq_length = spec.seq_length
    burn = spec.burn_in
    n_regions = spec.n_regions
    latent_dim = spec.latent_dim
    state_count = int(system["A_states"].shape[0])  # type: ignore[index]

    total_steps = seq_length + burn
    data = np.zeros((n_seq, seq_length, n_regions), dtype=np.float64)
    state_hist = np.zeros((n_seq, total_steps), dtype=np.int64)
    spike_hist = np.zeros((n_seq, total_steps, n_regions), dtype=np.float64)

    for seq_idx in range(n_seq):
        state = int(rng.integers(0, state_count))
        z = rng.normal(scale=0.15, size=latent_dim)
        latent_prev = np.zeros(latent_dim, dtype=np.float64)
        region_prev = np.zeros(n_regions, dtype=np.float64)
        calcium_prev1 = np.zeros(n_regions, dtype=np.float64)
        calcium_prev2 = np.zeros(n_regions, dtype=np.float64)
        state_dwell = 0

        for t in range(total_steps):
            switch_prob = spec.state_switch_prob
            if spec.perturbation_name in {"temporal_kernel_switch", "temporal_combo"}:
                switch_prob *= (1.0 + 4.0 * float(spec.perturbation_level))
            if state_dwell > 8 and rng.random() < switch_prob:
                state = int(rng.integers(0, state_count - 1))
                if state >= state_hist[seq_idx, max(0, t - 1)]:
                    state += 1
                state_dwell = 0
            state_dwell += 1

            z = (
                system["A_states"][state] @ z  # type: ignore[index]
                + system["state_biases"][state]  # type: ignore[index]
                + rng.normal(scale=spec.latent_noise_scale, size=latent_dim)
            )
            z_eff = system["latent_transform"] @ z  # type: ignore[operator]

            latent_logits = z_eff + system["latent_event_bias"][state] - (spec.refractory_strength * latent_prev)  # type: ignore[index]
            latent_prob = _sigmoid(latent_logits)
            if rng.random() < spec.shared_event_prob * (1.0 + 0.6 * (state == 1)):
                latent_prob = np.maximum(latent_prob, system["shared_event_profile"][state])  # type: ignore[index]
            latent_spikes = rng.binomial(1, np.clip(latent_prob, 1e-4, 0.995)).astype(np.float64)
            latent_prev = latent_spikes

            region_drive = (
                system["region_bias"]  # type: ignore[operator]
                + system["state_region_bias"][state]  # type: ignore[index]
                + (system["W_spike"] @ latent_spikes)  # type: ignore[operator]
                + 0.35 * (system["W_cont"] @ z_eff)  # type: ignore[operator]
                - (system["refractory_region"] * region_prev)  # type: ignore[operator]
            )
            region_prob = _sigmoid(region_drive)
            if rng.random() < spec.shared_event_prob * 0.6:
                region_prob = np.maximum(region_prob, system["region_event_profile"][state])  # type: ignore[index]
            region_spikes = rng.binomial(1, np.clip(region_prob, 1e-4, 0.995)).astype(np.float64)
            region_prev = region_spikes

            drift = spec.baseline_drift_scale * np.sin(
                (2.0 * np.pi * system["drift_frequency"] * t) + system["drift_phase"]  # type: ignore[operator]
            )
            calcium = (
                system["ar1"] * calcium_prev1  # type: ignore[operator]
                + system["ar2"] * calcium_prev2  # type: ignore[operator]
                + (system["spike_amplitudes"] * region_spikes)  # type: ignore[operator]
                + drift
            )
            calcium_prev2 = calcium_prev1
            calcium_prev1 = calcium
            neuropil = rng.normal(scale=spec.neuropil_noise)
            observed = calcium + (0.25 * neuropil) + rng.normal(scale=spec.observation_noise, size=n_regions)

            state_hist[seq_idx, t] = state
            spike_hist[seq_idx, t] = region_spikes
            if t >= burn:
                data[seq_idx, t - burn] = observed

    region_names = _region_names(n_regions)
    flat = data.reshape(-1, n_regions)
    summary = {
        "perturbation_name": spec.perturbation_name,
        "perturbation_level": float(spec.perturbation_level),
        "n_sequences": int(spec.n_sequences),
        "seq_length": int(spec.seq_length),
        "n_regions": int(spec.n_regions),
        "latent_dim": int(spec.latent_dim),
        "sample_seed": int(sample_seed),
        "system_seed": int(spec.system_seed),
        "mean_trace_value": float(np.mean(flat)),
        "std_trace_value": float(np.std(flat)),
        "mean_spike_rate": float(np.mean(spike_hist[:, burn:])),
        "mean_state_switch_rate": float(np.mean(np.diff(state_hist, axis=1) != 0)),
        "mean_tau_decay": float(np.mean(system["tau_decay"])),  # type: ignore[arg-type]
        "mean_tau_rise": float(np.mean(system["tau_rise"])),  # type: ignore[arg-type]
        "coupling_strength": float(spec.coupling_strength),
    }
    return {
        "signals": data,
        "spikes": spike_hist[:, burn:].copy(),
        "states": state_hist.copy(),
        "region_names": region_names,
        "summary": summary,
    }


def run_biophysical_synthetic_neuro_validation(
    *,
    output_root: Path,
    spec: BiophysicalSyntheticNeuralSpec | None = None,
    oracle_replicates: int = 3,
    perturbations: Iterable[PerturbationSpec] = DEFAULT_BIOPHYSICAL_PERTURBATIONS,
    save_datasets: bool = True,
) -> dict[str, object]:
    spec = spec or BiophysicalSyntheticNeuralSpec()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    datasets_dir = output_root / "datasets"
    for path in [tables_dir, figures_dir, datasets_dir]:
        path.mkdir(parents=True, exist_ok=True)

    perturbation_defs = tuple(perturbations)
    target_lookup = {perturbation.name: perturbation.target_family for perturbation in perturbation_defs}

    reference = generate_biophysical_synthetic_dataset(spec, sample_seed=101)
    if save_datasets:
        dataset_to_sequence_frame(reference.array, reference.region_names).to_csv(
            datasets_dir / "biophysical_ground_truth.csv", index=False
        )

    rows: list[dict[str, object]] = []
    for rep_idx in range(oracle_replicates):
        oracle_seed = 701 + rep_idx
        oracle = generate_biophysical_synthetic_dataset(spec, sample_seed=oracle_seed)
        if save_datasets and rep_idx == 0:
            dataset_to_sequence_frame(oracle.array, oracle.region_names).to_csv(
                datasets_dir / "biophysical_oracle_prediction.csv", index=False
            )
        scores = _quiet_scores_from_arrays(reference.array, oracle.array, region_names=reference.region_names)
        scores["ORACLE_VALIDATION_COMPOSITE_SCORE"] = _oracle_validation_composite(scores)
        scores.update(_quiet_fidelity_from_arrays(reference.array, oracle.array, region_names=reference.region_names))
        row = _score_row_metadata(spec, "oracle", oracle_seed, target_lookup=target_lookup)
        row.update(scores)
        rows.append(row)

    for perturbation in perturbation_defs:
        for level in perturbation.levels:
            perturbed_spec = replace(spec, perturbation_name=perturbation.name, perturbation_level=float(level))
            perturbed = generate_biophysical_synthetic_dataset(perturbed_spec, sample_seed=1701)
            if save_datasets and level == max(perturbation.levels):
                out_name = f"{perturbation.name}-level-{int(level * 100):03d}.csv"
                dataset_to_sequence_frame(perturbed.array, perturbed.region_names).to_csv(datasets_dir / out_name, index=False)
            scores = _quiet_scores_from_arrays(reference.array, perturbed.array, region_names=reference.region_names)
            scores["ORACLE_VALIDATION_COMPOSITE_SCORE"] = _oracle_validation_composite(scores)
            scores.update(_quiet_fidelity_from_arrays(reference.array, perturbed.array, region_names=reference.region_names))
            row = _score_row_metadata(perturbed_spec, "perturbation", 1701, target_lookup=target_lookup)
            row.update(scores)
            rows.append(row)

    scores_df = pd.DataFrame(rows)
    metric_columns = _metric_score_columns(scores_df)
    family_oracle_summary = _build_oracle_summary(scores_df, FAMILY_COLUMNS)
    metric_oracle_summary = _build_oracle_summary(scores_df, metric_columns)
    fidelity_oracle_summary = _build_oracle_summary(scores_df, FIDELITY_COLUMNS)
    if perturbation_defs:
        family_selectivity_df = _build_selectivity_table(scores_df, FAMILY_COLUMNS, perturbation_defs)
        metric_selectivity_df = _build_selectivity_table(scores_df, metric_columns, perturbation_defs)
        fidelity_selectivity_df = _build_selectivity_table(scores_df, FIDELITY_COLUMNS, perturbation_defs)
        family_dose_df = _build_dose_response(scores_df, FAMILY_COLUMNS, perturbation_defs)
        metric_dose_df = _build_dose_response(scores_df, metric_columns, perturbation_defs)
        fidelity_dose_df = _build_dose_response(scores_df, FIDELITY_COLUMNS, perturbation_defs)
    else:
        family_selectivity_df = pd.DataFrame(columns=["perturbation_name", "target_family", *FAMILY_COLUMNS])
        metric_selectivity_df = pd.DataFrame(columns=["perturbation_name", "target_family", *metric_columns])
        fidelity_selectivity_df = pd.DataFrame(columns=["perturbation_name", "target_family", *FIDELITY_COLUMNS])
        family_dose_df = pd.DataFrame(columns=["perturbation_name", "target_family", "level", "score_name", "mean_score"])
        metric_dose_df = pd.DataFrame(columns=["perturbation_name", "target_family", "level", "score_name", "mean_score"])
        fidelity_dose_df = pd.DataFrame(columns=["perturbation_name", "target_family", "level", "score_name", "mean_score"])

    metadata = {
        "generator_spec": asdict(spec),
        "reference_summary": reference.summary,
        "oracle_replicates": int(oracle_replicates),
        "perturbations": [asdict(p) for p in perturbation_defs],
    }

    scores_path = tables_dir / "biophysical_scores.csv"
    family_oracle_summary_path = tables_dir / "biophysical_family_oracle_summary.csv"
    metric_oracle_summary_path = tables_dir / "biophysical_metric_oracle_summary.csv"
    fidelity_oracle_summary_path = tables_dir / "biophysical_fidelity_oracle_summary.csv"
    family_selectivity_path = tables_dir / "biophysical_family_selectivity.csv"
    metric_selectivity_path = tables_dir / "biophysical_metric_selectivity.csv"
    fidelity_selectivity_path = tables_dir / "biophysical_fidelity_selectivity.csv"
    family_dose_response_path = tables_dir / "biophysical_family_dose_response.csv"
    metric_dose_response_path = tables_dir / "biophysical_metric_dose_response.csv"
    fidelity_dose_response_path = tables_dir / "biophysical_fidelity_dose_response.csv"
    metadata_path = output_root / "biophysical_metadata.json"

    scores_df.to_csv(scores_path, index=False)
    family_oracle_summary.to_csv(family_oracle_summary_path, index=False)
    metric_oracle_summary.to_csv(metric_oracle_summary_path, index=False)
    fidelity_oracle_summary.to_csv(fidelity_oracle_summary_path, index=False)
    family_selectivity_df.to_csv(family_selectivity_path, index=False)
    metric_selectivity_df.to_csv(metric_selectivity_path, index=False)
    fidelity_selectivity_df.to_csv(fidelity_selectivity_path, index=False)
    family_dose_df.to_csv(family_dose_response_path, index=False)
    metric_dose_df.to_csv(metric_dose_response_path, index=False)
    fidelity_dose_df.to_csv(fidelity_dose_response_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    ceiling_plot = figures_dir / "biophysical_family_ceiling_floor.png"
    family_selectivity_plot = figures_dir / "biophysical_family_selectivity.png"
    metric_selectivity_plot = figures_dir / "biophysical_metric_selectivity.png"
    dose_plot = figures_dir / "biophysical_dose_response.png"

    if perturbation_defs:
        _plot_family_ceiling_floor(scores_df, ceiling_plot, perturbation_defs=perturbation_defs)
        _plot_selectivity_heatmap(
            family_selectivity_df,
            family_selectivity_plot,
            score_columns=FAMILY_COLUMNS,
            title="Biophysical synthetic family selectivity",
        )
        _plot_selectivity_heatmap(
            metric_selectivity_df,
            metric_selectivity_plot,
            score_columns=metric_columns,
            title="Biophysical synthetic metric selectivity",
        )
        _plot_dose_response(family_dose_df, dose_plot, perturbation_defs=perturbation_defs)

    return {
        "output_root": str(output_root),
        "scores_path": str(scores_path),
        "family_oracle_summary_path": str(family_oracle_summary_path),
        "metric_oracle_summary_path": str(metric_oracle_summary_path),
        "fidelity_oracle_summary_path": str(fidelity_oracle_summary_path),
        "family_selectivity_path": str(family_selectivity_path),
        "metric_selectivity_path": str(metric_selectivity_path),
        "fidelity_selectivity_path": str(fidelity_selectivity_path),
        "family_dose_response_path": str(family_dose_response_path),
        "metric_dose_response_path": str(metric_dose_response_path),
        "fidelity_dose_response_path": str(fidelity_dose_response_path),
        "ceiling_plot": str(ceiling_plot),
        "family_selectivity_plot": str(family_selectivity_plot),
        "metric_selectivity_plot": str(metric_selectivity_plot),
        "dose_plot": str(dose_plot),
        "scores_df": scores_df,
        "family_oracle_summary": family_oracle_summary,
        "metric_oracle_summary": metric_oracle_summary,
        "fidelity_oracle_summary": fidelity_oracle_summary,
        "family_selectivity_df": family_selectivity_df,
        "metric_selectivity_df": metric_selectivity_df,
        "fidelity_selectivity_df": fidelity_selectivity_df,
        "family_dose_df": family_dose_df,
        "metric_dose_df": metric_dose_df,
        "fidelity_dose_df": fidelity_dose_df,
        "metadata_path": str(metadata_path),
    }
