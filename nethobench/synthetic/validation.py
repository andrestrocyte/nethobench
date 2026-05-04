from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, replace
import io
import json
from pathlib import Path
import tempfile
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nethobench.neuro.metrics.composites import load_and_run_neuro_full_analysis
from nethobench.neuro.metrics.definitions import NEURO_FAMILY_WEIGHTS
from nethobench.utils.helpers import (
    get_region_names,
    generate_orthogonal_matrix,
    get_module_assignments
)
from nethobench.utils.calculation import (
    dataset_to_sequence_frame
)

FAMILY_COLUMNS = [f"family_{name}" for name in NEURO_FAMILY_WEIGHTS] + [
    "FINAL_COMPOSITE_SCORE",
    "ORACLE_VALIDATION_COMPOSITE_SCORE",
]

ORACLE_VALIDATION_WEIGHTS = {
    f"family_{name}": weight for name, weight in NEURO_FAMILY_WEIGHTS.items()
}
FIDELITY_COLUMNS = ["Error_score", "MI_score", "family_fidelity", "FIDELITY_SCORE"]


@dataclass(frozen=True)
class SyntheticNeuralSpec:
    n_sequences: int = 12
    seq_length: int = 240
    n_regions: int = 16
    latent_dim: int = 4
    burn_in: int = 64
    system_seed: int = 11
    latent_noise_scale: float = 0.22
    coupling_strength: float = 0.16
    calcium_alpha: float = 0.92
    oscillator_amplitude: float = 0.35
    oscillator_frequency: float = 0.045
    observation_noise: float = 0.035
    perturbation_name: str = "oracle"
    perturbation_level: float = 0.0


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    target_family: str
    description: str
    levels: tuple[float, ...]


@dataclass(frozen=True)
class SyntheticDataset:
    array: np.ndarray
    region_names: list[str]
    summary: dict[str, float | int | str]


DEFAULT_PERTURBATIONS = (
    PerturbationSpec(
        name="distribution_gain_bias",
        target_family="family_distribution",
        description="Change per-region gains and offsets while preserving latent dynamics.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="temporal_tau_frequency",
        target_family="family_temporal_spectral",
        description="Shift decay constants and oscillation frequencies to distort temporal spectra and memory.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="temporal_phase_desync",
        target_family="family_temporal_spectral",
        description="Desynchronize latent oscillatory phases and spread temporal frequencies across modes.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="temporal_kernel_mismatch",
        target_family="family_temporal_spectral",
        description="Impose heterogeneous calcium kernels and stronger observation jitter across regions.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="temporal_combo",
        target_family="family_temporal_spectral",
        description="Combine decay, phase, and kernel perturbations into a stronger temporal mismatch.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="relational_coupling_rewire",
        target_family="family_relational",
        description="Reduce latent coupling and partially rewire region loadings.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="relational_loading_swap",
        target_family="family_relational",
        description="Swap region-to-latent assignments to break conditional interaction structure.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="relational_sign_flip",
        target_family="family_relational",
        description="Flip interaction signs and invert subsets of region loadings to distort direct dependencies.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="relational_combo",
        target_family="family_relational",
        description="Combine rewiring, assignment swaps, and sign inversions for a strong relational mismatch.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="geometry_subspace_rotation",
        target_family="family_geometry",
        description="Rotate the dominant latent subspace while preserving total variance scale.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="geometry_variance_redistribution",
        target_family="family_geometry",
        description="Redistribute variance sharply across latent dimensions while preserving overall power.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="geometry_rank_truncation",
        target_family="family_geometry",
        description="Collapse latent rank and add nonlinear latent warping to distort manifold structure.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
    PerturbationSpec(
        name="geometry_combo",
        target_family="family_geometry",
        description="Combine subspace rotation, variance redistribution, and rank truncation into a strong geometry mismatch.",
        levels=(0.0, 0.25, 0.50, 0.75, 1.0),
    ),
)


def _blended_rotation(dim: int, level: float, rng: np.random.Generator) -> np.ndarray:
    base = np.eye(dim)
    target = generate_orthogonal_matrix(dim, rng)
    mixed = ((1.0 - level) * base) + (level * target)
    q, _ = np.linalg.qr(mixed)
    return q


def _build_system(spec: SyntheticNeuralSpec) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(spec.system_seed)
    latent_dim = spec.latent_dim
    n_regions = spec.n_regions
    level = float(spec.perturbation_level)
    perturbation = spec.perturbation_name

    base_decay = np.linspace(0.93, 0.68, latent_dim)
    cross = rng.normal(scale=0.10, size=(latent_dim, latent_dim))
    cross -= np.diag(np.diag(cross))
    A = np.diag(base_decay) + spec.coupling_strength * cross / np.sqrt(
        max(latent_dim, 1)
    )
    eig = np.max(np.abs(np.linalg.eigvals(A)))
    if np.isfinite(eig) and eig > 0.97:
        A *= 0.97 / float(eig)

    assignments = get_module_assignments(n_regions, latent_dim)
    W = rng.normal(scale=0.10, size=(n_regions, latent_dim))
    for region in range(n_regions):
        dom = int(assignments[region])
        W[region, dom] += 1.15
        W[region, (dom + 1) % latent_dim] += 0.20 * (-1 if region % 2 else 1)
    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-9

    gains = rng.lognormal(mean=0.0, sigma=0.14, size=n_regions)
    biases = np.linspace(-0.3, 0.4, n_regions) + rng.normal(scale=0.08, size=n_regions)
    alphas = np.clip(
        spec.calcium_alpha + rng.normal(scale=0.015, size=n_regions),
        0.75,
        0.98,
    )
    latent_phases = rng.uniform(0.0, 2.0 * np.pi, size=latent_dim)
    freq_mult = 1.0 + 0.12 * rng.normal(size=latent_dim)
    oscillation_frequency = spec.oscillator_frequency * np.clip(freq_mult, 0.6, 1.4)
    oscillator_weights = np.zeros(latent_dim, dtype=np.float64)
    oscillator_weights[: min(2, latent_dim)] = [1.0, -0.75][: min(2, latent_dim)]
    if latent_dim > 2:
        oscillator_weights[2:] = 0.15 * rng.normal(size=latent_dim - 2)

    observation_noise = spec.observation_noise
    latent_transform = np.eye(latent_dim)
    latent_warp_strength = 0.0

    if perturbation == "distribution_gain_bias":
        pattern = np.linspace(-0.5, 1.0, n_regions)
        gains = gains * (1.0 + 1.2 * level * pattern)
        biases = biases + 0.45 * level * pattern
    elif perturbation in {"temporal_tau_frequency", "temporal_combo"}:
        alphas = np.clip(alphas - 0.25 * level, 0.55, 0.98)
        oscillation_frequency = oscillation_frequency * (
            1.0 + level * np.linspace(0.7, 1.1, latent_dim)
        )
        oscillator_weights = oscillator_weights * (1.0 - 0.65 * level)
        A = A.copy()
        np.fill_diagonal(A, np.clip(np.diag(A) - 0.22 * level, 0.28, 0.99))
    if perturbation in {"temporal_phase_desync", "temporal_combo"}:
        latent_phases = latent_phases + (
            1.4 * level * rng.uniform(-np.pi, np.pi, size=latent_dim)
        )
        oscillation_frequency = oscillation_frequency * np.clip(
            1.0 + (0.55 * level * rng.normal(size=latent_dim)),
            0.30,
            1.80,
        )
    if perturbation in {"temporal_kernel_mismatch", "temporal_combo"}:
        pattern = np.sin(np.linspace(0.0, 2.0 * np.pi, n_regions, endpoint=False))
        alphas = np.clip(alphas + (0.45 * level * pattern), 0.25, 0.995)
        observation_noise = observation_noise * (1.0 + 3.0 * level)
    if perturbation in {"relational_coupling_rewire", "relational_combo"}:
        A = np.diag(np.diag(A)) + (1.0 - level) * (A - np.diag(np.diag(A)))
        perm = rng.permutation(n_regions)
        mix = level * 0.80
        W = ((1.0 - mix) * W) + (mix * W[perm])
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    if perturbation in {"relational_loading_swap", "relational_combo"}:
        perm = rng.permutation(n_regions)
        swap = 0.90 * level
        W = ((1.0 - swap) * W) + (swap * W[perm])
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    if perturbation in {"relational_sign_flip", "relational_combo"}:
        # Zero out connections to break absolute relational graphs 
        drop_mask = np.ones_like(W)
        drop_mask[rng.random(size=W.shape) < (0.35 * level)] = 0.10
        W = W * drop_mask
        W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
        A = np.diag(np.diag(A)) + ((1.0 - 2.0 * level) * (A - np.diag(np.diag(A))))
    if perturbation in {"geometry_subspace_rotation", "geometry_combo"}:
        # Rotate the ambient space of W to change the Subspace Angle
        Q = _blended_rotation(n_regions, level, rng)
        W = Q @ W
    if perturbation in {"geometry_variance_redistribution", "geometry_combo"}:
        scales = np.exp(level * np.linspace(1.2, -1.2, latent_dim))
        scales = scales / np.mean(scales)
        latent_transform = np.diag(scales) @ latent_transform
    if perturbation in {"geometry_rank_truncation", "geometry_combo"}:
         keep = np.ones(latent_dim, dtype=np.float64)
         if latent_dim > 1:
             # Keep the aggressive collapse rate (1.5 to 2.5), but 
             # restore the 0.02 floor so linear metrics don't break.
             keep[1:] = np.maximum(
                 1.0 - (level * np.linspace(1.5, 2.5, latent_dim - 1)), 0.02
             )
         latent_transform = np.diag(keep) @ latent_transform
         
         # Reduce the warp multiplier from 2.0 back down to 1.0
         latent_warp_strength = max(latent_warp_strength, 1.0 * level)

    return {
        "A": A.astype(np.float64),
        "W": W.astype(np.float64),
        "gains": gains.astype(np.float64),
        "biases": biases.astype(np.float64),
        "alphas": alphas.astype(np.float64),
        "latent_phases": latent_phases.astype(np.float64),
        "oscillation_frequency": oscillation_frequency.astype(np.float64),
        "oscillator_weights": oscillator_weights.astype(np.float64),
        "observation_noise": float(observation_noise),
        "assignments": assignments.astype(int),
        "latent_transform": latent_transform.astype(np.float64),
        "latent_warp_strength": float(latent_warp_strength),
    }


def generate_synthetic_neural_dataset(
    spec: SyntheticNeuralSpec,
    *,
    sample_seed: int,
) -> SyntheticDataset:
    """
    Generate a synthetic neural dataset from a latent dynamical system.

    Simulates latent latent oscillatory dynamics, projects them through a
    region loading matrix, and applies calcium-filter-like smoothing and
    observation noise to produce realistic synthetic neural traces.

    Args:
        spec: Specification object controlling sequences, length, regions,
            latent dimension, perturbations, and system hyperparameters.
        sample_seed: Random seed for sampling stochastic noise and initial
            conditions.

    Returns:
        A ``SyntheticDataset`` containing the 3D array of traces
        ``[n_sequences, seq_length, n_regions]``, region names, and a summary
        dictionary.
    """
    system = _build_system(spec)
    rng = np.random.default_rng(sample_seed)
    n_seq = spec.n_sequences
    seq_length = spec.seq_length
    burn = spec.burn_in
    n_regions = spec.n_regions
    latent_dim = spec.latent_dim

    data = np.zeros((n_seq, seq_length, n_regions), dtype=np.float64)
    total_steps = seq_length + burn
    for seq_idx in range(n_seq):
        z = rng.normal(scale=0.2, size=latent_dim)
        calcium = rng.normal(scale=0.05, size=n_regions)
        seq_trace = np.zeros((total_steps, n_regions), dtype=np.float64)
        for t in range(total_steps):
            osc = (
                spec.oscillator_amplitude
                * system["oscillator_weights"]
                * np.sin(
                    (2.0 * np.pi * system["oscillation_frequency"] * t)
                    + system["latent_phases"]
                )
            )
            z = (
                system["A"] @ z
                + osc
                + rng.normal(scale=spec.latent_noise_scale, size=latent_dim)
            )
            z_eff = system["latent_transform"] @ z
            if system["latent_warp_strength"] > 0 and latent_dim >= 2:
                z_eff = z_eff.copy()
                z_eff[1] = z_eff[1] + (
                    system["latent_warp_strength"] * ((z_eff[0] ** 2) - 1.0)
                )
            drive = system["W"] @ z_eff + system["biases"]
            activity = np.log1p(np.exp(drive))
            activity = system["gains"] * activity
            calcium = (system["alphas"] * calcium) + (
                (1.0 - system["alphas"]) * activity
            )
            calcium = calcium + rng.normal(
                scale=system["observation_noise"], size=n_regions
            )
            seq_trace[t] = calcium
        data[seq_idx] = seq_trace[burn:]

    region_names = get_region_names(n_regions)
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
        "mean_calcium_alpha": float(np.mean(system["alphas"])),
        "oscillator_frequency_mean": float(np.mean(system["oscillation_frequency"])),
        "coupling_strength": float(spec.coupling_strength),
    }
    return SyntheticDataset(array=data, region_names=region_names, summary=summary)


def _oracle_validation_composite(score_row: dict[str, object]) -> float:
    values = []
    weights = []
    for key, weight in ORACLE_VALIDATION_WEIGHTS.items():
        value = float(score_row.get(key, np.nan))
        if np.isfinite(value):
            values.append(value)
            weights.append(weight)
    if not values:
        return np.nan
    return float(np.sum(np.asarray(values) * np.asarray(weights)) / np.sum(weights))


def _relative_drop(reference: float, value: float) -> float:
    if not np.isfinite(reference) or not np.isfinite(value):
        return np.nan
    return float((reference - value) / max(reference, 1e-9))


def _score_row_metadata(
    spec: SyntheticNeuralSpec,
    kind: str,
    sample_seed: int,
    *,
    target_lookup: dict[str, str],
) -> dict[str, object]:
    return {
        "comparison_kind": kind,
        "perturbation_name": spec.perturbation_name,
        "perturbation_level": float(spec.perturbation_level),
        "target_family": target_lookup.get(spec.perturbation_name, "oracle"),
        "sample_seed": int(sample_seed),
        "n_sequences": int(spec.n_sequences),
        "seq_length": int(spec.seq_length),
        "n_regions": int(spec.n_regions),
        "latent_dim": int(spec.latent_dim),
    }


def _metric_score_columns(scores_df: pd.DataFrame) -> list[str]:
    return [column for column in scores_df.columns if column.endswith("_score")]


def _build_selectivity_table(
    scores_df: pd.DataFrame,
    score_columns: list[str],
    perturbation_defs: tuple[PerturbationSpec, ...],
) -> pd.DataFrame:
    oracle_means = (
        scores_df[scores_df["comparison_kind"] == "oracle"][score_columns]
        .mean(numeric_only=True)
        .to_dict()
    )
    rows: list[dict[str, object]] = []
    perturb_only = scores_df[scores_df["comparison_kind"] == "perturbation"]
    for perturbation in perturbation_defs:
        max_level = max(perturbation.levels)
        sub = perturb_only[
            (perturb_only["perturbation_name"] == perturbation.name)
            & (np.isclose(perturb_only["perturbation_level"], max_level))
        ]
        if sub.empty:
            continue
        score_means = sub[score_columns].mean(numeric_only=True).to_dict()
        row = {
            "perturbation_name": perturbation.name,
            "target_family": perturbation.target_family,
        }
        for score_col in score_columns:
            row[score_col] = _relative_drop(
                oracle_means.get(score_col, np.nan), score_means.get(score_col, np.nan)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_oracle_summary(
    scores_df: pd.DataFrame, score_columns: list[str]
) -> pd.DataFrame:
    sub = scores_df[scores_df["comparison_kind"] == "oracle"].copy()
    rows = []
    for column in score_columns:
        rows.append(
            {
                "score_name": column,
                "mean": float(sub[column].mean()),
                "std": float(sub[column].std(ddof=0)),
                "min": float(sub[column].min()),
                "max": float(sub[column].max()),
            }
        )
    return pd.DataFrame(rows)


def _build_dose_response(
    scores_df: pd.DataFrame,
    score_columns: list[str],
    perturbation_defs: tuple[PerturbationSpec, ...],
) -> pd.DataFrame:
    sub = scores_df[scores_df["comparison_kind"] == "perturbation"].copy()
    rows: list[dict[str, object]] = []
    for perturbation in perturbation_defs:
        for level in perturbation.levels:
            level_df = sub[
                (sub["perturbation_name"] == perturbation.name)
                & (np.isclose(sub["perturbation_level"], level))
            ]
            if level_df.empty:
                continue
            for score_name in score_columns:
                rows.append(
                    {
                        "perturbation_name": perturbation.name,
                        "target_family": perturbation.target_family,
                        "level": float(level),
                        "score_name": score_name,
                        "mean_score": float(level_df[score_name].mean()),
                    }
                )
    return pd.DataFrame(rows)


def _empty_selectivity_frame(score_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=["perturbation_name", "target_family", *score_columns])


def _empty_dose_frame(score_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "perturbation_name",
            "target_family",
            "level",
            "score_name",
            "mean_score",
        ]
    )


def _plot_family_ceiling_floor(
    scores_df: pd.DataFrame,
    output_path: Path,
    *,
    perturbation_defs: tuple[PerturbationSpec, ...],
) -> None:
    oracle_means = scores_df[scores_df["comparison_kind"] == "oracle"][
        FAMILY_COLUMNS
    ].mean(numeric_only=True)
    perturb_only = scores_df[scores_df["comparison_kind"] == "perturbation"]
    labels = [
        col.replace("family_", "").replace("_", " ").title() for col in FAMILY_COLUMNS
    ]
    n_cols = 3
    n_rows = int(np.ceil(len(FAMILY_COLUMNS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.0 * n_rows), sharey=True)
    axes = np.asarray(axes).ravel()
    for ax, family_col, label in zip(axes, FAMILY_COLUMNS, labels):
        values = [float(oracle_means[family_col])]
        names = ["oracle"]
        for perturbation in perturbation_defs:
            sub = perturb_only[
                (perturb_only["perturbation_name"] == perturbation.name)
                & (
                    np.isclose(
                        perturb_only["perturbation_level"], max(perturbation.levels)
                    )
                )
            ]
            values.append(float(sub[family_col].mean()))
            names.append(perturbation.name.replace("_", "\n"))
        colors = ["#2E86AB"] + ["#D1495B"] * (len(values) - 1)
        ax.bar(np.arange(len(values)), values, color=colors)
        ax.set_title(label)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(names, rotation=35, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(FAMILY_COLUMNS) :]:
        ax.axis("off")
    fig.suptitle("Oracle ceilings versus targeted synthetic mismatches", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_selectivity_heatmap(
    selectivity_df: pd.DataFrame,
    output_path: Path,
    *,
    score_columns: list[str],
    title: str,
) -> None:
    matrix = selectivity_df[score_columns].to_numpy(dtype=np.float64)
    fig_width = max(8.5, 0.48 * len(score_columns))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    vmax = np.nanmax(matrix) if np.isfinite(matrix).any() else 0.30
    im = ax.imshow(
        matrix, cmap="viridis", aspect="auto", vmin=0.0, vmax=max(0.30, float(vmax))
    )
    ax.set_xticks(np.arange(len(score_columns)))
    ax.set_xticklabels(
        [
            col.replace("family_", "").replace("_score", "").replace("_", " ")
            for col in score_columns
        ],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(np.arange(len(selectivity_df)))
    ax.set_yticklabels(selectivity_df["perturbation_name"])
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            val = matrix[row_idx, col_idx]
            if np.isfinite(val):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )
    fig.colorbar(im, ax=ax, shrink=0.85, label="relative drop from oracle")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_dose_response(
    dose_df: pd.DataFrame,
    output_path: Path,
    *,
    perturbation_defs: tuple[PerturbationSpec, ...],
) -> None:
    fig, axes = plt.subplots(
        len(perturbation_defs),
        1,
        figsize=(10, 3.0 * len(perturbation_defs)),
        sharex=False,
    )
    if len(perturbation_defs) == 1:
        axes = [axes]
    for ax, perturbation in zip(axes, perturbation_defs):
        sub = dose_df[dose_df["perturbation_name"] == perturbation.name]
        target = perturbation.target_family
        target_df = sub[sub["score_name"] == target].sort_values("level")
        composite_df = sub[sub["score_name"] == "FINAL_COMPOSITE_SCORE"].sort_values(
            "level"
        )
        ax.plot(
            target_df["level"],
            target_df["mean_score"],
            marker="o",
            linewidth=2.6,
            color="#1f77b4",
            label=target.replace("family_", "target: "),
        )
        ax.plot(
            composite_df["level"],
            composite_df["mean_score"],
            marker="s",
            linewidth=2.0,
            color="#d62728",
            label="final composite",
        )
        ax.set_title(perturbation.name)
        ax.set_ylabel("mean score")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("perturbation magnitude")
    fig.suptitle("Dose-response curves for targeted synthetic perturbations", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_validation_pipeline(
    *,
    spec,
    generator_fn,
    output_root: Path,
    oracle_replicates: int,
    perturbations: Iterable[PerturbationSpec],
    save_datasets: bool,
    save_artifacts: bool = True,
    prefix: str,
    oracle_seed_base: int,
    perturbation_seed: int,
    family_selectivity_title: str,
    metric_selectivity_title: str,
) -> dict[str, object]:
    output_root = Path(output_root)
    if save_artifacts:
        output_root.mkdir(parents=True, exist_ok=True)
        tables_dir = output_root / "tables"
        figures_dir = output_root / "figures"
        datasets_dir = output_root / "datasets"
        for path in [tables_dir, figures_dir, datasets_dir]:
            path.mkdir(parents=True, exist_ok=True)
    else:
        tables_dir = output_root / "tables"
        figures_dir = output_root / "figures"
        datasets_dir = output_root / "datasets"

    perturbation_defs = tuple(perturbations)
    target_lookup = {
        perturbation.name: perturbation.target_family
        for perturbation in perturbation_defs
    }

    reference = generator_fn(spec, sample_seed=101)
    if save_artifacts and save_datasets:
        dataset_to_sequence_frame(reference.array, reference.region_names).to_csv(
            datasets_dir / f"{prefix}_ground_truth.csv", index=False
        )

    rows: list[dict[str, object]] = []
    for rep_idx in range(oracle_replicates):
        oracle_seed = oracle_seed_base + rep_idx
        oracle = generator_fn(spec, sample_seed=oracle_seed)
        if save_artifacts and save_datasets and rep_idx == 0:
            dataset_to_sequence_frame(oracle.array, oracle.region_names).to_csv(
                datasets_dir / f"{prefix}_oracle_prediction.csv", index=False
            )
        scores = load_and_run_neuro_full_analysis(reference.array, oracle.array)
        scores["ORACLE_VALIDATION_COMPOSITE_SCORE"] = _oracle_validation_composite(
            scores
        )
        scores["FIDELITY_SCORE"] = scores.get("family_fidelity", float("nan"))
        row = _score_row_metadata(
            spec, "oracle", oracle_seed, target_lookup=target_lookup
        )
        row.update(scores)
        rows.append(row)

    for perturbation in perturbation_defs:
        for level in perturbation.levels:
            perturbed_spec = replace(
                spec,
                perturbation_name=perturbation.name,
                perturbation_level=float(level),
            )
            perturbed = generator_fn(perturbed_spec, sample_seed=perturbation_seed)
            if save_artifacts and save_datasets and level == max(perturbation.levels):
                out_name = f"{perturbation.name}-level-{int(level * 100):03d}.csv"
                dataset_to_sequence_frame(
                    perturbed.array, perturbed.region_names
                ).to_csv(datasets_dir / out_name, index=False)
            scores = load_and_run_neuro_full_analysis(
                reference.array, perturbed.array
            )
            scores["ORACLE_VALIDATION_COMPOSITE_SCORE"] = _oracle_validation_composite(
                scores
            )
            scores["FIDELITY_SCORE"] = scores.get("family_fidelity", float("nan"))
            row = _score_row_metadata(
                perturbed_spec, "perturbation", perturbation_seed, target_lookup=target_lookup
            )
            row.update(scores)
            rows.append(row)

    scores_df = pd.DataFrame(rows)
    metric_columns = _metric_score_columns(scores_df)
    family_oracle_summary = _build_oracle_summary(scores_df, FAMILY_COLUMNS)
    metric_oracle_summary = _build_oracle_summary(scores_df, metric_columns)
    fidelity_oracle_summary = _build_oracle_summary(scores_df, FIDELITY_COLUMNS)
    if perturbation_defs:
        family_selectivity_df = _build_selectivity_table(
            scores_df, FAMILY_COLUMNS, perturbation_defs
        )
        metric_selectivity_df = _build_selectivity_table(
            scores_df, metric_columns, perturbation_defs
        )
        fidelity_selectivity_df = _build_selectivity_table(
            scores_df, FIDELITY_COLUMNS, perturbation_defs
        )
        family_dose_df = _build_dose_response(
            scores_df, FAMILY_COLUMNS, perturbation_defs
        )
        metric_dose_df = _build_dose_response(
            scores_df, metric_columns, perturbation_defs
        )
        fidelity_dose_df = _build_dose_response(
            scores_df, FIDELITY_COLUMNS, perturbation_defs
        )
    else:
        family_selectivity_df = _empty_selectivity_frame(FAMILY_COLUMNS)
        metric_selectivity_df = _empty_selectivity_frame(metric_columns)
        fidelity_selectivity_df = _empty_selectivity_frame(FIDELITY_COLUMNS)
        family_dose_df = _empty_dose_frame(FAMILY_COLUMNS)
        metric_dose_df = _empty_dose_frame(metric_columns)
        fidelity_dose_df = _empty_dose_frame(FIDELITY_COLUMNS)

    metadata = {
        "generator_spec": asdict(spec),
        "reference_summary": reference.summary,
        "perturbations": [asdict(p) for p in perturbation_defs],
        "n_oracle_replicates": int(oracle_replicates),
    }

    scores_path = tables_dir / f"{prefix}_score_runs.csv"
    family_oracle_path = tables_dir / f"{prefix}_family_oracle_summary.csv"
    metric_oracle_path = tables_dir / f"{prefix}_metric_oracle_summary.csv"
    fidelity_oracle_path = tables_dir / f"{prefix}_fidelity_oracle_summary.csv"
    family_selectivity_path = tables_dir / f"{prefix}_family_selectivity.csv"
    metric_selectivity_path = tables_dir / f"{prefix}_metric_selectivity.csv"
    fidelity_selectivity_path = tables_dir / f"{prefix}_fidelity_selectivity.csv"
    family_dose_path = tables_dir / f"{prefix}_family_dose_response.csv"
    metric_dose_path = tables_dir / f"{prefix}_metric_dose_response.csv"
    fidelity_dose_path = tables_dir / f"{prefix}_fidelity_dose_response.csv"
    metadata_path = output_root / f"{prefix}_validation_metadata.json"
    if save_artifacts:
        scores_df.to_csv(scores_path, index=False)
        family_oracle_summary.to_csv(family_oracle_path, index=False)
        metric_oracle_summary.to_csv(metric_oracle_path, index=False)
        fidelity_oracle_summary.to_csv(fidelity_oracle_path, index=False)
        family_selectivity_df.to_csv(family_selectivity_path, index=False)
        metric_selectivity_df.to_csv(metric_selectivity_path, index=False)
        fidelity_selectivity_df.to_csv(fidelity_selectivity_path, index=False)
        family_dose_df.to_csv(family_dose_path, index=False)
        metric_dose_df.to_csv(metric_dose_path, index=False)
        fidelity_dose_df.to_csv(fidelity_dose_path, index=False)
        metadata_path.write_text(json.dumps(metadata, indent=2))

    ceiling_plot = figures_dir / f"{prefix}_family_ceiling_floor.png"
    family_selectivity_plot = figures_dir / f"{prefix}_family_selectivity.png"
    metric_selectivity_plot = figures_dir / f"{prefix}_metric_selectivity.png"
    dose_plot = figures_dir / f"{prefix}_dose_response.png"
    if save_artifacts and perturbation_defs:
        _plot_family_ceiling_floor(
            scores_df, ceiling_plot, perturbation_defs=perturbation_defs
        )
        _plot_selectivity_heatmap(
            family_selectivity_df,
            family_selectivity_plot,
            score_columns=FAMILY_COLUMNS,
            title=family_selectivity_title,
        )
        _plot_selectivity_heatmap(
            metric_selectivity_df,
            metric_selectivity_plot,
            score_columns=metric_columns,
            title=metric_selectivity_title,
        )
        _plot_dose_response(
            family_dose_df, dose_plot, perturbation_defs=perturbation_defs
        )

    return {
        "output_root": output_root,
        "scores_path": scores_path,
        "family_oracle_summary_path": family_oracle_path,
        "metric_oracle_summary_path": metric_oracle_path,
        "fidelity_oracle_summary_path": fidelity_oracle_path,
        "family_selectivity_path": family_selectivity_path,
        "metric_selectivity_path": metric_selectivity_path,
        "fidelity_selectivity_path": fidelity_selectivity_path,
        "family_dose_response_path": family_dose_path,
        "metric_dose_response_path": metric_dose_path,
        "fidelity_dose_response_path": fidelity_dose_path,
        "metadata_path": metadata_path,
        "ceiling_plot": ceiling_plot,
        "family_selectivity_plot": family_selectivity_plot,
        "metric_selectivity_plot": metric_selectivity_plot,
        "dose_plot": dose_plot,
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
    }


def run_synthetic_neuro_validation(
    *,
    output_root: Path,
    spec: SyntheticNeuralSpec | None = None,
    oracle_replicates: int = 3,
    perturbations: Iterable[PerturbationSpec] = DEFAULT_PERTURBATIONS,
    save_datasets: bool = True,
    save_artifacts: bool = True,
) -> dict[str, object]:
    """
    Run the full synthetic neural validation pipeline.

    Generates oracle and perturbed synthetic datasets, scores them with the
    neuro composite metrics, and produces summary tables, selectivity
    heatmaps, dose-response curves, and optional CSV artifacts.

    Args:
        output_root: Directory where tables, figures, and datasets are saved.
        spec: ``SyntheticNeuralSpec`` instance. If None, default spec values
            are used.
        oracle_replicates: Number of independent oracle runs to average for
            the baseline ceiling. Defaults to 3.
        perturbations: Iterable of ``PerturbationSpec`` definitions to test.
            Defaults to ``DEFAULT_PERTURBATIONS``.
        save_datasets: Whether to write generated datasets to CSV.
            Defaults to True.
        save_artifacts: Whether to write tables, figures, and metadata to disk.
            Defaults to True.

    Returns:
        Dictionary of output paths and resulting DataFrames from the
        validation pipeline.
    """
    return run_validation_pipeline(
        spec=spec or SyntheticNeuralSpec(),
        generator_fn=generate_synthetic_neural_dataset,
        output_root=output_root,
        oracle_replicates=oracle_replicates,
        perturbations=perturbations,
        save_datasets=save_datasets,
        save_artifacts=save_artifacts,
        prefix="synthetic",
        oracle_seed_base=501,
        perturbation_seed=701,
        family_selectivity_title="Family selectivity heatmap",
        metric_selectivity_title="Metric selectivity heatmap",
    )
