from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .additional_neuro_metrics import compute_additional_structural_metrics
from .direct_neuro_metrics import (
    compute_graph_score01,
    compute_manifold_score01,
    compute_moment_score01,
    compute_trajectory_score01,
)
from .score_definitions import compute_fidelity_composite, compute_neuro_composite, compute_neuro_family_scores


FAMILY_COLUMNS = [
    "family_distribution",
    "family_temporal_spectral",
    "family_relational",
    "family_geometry",
    "family_state_dynamics",
]
SCORE_COLUMNS = FAMILY_COLUMNS + ["FINAL_COMPOSITE_SCORE", "FIDELITY_SCORE", "mse", "corr"]
DEFAULT_OUTPUT_ROOT = Path("outputs") / "bulletproof-validation"


@dataclass(frozen=True)
class BulletproofValidationConfig:
    mode: str = "quick"
    output_root: Path = DEFAULT_OUTPUT_ROOT
    n_sequences: int = 8
    seq_length: int = 160
    n_regions: int = 12
    latent_dim: int = 5
    seeds: tuple[int, ...] = (7, 11)
    min_free_disk_gb: float = 8.0
    run_full_notebook_scores: bool = False

    @classmethod
    def for_mode(
        cls,
        mode: str,
        *,
        output_root: Path | None = None,
        min_free_disk_gb: float = 8.0,
        run_full_notebook_scores: bool = False,
    ) -> "BulletproofValidationConfig":
        if mode not in {"quick", "full"}:
            raise ValueError("mode must be 'quick' or 'full'.")
        if mode == "quick":
            return cls(
                mode=mode,
                output_root=output_root or DEFAULT_OUTPUT_ROOT,
                n_sequences=8,
                seq_length=160,
                n_regions=12,
                latent_dim=5,
                seeds=(7, 11),
                min_free_disk_gb=min_free_disk_gb,
                run_full_notebook_scores=run_full_notebook_scores,
            )
        return cls(
            mode=mode,
            output_root=output_root or DEFAULT_OUTPUT_ROOT,
            n_sequences=14,
            seq_length=240,
            n_regions=16,
            latent_dim=6,
            seeds=(7, 11, 19),
            min_free_disk_gb=min_free_disk_gb,
            run_full_notebook_scores=run_full_notebook_scores,
        )


@dataclass
class SyntheticSystemResult:
    system: str
    seed: int
    neural: np.ndarray
    oracle: np.ndarray
    latent: np.ndarray
    labels: np.ndarray
    behavior: np.ndarray
    metadata: dict[str, object]


def _free_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return float(usage.free / (1024**3))


def _check_disk(path: Path, min_free_gb: float, phase: str) -> dict[str, object]:
    path.mkdir(parents=True, exist_ok=True)
    free = _free_disk_gb(path)
    if free < min_free_gb:
        raise RuntimeError(
            f"Refusing {phase}: only {free:.2f} GB free at {path}, below reserve {min_free_gb:.2f} GB."
        )
    return {"phase": phase, "path": str(path), "free_gb": free, "min_free_gb": float(min_free_gb)}


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _standardize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    mu = np.nanmean(arr.reshape(-1, arr.shape[-1]), axis=0, keepdims=True)
    sd = np.nanstd(arr.reshape(-1, arr.shape[-1]), axis=0, keepdims=True)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    out = (arr - mu.reshape(1, 1, -1)) / sd.reshape(1, 1, -1)
    out[~np.isfinite(out)] = 0.0
    return out


def _flatten_finite(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).reshape(-1)
    return out[np.isfinite(out)]


def _corr(gt: np.ndarray, pred: np.ndarray) -> float:
    x = _flatten_finite(gt)
    y = _flatten_finite(pred)
    n = min(x.size, y.size)
    if n < 4:
        return float("nan")
    x = x[:n]
    y = y[:n]
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _hist_score(gt: np.ndarray, pred: np.ndarray, bins: int = 60) -> float:
    gt_flat = _flatten_finite(gt)
    pred_flat = _flatten_finite(pred)
    if gt_flat.size < 10 or pred_flat.size < 10:
        return float("nan")
    lo, hi = np.nanquantile(gt_flat, [0.001, 0.999])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(gt_flat)), float(np.nanmax(gt_flat) + 1e-6)
    p, edges = np.histogram(gt_flat, bins=bins, range=(lo, hi), density=False)
    q, _ = np.histogram(pred_flat, bins=edges, density=False)
    p = p.astype(np.float64) + 1e-8
    q = q.astype(np.float64) + 1e-8
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return float(np.clip(1.0 - js / math.log(2.0), 0.0, 1.0))


def _quantile_score(gt: np.ndarray, pred: np.ndarray) -> float:
    qs = np.asarray([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    gt_q = np.nanquantile(gt.reshape(-1, gt.shape[-1]), qs, axis=0)
    pr_q = np.nanquantile(pred.reshape(-1, pred.shape[-1]), qs, axis=0)
    err = float(np.nanmean(np.abs(gt_q - pr_q)))
    scale = float(np.nanmean(np.nanstd(gt.reshape(-1, gt.shape[-1]), axis=0)))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return float(1.0 / (1.0 + err / scale))


def _mean_score(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_mean = np.nanmean(gt.reshape(-1, gt.shape[-1]), axis=0)
    pr_mean = np.nanmean(pred.reshape(-1, pred.shape[-1]), axis=0)
    err = float(np.nanmean(np.abs(gt_mean - pr_mean)))
    scale = float(np.nanmean(np.nanstd(gt.reshape(-1, gt.shape[-1]), axis=0)))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return float(1.0 / (1.0 + err / scale))


def _fast_fidelity_scores(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mse = float(np.nanmean((gt - pred) ** 2))
    rmse = math.sqrt(max(mse, 0.0))
    scale = float(np.nanstd(gt))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    error_score = float(1.0 / (1.0 + rmse / scale))
    mi_proxy = float(np.clip(0.5 * (_corr(gt, pred) + 1.0), 0.0, 1.0)) if np.isfinite(_corr(gt, pred)) else float("nan")
    out = {"Error_score01": error_score, "MI_score01": mi_proxy}
    out["family_fidelity"] = compute_fidelity_composite(out)
    out["FIDELITY_SCORE"] = out["family_fidelity"]
    return out


def score_arrays_fast(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Compute NethoBench-compatible scores without executing the full notebook."""
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if gt.shape != pred.shape or gt.ndim != 3:
        raise ValueError(f"Expected matching [sequence,time,channel] arrays, got {gt.shape} and {pred.shape}.")
    scores: dict[str, float] = {}

    def _safe(fn):
        try:
            return fn()
        except Exception:
            return float("nan")

    scores["KL_or_JSD_score01"] = _safe(lambda: _hist_score(gt, pred))
    scores["QNT_score01"] = _safe(lambda: _quantile_score(gt, pred))
    scores["Mean_score01"] = _safe(lambda: _mean_score(gt, pred))
    scores["MOM_score01"] = _safe(lambda: compute_moment_score01(gt, pred)["scores"]["MOM_score01"])
    scores["TRJDIST_score01"] = _safe(lambda: compute_trajectory_score01(gt, pred)["scores"]["TRJDIST_score01"])
    scores["GRAPH_score01"] = _safe(lambda: compute_graph_score01(gt, pred)["scores"]["GRAPH_score01"])
    scores["MANI_score01"] = _safe(lambda: compute_manifold_score01(gt, pred)["scores"]["MANI_score01"])
    extra = _safe(lambda: compute_additional_structural_metrics(gt, pred).get("scores", {}))
    if isinstance(extra, dict):
        for key, value in extra.items():
            scores[key] = _safe_float(value)
    families = compute_neuro_family_scores(scores)
    scores.update(families)
    scores["FINAL_COMPOSITE_SCORE"] = compute_neuro_composite(scores)
    scores.update(_fast_fidelity_scores(gt, pred))
    scores["mse"] = float(np.nanmean((gt - pred) ** 2))
    scores["corr"] = _corr(gt, pred)
    return {key: _safe_float(value) for key, value in scores.items()}


def _stabilize_matrix(mat: np.ndarray, radius: float = 0.94) -> np.ndarray:
    eig = np.max(np.abs(np.linalg.eigvals(mat)))
    if np.isfinite(eig) and eig > radius:
        mat = mat * (radius / float(eig))
    return np.asarray(mat, dtype=np.float64)


def _make_loading(rng: np.random.Generator, n_regions: int, latent_dim: int) -> np.ndarray:
    W = rng.normal(scale=0.25, size=(n_regions, latent_dim))
    for region in range(n_regions):
        W[region, region % latent_dim] += 1.0
        W[region, (region + 1) % latent_dim] += 0.25
    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    return W


def _observe(latent: np.ndarray, W: np.ndarray, rng: np.random.Generator, noise: float = 0.08) -> np.ndarray:
    y = np.tanh(latent @ W.T) + 0.12 * (latent @ W.T) ** 2
    y += rng.normal(scale=noise, size=y.shape)
    return _standardize(y)


def _behavior_from_latent(latent: np.ndarray, labels: np.ndarray) -> np.ndarray:
    x = np.tanh(latent[..., 0])
    y = np.tanh(latent[..., 1 % latent.shape[-1]])
    speed = np.concatenate(
        [np.zeros((latent.shape[0], 1)), np.linalg.norm(np.diff(latent[..., :2], axis=1), axis=2)],
        axis=1,
    )
    state = labels / max(float(np.nanmax(labels)), 1.0)
    return np.stack([x, y, speed, state], axis=2)


def _simulate_nonlinear_lds(
    seed: int,
    n_sequences: int,
    seq_length: int,
    n_regions: int,
    latent_dim: int,
    *,
    oracle: bool = False,
) -> SyntheticSystemResult:
    rng = np.random.default_rng(seed)
    param_rng = np.random.default_rng(seed + 1000)
    A = _stabilize_matrix(0.78 * np.eye(latent_dim) + param_rng.normal(scale=0.09, size=(latent_dim, latent_dim)))
    B = param_rng.normal(scale=0.16, size=(latent_dim, latent_dim))
    U = param_rng.normal(scale=0.28, size=(latent_dim, 3))
    W = _make_loading(param_rng, n_regions, latent_dim)
    sim_rng = np.random.default_rng(seed + (101 if oracle else 0))
    z = np.zeros((n_sequences, seq_length, latent_dim), dtype=np.float64)
    labels = np.zeros((n_sequences, seq_length), dtype=int)
    for s in range(n_sequences):
        z[s, 0] = sim_rng.normal(scale=0.5, size=latent_dim)
        for t in range(seq_length - 1):
            label = int((t // max(10, seq_length // 5) + s) % 3)
            labels[s, t] = label
            u = np.zeros(3)
            u[label] = 1.0
            z[s, t + 1] = A @ z[s, t] + B @ np.tanh(z[s, t]) + U @ u + sim_rng.normal(scale=0.20, size=latent_dim)
    labels[:, -1] = labels[:, -2]
    neural = _observe(z, W, sim_rng)
    return SyntheticSystemResult("nonlinear_lds", seed, neural, neural.copy(), z, labels, _behavior_from_latent(z, labels), {"A": A.tolist()})


def _simulate_slds(
    seed: int,
    n_sequences: int,
    seq_length: int,
    n_regions: int,
    latent_dim: int,
    *,
    oracle: bool = False,
) -> SyntheticSystemResult:
    param_rng = np.random.default_rng(seed + 2000)
    sim_rng = np.random.default_rng(seed + (202 if oracle else 0))
    n_states = 3
    P = np.asarray([[0.92, 0.06, 0.02], [0.04, 0.90, 0.06], [0.05, 0.05, 0.90]], dtype=np.float64)
    As = []
    for k in range(n_states):
        base = 0.65 * np.eye(latent_dim) + param_rng.normal(scale=0.07, size=(latent_dim, latent_dim))
        base += 0.16 * np.roll(np.eye(latent_dim), k + 1, axis=1)
        As.append(_stabilize_matrix(base, radius=0.90))
    offsets = param_rng.normal(scale=0.35, size=(n_states, latent_dim))
    W = _make_loading(param_rng, n_regions, latent_dim)
    z = np.zeros((n_sequences, seq_length, latent_dim), dtype=np.float64)
    labels = np.zeros((n_sequences, seq_length), dtype=int)
    for s in range(n_sequences):
        state = int(sim_rng.integers(n_states))
        z[s, 0] = sim_rng.normal(scale=0.5, size=latent_dim)
        for t in range(seq_length - 1):
            labels[s, t] = state
            z[s, t + 1] = As[state] @ z[s, t] + offsets[state] + sim_rng.normal(scale=0.16, size=latent_dim)
            state = int(sim_rng.choice(n_states, p=P[state]))
    labels[:, -1] = labels[:, -2]
    neural = _observe(z, W, sim_rng, noise=0.07)
    return SyntheticSystemResult("switching_lds", seed, neural, neural.copy(), z, labels, _behavior_from_latent(z, labels), {"transition_matrix": P.tolist()})


def _lorenz_rhs(x: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> np.ndarray:
    return np.asarray([sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]], dtype=np.float64)


def _simulate_lorenz(
    seed: int,
    n_sequences: int,
    seq_length: int,
    n_regions: int,
    latent_dim: int,
    *,
    oracle: bool = False,
) -> SyntheticSystemResult:
    param_rng = np.random.default_rng(seed + 3000)
    sim_rng = np.random.default_rng(seed + (303 if oracle else 0))
    W = _make_loading(param_rng, n_regions, max(latent_dim, 3))
    z = np.zeros((n_sequences, seq_length, max(latent_dim, 3)), dtype=np.float64)
    labels = np.zeros((n_sequences, seq_length), dtype=int)
    dt = 0.012
    skip = 6
    for s in range(n_sequences):
        x = np.asarray([1.0, 1.0, 20.0], dtype=np.float64) + sim_rng.normal(scale=0.4, size=3)
        for t in range(seq_length):
            for _ in range(skip):
                k1 = _lorenz_rhs(x)
                k2 = _lorenz_rhs(x + 0.5 * dt * k1)
                k3 = _lorenz_rhs(x + 0.5 * dt * k2)
                k4 = _lorenz_rhs(x + dt * k3)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            z[s, t, :3] = x / np.asarray([12.0, 16.0, 35.0])
            if z.shape[-1] > 3:
                z[s, t, 3:] = np.sin((t + 1) * np.linspace(0.03, 0.09, z.shape[-1] - 3))
            labels[s, t] = int(x[2] > 25.0) + int(x[0] > 0.0)
    neural = _observe(z, W, sim_rng, noise=0.06)
    return SyntheticSystemResult("lorenz", seed, neural, neural.copy(), z, labels, _behavior_from_latent(z, labels), {"dt": dt, "skip": skip})


def _simulate_low_rank_rnn(
    seed: int,
    n_sequences: int,
    seq_length: int,
    n_regions: int,
    latent_dim: int,
    *,
    oracle: bool = False,
) -> SyntheticSystemResult:
    param_rng = np.random.default_rng(seed + 4000)
    sim_rng = np.random.default_rng(seed + (404 if oracle else 0))
    rank = min(3, latent_dim)
    M = param_rng.normal(scale=1.1 / math.sqrt(latent_dim), size=(latent_dim, rank))
    N = param_rng.normal(scale=1.1 / math.sqrt(latent_dim), size=(latent_dim, rank))
    J = (M @ N.T) / rank
    U = param_rng.normal(scale=0.55, size=(latent_dim, 2))
    W = _make_loading(param_rng, n_regions, latent_dim)
    z = np.zeros((n_sequences, seq_length, latent_dim), dtype=np.float64)
    labels = np.zeros((n_sequences, seq_length), dtype=int)
    alpha = 0.25
    for s in range(n_sequences):
        z[s, 0] = sim_rng.normal(scale=0.4, size=latent_dim)
        for t in range(seq_length - 1):
            label = int(np.sin(2 * np.pi * t / max(32, seq_length // 2) + s) > 0)
            labels[s, t] = label
            u = np.asarray([1.0, -1.0]) if label else np.asarray([-1.0, 1.0])
            dz = -z[s, t] + np.tanh(J @ z[s, t] + U @ u)
            z[s, t + 1] = z[s, t] + alpha * dz + sim_rng.normal(scale=0.11, size=latent_dim)
    labels[:, -1] = labels[:, -2]
    neural = _observe(z, W, sim_rng, noise=0.06)
    return SyntheticSystemResult("low_rank_rnn", seed, neural, neural.copy(), z, labels, _behavior_from_latent(z, labels), {"rank": rank, "J": J.tolist()})


def _simulate_spiking_calcium(
    seed: int,
    n_sequences: int,
    seq_length: int,
    n_regions: int,
    latent_dim: int,
    *,
    oracle: bool = False,
) -> SyntheticSystemResult:
    base = _simulate_slds(seed, n_sequences, seq_length, n_regions, latent_dim, oracle=oracle)
    sim_rng = np.random.default_rng(seed + (505 if oracle else 5))
    drive = 0.7 * base.neural + 0.35 * np.roll(base.neural, shift=1, axis=2)
    rate = np.exp(np.clip(drive, -2.4, 2.1)) * 0.18
    spikes = sim_rng.poisson(rate).astype(np.float64)
    calcium = np.zeros_like(spikes)
    alpha = np.linspace(0.86, 0.96, n_regions)
    for t in range(1, seq_length):
        calcium[:, t] = alpha.reshape(1, -1) * calcium[:, t - 1] + spikes[:, t]
    calcium += sim_rng.normal(scale=0.04, size=calcium.shape)
    return SyntheticSystemResult(
        "spiking_calcium",
        seed,
        _standardize(calcium),
        _standardize(calcium.copy()),
        base.latent,
        base.labels,
        base.behavior,
        {"observation": "poisson_spikes_convolved_to_calcium"},
    )


GENERATORS = {
    "nonlinear_lds": _simulate_nonlinear_lds,
    "switching_lds": _simulate_slds,
    "lorenz": _simulate_lorenz,
    "low_rank_rnn": _simulate_low_rank_rnn,
    "spiking_calcium": _simulate_spiking_calcium,
}


def generate_rich_system(
    name: str,
    seed: int,
    *,
    n_sequences: int,
    seq_length: int,
    n_regions: int,
    latent_dim: int,
) -> SyntheticSystemResult:
    generator = GENERATORS[name]
    primary = generator(seed, n_sequences, seq_length, n_regions, latent_dim, oracle=False)
    oracle = generator(seed, n_sequences, seq_length, n_regions, latent_dim, oracle=True)
    primary.oracle = oracle.neural
    return primary


def _moving_average(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    return gaussian_filter1d(arr, sigma=sigma, axis=1, mode="nearest")


def _marginal_shuffle(gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(gt)
    flat = gt.reshape(-1, gt.shape[-1])
    for c in range(gt.shape[-1]):
        out[..., c] = rng.permutation(flat[:, c]).reshape(gt.shape[0], gt.shape[1])
    return out


def _covariance_gaussian(gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    flat = gt.reshape(-1, gt.shape[-1])
    mu = np.mean(flat, axis=0)
    cov = np.cov(flat, rowvar=False) + 1e-4 * np.eye(gt.shape[-1])
    sample = rng.multivariate_normal(mu, cov, size=flat.shape[0])
    return sample.reshape(gt.shape)


def _block_transition_shuffle(gt: np.ndarray, rng: np.random.Generator, block: int = 8) -> np.ndarray:
    blocks = []
    for start in range(0, gt.shape[1], block):
        blocks.append(gt[:, start : start + block, :])
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order], axis=1)[:, : gt.shape[1], :]


def _short_horizon_good(gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pred = gt.copy()
    cutoff = max(8, gt.shape[1] // 3)
    tail_mean = np.mean(gt[:, :cutoff], axis=1, keepdims=True)
    drift = np.linspace(0.0, 1.0, gt.shape[1] - cutoff).reshape(1, -1, 1)
    pred[:, cutoff:] = (1.0 - drift) * pred[:, cutoff:] + drift * tail_mean
    pred += rng.normal(scale=0.03, size=pred.shape)
    return pred


def _calibrate_mse(gt: np.ndarray, raw: np.ndarray, target_mse: float | None = None) -> np.ndarray:
    base_mse = float(np.nanmean((raw - gt) ** 2))
    if not np.isfinite(base_mse) or base_mse < 1e-12:
        return raw
    if target_mse is None:
        target_mse = 0.18 * float(np.nanvar(gt))
    alpha = math.sqrt(max(target_mse, 1e-12) / base_mse)
    alpha = float(np.clip(alpha, 0.05, 2.5))
    return gt + alpha * (raw - gt)


def build_controls(system: SyntheticSystemResult, *, seed: int, target_mse: float | None = None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 9000)
    gt = system.neural
    controls = {
        "split_half_ceiling": gt + rng.normal(scale=0.05, size=gt.shape),
        "oracle": system.oracle,
        "smoothed_mean": _moving_average(gt, sigma=2.5),
        "short_horizon_good": _short_horizon_good(gt, rng),
        "marginal_shuffle": _marginal_shuffle(gt, rng),
        "covariance_gaussian": _covariance_gaussian(gt, rng),
        "transition_shuffle": _block_transition_shuffle(gt, rng),
    }
    matched = {}
    for name, arr in controls.items():
        if name in {"oracle", "split_half_ceiling"}:
            matched[name] = arr
        else:
            matched[name] = _calibrate_mse(gt, arr, target_mse=target_mse)
    return matched


def _hrf_kernel(tr: float = 1.0, dt: float = 0.1, duration: float = 32.0) -> np.ndarray:
    t = np.arange(0.0, duration, dt)
    peak = (t ** 8.6) * np.exp(-t / 0.547)
    undershoot = 0.35 * (t ** 9.0) * np.exp(-t / 0.9)
    hrf = peak / (np.max(peak) + 1e-12) - undershoot / (np.max(undershoot) + 1e-12)
    hrf = hrf / (np.sum(np.abs(hrf)) + 1e-12)
    step = max(1, int(round(tr / dt)))
    return hrf[::step]


def neural_to_synthetic_bold(neural: np.ndarray, *, tr_bins: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    kernel = _hrf_kernel(tr=1.0, dt=1.0)
    convolved = np.zeros_like(neural)
    for s in range(neural.shape[0]):
        for c in range(neural.shape[2]):
            convolved[s, :, c] = fftconvolve(neural[s, :, c], kernel, mode="full")[: neural.shape[1]]
    bold = convolved[:, ::tr_bins, :]
    drift_t = np.linspace(-1.0, 1.0, bold.shape[1])
    drift = 0.08 * drift_t.reshape(1, -1, 1) * rng.normal(size=(1, 1, bold.shape[2]))
    bold = bold + drift + rng.normal(scale=0.05, size=bold.shape)
    return _standardize(bold)


def _state_transition_matrix(labels: np.ndarray, n_states: int | None = None) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    n_states = int(n_states or (np.nanmax(labels) + 1))
    mat = np.zeros((n_states, n_states), dtype=np.float64)
    for seq in labels:
        np.add.at(mat, (seq[:-1], seq[1:]), 1.0)
    row = mat.sum(axis=1, keepdims=True)
    return mat / np.maximum(row, 1.0)


def _decode_labels(activity: np.ndarray, labels: np.ndarray, seed: int = 0) -> float:
    X = activity.reshape(-1, activity.shape[-1])
    y = labels.reshape(-1)
    if np.unique(y).size < 2:
        return float("nan")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=seed, stratify=y)
    pipe_x = StandardScaler().fit_transform(X_train)
    pipe_xt = StandardScaler().fit(X_train).transform(X_test)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(pipe_x, y_train)
    return float(accuracy_score(y_test, clf.predict(pipe_xt)))


def _neural_to_behavior_r2(activity: np.ndarray, behavior: np.ndarray, seed: int = 0) -> float:
    X = activity.reshape(-1, activity.shape[-1])
    Y = behavior.reshape(-1, behavior.shape[-1])
    idx = np.arange(X.shape[0])
    train, test = train_test_split(idx, test_size=0.35, random_state=seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train])
    X_test = scaler.transform(X[test])
    reg = Ridge(alpha=1.0)
    reg.fit(X_train, Y[train])
    pred = reg.predict(X_test)
    ss_res = float(np.sum((Y[test] - pred) ** 2))
    ss_tot = float(np.sum((Y[test] - np.mean(Y[test], axis=0, keepdims=True)) ** 2)) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def _condition_geometry_score(activity_gt: np.ndarray, activity_pred: np.ndarray, labels: np.ndarray) -> float:
    states = np.unique(labels)
    if states.size < 2:
        return float("nan")
    def centroids(arr: np.ndarray) -> np.ndarray:
        flat = arr.reshape(-1, arr.shape[-1])
        y = labels.reshape(-1)
        return np.vstack([np.mean(flat[y == state], axis=0) for state in states])
    Cg = centroids(activity_gt)
    Cp = centroids(activity_pred)
    dg = np.linalg.norm(Cg[:, None, :] - Cg[None, :, :], axis=2)
    dp = np.linalg.norm(Cp[:, None, :] - Cp[None, :, :], axis=2)
    v1 = dg[np.triu_indices_from(dg, k=1)]
    v2 = dp[np.triu_indices_from(dp, k=1)]
    if np.std(v1) < 1e-9 or np.std(v2) < 1e-9:
        return float("nan")
    return float(np.clip(0.5 * (np.corrcoef(v1, v2)[0, 1] + 1.0), 0.0, 1.0))


def task_behavior_validity(system: SyntheticSystemResult, candidate: np.ndarray, *, candidate_name: str) -> dict[str, float | str | int]:
    gt_acc = _decode_labels(system.neural, system.labels, seed=system.seed)
    pred_acc = _decode_labels(candidate, system.labels, seed=system.seed)
    gt_r2 = _neural_to_behavior_r2(system.neural, system.behavior, seed=system.seed)
    pred_r2 = _neural_to_behavior_r2(candidate, system.behavior, seed=system.seed)
    traj = _condition_geometry_score(system.neural, candidate, system.labels)
    label_transition = _state_transition_matrix(system.labels)
    transition_entropy = -float(np.sum(label_transition * np.log(label_transition + 1e-12)) / label_transition.shape[0])
    return {
        "system": system.system,
        "seed": system.seed,
        "candidate": candidate_name,
        "gt_decoding_accuracy": gt_acc,
        "pred_decoding_accuracy": pred_acc,
        "decoding_preservation": float(1.0 / (1.0 + abs(gt_acc - pred_acc))) if np.isfinite(gt_acc + pred_acc) else float("nan"),
        "gt_neural_to_behavior_r2": gt_r2,
        "pred_neural_to_behavior_r2": pred_r2,
        "neural_behavior_preservation": float(1.0 / (1.0 + abs(gt_r2 - pred_r2))) if np.isfinite(gt_r2 + pred_r2) else float("nan"),
        "condition_geometry_score": traj,
        "label_transition_entropy": transition_entropy,
    }


def _impulse_response(arr: np.ndarray, impulse_time: int, channel: int = 0) -> np.ndarray:
    baseline = np.mean(arr[:, max(0, impulse_time - 5) : impulse_time, :], axis=1, keepdims=True)
    response = arr[:, impulse_time : min(arr.shape[1], impulse_time + 40), :] - baseline
    return np.mean(response, axis=0)


def perturbation_validation(system: SyntheticSystemResult, controls: dict[str, np.ndarray]) -> list[dict[str, float | str | int]]:
    impulse_time = system.neural.shape[1] // 3
    gt_response = _impulse_response(system.neural, impulse_time)
    rows = []
    for name, arr in controls.items():
        response = _impulse_response(arr, impulse_time)
        response_corr = _corr(gt_response[None, ...], response[None, ...])
        gt_peak = int(np.nanargmax(np.linalg.norm(gt_response, axis=1)))
        pr_peak = int(np.nanargmax(np.linalg.norm(response, axis=1)))
        peak_latency_score = float(1.0 - min(1.0, abs(gt_peak - pr_peak) / max(gt_response.shape[0] - 1, 1)))
        amplitude_score = float(
            1.0
            / (
                1.0
                + abs(float(np.nanmax(np.linalg.norm(gt_response, axis=1))) - float(np.nanmax(np.linalg.norm(response, axis=1))))
                / (float(np.nanmax(np.linalg.norm(gt_response, axis=1))) + 1e-9)
            )
        )
        rows.append(
            {
                "system": system.system,
                "seed": system.seed,
                "candidate": name,
                "response_corr": response_corr,
                "peak_latency_score": peak_latency_score,
                "amplitude_score": amplitude_score,
                "perturbation_response_score": float(np.nanmean([0.5 * (response_corr + 1.0), peak_latency_score, amplitude_score])),
            }
        )
    wrong = np.roll(system.neural, shift=system.neural.shape[1] // 4, axis=1)
    wrong_response = _impulse_response(wrong, impulse_time)
    rows.append(
        {
            "system": system.system,
            "seed": system.seed,
            "candidate": "wrong_impulse_response",
            "response_corr": _corr(gt_response[None, ...], wrong_response[None, ...]),
            "peak_latency_score": 0.0,
            "amplitude_score": 1.0
            / (1.0 + abs(float(np.nanstd(gt_response)) - float(np.nanstd(wrong_response))) / (float(np.nanstd(gt_response)) + 1e-9)),
            "perturbation_response_score": float("nan"),
        }
    )
    return rows


def _score_controls(system: SyntheticSystemResult, controls: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    for candidate, arr in controls.items():
        scores = score_arrays_fast(system.neural, arr)
        row: dict[str, object] = {"system": system.system, "seed": system.seed, "candidate": candidate, "representation": "neural"}
        row.update(scores)
        rows.append(row)
    return rows


def _score_bold(system: SyntheticSystemResult, controls: dict[str, np.ndarray]) -> list[dict[str, object]]:
    gt_bold = neural_to_synthetic_bold(system.neural, seed=system.seed)
    rows = []
    for candidate, arr in controls.items():
        pred_bold = neural_to_synthetic_bold(arr, seed=system.seed + 17)
        scores = score_arrays_fast(gt_bold, pred_bold)
        row: dict[str, object] = {"system": system.system, "seed": system.seed, "candidate": candidate, "representation": "synthetic_bold"}
        row.update(scores)
        rows.append(row)
    rng = np.random.default_rng(system.seed + 99)
    floor = _marginal_shuffle(gt_bold, rng)
    floor_scores = score_arrays_fast(gt_bold, floor)
    rows.append({"system": system.system, "seed": system.seed, "candidate": "bold_time_parcel_shuffle_floor", "representation": "synthetic_bold", **floor_scores})
    ceiling = neural_to_synthetic_bold(system.neural, seed=system.seed + 100)
    ceiling_scores = score_arrays_fast(gt_bold, ceiling)
    rows.append({"system": system.system, "seed": system.seed, "candidate": "bold_split_half_ceiling", "representation": "synthetic_bold", **ceiling_scores})
    return rows


def _write_outputs(
    output_root: Path,
    *,
    score_rows: list[dict[str, object]],
    behavior_rows: list[dict[str, object]],
    perturb_rows: list[dict[str, object]],
    bold_rows: list[dict[str, object]],
    disk_events: list[dict[str, object]],
    config: BulletproofValidationConfig,
) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    scores_df = pd.DataFrame(score_rows)
    behavior_df = pd.DataFrame(behavior_rows)
    perturb_df = pd.DataFrame(perturb_rows)
    bold_df = pd.DataFrame(bold_rows)

    rich_path = output_root / "rich_synthetic_scores_long.csv"
    wrong_path = output_root / "wrong_dynamics_controls.csv"
    behavior_path = output_root / "task_behavior_validity_scores.csv"
    perturb_path = output_root / "perturbation_response_scores.csv"
    bold_path = output_root / "fmri_bold_validation_scores.csv"

    scores_df.to_csv(rich_path, index=False)
    scores_df[scores_df["candidate"].isin(["smoothed_mean", "short_horizon_good", "marginal_shuffle", "covariance_gaussian", "transition_shuffle"])].to_csv(
        wrong_path,
        index=False,
    )
    behavior_df.to_csv(behavior_path, index=False)
    perturb_df.to_csv(perturb_path, index=False)
    bold_df.to_csv(bold_path, index=False)

    outputs = {
        "rich_synthetic_scores_long_csv": str(rich_path),
        "wrong_dynamics_controls_csv": str(wrong_path),
        "task_behavior_validity_scores_csv": str(behavior_path),
        "perturbation_response_scores_csv": str(perturb_path),
        "fmri_bold_validation_scores_csv": str(bold_path),
    }
    outputs.update(_make_figures(output_root, scores_df, behavior_df, perturb_df, bold_df))

    report = _build_report(scores_df, behavior_df, perturb_df, bold_df, disk_events, config, outputs)
    report_json = output_root / "bulletproof_analysis_report.json"
    report_md = output_root / "bulletproof_analysis_report.md"
    outputs["bulletproof_analysis_report_json"] = str(report_json)
    outputs["bulletproof_analysis_report_md"] = str(report_md)
    report["outputs"] = outputs
    report_json.write_text(json.dumps(report, indent=2))
    report_md.write_text(_report_markdown(report))
    return outputs


def _mean_by_candidate(df: pd.DataFrame, value: str, representation: str = "neural") -> pd.Series:
    sub = df[df["representation"] == representation] if "representation" in df.columns else df
    return pd.to_numeric(sub[value], errors="coerce").groupby(sub["candidate"]).mean().sort_values(ascending=False)


def _make_figures(
    output_root: Path,
    scores_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    perturb_df: pd.DataFrame,
    bold_df: pd.DataFrame,
) -> dict[str, str]:
    fig_paths: dict[str, str] = {}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    neural = scores_df[scores_df["representation"] == "neural"]
    means = neural.groupby("candidate")[FAMILY_COLUMNS + ["FINAL_COMPOSITE_SCORE", "FIDELITY_SCORE"]].mean(numeric_only=True)
    means = means.loc[
        [
            c
            for c in [
                "split_half_ceiling",
                "oracle",
                "short_horizon_good",
                "smoothed_mean",
                "marginal_shuffle",
                "covariance_gaussian",
                "transition_shuffle",
            ]
            if c in means.index
        ]
    ]
    axes[0, 0].imshow(means[FAMILY_COLUMNS].to_numpy(), vmin=0, vmax=1, aspect="auto", cmap="viridis")
    axes[0, 0].set_yticks(np.arange(means.shape[0]), means.index)
    axes[0, 0].set_xticks(np.arange(len(FAMILY_COLUMNS)), [c.replace("family_", "").replace("_", "\n") for c in FAMILY_COLUMNS], rotation=0)
    axes[0, 0].set_title("NethoBench family signatures")
    means[["FINAL_COMPOSITE_SCORE", "FIDELITY_SCORE"]].plot(kind="bar", ax=axes[0, 1], ylim=(0, 1), color=["#4C78A8", "#F58518"])
    axes[0, 1].set_title("Realism composite vs pointwise fidelity")
    axes[0, 1].set_xlabel("")
    axes[0, 1].tick_params(axis="x", rotation=35)
    axes[0, 2].scatter(neural["mse"], neural["FINAL_COMPOSITE_SCORE"], c=neural["FIDELITY_SCORE"], cmap="magma", s=42)
    axes[0, 2].set_xlabel("MSE")
    axes[0, 2].set_ylabel("NethoBench composite")
    axes[0, 2].set_title("Same-error controls can differ in realism")
    beh = behavior_df.groupby("candidate")[["decoding_preservation", "neural_behavior_preservation", "condition_geometry_score"]].mean(numeric_only=True)
    beh.plot(kind="bar", ax=axes[1, 0], ylim=(0, 1), color=["#54A24B", "#E45756", "#72B7B2"])
    axes[1, 0].set_title("Task/behavior validity preservation")
    axes[1, 0].set_xlabel("")
    axes[1, 0].tick_params(axis="x", rotation=35)
    pert = perturb_df.groupby("candidate")[["response_corr", "peak_latency_score", "amplitude_score"]].mean(numeric_only=True)
    pert.plot(kind="bar", ax=axes[1, 1], ylim=(-1, 1), color=["#4C78A8", "#B279A2", "#FF9DA6"])
    axes[1, 1].set_title("Perturbation-response preservation")
    axes[1, 1].set_xlabel("")
    axes[1, 1].tick_params(axis="x", rotation=35)
    bmeans = bold_df.groupby("candidate")[["FINAL_COMPOSITE_SCORE", "FIDELITY_SCORE"]].mean(numeric_only=True).sort_values("FINAL_COMPOSITE_SCORE", ascending=False)
    bmeans.plot(kind="bar", ax=axes[1, 2], ylim=(0, 1), color=["#4C78A8", "#F58518"])
    axes[1, 2].set_title("Synthetic fMRI/BOLD validation")
    axes[1, 2].set_xlabel("")
    axes[1, 2].tick_params(axis="x", rotation=35)
    path = output_root / "bulletproof_validation_summary.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    fig_paths["summary_svg"] = str(path)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    same = neural[
        neural["candidate"].isin(
            ["short_horizon_good", "smoothed_mean", "transition_shuffle", "marginal_shuffle", "covariance_gaussian"]
        )
    ]
    for candidate, sub in same.groupby("candidate"):
        axes[0].scatter(sub["mse"], sub["FINAL_COMPOSITE_SCORE"], label=candidate, s=50)
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel("MSE")
    axes[0].set_ylabel("Composite")
    axes[0].set_title("Matched-MSE realism split")
    same.groupby("candidate")[FAMILY_COLUMNS].mean(numeric_only=True).T.plot(ax=axes[1], marker="o", ylim=(0, 1))
    axes[1].set_title("Different families fail")
    axes[1].tick_params(axis="x", rotation=35)
    same.groupby("candidate")[["FINAL_COMPOSITE_SCORE", "FIDELITY_SCORE"]].mean(numeric_only=True).plot(kind="bar", ax=axes[2], ylim=(0, 1))
    axes[2].set_title("Fidelity is not sufficient")
    axes[2].tick_params(axis="x", rotation=35)
    path = output_root / "same_mse_different_realism.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    fig_paths["same_mse_different_realism_svg"] = str(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    system_summary = neural.groupby(["system", "candidate"])["FINAL_COMPOSITE_SCORE"].mean().reset_index()
    keep_candidates = ["split_half_ceiling", "oracle", "marginal_shuffle", "covariance_gaussian"]
    pivot = system_summary[system_summary["candidate"].isin(keep_candidates)].pivot(
        index="system",
        columns="candidate",
        values="FINAL_COMPOSITE_SCORE",
    )
    pivot[[c for c in keep_candidates if c in pivot.columns]].plot(kind="bar", ax=axes[0], ylim=(0, 1))
    axes[0].set_title("Richer known-ground-truth systems")
    axes[0].set_ylabel("NethoBench composite")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=35)
    fam = neural.groupby("system")[FAMILY_COLUMNS].mean(numeric_only=True)
    im = axes[1].imshow(fam.to_numpy(), vmin=0, vmax=1, aspect="auto", cmap="viridis")
    axes[1].set_yticks(np.arange(fam.shape[0]), fam.index)
    axes[1].set_xticks(np.arange(len(FAMILY_COLUMNS)), [c.replace("family_", "").replace("_", "\n") for c in FAMILY_COLUMNS])
    axes[1].set_title("Family coverage across systems")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    path = output_root / "richer_synthetic_systems_overview.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    fig_paths["richer_synthetic_systems_overview_svg"] = str(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    beh = behavior_df.groupby("candidate")[["decoding_preservation", "neural_behavior_preservation", "condition_geometry_score"]].mean(numeric_only=True)
    beh.sort_values("decoding_preservation", ascending=False).plot(kind="bar", ax=axes[0], ylim=(0, 1), color=["#54A24B", "#E45756", "#72B7B2"])
    axes[0].set_title("Task/behavior preservation")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=35)
    merged_behavior = behavior_df.merge(
        neural[["system", "seed", "candidate", "FINAL_COMPOSITE_SCORE", "mse"]],
        on=["system", "seed", "candidate"],
        how="inner",
    )
    axes[1].scatter(
        merged_behavior["FINAL_COMPOSITE_SCORE"],
        merged_behavior["decoding_preservation"],
        c=merged_behavior["mse"],
        cmap="magma",
        s=42,
    )
    axes[1].set_xlabel("NethoBench composite")
    axes[1].set_ylabel("Decoding preservation")
    axes[1].set_title("Realism vs task validity")
    path = output_root / "task_behavior_preservation.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    fig_paths["task_behavior_preservation_svg"] = str(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    pert = perturb_df.groupby("candidate")[["response_corr", "peak_latency_score", "amplitude_score"]].mean(numeric_only=True)
    pert.sort_values("response_corr", ascending=False).plot(kind="bar", ax=axes[0], ylim=(-1, 1), color=["#4C78A8", "#B279A2", "#FF9DA6"])
    axes[0].set_title("Perturbation-response preservation")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=35)
    pert_merge = perturb_df.merge(
        neural[["system", "seed", "candidate", "FINAL_COMPOSITE_SCORE"]],
        on=["system", "seed", "candidate"],
        how="inner",
    )
    axes[1].scatter(pert_merge["FINAL_COMPOSITE_SCORE"], pert_merge["response_corr"], s=48, color="#4C78A8")
    axes[1].set_xlabel("NethoBench composite")
    axes[1].set_ylabel("Response correlation")
    axes[1].set_title("Realism vs perturbation response")
    path = output_root / "perturbation_response_preservation.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    fig_paths["perturbation_response_preservation_svg"] = str(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    bmeans = bold_df.groupby("candidate")[["FINAL_COMPOSITE_SCORE", "FIDELITY_SCORE"]].mean(numeric_only=True).sort_values("FINAL_COMPOSITE_SCORE", ascending=False)
    bmeans.plot(kind="bar", ax=axes[0], ylim=(0, 1), color=["#4C78A8", "#F58518"])
    axes[0].set_title("Synthetic fMRI/BOLD validation")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=35)
    bfam = bold_df.groupby("candidate")[FAMILY_COLUMNS].mean(numeric_only=True).sort_values("family_temporal_spectral", ascending=False)
    im = axes[1].imshow(bfam.to_numpy(), vmin=0, vmax=1, aspect="auto", cmap="viridis")
    axes[1].set_yticks(np.arange(bfam.shape[0]), bfam.index)
    axes[1].set_xticks(np.arange(len(FAMILY_COLUMNS)), [c.replace("family_", "").replace("_", "\n") for c in FAMILY_COLUMNS])
    axes[1].set_title("BOLD family signatures")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    path = output_root / "fmri_bold_validation.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    fig_paths["fmri_bold_validation_svg"] = str(path)
    return fig_paths


def _build_report(
    scores_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    perturb_df: pd.DataFrame,
    bold_df: pd.DataFrame,
    disk_events: list[dict[str, object]],
    config: BulletproofValidationConfig,
    outputs: dict[str, str],
) -> dict[str, object]:
    neural = scores_df[scores_df["representation"] == "neural"]
    mean_composite = _mean_by_candidate(neural, "FINAL_COMPOSITE_SCORE").to_dict()
    mean_fidelity = _mean_by_candidate(neural, "FIDELITY_SCORE").to_dict()
    bold_composite = _mean_by_candidate(bold_df, "FINAL_COMPOSITE_SCORE", representation="synthetic_bold").to_dict()
    behavior_summary = behavior_df.groupby("candidate")[
        ["decoding_preservation", "neural_behavior_preservation", "condition_geometry_score"]
    ].mean(numeric_only=True).to_dict()
    perturb_summary = perturb_df.groupby("candidate")[["response_corr", "peak_latency_score", "amplitude_score"]].mean(numeric_only=True).to_dict()
    controls = neural[
        neural["candidate"].isin(
            ["smoothed_mean", "short_horizon_good", "transition_shuffle", "marginal_shuffle", "covariance_gaussian"]
        )
    ]
    same_mse_spread = float(controls.groupby("candidate")["mse"].mean().max() - controls.groupby("candidate")["mse"].mean().min()) if not controls.empty else float("nan")
    same_mse_composite_spread = (
        float(controls.groupby("candidate")["FINAL_COMPOSITE_SCORE"].mean().max() - controls.groupby("candidate")["FINAL_COMPOSITE_SCORE"].mean().min())
        if not controls.empty
        else float("nan")
    )
    return {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "systems": sorted(scores_df["system"].unique().tolist()),
        "n_score_rows": int(scores_df.shape[0]),
        "disk_events": disk_events,
        "mean_neural_composite_by_candidate": mean_composite,
        "mean_fidelity_by_candidate": mean_fidelity,
        "mean_bold_composite_by_candidate": bold_composite,
        "behavior_summary": behavior_summary,
        "perturbation_summary": perturb_summary,
        "same_mse_control_spread": {
            "mean_mse_spread": same_mse_spread,
            "mean_composite_spread": same_mse_composite_spread,
        },
        "outputs": outputs,
    }


def _report_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Bulletproof NethoBench Validation Report",
        "",
        "## What Ran",
        f"- Systems: {', '.join(report['systems'])}",
        f"- Score rows: {report['n_score_rows']}",
        "",
        "## Neural Composite By Candidate",
    ]
    for key, value in report["mean_neural_composite_by_candidate"].items():
        lines.append(f"- {key}: {float(value):.3f}")
    lines += ["", "## Synthetic BOLD Composite By Candidate"]
    for key, value in report["mean_bold_composite_by_candidate"].items():
        lines.append(f"- {key}: {float(value):.3f}")
    spread = report["same_mse_control_spread"]
    lines += [
        "",
        "## Same-MSE Control Check",
        f"- Mean MSE spread among selected controls: {float(spread['mean_mse_spread']):.4f}",
        f"- Mean NethoBench composite spread among selected controls: {float(spread['mean_composite_spread']):.4f}",
        "",
        "## Disk Events",
    ]
    for event in report["disk_events"]:
        lines.append(f"- {event['phase']}: {float(event['free_gb']):.2f} GB free, reserve {float(event['min_free_gb']):.2f} GB")
    return "\n".join(lines) + "\n"


def run_bulletproof_validation(
    *,
    output_root: Path | None = None,
    mode: str = "quick",
    min_free_disk_gb: float = 8.0,
    run_full_notebook_scores: bool = False,
) -> dict[str, object]:
    config = BulletproofValidationConfig.for_mode(
        mode,
        output_root=output_root,
        min_free_disk_gb=min_free_disk_gb,
        run_full_notebook_scores=run_full_notebook_scores,
    )
    output_root = Path(config.output_root)
    disk_events = [_check_disk(output_root, config.min_free_disk_gb, "start")]
    score_rows: list[dict[str, object]] = []
    behavior_rows: list[dict[str, object]] = []
    perturb_rows: list[dict[str, object]] = []
    bold_rows: list[dict[str, object]] = []
    target_mse = 0.16
    systems = ["nonlinear_lds", "switching_lds", "lorenz", "low_rank_rnn", "spiking_calcium"]
    if mode == "quick":
        systems = systems[:5]

    for seed in config.seeds:
        for system_name in systems:
            disk_events.append(_check_disk(output_root, config.min_free_disk_gb, f"before_{system_name}_{seed}"))
            system = generate_rich_system(
                system_name,
                seed,
                n_sequences=config.n_sequences,
                seq_length=config.seq_length,
                n_regions=config.n_regions,
                latent_dim=config.latent_dim,
            )
            controls = build_controls(system, seed=seed, target_mse=target_mse)
            score_rows.extend(_score_controls(system, controls))
            for candidate_name, arr in controls.items():
                behavior_rows.append(task_behavior_validity(system, arr, candidate_name=candidate_name))
            if system_name in {"nonlinear_lds", "low_rank_rnn"}:
                perturb_rows.extend(perturbation_validation(system, controls))
            if system_name in {"nonlinear_lds", "switching_lds", "low_rank_rnn"}:
                bold_rows.extend(_score_bold(system, controls))

    disk_events.append(_check_disk(output_root, config.min_free_disk_gb, "before_write_outputs"))
    outputs = _write_outputs(
        output_root,
        score_rows=score_rows,
        behavior_rows=behavior_rows,
        perturb_rows=perturb_rows,
        bold_rows=bold_rows,
        disk_events=disk_events,
        config=config,
    )
    disk_events.append(_check_disk(output_root, config.min_free_disk_gb, "finished"))

    report = json.loads(Path(outputs["bulletproof_analysis_report_json"]).read_text())
    report["disk_events"] = disk_events
    report["outputs"] = outputs
    Path(outputs["bulletproof_analysis_report_json"]).write_text(json.dumps(report, indent=2))
    Path(outputs["bulletproof_analysis_report_md"]).write_text(_report_markdown(report))
    return {"output_root": str(output_root), "outputs": outputs, "report": report}


def run_bulletproof_synthetic(**kwargs) -> dict[str, object]:
    return run_bulletproof_validation(**kwargs)


def run_bulletproof_fmri(
    *,
    output_root: Path | None = None,
    mode: str = "quick",
    min_free_disk_gb: float = 8.0,
) -> dict[str, object]:
    config = BulletproofValidationConfig.for_mode(mode, output_root=output_root, min_free_disk_gb=min_free_disk_gb)
    output_root = Path(config.output_root)
    disk_events = [_check_disk(output_root, config.min_free_disk_gb, "fmri_start")]
    bold_rows: list[dict[str, object]] = []
    for seed in config.seeds:
        for system_name in ["nonlinear_lds", "switching_lds", "low_rank_rnn"]:
            system = generate_rich_system(
                system_name,
                seed,
                n_sequences=config.n_sequences,
                seq_length=config.seq_length,
                n_regions=config.n_regions,
                latent_dim=config.latent_dim,
            )
            controls = build_controls(system, seed=seed, target_mse=0.16)
            bold_rows.extend(_score_bold(system, controls))
    bold_df = pd.DataFrame(bold_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    bold_path = output_root / "fmri_bold_validation_scores.csv"
    bold_df.to_csv(bold_path, index=False)
    fig_paths = _make_figures(
        output_root,
        pd.DataFrame(columns=["representation", "candidate", *SCORE_COLUMNS]),
        pd.DataFrame(columns=["candidate", "decoding_preservation", "neural_behavior_preservation", "condition_geometry_score"]),
        pd.DataFrame(columns=["candidate", "response_corr", "peak_latency_score", "amplitude_score"]),
        bold_df,
    )
    disk_events.append(_check_disk(output_root, config.min_free_disk_gb, "fmri_finished"))
    report = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "disk_events": disk_events,
        "mean_bold_composite_by_candidate": _mean_by_candidate(bold_df, "FINAL_COMPOSITE_SCORE", representation="synthetic_bold").to_dict(),
        "outputs": {"fmri_bold_validation_scores_csv": str(bold_path), **fig_paths},
    }
    report_path = output_root / "fmri_bold_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return {"output_root": str(output_root), "outputs": report["outputs"], "report": report}


def rerun_bulletproof_score(output_root: Path) -> dict[str, object]:
    root = Path(output_root)
    required = [
        root / "rich_synthetic_scores_long.csv",
        root / "task_behavior_validity_scores.csv",
        root / "perturbation_response_scores.csv",
        root / "fmri_bold_validation_scores.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing score inputs: {missing}")
    scores_df = pd.read_csv(required[0])
    behavior_df = pd.read_csv(required[1])
    perturb_df = pd.read_csv(required[2])
    bold_df = pd.read_csv(required[3])
    disk_events = [_check_disk(root, 0.0, "score_regeneration_start")]
    outputs = _make_figures(root, scores_df, behavior_df, perturb_df, bold_df)
    disk_events.append(_check_disk(root, 0.0, "score_regeneration_finished"))
    report = _build_report(scores_df, behavior_df, perturb_df, bold_df, disk_events, BulletproofValidationConfig(output_root=root), outputs)
    report_json = root / "bulletproof_analysis_report.json"
    report_md = root / "bulletproof_analysis_report.md"
    outputs["bulletproof_analysis_report_json"] = str(report_json)
    outputs["bulletproof_analysis_report_md"] = str(report_md)
    report["outputs"] = outputs
    report_json.write_text(json.dumps(report, indent=2))
    report_md.write_text(_report_markdown(report))
    return {"output_root": str(root), "outputs": {**outputs, "bulletproof_analysis_report_json": str(report_json), "bulletproof_analysis_report_md": str(report_md)}, "report": report}


__all__ = [
    "BulletproofValidationConfig",
    "SyntheticSystemResult",
    "build_controls",
    "generate_rich_system",
    "neural_to_synthetic_bold",
    "rerun_bulletproof_score",
    "run_bulletproof_fmri",
    "run_bulletproof_synthetic",
    "run_bulletproof_validation",
    "score_arrays_fast",
    "task_behavior_validity",
]
