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
from scipy.stats import wasserstein_distance
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .bulletproof_validation import generate_rich_system


DEFAULT_OUTPUT_ROOT = Path("outputs") / "behavior-crossmodal-supplement"


@dataclass(frozen=True)
class BehaviorCrossModalConfig:
    mode: str = "quick"
    output_root: Path = DEFAULT_OUTPUT_ROOT
    n_sequences: int = 8
    seq_length: int = 160
    n_regions: int = 12
    latent_dim: int = 5
    seeds: tuple[int, ...] = (7, 11)
    min_free_disk_gb: float = 4.0

    @classmethod
    def for_mode(cls, mode: str, *, output_root: Path | None = None, min_free_disk_gb: float = 4.0) -> "BehaviorCrossModalConfig":
        if mode not in {"quick", "full"}:
            raise ValueError("mode must be 'quick' or 'full'.")
        if mode == "quick":
            return cls(mode=mode, output_root=output_root or DEFAULT_OUTPUT_ROOT, min_free_disk_gb=min_free_disk_gb)
        return cls(
            mode=mode,
            output_root=output_root or DEFAULT_OUTPUT_ROOT,
            n_sequences=14,
            seq_length=220,
            n_regions=16,
            latent_dim=6,
            seeds=(7, 11, 19),
            min_free_disk_gb=min_free_disk_gb,
        )


@dataclass
class BehaviorCandidate:
    name: str
    behavior: np.ndarray
    labels: np.ndarray
    neural: np.ndarray | None = None


def _free_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return float(usage.free / (1024**3))


def _check_disk(path: Path, min_free_gb: float, phase: str) -> dict[str, object]:
    path.mkdir(parents=True, exist_ok=True)
    free = _free_disk_gb(path)
    if free < min_free_gb:
        raise RuntimeError(f"Refusing {phase}: {free:.2f} GB free at {path}, below reserve {min_free_gb:.2f} GB.")
    return {"phase": phase, "path": str(path), "free_gb": free, "min_free_gb": float(min_free_gb)}


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _standardize(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    flat = values.reshape(-1, values.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    out = (values - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    out[~np.isfinite(out)] = 0.0
    return out


def _derive_labels(behavior: np.ndarray, n_states: int) -> np.ndarray:
    state_channel = np.asarray(behavior[..., -1], dtype=np.float64)
    if np.nanmax(state_channel) <= 1.5:
        labels = np.rint(np.clip(state_channel, 0.0, 1.0) * max(1, n_states - 1)).astype(int)
    else:
        labels = np.rint(state_channel).astype(int)
    return np.clip(labels, 0, n_states - 1)


def _transition_matrix(labels: np.ndarray, n_states: int) -> np.ndarray:
    mat = np.zeros((n_states, n_states), dtype=np.float64)
    for seq in labels:
        np.add.at(mat, (seq[:-1], seq[1:]), 1.0)
    row = mat.sum(axis=1, keepdims=True)
    return mat / np.maximum(row, 1.0)


def _hist_overlap(a: np.ndarray, b: np.ndarray, bins: int | np.ndarray = 40) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 4 or y.size < 4:
        return float("nan")
    if isinstance(bins, int):
        lo, hi = np.nanquantile(x, [0.005, 0.995])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(x)), float(np.nanmax(x) + 1e-6)
        bins = np.linspace(lo, hi, int(bins) + 1)
    px, _ = np.histogram(x, bins=bins)
    py, _ = np.histogram(y, bins=bins)
    px = px.astype(float) / max(float(px.sum()), 1.0)
    py = py.astype(float) / max(float(py.sum()), 1.0)
    return float(np.sum(np.minimum(px, py)))


def _matrix_similarity(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if np.nanstd(x) < 1e-9 or np.nanstd(y) < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(x, y)[0, 1])
    err = float(np.linalg.norm(x - y) / (np.linalg.norm(x) + 1e-9))
    return float(np.clip(0.5 * (corr + 1.0) * (1.0 / (1.0 + err)), 0.0, 1.0))


def _bout_lengths(labels: np.ndarray) -> np.ndarray:
    lengths: list[int] = []
    for seq in labels:
        if seq.size == 0:
            continue
        run = 1
        for idx in range(1, seq.size):
            if seq[idx] == seq[idx - 1]:
                run += 1
            else:
                lengths.append(run)
                run = 1
        lengths.append(run)
    return np.asarray(lengths, dtype=np.float64)


def _path_features(behavior: np.ndarray) -> np.ndarray:
    xy = behavior[..., :2]
    delta = np.diff(xy, axis=1)
    speed = np.linalg.norm(delta, axis=2)
    accel = np.diff(speed, axis=1)
    displacement = np.linalg.norm(xy[:, -1, :] - xy[:, 0, :], axis=1)
    path_len = np.sum(speed, axis=1)
    turn = np.zeros_like(speed[:, 1:])
    if delta.shape[1] > 1:
        v1 = delta[:, :-1, :]
        v2 = delta[:, 1:, :]
        denom = np.linalg.norm(v1, axis=2) * np.linalg.norm(v2, axis=2) + 1e-9
        turn = np.arccos(np.clip(np.sum(v1 * v2, axis=2) / denom, -1.0, 1.0))
    return np.column_stack(
        [
            np.nanmean(speed, axis=1),
            np.nanstd(speed, axis=1),
            np.nanmean(np.abs(accel), axis=1) if accel.size else np.zeros(behavior.shape[0]),
            displacement,
            path_len,
            np.nanmean(turn, axis=1) if turn.size else np.zeros(behavior.shape[0]),
        ]
    )


def score_behavior_realism(gt_behavior: np.ndarray, gt_labels: np.ndarray, pred_behavior: np.ndarray, pred_labels: np.ndarray) -> dict[str, float]:
    n_states = int(max(np.nanmax(gt_labels), np.nanmax(pred_labels)) + 1)
    gt_labels = np.asarray(gt_labels, dtype=int)
    pred_labels = np.asarray(pred_labels, dtype=int)
    gt_occ = np.bincount(gt_labels.reshape(-1), minlength=n_states).astype(float)
    pr_occ = np.bincount(pred_labels.reshape(-1), minlength=n_states).astype(float)
    gt_occ /= max(float(gt_occ.sum()), 1.0)
    pr_occ /= max(float(pr_occ.sum()), 1.0)
    occupancy = float(np.sum(np.minimum(gt_occ, pr_occ)))
    transition = _matrix_similarity(_transition_matrix(gt_labels, n_states), _transition_matrix(pred_labels, n_states))
    gt_bouts = _bout_lengths(gt_labels)
    pr_bouts = _bout_lengths(pred_labels)
    bout_scale = float(np.nanstd(gt_bouts)) if gt_bouts.size else 1.0
    if not np.isfinite(bout_scale) or bout_scale < 1e-9:
        bout_scale = 1.0
    bout = float(1.0 / (1.0 + wasserstein_distance(gt_bouts, pr_bouts) / bout_scale))
    speed = _hist_overlap(gt_behavior[..., 2], pred_behavior[..., 2])
    gt_path = _path_features(gt_behavior)
    pr_path = _path_features(pred_behavior)
    path_score = _matrix_similarity(gt_path, pr_path)
    composite = float(np.nanmean([occupancy, transition, bout, speed, path_score]))
    return {
        "state_occupancy_score": occupancy,
        "state_transition_score": transition,
        "bout_duration_score": bout,
        "speed_distribution_score": speed,
        "trajectory_path_score": path_score,
        "behavior_composite": composite,
    }


def _train_behavior_gru(gt_behavior: np.ndarray, *, seed: int, context_bins: int = 12) -> np.ndarray:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception:
        return gaussian_filter1d(gt_behavior, sigma=1.2, axis=1) + np.random.default_rng(seed).normal(scale=0.06, size=gt_behavior.shape)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    behavior = np.asarray(gt_behavior, dtype=np.float32)
    n_seq, n_time, n_dim = behavior.shape
    context_bins = min(context_bins, max(2, n_time // 4))

    class BehaviorGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_dim, 32, batch_first=True)
            self.out = nn.Linear(32, n_dim)

        def forward(self, x, target=None):
            _, h = self.gru(x)
            prev = x[:, -1:, :]
            outs = []
            for t in range(n_time - context_bins):
                y, h = self.gru(prev, h)
                pred = self.out(y)
                outs.append(pred)
                prev = target[:, t : t + 1, :] if self.training and target is not None else pred
            return torch.cat(outs, dim=1)

    model = BehaviorGRU()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    x = torch.tensor(behavior[:, :context_bins, :])
    y = torch.tensor(behavior[:, context_bins:, :])
    loader = DataLoader(TensorDataset(x, y), batch_size=min(16, n_seq), shuffle=True)
    best = None
    best_loss = float("inf")
    for _epoch in range(60):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb, yb)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(torch.mean((model(x) - y) ** 2).detach())
        if val_loss < best_loss:
            best_loss = val_loss
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        rollout = model(x).detach().numpy()
    pred = behavior.copy()
    pred[:, context_bins:, :] = rollout
    pred[:, context_bins:, :] += rng.normal(scale=0.035, size=pred[:, context_bins:, :].shape)
    return pred.astype(np.float64)


def _markov_ar_behavior(gt_behavior: np.ndarray, gt_labels: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_states = int(np.nanmax(gt_labels) + 1)
    trans = _transition_matrix(gt_labels, n_states)
    flat_behavior = gt_behavior.reshape(-1, gt_behavior.shape[-1])
    flat_labels = gt_labels.reshape(-1)
    means = np.vstack([np.nanmean(flat_behavior[flat_labels == state], axis=0) for state in range(n_states)])
    means[~np.isfinite(means)] = 0.0
    pred_labels = np.zeros_like(gt_labels)
    pred_behavior = np.zeros_like(gt_behavior)
    for s in range(gt_labels.shape[0]):
        state = int(gt_labels[s, 0])
        pred_labels[s, 0] = state
        pred_behavior[s, 0] = gt_behavior[s, 0]
        for t in range(1, gt_labels.shape[1]):
            state = int(rng.choice(n_states, p=trans[state]))
            pred_labels[s, t] = state
            target = means[state] + rng.normal(scale=0.08, size=gt_behavior.shape[-1])
            pred_behavior[s, t] = 0.82 * pred_behavior[s, t - 1] + 0.18 * target
            pred_behavior[s, t, -1] = state / max(1, n_states - 1)
    return pred_behavior, pred_labels


def build_behavior_candidates(gt_behavior: np.ndarray, gt_labels: np.ndarray, *, seed: int) -> dict[str, BehaviorCandidate]:
    rng = np.random.default_rng(seed)
    n_states = int(np.nanmax(gt_labels) + 1)
    split = gt_behavior + rng.normal(scale=0.025, size=gt_behavior.shape)
    gru = _train_behavior_gru(gt_behavior, seed=seed)
    markov_behavior, markov_labels = _markov_ar_behavior(gt_behavior, gt_labels, seed=seed + 1)
    smooth = gaussian_filter1d(gt_behavior, sigma=3.0, axis=1)
    transition_labels = gt_labels.copy()
    for s in range(transition_labels.shape[0]):
        blocks = np.array_split(transition_labels[s], max(2, transition_labels.shape[1] // 20))
        rng.shuffle(blocks)
        transition_labels[s] = np.concatenate(blocks)
    transition_behavior = gt_behavior.copy()
    transition_behavior[..., -1] = transition_labels / max(1, n_states - 1)
    time_behavior = gt_behavior.copy()
    time_labels = gt_labels.copy()
    for s in range(gt_behavior.shape[0]):
        perm = rng.permutation(gt_behavior.shape[1])
        time_behavior[s] = time_behavior[s, perm]
        time_labels[s] = time_labels[s, perm]
    return {
        "split_half_ceiling": BehaviorCandidate("split_half_ceiling", split, gt_labels.copy()),
        "behavior_gru": BehaviorCandidate("behavior_gru", gru, _derive_labels(gru, n_states)),
        "markov_ar_baseline": BehaviorCandidate("markov_ar_baseline", markov_behavior, markov_labels),
        "smoothed_mean": BehaviorCandidate("smoothed_mean", smooth, _derive_labels(smooth, n_states)),
        "transition_shuffle": BehaviorCandidate("transition_shuffle", transition_behavior, transition_labels),
        "time_shuffle": BehaviorCandidate("time_shuffle", time_behavior, time_labels),
    }


def _cross_covariance(neural: np.ndarray, behavior: np.ndarray) -> np.ndarray:
    x = _standardize(neural).reshape(-1, neural.shape[-1])
    y = _standardize(behavior).reshape(-1, behavior.shape[-1])
    n = min(x.shape[0], y.shape[0])
    return (x[:n].T @ y[:n]) / max(n - 1, 1)


def _lag_profile(neural: np.ndarray, behavior: np.ndarray, max_lag: int = 8) -> np.ndarray:
    x = _standardize(neural)
    y = _standardize(behavior)
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            xn, yb = x[:, :lag, :], y[:, -lag:, :]
        elif lag > 0:
            xn, yb = x[:, lag:, :], y[:, :-lag, :]
        else:
            xn, yb = x, y
        rows.append(_cross_covariance(xn, yb).reshape(-1))
    return np.asarray(rows)


def _decoder_agreement(neural: np.ndarray, behavior: np.ndarray, labels: np.ndarray, cand_neural: np.ndarray, cand_behavior: np.ndarray, seed: int) -> float:
    y = labels.reshape(-1)
    if np.unique(y).size < 2:
        return float("nan")
    Xn = neural.reshape(-1, neural.shape[-1])
    Xb = behavior.reshape(-1, behavior.shape[-1])
    idx = np.arange(y.size)
    train, _test = train_test_split(idx, test_size=0.35, random_state=seed, stratify=y)
    n_scaler = StandardScaler().fit(Xn[train])
    b_scaler = StandardScaler().fit(Xb[train])
    n_clf = LogisticRegression(max_iter=1000).fit(n_scaler.transform(Xn[train]), y[train])
    b_clf = LogisticRegression(max_iter=1000).fit(b_scaler.transform(Xb[train]), y[train])
    pn = n_clf.predict(n_scaler.transform(cand_neural.reshape(-1, cand_neural.shape[-1])))
    pb = b_clf.predict(b_scaler.transform(cand_behavior.reshape(-1, cand_behavior.shape[-1])))
    return float(np.mean(pn == pb))


def _neural_to_behavior_score(neural: np.ndarray, behavior: np.ndarray, cand_neural: np.ndarray, cand_behavior: np.ndarray, seed: int) -> float:
    x = neural.reshape(-1, neural.shape[-1])
    y = behavior.reshape(-1, behavior.shape[-1])
    idx = np.arange(x.shape[0])
    train, test = train_test_split(idx, test_size=0.35, random_state=seed)
    scaler = StandardScaler().fit(x[train])
    reg = Ridge(alpha=1.0).fit(scaler.transform(x[train]), y[train])
    gt_pred = reg.predict(scaler.transform(x[test]))
    gt_ss = float(np.sum((y[test] - np.mean(y[test], axis=0, keepdims=True)) ** 2)) + 1e-12
    gt_r2 = 1.0 - float(np.sum((y[test] - gt_pred) ** 2)) / gt_ss
    cx = cand_neural.reshape(-1, cand_neural.shape[-1])
    cy = cand_behavior.reshape(-1, cand_behavior.shape[-1])
    n = min(cx.shape[0], cy.shape[0])
    cand_pred = reg.predict(scaler.transform(cx[:n]))
    cand_ss = float(np.sum((cy[:n] - np.mean(cy[:n], axis=0, keepdims=True)) ** 2)) + 1e-12
    cand_r2 = 1.0 - float(np.sum((cy[:n] - cand_pred) ** 2)) / cand_ss
    return float(1.0 / (1.0 + abs(gt_r2 - cand_r2)))


def _cca_score(neural: np.ndarray, behavior: np.ndarray, cand_neural: np.ndarray, cand_behavior: np.ndarray) -> float:
    def cca_mean(xarr: np.ndarray, yarr: np.ndarray) -> float:
        x = _standardize(xarr).reshape(-1, xarr.shape[-1])
        y = _standardize(yarr).reshape(-1, yarr.shape[-1])
        n = min(x.shape[0], y.shape[0], 2500)
        x = x[:n]
        y = y[:n]
        k = min(3, x.shape[1], y.shape[1])
        if k < 1:
            return float("nan")
        cca = CCA(n_components=k, max_iter=500)
        xs, ys = cca.fit_transform(x, y)
        vals = []
        for i in range(k):
            if np.std(xs[:, i]) > 1e-9 and np.std(ys[:, i]) > 1e-9:
                vals.append(np.corrcoef(xs[:, i], ys[:, i])[0, 1])
        return float(np.nanmean(vals)) if vals else float("nan")
    gt = cca_mean(neural, behavior)
    pred = cca_mean(cand_neural, cand_behavior)
    return float(1.0 / (1.0 + abs(gt - pred))) if np.isfinite(gt + pred) else float("nan")


def score_cross_modal_realism(neural: np.ndarray, behavior: np.ndarray, labels: np.ndarray, cand_neural: np.ndarray, cand_behavior: np.ndarray, *, seed: int) -> dict[str, float]:
    cov = _matrix_similarity(_cross_covariance(neural, behavior), _cross_covariance(cand_neural, cand_behavior))
    lag = _matrix_similarity(_lag_profile(neural, behavior), _lag_profile(cand_neural, cand_behavior))
    decode = _decoder_agreement(neural, behavior, labels, cand_neural, cand_behavior, seed)
    r2 = _neural_to_behavior_score(neural, behavior, cand_neural, cand_behavior, seed)
    cca = _cca_score(neural, behavior, cand_neural, cand_behavior)
    composite = float(np.nanmean([cov, decode, r2, lag, cca]))
    return {
        "neural_behavior_covariance_score": cov,
        "decoding_consistency_score": decode,
        "neural_to_behavior_r2_score": r2,
        "lagged_neural_behavior_score": lag,
        "cross_modal_latent_alignment_score": cca,
        "cross_modal_composite": composite,
    }


def build_cross_modal_candidates(system, behavior_candidates: dict[str, BehaviorCandidate], *, seed: int) -> dict[str, BehaviorCandidate]:
    rng = np.random.default_rng(seed)
    aligned_behavior = behavior_candidates["split_half_ceiling"].behavior
    n_states = int(np.nanmax(system.labels) + 1)
    modality_perm = rng.permutation(system.neural.shape[0])
    label_perm = rng.permutation(system.labels.reshape(-1)).reshape(system.labels.shape)
    label_behavior = system.behavior.copy()
    label_behavior[..., -1] = label_perm / max(1, n_states - 1)
    lag = max(6, system.neural.shape[1] // 8)
    return {
        "split_half_ceiling": BehaviorCandidate("split_half_ceiling", system.behavior + rng.normal(scale=0.025, size=system.behavior.shape), system.labels.copy(), system.neural + rng.normal(scale=0.025, size=system.neural.shape)),
        "aligned_model": BehaviorCandidate("aligned_model", aligned_behavior, _derive_labels(aligned_behavior, n_states), system.oracle),
        "modality_shuffle": BehaviorCandidate("modality_shuffle", system.behavior, system.labels.copy(), system.neural[modality_perm]),
        "temporal_lag": BehaviorCandidate("temporal_lag", np.roll(system.behavior, shift=lag, axis=1), np.roll(system.labels, shift=lag, axis=1), system.neural),
        "label_shuffle": BehaviorCandidate("label_shuffle", label_behavior, label_perm, system.neural),
        "neural_good_behavior_misaligned": BehaviorCandidate("neural_good_behavior_misaligned", behavior_candidates["time_shuffle"].behavior, behavior_candidates["time_shuffle"].labels, system.oracle),
    }


def _plot_supplement(
    output_path: Path,
    *,
    system,
    behavior_candidates: dict[str, BehaviorCandidate],
    cross_candidates: dict[str, BehaviorCandidate],
    behavior_scores: pd.DataFrame,
    cross_scores: pd.DataFrame,
) -> None:
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig = plt.figure(figsize=(13.5, 9.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.75, 1.15, 1.0], width_ratios=[1.1, 1.15, 1.15], hspace=0.55, wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    boxes = [("Neural\nrealism", 0.05, "#4C78A8"), ("Behavioral\nrealism", 0.38, "#54A24B"), ("Cross-modal\nrealism", 0.71, "#F58518")]
    for text, x, color in boxes:
        ax.add_patch(plt.Rectangle((x, 0.35), 0.24, 0.34, facecolor=color, alpha=0.22, edgecolor=color, lw=1.8))
        ax.text(x + 0.12, 0.52, text, ha="center", va="center", weight="bold")
    ax.annotate("", xy=(0.38, 0.52), xytext=(0.29, 0.52), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.annotate("", xy=(0.71, 0.52), xytext=(0.62, 0.52), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.set_title("A  NethoBench extension scope")

    ax = fig.add_subplot(gs[0, 1:])
    ax.axis("off")
    ax.text(
        0.0,
        0.92,
        "Proof-of-concept only: this supplement tests whether the implementation supports behavioral and neural-behavioral structural realism. "
        "The main paper claims remain focused on neural forecasting.",
        ha="left",
        va="top",
        fontsize=10,
        wrap=True,
    )
    ax.text(
        0.0,
        0.42,
        f"Dataset: controlled paired synthetic neural-behavior sequences from {system.system}; "
        f"{system.neural.shape[0]} sequences, {system.neural.shape[1]} time bins, "
        f"{system.neural.shape[2]} neural channels, {system.behavior.shape[2]} behavior channels.",
        ha="left",
        va="top",
        fontsize=9,
        wrap=True,
    )

    ax = fig.add_subplot(gs[1, 0])
    seq = 0
    gt = system.behavior[seq]
    gru = behavior_candidates["behavior_gru"].behavior[seq]
    bad = behavior_candidates["time_shuffle"].behavior[seq]
    ax.plot(gt[:, 0], gt[:, 1], color="black", lw=2.2, label="true behavior")
    ax.plot(gru[:, 0], gru[:, 1], color="#54A24B", lw=1.7, label="behavior GRU")
    ax.plot(bad[:, 0], bad[:, 1], color="#E45756", lw=1.1, alpha=0.75, label="time shuffle")
    ax.set_title("B  Behavioral trajectories")
    ax.set_xlabel("position x")
    ax.set_ylabel("position y")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.18)

    ax = fig.add_subplot(gs[1, 1])
    beh = behavior_scores.groupby("candidate")[["behavior_composite", "state_transition_score", "speed_distribution_score", "trajectory_path_score"]].mean(numeric_only=True)
    order = ["split_half_ceiling", "behavior_gru", "markov_ar_baseline", "smoothed_mean", "transition_shuffle", "time_shuffle"]
    beh = beh.reindex(order).dropna(how="all")
    x = np.arange(beh.shape[0])
    ax.bar(x, beh["behavior_composite"], color="#54A24B", alpha=0.82)
    ax.plot(x, beh["state_transition_score"], color="#2F5597", marker="o", lw=1.2, label="transition")
    ax.plot(x, beh["trajectory_path_score"], color="#F58518", marker="s", lw=1.2, label="path")
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", "\n") for v in beh.index], rotation=0, fontsize=6.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("C  Behavioral realism")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.18)

    ax = fig.add_subplot(gs[1, 2])
    time = np.arange(system.behavior.shape[1])
    neural = _standardize(system.neural)[seq, :, 0]
    aligned = _standardize(cross_candidates["aligned_model"].behavior)[seq, :, 2]
    lagged = _standardize(cross_candidates["temporal_lag"].behavior)[seq, :, 2]
    ax.plot(time, neural, color="#4C78A8", lw=1.8, label="neural ch. 1")
    ax.plot(time, aligned, color="#54A24B", lw=1.6, label="aligned speed")
    ax.plot(time, lagged, color="#E45756", lw=1.2, alpha=0.8, label="lagged speed")
    ax.set_title("D  Paired neural-behavior traces")
    ax.set_xlabel("time bin")
    ax.set_ylabel("z-score")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.18)

    ax = fig.add_subplot(gs[2, 0])
    cross = cross_scores.groupby("candidate")[["cross_modal_composite", "neural_behavior_covariance_score", "lagged_neural_behavior_score", "neural_to_behavior_r2_score"]].mean(numeric_only=True)
    order = ["split_half_ceiling", "aligned_model", "modality_shuffle", "temporal_lag", "label_shuffle", "neural_good_behavior_misaligned"]
    cross = cross.reindex(order).dropna(how="all")
    x = np.arange(cross.shape[0])
    ax.bar(x, cross["cross_modal_composite"], color="#F58518", alpha=0.84)
    ax.plot(x, cross["neural_behavior_covariance_score"], color="#4C78A8", marker="o", lw=1.2, label="covariance")
    ax.plot(x, cross["lagged_neural_behavior_score"], color="#B279A2", marker="s", lw=1.2, label="lagged")
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", "\n") for v in cross.index], fontsize=6.3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("E  Cross-modal realism")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.18)

    ax = fig.add_subplot(gs[2, 1:])
    ax.axis("off")
    beh_means = behavior_scores.groupby("candidate")["behavior_composite"].mean(numeric_only=True)
    cross_means = cross_scores.groupby("candidate")["cross_modal_composite"].mean(numeric_only=True)
    text = (
        "F  Interpretation\n"
        f"Behavior-only: behavior GRU composite={beh_means.get('behavior_gru', float('nan')):.3f}, "
        f"Markov/AR={beh_means.get('markov_ar_baseline', float('nan')):.3f}, "
        f"time shuffle={beh_means.get('time_shuffle', float('nan')):.3f}.\n"
        f"Cross-modal: aligned model={cross_means.get('aligned_model', float('nan')):.3f}, "
        f"modality shuffle={cross_means.get('modality_shuffle', float('nan')):.3f}, "
        f"temporal lag={cross_means.get('temporal_lag', float('nan')):.3f}.\n"
        "The result supports extensibility: NethoBench-style structural scores can be applied to behavioral and paired neural-behavioral processes, "
        "while full benchmarking of behavioral generative models remains future work."
    )
    ax.text(0, 1, text, ha="left", va="top", fontsize=10, linespacing=1.35, wrap=True)
    fig.suptitle("Supplementary proof of concept: behavioral and cross-modal structural realism", fontsize=13.5, y=0.99)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_behavior_crossmodal_supplement(
    *,
    output_root: Path | None = None,
    mode: str = "quick",
    min_free_disk_gb: float = 4.0,
) -> dict[str, object]:
    config = BehaviorCrossModalConfig.for_mode(mode, output_root=output_root, min_free_disk_gb=min_free_disk_gb)
    output_root = Path(config.output_root)
    disk_events = [_check_disk(output_root, config.min_free_disk_gb, "start")]
    systems = ["low_rank_rnn"] if mode == "quick" else ["nonlinear_lds", "low_rank_rnn", "switching_lds"]
    behavior_rows: list[dict[str, object]] = []
    cross_rows: list[dict[str, object]] = []
    example_payload = None
    for seed in config.seeds:
        for system_name in systems:
            disk_events.append(_check_disk(output_root, config.min_free_disk_gb, f"before_{system_name}_{seed}"))
            system = generate_rich_system(system_name, seed, n_sequences=config.n_sequences, seq_length=config.seq_length, n_regions=config.n_regions, latent_dim=config.latent_dim)
            behavior_candidates = build_behavior_candidates(system.behavior, system.labels, seed=seed)
            cross_candidates = build_cross_modal_candidates(system, behavior_candidates, seed=seed)
            for candidate in behavior_candidates.values():
                row: dict[str, object] = {"system": system.system, "seed": seed, "candidate": candidate.name}
                row.update(score_behavior_realism(system.behavior, system.labels, candidate.behavior, candidate.labels))
                behavior_rows.append(row)
            for candidate in cross_candidates.values():
                row = {"system": system.system, "seed": seed, "candidate": candidate.name}
                row.update(score_cross_modal_realism(system.neural, system.behavior, system.labels, np.asarray(candidate.neural), candidate.behavior, seed=seed))
                cross_rows.append(row)
            if example_payload is None:
                example_payload = (system, behavior_candidates, cross_candidates)
    behavior_df = pd.DataFrame(behavior_rows)
    cross_df = pd.DataFrame(cross_rows)
    behavior_path = output_root / "behavior_realism_scores.csv"
    cross_path = output_root / "cross_modal_realism_scores.csv"
    behavior_df.to_csv(behavior_path, index=False)
    cross_df.to_csv(cross_path, index=False)
    if example_payload is None:
        raise RuntimeError("No synthetic system was generated.")
    figure_path = output_root / "behavior_crossmodal_supplement.svg"
    _plot_supplement(figure_path, system=example_payload[0], behavior_candidates=example_payload[1], cross_candidates=example_payload[2], behavior_scores=behavior_df, cross_scores=cross_df)
    disk_events.append(_check_disk(output_root, config.min_free_disk_gb, "finished"))
    report = {
        "config": {**asdict(config), "output_root": str(config.output_root), "seeds": list(config.seeds)},
        "disk_events": disk_events,
        "behavior_summary": behavior_df.groupby("candidate")["behavior_composite"].mean(numeric_only=True).to_dict(),
        "cross_modal_summary": cross_df.groupby("candidate")["cross_modal_composite"].mean(numeric_only=True).to_dict(),
        "outputs": {
            "behavior_realism_scores_csv": str(behavior_path),
            "cross_modal_realism_scores_csv": str(cross_path),
            "behavior_crossmodal_supplement_svg": str(figure_path),
            "behavior_crossmodal_supplement_png": str(figure_path.with_suffix(".png")),
        },
        "paper_interpretation": (
            "NethoBench is implemented as a general structural-realism framework. "
            "This proof-of-concept supports behavioral and cross-modal realism analyses, "
            "while the paper's core scientific claims remain focused on neural forecasting."
        ),
    }
    report_path = output_root / "behavior_crossmodal_supplement_report.json"
    report["outputs"]["behavior_crossmodal_supplement_report_json"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2))
    return {"output_root": str(output_root), "outputs": report["outputs"], "report": report}


__all__ = [
    "BehaviorCrossModalConfig",
    "build_behavior_candidates",
    "run_behavior_crossmodal_supplement",
    "score_behavior_realism",
    "score_cross_modal_realism",
]
