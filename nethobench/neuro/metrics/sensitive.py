from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import importlib.util

import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from nethobench.utils.calculation import align_arrays, robust_scale, correlation_score, EPS
from nethobench.utils.evaluation_constants import (
    MIN_ROWS_PCA,
    MIN_SAMPLES_MANIFOLD,
    CROSS_CORR_MAX_LAG_SMALL,
    MIN_ROWS_CORRCOEF,
    AUTOCORR_MAX_LAG,
    LATENT_SUBSAMPLE_SIZE,
    N_POINTS_MANIFOLD_PH_KNN,
    MAX_SEQUENCES_STRATIFIED,
    DISTRIBUTION_GRID_QUANTILES,
    WELCH_SAMPLING_FREQUENCY,
    WELCH_NPERSEG,
    HISTOGRAM_BINS_DEFAULT,
)
if importlib.util.find_spec("ripser") is not None:
    from ripser import ripser
else: 
    ripser = None


Array3D = np.ndarray
MetricFn = Callable[[Array3D, Array3D], dict]
CorruptionFn = Callable[[Array3D, float, int], Array3D]




@dataclass(frozen=True)
class CandidateSpec:
    metric_group: str
    name: str
    description: str
    score_fn: MetricFn


@dataclass(frozen=True)
class CorruptionSpec:
    metric_group: str
    name: str
    description: str
    levels: tuple[float, ...]
    apply_fn: CorruptionFn




def finite_rows(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    return X[mask], Y[mask]


def _standardize_with_gt(
    Xg: np.ndarray, Xp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(Xg, axis=0, keepdims=True)
    sd = np.std(Xg, axis=0, keepdims=True)
    sd = np.where(sd < EPS, 1.0, sd)
    return (Xg - mu) / sd, (Xp - mu) / sd


def score_from_distance(distance: float) -> float:
    return float(1.0 / (1.0 + distance)) if np.isfinite(distance) else np.nan



def _quantile_distance(
    x: np.ndarray, y: np.ndarray, qs: tuple[float, ...] = DISTRIBUTION_GRID_QUANTILES
) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 8 or y.size < 8:
        return np.nan
    qx = np.quantile(x, qs)
    qy = np.quantile(y, qs)
    scale = robust_scale(x)
    return float(np.mean(np.abs(qx - qy) / (scale)))


def _tv_score_from_hist(hist_g: np.ndarray, hist_p: np.ndarray) -> float:
    hist_g = np.asarray(hist_g, dtype=np.float64)
    hist_p = np.asarray(hist_p, dtype=np.float64)
    if hist_g.shape != hist_p.shape or hist_g.size == 0:
        return np.nan
    return float(np.clip(1.0 - 0.5 * np.sum(np.abs(hist_g - hist_p)), 0.0, 1.0))


def _gt_hist_score(xg: np.ndarray, xp: np.ndarray, bins: int = HISTOGRAM_BINS_DEFAULT) -> float:
    xg = np.asarray(xg, dtype=np.float64)
    xp = np.asarray(xp, dtype=np.float64)
    xg = xg[np.isfinite(xg)]
    xp = xp[np.isfinite(xp)]
    if xg.size < bins + 2 or xp.size < bins + 2:
        return np.nan
    lo, hi = np.quantile(xg, [0.02, 0.98])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.min(xg))
        hi = float(np.max(xg) + EPS)
    edges = np.linspace(lo, hi, bins + 1)
    hg, _ = np.histogram(np.clip(xg, lo, hi), bins=edges, density=False)
    hp, _ = np.histogram(np.clip(xp, lo, hi), bins=edges, density=False)
    hg = hg.astype(np.float64)
    hp = hp.astype(np.float64)
    hg = hg / (np.sum(hg) + EPS)
    hp = hp / (np.sum(hp) + EPS)
    return _tv_score_from_hist(hg, hp)


def _choose_k(
    Xg: np.ndarray, var_target: float = 0.9, k_min: int = 2, k_max: int = 8
) -> int:
    k_fit = int(min(k_max, Xg.shape[1], Xg.shape[0] - 1))
    if k_fit < 2:
        return 1
    pca = PCA(n_components=k_fit, svd_solver="full", random_state=0).fit(Xg)
    csum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(csum, var_target) + 1)
    return int(np.clip(k, k_min, k_fit))


def _fit_gt_pca_pooled(
    gt: Array3D, pred: Array3D, k_max: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = finite_rows(Xg, Xp)
    if Xg.shape[0] < MIN_ROWS_PCA:
        raise ValueError("Not enough valid rows for pooled PCA.")
    Xg_z, Xp_z = _standardize_with_gt(Xg, Xp)
    k = _choose_k(Xg_z, k_max=k_max)
    pca = PCA(n_components=k, svd_solver="full", random_state=0).fit(Xg_z)
    return pca.transform(Xg_z), pca.transform(Xp_z), pca.components_[:k], pca


def _autocorr_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < max_lag + 5:
        return np.full(max_lag, np.nan, dtype=np.float64)
    x = x - np.mean(x)
    denom = np.dot(x, x)
    if denom < EPS:
        return np.full(max_lag, np.nan, dtype=np.float64)
    full = np.correlate(x, x, mode="full")
    mid = x.size - 1
    ac = full[mid + 1 : mid + max_lag + 1] / denom
    return ac.astype(np.float64)


def _crosscorr_1d(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < max_lag + 5:
        return np.full((2 * max_lag) + 1, np.nan, dtype=np.float64)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if denom < EPS:
        return np.full((2 * max_lag) + 1, np.nan, dtype=np.float64)
    full = signal.correlate(x, y, mode="full", method="fft") / denom
    center = x.size - 1
    return full[center - max_lag : center + max_lag + 1].astype(np.float64)


def _welch_relative_psd(
    x: np.ndarray, fs: float = WELCH_SAMPLING_FREQUENCY
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < MIN_SAMPLES_MANIFOLD:
        return np.array([]), np.array([])
    nperseg = min(WELCH_NPERSEG, x.size)
    freqs, pxx = signal.welch(
        x, fs=fs, nperseg=nperseg, noverlap=min(nperseg // 2, nperseg - 1)
    )
    mask = freqs > 0
    freqs = freqs[mask]
    pxx = pxx[mask]
    total = np.sum(pxx)
    if total < EPS:
        return freqs, np.full_like(freqs, np.nan)
    return freqs, pxx / total


def safe_corrcoef(X: np.ndarray) -> np.ndarray | None:
    X = np.asarray(X, dtype=np.float64)
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    if X.shape[0] < MIN_ROWS_CORRCOEF:
        return None
    C = np.corrcoef(X, rowvar=False)
    if not np.all(np.isfinite(C)):
        return None
    C = np.clip(C, -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    return C


def _upper_triangle(C: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(C, k=1)
    return np.asarray(C[iu], dtype=np.float64)


def _lagged_corr_matrix(arr: Array3D, lag: int) -> np.ndarray | None:
    if arr.shape[1] <= lag + 2:
        return None
    left = (
        arr[:, :-lag, :].reshape(-1, arr.shape[-1])
        if lag > 0
        else arr.reshape(-1, arr.shape[-1])
    )
    right = (
        arr[:, lag:, :].reshape(-1, arr.shape[-1])
        if lag > 0
        else arr.reshape(-1, arr.shape[-1])
    )
    mask = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    left = left[mask]
    right = right[mask]
    if left.shape[0] < MIN_ROWS_CORRCOEF:
        return None
    out = np.empty((arr.shape[-1], arr.shape[-1]), dtype=np.float64)
    for i in range(arr.shape[-1]):
        xi = left[:, i]
        xi = xi - np.mean(xi)
        dxi = np.sqrt(np.sum(xi * xi)) + EPS
        for j in range(arr.shape[-1]):
            yj = right[:, j]
            yj = yj - np.mean(yj)
            dyj = np.sqrt(np.sum(yj * yj)) + EPS
            out[i, j] = np.sum(xi * yj) / (dxi * dyj)
    return np.clip(out, -1.0, 1.0)


def _pooled_latent_clouds(
    gt: Array3D,
    pred: Array3D,
    *,
    k_max: int = 3,
    n_points: int = LATENT_SUBSAMPLE_SIZE,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    gt, pred = align_arrays(gt, pred)
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = finite_rows(Xg, Xp)
    if Xg.shape[0] < MIN_SAMPLES_MANIFOLD:
        return None, None
    Xg_z, Xp_z = _standardize_with_gt(Xg, Xp)
    k = _choose_k(Xg_z, k_max=min(k_max, Xg_z.shape[1]))
    pca = PCA(n_components=k, svd_solver="full", random_state=seed).fit(Xg_z)
    Zg = pca.transform(Xg_z)
    Zp = pca.transform(Xp_z)
    rng = np.random.default_rng(seed)
    if Zg.shape[0] > n_points:
        idx_g = rng.choice(Zg.shape[0], size=n_points, replace=False)
        Zg = Zg[idx_g]
    if Zp.shape[0] > n_points:
        idx_p = rng.choice(Zp.shape[0], size=n_points, replace=False)
        Zp = Zp[idx_p]
    return Zg.astype(np.float64), Zp.astype(np.float64)


def _fit_pooled_gt_pca(
    gt: Array3D,
    pred: Array3D,
    *,
    k_max: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, PCA] | tuple[None, None, None]:
    gt, pred = align_arrays(gt, pred)
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = finite_rows(Xg, Xp)
    if Xg.shape[0] < MIN_SAMPLES_MANIFOLD:
        return None, None, None
    mu = np.mean(Xg, axis=0, keepdims=True)
    sd = np.std(Xg, axis=0, keepdims=True)
    sd = np.where(sd < EPS, 1.0, sd)
    Xg_z = (Xg - mu) / sd
    k = _choose_k(Xg_z, k_max=min(k_max, Xg_z.shape[1]))
    pca = PCA(n_components=k, svd_solver="full", random_state=seed).fit(Xg_z)
    return mu, sd, pca


def _stratified_latent_clouds(
    gt: Array3D,
    pred: Array3D,
    *,
    k_max: int = 3,
    points_per_seq: int = 2,
    max_sequences: int = MAX_SEQUENCES_STRATIFIED,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    gt, pred = align_arrays(gt, pred)
    mu, sd, pca = _fit_pooled_gt_pca(gt, pred, k_max=k_max, seed=seed)
    if pca is None:
        return None, None
    seq_indices = np.arange(gt.shape[0], dtype=int)
    if seq_indices.size > max_sequences:
        keep_idx = np.linspace(0, seq_indices.size - 1, num=max_sequences, dtype=int)
        seq_indices = seq_indices[keep_idx]
    clouds_g = []
    clouds_p = []
    for seq_idx in seq_indices:
        Xg, Xp = finite_rows(gt[seq_idx], pred[seq_idx])
        if Xg.shape[0] < max(8, points_per_seq + 1):
            continue
        Zg = pca.transform((Xg - mu) / sd)
        Zp = pca.transform((Xp - mu) / sd)
        take = min(points_per_seq, Zg.shape[0], Zp.shape[0])
        idx = np.linspace(0, min(Zg.shape[0], Zp.shape[0]) - 1, num=take, dtype=int)
        clouds_g.append(Zg[idx])
        clouds_p.append(Zp[idx])
    if not clouds_g or not clouds_p:
        return None, None
    return np.vstack(clouds_g).astype(np.float64), np.vstack(clouds_p).astype(
        np.float64
    )


def _ripser_diagrams(X: np.ndarray, n_perm: int, *, maxdim: int = 1):
    if ripser is None or X is None or X.shape[0] < CROSS_CORR_MAX_LAG_SMALL:
        return None
    return ripser(X, maxdim=maxdim, n_perm=min(n_perm, X.shape[0]))["dgms"]


def _lifetimes(dgm: np.ndarray | None) -> np.ndarray:
    if dgm is None or len(dgm) == 0:
        return np.array([], dtype=np.float64)
    arr = np.asarray(dgm, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.array([], dtype=np.float64)
    life = arr[:, 1] - arr[:, 0]
    mask = np.isfinite(life) & (life > 0)
    return np.sort(life[mask].astype(np.float64))


def _lifetime_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 and y.size == 0:
        return 1.0
    if x.size == 0 or y.size == 0:
        return 0.0
    dist = _quantile_distance(x, y, qs=(0.25, 0.5, 0.75, 0.9))
    total = abs(np.sum(x) - np.sum(y)) / (np.sum(x) + 0.05)
    return float(np.nanmean([score_from_distance(dist), score_from_distance(total)]))


def _betti_curve(dgm: np.ndarray | None, grid: np.ndarray) -> np.ndarray:
    if dgm is None or len(dgm) == 0:
        return np.zeros_like(grid, dtype=np.float64)
    arr = np.asarray(dgm, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return np.zeros_like(grid, dtype=np.float64)
    births = arr[:, 0]
    deaths = arr[:, 1]
    mask = np.isfinite(births) & np.isfinite(deaths)
    births = births[mask]
    deaths = deaths[mask]
    if births.size == 0:
        return np.zeros_like(grid, dtype=np.float64)
    return np.asarray(
        [np.sum((births <= t) & (t < deaths)) for t in grid], dtype=np.float64
    )


def _betti_curve_similarity(
    dgm_g: np.ndarray | None, dgm_p: np.ndarray | None
) -> float:
    finite = []
    for dgm in (dgm_g, dgm_p):
        if dgm is None or len(dgm) == 0:
            continue
        arr = np.asarray(dgm, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 2:
            vals = arr[np.isfinite(arr[:, 1]), 1]
            if vals.size:
                finite.append(float(np.max(vals)))
    max_scale = max(finite) if finite else 0.0
    if max_scale <= 0:
        return 1.0
    grid = np.linspace(0.0, max_scale, 32, dtype=np.float64)
    bg = _betti_curve(dgm_g, grid)
    bp = _betti_curve(dgm_p, grid)
    corr = correlation_score(bg, bp)
    dist = np.mean(np.abs(bp - bg)) / (np.mean(bg) + 1.0)
    return float(np.nanmean([corr, score_from_distance(dist)]))


def _participation_ratio(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 8:
        return np.nan
    C = np.cov(X, rowvar=False)
    vals = np.linalg.eigvalsh(C)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return np.nan
    return float((np.sum(vals) ** 2) / (np.sum(vals**2) + EPS))


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _knn_distance_profile(X: np.ndarray, k: int = 5) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < max(8, k + 2):
        return np.array([], dtype=np.float64)
    D = _pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    kth = np.partition(D, kth=min(k - 1, D.shape[1] - 1), axis=1)[
        :, min(k - 1, D.shape[1] - 1)
    ]
    return kth[np.isfinite(kth)]


def _radial_profile(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 8:
        return np.array([], dtype=np.float64)
    center = np.mean(X, axis=0, keepdims=True)
    return np.linalg.norm(X - center, axis=1).astype(np.float64)


def _iter_region_series(arr: Array3D):
    for region in range(arr.shape[-1]):
        yield arr[:, :, region].reshape(-1)


def trajectory_path_features(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = align_arrays(gt, pred)
    k_max = min(4, gt.shape[-1])
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = finite_rows(Xg, Xp)
    if Xg.shape[0] < MIN_SAMPLES_MANIFOLD:
        return {"score": np.nan}
    Xg_z, Xp_z = _standardize_with_gt(Xg, Xp)
    k = _choose_k(Xg_z, k_max=k_max)
    pca = PCA(n_components=k, svd_solver="full", random_state=0).fit(Xg_z)
    Zg = pca.transform(Xg_z).reshape(gt.shape[0], -1, k)
    Zp = pca.transform(Xp_z).reshape(pred.shape[0], -1, k)
    features_g = []
    features_p = []
    for seq_g, seq_p in zip(Zg, Zp):
        vg = np.diff(seq_g, axis=0)
        vp = np.diff(seq_p, axis=0)
        if vg.shape[0] < 4 or vp.shape[0] < 4:
            continue

        def _seq_features(z, v):
            path = np.sum(np.linalg.norm(v, axis=1))
            disp = np.linalg.norm(z[-1] - z[0])
            radius = np.mean(
                np.linalg.norm(z - np.mean(z, axis=0, keepdims=True), axis=1)
            )
            persistence = disp / (path + EPS)
            speed = np.linalg.norm(v, axis=1)
            speed_lag1 = correlation_score(speed[1:], speed[:-1])
            return np.array(
                [path, disp, radius, persistence, speed_lag1], dtype=np.float64
            )

        features_g.append(_seq_features(seq_g, vg))
        features_p.append(_seq_features(seq_p, vp))
    if not features_g:
        return {"score": np.nan}
    fg = np.vstack(features_g)
    fp = np.vstack(features_p)
    denom = np.nanmean(np.abs(fg), axis=0) + EPS
    distance = np.nanmean(
        np.abs(np.nanmean(fp, axis=0) - np.nanmean(fg, axis=0)) / denom
    )
    return {"score": score_from_distance(distance)}


def trajectory_occupancy_velocity(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = align_arrays(gt, pred)
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = finite_rows(Xg, Xp)
    if Xg.shape[0] < MIN_SAMPLES_MANIFOLD:
        return {"score": np.nan}
    Xg_z, Xp_z = _standardize_with_gt(Xg, Xp)
    k = _choose_k(Xg_z, k_max=min(3, Xg_z.shape[1]))
    pca = PCA(n_components=k, svd_solver="full", random_state=0).fit(Xg_z)
    Zg = pca.transform(Xg_z)
    Zp = pca.transform(Xp_z)
    occ_scores = []
    for dim in range(k):
        occ_scores.append(_gt_hist_score(Zg[:, dim], Zp[:, dim], bins=HISTOGRAM_BINS_DEFAULT))
    Zg_seq = pca.transform(gt.reshape(-1, gt.shape[-1])).reshape(
        gt.shape[0], gt.shape[1], k
    )
    Zp_seq = pca.transform(pred.reshape(-1, pred.shape[-1])).reshape(
        pred.shape[0], pred.shape[1], k
    )
    vg = np.diff(Zg_seq, axis=1).reshape(-1, k)
    vp = np.diff(Zp_seq, axis=1).reshape(-1, k)
    speed_score = score_from_distance(
        _quantile_distance(np.linalg.norm(vg, axis=1), np.linalg.norm(vp, axis=1))
    )
    turn_g = np.sum(vg[1:] * vg[:-1], axis=1) / (
        (np.linalg.norm(vg[1:], axis=1) * np.linalg.norm(vg[:-1], axis=1)) + EPS
    )
    turn_p = np.sum(vp[1:] * vp[:-1], axis=1) / (
        (np.linalg.norm(vp[1:], axis=1) * np.linalg.norm(vp[:-1], axis=1)) + EPS
    )
    turn_score = score_from_distance(
        _quantile_distance(turn_g, turn_p, qs=DISTRIBUTION_GRID_QUANTILES)
    )
    return {"score": float(np.nanmean(occ_scores + [speed_score, turn_score]))}


def bandpower_band_fraction(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = align_arrays(gt, pred)
    specs_g = []
    specs_p = []
    for xg, xp in zip(_iter_region_series(gt), _iter_region_series(pred)):
        freqs_g, psg = _welch_relative_psd(xg)
        freqs_p, psp = _welch_relative_psd(xp)
        if psg.size == 0 or psp.size == 0 or psg.shape != psp.shape:
            continue
        specs_g.append(psg)
        specs_p.append(psp)
    if not specs_g:
        return {"score": np.nan}
    mean_g = np.nanmean(np.vstack(specs_g), axis=0)
    mean_p = np.nanmean(np.vstack(specs_p), axis=0)
    csum = np.cumsum(mean_g)
    splits = [np.searchsorted(csum, q) for q in (1.0 / 3.0, 2.0 / 3.0)]
    bands = [
        slice(0, splits[0]),
        slice(splits[0], splits[1]),
        slice(splits[1], mean_g.size),
    ]
    frac_g = np.array([np.sum(mean_g[band]) for band in bands], dtype=np.float64)
    frac_p = np.array([np.sum(mean_p[band]) for band in bands], dtype=np.float64)
    score = 1.0 - 0.5 * np.sum(np.abs(frac_g - frac_p))
    return {"score": float(np.clip(score, 0.0, 1.0))}


def pca_reconstruction_transfer(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = align_arrays(gt, pred)
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = finite_rows(Xg, Xp)
    if Xg.shape[0] < 64:
        return {"score": np.nan}
    Xg_z, Xp_z = _standardize_with_gt(Xg, Xp)
    k = _choose_k(Xg_z, k_max=min(8, Xg_z.shape[1]))
    pca = PCA(n_components=k, svd_solver="full", random_state=0).fit(Xg_z)
    Ug = pca.components_
    Zg = pca.transform(Xg_z)
    Zp = pca.transform(Xp_z)
    Xg_rec = pca.inverse_transform(Zg)
    Xp_rec = pca.inverse_transform(Zp)
    err_g = np.mean((Xg_z - Xg_rec) ** 2)
    err_p = np.mean((Xp_z - Xp_rec) ** 2)
    var_g = np.mean(Xg_z**2)
    var_p = np.mean(Xp_z**2)
    frac_g = 1.0 - (err_g / (var_g + EPS))
    frac_p = 1.0 - (err_p / (var_p + EPS))
    var_score = score_from_distance(
        np.mean(
            np.abs(np.var(Zp, axis=0) - np.var(Zg, axis=0))
            / (np.var(Zg, axis=0) + 0.05)
        )
    )
    rec_score = float(np.clip(frac_p / (frac_g + EPS), 0.0, 1.0))
    return {"score": float(np.clip(0.5 * rec_score + 0.5 * var_score, 0.0, 1.0))}


def pca_reconstruction_product(gt: Array3D, pred: Array3D) -> dict:
    base = pca_reconstruction_transfer(gt, pred).get("score", np.nan)
    if not np.isfinite(base):
        return {"score": np.nan}
    return {"score": float(np.clip(base**2, 0.0, 1.0))}


def autocorr_weighted_rmse(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = align_arrays(gt, pred)
    max_lag = min(AUTOCORR_MAX_LAG, gt.shape[1] // 6)
    weights = 1.0 / np.sqrt(np.arange(1, max_lag + 1, dtype=np.float64))
    region_scores = []
    for region in range(gt.shape[-1]):
        ac_g = np.nanmean(
            np.vstack(
                [_autocorr_1d(gt[s, :, region], max_lag) for s in range(gt.shape[0])]
            ),
            axis=0,
        )
        ac_p = np.nanmean(
            np.vstack(
                [
                    _autocorr_1d(pred[s, :, region], max_lag)
                    for s in range(pred.shape[0])
                ]
            ),
            axis=0,
        )
        if not np.isfinite(ac_g).any() or not np.isfinite(ac_p).any():
            continue
        diff = (ac_p - ac_g) * weights
        denom = np.sqrt(np.mean((ac_g * weights) ** 2)) + 0.05
        region_scores.append(score_from_distance(np.sqrt(np.mean(diff**2)) / denom))
    return {"score": float(np.nanmean(region_scores)) if region_scores else np.nan}


def autocorr_weighted_rmse_power(gt: Array3D, pred: Array3D) -> dict:
    base = autocorr_weighted_rmse(gt, pred).get("score", np.nan)
    if not np.isfinite(base):
        return {"score": np.nan}
    return {"score": float(np.clip(base**2, 0.0, 1.0))}


def crosscorr_lagged_matrix(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = align_arrays(gt, pred)
    Cg0 = safe_corrcoef(gt.reshape(-1, gt.shape[-1]))
    Cp0 = safe_corrcoef(pred.reshape(-1, pred.shape[-1]))
    Cg1 = _lagged_corr_matrix(gt, lag=1)
    Cp1 = _lagged_corr_matrix(pred, lag=1)
    if Cg0 is None or Cp0 is None or Cg1 is None or Cp1 is None:
        return {"score": np.nan}
    e0g = _upper_triangle(Cg0)
    e0p = _upper_triangle(Cp0)
    e1g = Cg1.reshape(-1)
    e1p = Cp1.reshape(-1)
    score0 = 0.5 * correlation_score(e0g, e0p) + 0.5 * score_from_distance(
        np.mean(np.abs(e0p - e0g)) / (robust_scale(e0g))
    )
    score1 = 0.5 * correlation_score(e1g, e1p) + 0.5 * score_from_distance(
        np.mean(np.abs(e1p - e1g)) / (robust_scale(e1g))
    )
    return {"score": float(np.nanmean([score0, score1]))}


def crosscorr_topedge_profiles(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = align_arrays(gt, pred)
    Cg0 = safe_corrcoef(gt.reshape(-1, gt.shape[-1]))
    if Cg0 is None:
        return {"score": np.nan}
    iu = np.triu_indices_from(Cg0, k=1)
    edge_strength = np.abs(Cg0[iu])
    if edge_strength.size == 0:
        return {"score": np.nan}
    thresh = np.quantile(edge_strength, 0.75)
    pairs = [(i, j) for i, j, w in zip(iu[0], iu[1], edge_strength) if w >= thresh]
    if not pairs:
        return {"score": np.nan}
    max_lag = min(CROSS_CORR_MAX_LAG_SMALL, gt.shape[1] // 12)
    scores = []
    for i, j in pairs:
        cc_g = np.nanmean(
            np.vstack(
                [
                    _crosscorr_1d(gt[s, :, i], gt[s, :, j], max_lag)
                    for s in range(gt.shape[0])
                ]
            ),
            axis=0,
        )
        cc_p = np.nanmean(
            np.vstack(
                [
                    _crosscorr_1d(pred[s, :, i], pred[s, :, j], max_lag)
                    for s in range(pred.shape[0])
                ]
            ),
            axis=0,
        )
        if not np.isfinite(cc_g).any() or not np.isfinite(cc_p).any():
            continue
        score_curve = correlation_score(cc_g, cc_p)
        lag_g = int(np.nanargmax(np.abs(cc_g))) - max_lag
        lag_p = int(np.nanargmax(np.abs(cc_p))) - max_lag
        lag_score = 1.0 - (abs(lag_p - lag_g) / max_lag)
        amp_g = cc_g[lag_g + max_lag]
        amp_p = cc_p[lag_p + max_lag]
        amp_score = score_from_distance(
            abs(amp_p - amp_g) / (robust_scale(cc_g))
        )
        scores.append(np.nanmean([score_curve, lag_score, amp_score]))
    return {"score": float(np.nanmean(scores)) if scores else np.nan}


def manifold_ph_lifetime_profile(gt: Array3D, pred: Array3D) -> dict:
    Zg, Zp = _pooled_latent_clouds(gt, pred, k_max=3, n_points=LATENT_SUBSAMPLE_SIZE, seed=0)
    if Zg is None or Zp is None:
        return {"score": np.nan}
    dgms_g = _ripser_diagrams(Zg, maxdim=1, n_perm=64)
    dgms_p = _ripser_diagrams(Zp, maxdim=1, n_perm=64)
    if dgms_g is None or dgms_p is None:
        return {"score": np.nan}
    score_h0 = _lifetime_similarity(_lifetimes(dgms_g[0]), _lifetimes(dgms_p[0]))
    score_h1 = _lifetime_similarity(_lifetimes(dgms_g[1]), _lifetimes(dgms_p[1]))
    return {"score": float(np.nanmean([score_h0, score_h1]))}


def manifold_ph_knn_profile(gt: Array3D, pred: Array3D) -> dict:
    Zg, Zp = _pooled_latent_clouds(gt, pred, k_max=3, n_points=N_POINTS_MANIFOLD_PH_KNN, seed=3)
    if Zg is None or Zp is None:
        return {"score": np.nan}
    score_life = manifold_ph_lifetime_profile(gt, pred).get("score", np.nan)
    knn_g = _knn_distance_profile(Zg, k=5)
    knn_p = _knn_distance_profile(Zp, k=5)
    score_knn = score_from_distance(
        _quantile_distance(knn_g, knn_p, qs=(0.1, 0.25, 0.5, 0.75, 0.9))
    )
    if not np.isfinite(score_life) or not np.isfinite(score_knn):
        return {"score": np.nan}
    return {"score": float(np.sqrt(score_life * score_knn))}


def manifold_ph_stratified_lifetime(gt: Array3D, pred: Array3D) -> dict:
    Zg, Zp = _stratified_latent_clouds(
        gt, pred, k_max=3, points_per_seq=2, max_sequences=48, seed=5
    )
    if Zg is None or Zp is None:
        return {"score": np.nan}
    dgms_g = _ripser_diagrams(Zg, maxdim=1, n_perm=64)
    dgms_p = _ripser_diagrams(Zp, maxdim=1, n_perm=64)
    if dgms_g is None or dgms_p is None:
        return {"score": np.nan}
    score_h0 = _lifetime_similarity(_lifetimes(dgms_g[0]), _lifetimes(dgms_p[0]))
    score_h1 = _lifetime_similarity(_lifetimes(dgms_g[1]), _lifetimes(dgms_p[1]))
    return {"score": float(np.nanmean([score_h0, score_h1]))}


def corruption_time_shuffle_blend(data: Array3D, level: float, seed: int) -> Array3D:
    rng = np.random.default_rng(seed)
    out = np.array(data, copy=True)
    for seq in range(out.shape[0]):
        for region in range(out.shape[-1]):
            shuffled = np.array(out[seq, :, region], copy=True)
            rng.shuffle(shuffled)
            out[seq, :, region] = (1.0 - level) * out[seq, :, region] + level * shuffled
    return out


def corruption_region_permute_blend(data: Array3D, level: float, seed: int) -> Array3D:
    rng = np.random.default_rng(seed)
    out = np.array(data, copy=True)
    perm = rng.permutation(out.shape[-1])
    permuted = out[:, :, perm]
    return (1.0 - level) * out + level * permuted


def corruption_region_shift_desync(data: Array3D, level: float, seed: int) -> Array3D:
    rng = np.random.default_rng(seed)
    out = np.array(data, copy=True)
    max_shift = max(1, int(round(level * min(24, out.shape[1] // 6))))
    for seq in range(out.shape[0]):
        for region in range(out.shape[-1]):
            shift = rng.integers(-max_shift, max_shift + 1)
            out[seq, :, region] = np.roll(out[seq, :, region], int(shift))
    return out
