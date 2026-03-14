from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import signal
from sklearn.decomposition import PCA

try:
    from ripser import ripser
except Exception:  # pragma: no cover - optional dependency during development
    ripser = None


Array3D = np.ndarray
MetricFn = Callable[[Array3D, Array3D], dict]
CorruptionFn = Callable[[Array3D, float, int], Array3D]


EPS = 1e-8


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


def _align_arrays(gt: Array3D, pred: Array3D) -> tuple[Array3D, Array3D]:
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if gt.shape[1] != pred.shape[1]:
        m = min(gt.shape[1], pred.shape[1])
        gt = gt[:, :m, :]
        pred = pred[:, :m, :]
    if gt.shape != pred.shape:
        raise ValueError(f"Aligned mismatch: {gt.shape} vs {pred.shape}")
    return gt, pred


def _finite_rows(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    return X[mask], Y[mask]


def _standardize_with_gt(Xg: np.ndarray, Xp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(Xg, axis=0, keepdims=True)
    sd = np.std(Xg, axis=0, keepdims=True)
    sd = np.where(sd < EPS, 1.0, sd)
    return (Xg - mu) / sd, (Xp - mu) / sd


def _score_from_distance(distance: float) -> float:
    return float(1.0 / (1.0 + distance)) if np.isfinite(distance) else np.nan


def _corr_score(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 3:
        return np.nan
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) < EPS or np.std(bb) < EPS:
        return np.nan
    return float(np.clip((np.corrcoef(aa, bb)[0, 1] + 1.0) / 2.0, 0.0, 1.0))


def _robust_scale(x: np.ndarray, floor: float = 0.05) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return float(floor)
    q25, q75 = np.quantile(x, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < floor:
        scale = float(np.std(x))
    return float(scale if np.isfinite(scale) and scale >= floor else floor)


def _quantile_distance(x: np.ndarray, y: np.ndarray, qs: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 8 or y.size < 8:
        return np.nan
    qx = np.quantile(x, qs)
    qy = np.quantile(y, qs)
    scale = _robust_scale(x)
    return float(np.mean(np.abs(qx - qy) / (scale + EPS)))


def _tv_score_from_hist(hist_g: np.ndarray, hist_p: np.ndarray) -> float:
    hist_g = np.asarray(hist_g, dtype=np.float64)
    hist_p = np.asarray(hist_p, dtype=np.float64)
    if hist_g.shape != hist_p.shape or hist_g.size == 0:
        return np.nan
    return float(np.clip(1.0 - 0.5 * np.sum(np.abs(hist_g - hist_p)), 0.0, 1.0))


def _gt_hist_score(xg: np.ndarray, xp: np.ndarray, bins: int = 12) -> float:
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


def _choose_k(Xg: np.ndarray, var_target: float = 0.9, k_min: int = 2, k_max: int = 8) -> int:
    k_fit = int(min(k_max, Xg.shape[1], Xg.shape[0] - 1))
    if k_fit < 2:
        return 1
    pca = PCA(n_components=k_fit, svd_solver="full", random_state=0).fit(Xg)
    csum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(csum, var_target) + 1)
    return int(np.clip(k, k_min, k_fit))


def _fit_gt_pca_pooled(gt: Array3D, pred: Array3D, k_max: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = _finite_rows(Xg, Xp)
    if Xg.shape[0] < 32:
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


def _welch_relative_psd(x: np.ndarray, fs: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 64:
        return np.array([]), np.array([])
    nperseg = min(256, x.size)
    freqs, pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=min(nperseg // 2, nperseg - 1))
    mask = freqs > 0
    freqs = freqs[mask]
    pxx = pxx[mask]
    total = np.sum(pxx)
    if total < EPS:
        return freqs, np.full_like(freqs, np.nan)
    return freqs, pxx / total


def _safe_corrcoef(X: np.ndarray) -> np.ndarray | None:
    X = np.asarray(X, dtype=np.float64)
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    if X.shape[0] < 16:
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
    left = arr[:, :-lag, :].reshape(-1, arr.shape[-1]) if lag > 0 else arr.reshape(-1, arr.shape[-1])
    right = arr[:, lag:, :].reshape(-1, arr.shape[-1]) if lag > 0 else arr.reshape(-1, arr.shape[-1])
    mask = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    left = left[mask]
    right = right[mask]
    if left.shape[0] < 16:
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
    n_points: int = 96,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    gt, pred = _align_arrays(gt, pred)
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = _finite_rows(Xg, Xp)
    if Xg.shape[0] < 64:
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
    gt, pred = _align_arrays(gt, pred)
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = _finite_rows(Xg, Xp)
    if Xg.shape[0] < 64:
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
    max_sequences: int = 48,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    gt, pred = _align_arrays(gt, pred)
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
        Xg, Xp = _finite_rows(gt[seq_idx], pred[seq_idx])
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
    return np.vstack(clouds_g).astype(np.float64), np.vstack(clouds_p).astype(np.float64)


def _ripser_diagrams(X: np.ndarray, *, maxdim: int = 1, n_perm: int = 64):
    if ripser is None or X is None or X.shape[0] < 16:
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
    return float(np.nanmean([_score_from_distance(dist), _score_from_distance(total)]))


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
    return np.asarray([np.sum((births <= t) & (t < deaths)) for t in grid], dtype=np.float64)


def _betti_curve_similarity(dgm_g: np.ndarray | None, dgm_p: np.ndarray | None) -> float:
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
    corr = _corr_score(bg, bp)
    dist = np.mean(np.abs(bp - bg)) / (np.mean(bg) + 1.0)
    return float(np.nanmean([corr, _score_from_distance(dist)]))


def _participation_ratio(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 8:
        return np.nan
    C = np.cov(X, rowvar=False)
    vals = np.linalg.eigvalsh(C)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return np.nan
    return float((np.sum(vals) ** 2) / (np.sum(vals ** 2) + EPS))


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
    kth = np.partition(D, kth=min(k - 1, D.shape[1] - 1), axis=1)[:, min(k - 1, D.shape[1] - 1)]
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


def trajectory_transition_matrix_v1(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    k_max = min(6, gt.shape[-1])
    scores = []
    for seq_idx in range(gt.shape[0]):
        Xg = gt[seq_idx]
        Xp = pred[seq_idx]
        Xg, Xp = _finite_rows(Xg, Xp)
        if Xg.shape[0] < 32:
            continue
        Xg_z, Xp_z = _standardize_with_gt(Xg, Xp)
        k = _choose_k(Xg_z, k_max=k_max)
        pca = PCA(n_components=k, svd_solver="full", random_state=0).fit(Xg_z)
        Zg = pca.transform(Xg_z)
        Zp = pca.transform(Xp_z)
        Ag, *_ = np.linalg.lstsq(Zg[:-1], Zg[1:], rcond=None)
        Ap, *_ = np.linalg.lstsq(Zp[:-1], Zp[1:], rcond=None)
        denom = np.linalg.norm(Ag) + EPS
        scores.append(_score_from_distance(np.linalg.norm(Ap - Ag) / denom))
    return {"score": float(np.nanmean(scores)) if scores else np.nan}


def trajectory_speed_turn_v2(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    k_max = min(4, gt.shape[-1])
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = _finite_rows(Xg, Xp)
    if Xg.shape[0] < 64:
        return {"score": np.nan}
    Xg_z, Xp_z = _standardize_with_gt(Xg, Xp)
    k = _choose_k(Xg_z, k_max=k_max)
    pca = PCA(n_components=k, svd_solver="full", random_state=0).fit(Xg_z)
    Zg = pca.transform(Xg_z).reshape(gt.shape[0], -1, k)
    Zp = pca.transform(Xp_z).reshape(pred.shape[0], -1, k)
    vg = np.diff(Zg, axis=1).reshape(-1, k)
    vp = np.diff(Zp, axis=1).reshape(-1, k)
    speed_g = np.linalg.norm(vg, axis=1)
    speed_p = np.linalg.norm(vp, axis=1)
    q = np.linspace(0.1, 0.9, 9)
    speed_d = np.mean(np.abs(np.quantile(speed_g, q) - np.quantile(speed_p, q)))
    turn_g = np.sum(vg[1:] * vg[:-1], axis=1) / ((np.linalg.norm(vg[1:], axis=1) * np.linalg.norm(vg[:-1], axis=1)) + EPS)
    turn_p = np.sum(vp[1:] * vp[:-1], axis=1) / ((np.linalg.norm(vp[1:], axis=1) * np.linalg.norm(vp[:-1], axis=1)) + EPS)
    turn_d = np.mean(np.abs(np.quantile(turn_g[np.isfinite(turn_g)], q) - np.quantile(turn_p[np.isfinite(turn_p)], q)))
    return {"score": float(np.clip(0.5 * _score_from_distance(speed_d) + 0.5 * (1.0 - 0.5 * np.clip(turn_d, 0.0, 2.0)), 0.0, 1.0))}


def trajectory_path_features_v3(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    k_max = min(4, gt.shape[-1])
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = _finite_rows(Xg, Xp)
    if Xg.shape[0] < 64:
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
            radius = np.mean(np.linalg.norm(z - np.mean(z, axis=0, keepdims=True), axis=1))
            persistence = disp / (path + EPS)
            speed = np.linalg.norm(v, axis=1)
            speed_lag1 = _corr_score(speed[1:], speed[:-1])
            return np.array([path, disp, radius, persistence, speed_lag1], dtype=np.float64)
        features_g.append(_seq_features(seq_g, vg))
        features_p.append(_seq_features(seq_p, vp))
    if not features_g:
        return {"score": np.nan}
    fg = np.vstack(features_g)
    fp = np.vstack(features_p)
    denom = np.nanmean(np.abs(fg), axis=0) + EPS
    distance = np.nanmean(np.abs(np.nanmean(fp, axis=0) - np.nanmean(fg, axis=0)) / denom)
    return {"score": _score_from_distance(distance)}


def trajectory_occupancy_velocity_v4(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = _finite_rows(Xg, Xp)
    if Xg.shape[0] < 64:
        return {"score": np.nan}
    Xg_z, Xp_z = _standardize_with_gt(Xg, Xp)
    k = _choose_k(Xg_z, k_max=min(3, Xg_z.shape[1]))
    pca = PCA(n_components=k, svd_solver="full", random_state=0).fit(Xg_z)
    Zg = pca.transform(Xg_z)
    Zp = pca.transform(Xp_z)
    occ_scores = []
    for dim in range(k):
        occ_scores.append(_gt_hist_score(Zg[:, dim], Zp[:, dim], bins=12))
    Zg_seq = pca.transform(gt.reshape(-1, gt.shape[-1])).reshape(gt.shape[0], gt.shape[1], k)
    Zp_seq = pca.transform(pred.reshape(-1, pred.shape[-1])).reshape(pred.shape[0], pred.shape[1], k)
    vg = np.diff(Zg_seq, axis=1).reshape(-1, k)
    vp = np.diff(Zp_seq, axis=1).reshape(-1, k)
    speed_score = _score_from_distance(_quantile_distance(np.linalg.norm(vg, axis=1), np.linalg.norm(vp, axis=1)))
    turn_g = np.sum(vg[1:] * vg[:-1], axis=1) / ((np.linalg.norm(vg[1:], axis=1) * np.linalg.norm(vg[:-1], axis=1)) + EPS)
    turn_p = np.sum(vp[1:] * vp[:-1], axis=1) / ((np.linalg.norm(vp[1:], axis=1) * np.linalg.norm(vp[:-1], axis=1)) + EPS)
    turn_score = _score_from_distance(_quantile_distance(turn_g, turn_p, qs=(0.1, 0.3, 0.5, 0.7, 0.9)))
    return {"score": float(np.nanmean(occ_scores + [speed_score, turn_score]))}


def trajectory_multistep_displacement_v5(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = _finite_rows(Xg, Xp)
    if Xg.shape[0] < 64:
        return {"score": np.nan}
    Xg_z, Xp_z = _standardize_with_gt(Xg, Xp)
    k = _choose_k(Xg_z, k_max=min(3, Xg_z.shape[1]))
    pca = PCA(n_components=k, svd_solver="full", random_state=0).fit(Xg_z)
    Zg = pca.transform(gt.reshape(-1, gt.shape[-1])).reshape(gt.shape[0], gt.shape[1], k)
    Zp = pca.transform(pred.reshape(-1, pred.shape[-1])).reshape(pred.shape[0], pred.shape[1], k)
    distances = []
    for lag in (1, 4, 12):
        dg = np.linalg.norm(Zg[:, lag:, :] - Zg[:, :-lag, :], axis=2).reshape(-1)
        dp = np.linalg.norm(Zp[:, lag:, :] - Zp[:, :-lag, :], axis=2).reshape(-1)
        distances.append(_score_from_distance(_quantile_distance(dg, dp)))
    radius_g = np.linalg.norm(Zg - np.mean(Zg.reshape(-1, k), axis=0, keepdims=True), axis=2).reshape(-1)
    radius_p = np.linalg.norm(Zp - np.mean(Zg.reshape(-1, k), axis=0, keepdims=True), axis=2).reshape(-1)
    distances.append(_score_from_distance(_quantile_distance(radius_g, radius_p)))
    return {"score": float(np.nanmean(distances))}


def bandpower_mean_spectrum_v1(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
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
    corr = _corr_score(np.log(mean_g + EPS), np.log(mean_p + EPS))
    l1 = 1.0 - 0.5 * np.sum(np.abs(mean_g - mean_p))
    return {"score": float(np.clip(0.5 * corr + 0.5 * l1, 0.0, 1.0))}


def bandpower_band_fraction_v2(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
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
    bands = [slice(0, splits[0]), slice(splits[0], splits[1]), slice(splits[1], mean_g.size)]
    frac_g = np.array([np.sum(mean_g[band]) for band in bands], dtype=np.float64)
    frac_p = np.array([np.sum(mean_p[band]) for band in bands], dtype=np.float64)
    score = 1.0 - 0.5 * np.sum(np.abs(frac_g - frac_p))
    return {"score": float(np.clip(score, 0.0, 1.0))}


def bandpower_shape_features_v3(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    feats_g = []
    feats_p = []
    for xg, xp in zip(_iter_region_series(gt), _iter_region_series(pred)):
        freqs_g, psg = _welch_relative_psd(xg)
        freqs_p, psp = _welch_relative_psd(xp)
        if psg.size == 0 or psp.size == 0 or psg.shape != psp.shape:
            continue
        def _features(freqs, psd):
            centroid = np.sum(freqs * psd)
            spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd))
            low_high = np.sum(psd[freqs <= np.median(freqs)]) / (np.sum(psd[freqs > np.median(freqs)]) + EPS)
            slope = np.polyfit(np.log(freqs + EPS), np.log(psd + EPS), deg=1)[0]
            return np.array([centroid, spread, low_high, slope], dtype=np.float64)
        feats_g.append(_features(freqs_g, psg))
        feats_p.append(_features(freqs_p, psp))
    if not feats_g:
        return {"score": np.nan}
    fg = np.nanmean(np.vstack(feats_g), axis=0)
    fp = np.nanmean(np.vstack(feats_p), axis=0)
    distance = np.mean(np.abs(fp - fg) / (np.abs(fg) + 1e-3))
    return {"score": _score_from_distance(distance)}


def bandpower_landmark_features_v4(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    feat_g = []
    feat_p = []
    for xg, xp in zip(_iter_region_series(gt), _iter_region_series(pred)):
        freqs, psg = _welch_relative_psd(xg)
        _, psp = _welch_relative_psd(xp)
        if psg.size == 0 or psg.shape != psp.shape:
            continue
        csg = np.cumsum(psg)
        csp = np.cumsum(psp)
        landmarks_g = [freqs[np.searchsorted(csg, q)] for q in (0.25, 0.5, 0.75, 0.9)]
        landmarks_p = [freqs[np.searchsorted(csp, q)] for q in (0.25, 0.5, 0.75, 0.9)]
        slope_g = np.polyfit(np.log(freqs + EPS), np.log(psg + EPS), deg=1)[0]
        slope_p = np.polyfit(np.log(freqs + EPS), np.log(psp + EPS), deg=1)[0]
        ent_g = -np.sum(psg * np.log(psg + EPS))
        ent_p = -np.sum(psp * np.log(psp + EPS))
        feat_g.append(np.array([*landmarks_g, slope_g, ent_g], dtype=np.float64))
        feat_p.append(np.array([*landmarks_p, slope_p, ent_p], dtype=np.float64))
    if not feat_g:
        return {"score": np.nan}
    fg = np.vstack(feat_g)
    fp = np.vstack(feat_p)
    scale = np.nanmean(np.abs(fg), axis=0) + 0.05
    distance = np.nanmean(np.abs(np.nanmean(fp, axis=0) - np.nanmean(fg, axis=0)) / scale)
    return {"score": _score_from_distance(distance)}


def bandpower_region_distribution_v5(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    region_feat_g = []
    region_feat_p = []
    for xg, xp in zip(_iter_region_series(gt), _iter_region_series(pred)):
        freqs, psg = _welch_relative_psd(xg)
        _, psp = _welch_relative_psd(xp)
        if psg.size == 0 or psg.shape != psp.shape:
            continue
        csg = np.cumsum(psg)
        split1 = freqs[min(np.searchsorted(csg, 1.0 / 3.0), freqs.size - 1)]
        split2 = freqs[min(np.searchsorted(csg, 2.0 / 3.0), freqs.size - 1)]
        mask1 = freqs <= split1
        mask2 = (freqs > split1) & (freqs <= split2)
        mask3 = freqs > split2
        def _feat(psd):
            return np.array([np.sum(psd[mask1]), np.sum(psd[mask2]), np.sum(psd[mask3])], dtype=np.float64)
        region_feat_g.append(_feat(psg))
        region_feat_p.append(_feat(psp))
    if not region_feat_g:
        return {"score": np.nan}
    fg = np.vstack(region_feat_g)
    fp = np.vstack(region_feat_p)
    pieces = []
    for dim in range(fg.shape[1]):
        if fg.shape[0] < 8:
            mean_g = float(np.mean(fg[:, dim]))
            mean_p = float(np.mean(fp[:, dim]))
            std_g = float(np.std(fg[:, dim]))
            std_p = float(np.std(fp[:, dim]))
            dist = (abs(mean_p - mean_g) / (abs(mean_g) + 0.05)) + (abs(std_p - std_g) / (std_g + 0.05))
            pieces.append(_score_from_distance(dist))
        else:
            pieces.append(_score_from_distance(_quantile_distance(fg[:, dim], fp[:, dim], qs=(0.1, 0.3, 0.5, 0.7, 0.9))))
    return {"score": float(np.nanmean(pieces))}


def pca_evr_profile_v1(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = _finite_rows(Xg, Xp)
    if Xg.shape[0] < 64:
        return {"score": np.nan}
    Xg_z, Xp_z = _standardize_with_gt(Xg, Xp)
    k_fit = min(8, Xg_z.shape[1], Xg_z.shape[0] - 1)
    pca_g = PCA(n_components=k_fit, svd_solver="full", random_state=0).fit(Xg_z)
    pca_p = PCA(n_components=k_fit, svd_solver="full", random_state=0).fit(Xp_z)
    evr_g = pca_g.explained_variance_ratio_
    evr_p = pca_p.explained_variance_ratio_
    score = 1.0 - 0.5 * np.mean(np.abs(evr_g - evr_p))
    return {"score": float(np.clip(score, 0.0, 1.0))}


def pca_subspace_overlap_v2(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = _finite_rows(Xg, Xp)
    if Xg.shape[0] < 64:
        return {"score": np.nan}
    Xg_z, Xp_z = _standardize_with_gt(Xg, Xp)
    k_fit = min(8, Xg_z.shape[1], Xg_z.shape[0] - 1)
    pca_g = PCA(n_components=k_fit, svd_solver="full", random_state=0).fit(Xg_z)
    pca_p = PCA(n_components=k_fit, svd_solver="full", random_state=0).fit(Xp_z)
    overlap = np.linalg.svd(pca_g.components_ @ pca_p.components_.T, compute_uv=False)
    weights = pca_g.explained_variance_ratio_[: overlap.size]
    weights = weights / (np.sum(weights) + EPS)
    score = float(np.sum(weights * (overlap[: weights.size] ** 2)))
    return {"score": float(np.clip(score, 0.0, 1.0))}


def pca_latent_covariance_v3(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = _finite_rows(Xg, Xp)
    if Xg.shape[0] < 64:
        return {"score": np.nan}
    Xg_z, Xp_z = _standardize_with_gt(Xg, Xp)
    k = _choose_k(Xg_z, k_max=min(8, Xg_z.shape[1]))
    pca_g = PCA(n_components=k, svd_solver="full", random_state=0).fit(Xg_z)
    Zg = pca_g.transform(Xg_z)
    Zp = pca_g.transform(Xp_z)
    Cg = np.cov(Zg, rowvar=False)
    Cp = np.cov(Zp, rowvar=False)
    distance = np.linalg.norm(Cp - Cg, ord="fro") / (np.linalg.norm(Cg, ord="fro") + EPS)
    return {"score": _score_from_distance(distance)}


def pca_reconstruction_transfer_v4(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = _finite_rows(Xg, Xp)
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
    var_g = np.mean(Xg_z ** 2)
    var_p = np.mean(Xp_z ** 2)
    frac_g = 1.0 - (err_g / (var_g + EPS))
    frac_p = 1.0 - (err_p / (var_p + EPS))
    var_score = _score_from_distance(np.mean(np.abs(np.var(Zp, axis=0) - np.var(Zg, axis=0)) / (np.var(Zg, axis=0) + 0.05)))
    rec_score = float(np.clip(frac_p / (frac_g + EPS), 0.0, 1.0))
    return {"score": float(np.clip(0.5 * rec_score + 0.5 * var_score, 0.0, 1.0))}


def pca_variance_landmarks_v5(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    Xg = gt.reshape(-1, gt.shape[-1])
    Xp = pred.reshape(-1, pred.shape[-1])
    Xg, Xp = _finite_rows(Xg, Xp)
    if Xg.shape[0] < 64:
        return {"score": np.nan}
    Xg_z, Xp_z = _standardize_with_gt(Xg, Xp)
    k_fit = min(8, Xg_z.shape[1], Xg_z.shape[0] - 1)
    pca_g = PCA(n_components=k_fit, svd_solver="full", random_state=0).fit(Xg_z)
    pca_p = PCA(n_components=k_fit, svd_solver="full", random_state=0).fit(Xp_z)
    cum_g = np.cumsum(pca_g.explained_variance_ratio_)
    cum_p = np.cumsum(pca_p.explained_variance_ratio_)
    idx = np.array(sorted(set([0, 1, min(3, k_fit - 1), k_fit - 1])), dtype=int)
    lm_g = cum_g[idx]
    lm_p = cum_p[idx]
    pr_g = (np.sum(pca_g.explained_variance_) ** 2) / (np.sum(pca_g.explained_variance_ ** 2) + EPS)
    pr_p = (np.sum(pca_p.explained_variance_) ** 2) / (np.sum(pca_p.explained_variance_ ** 2) + EPS)
    distance = np.mean(np.abs(lm_p - lm_g)) + abs(pr_p - pr_g) / (pr_g + 0.1)
    return {"score": _score_from_distance(distance)}


def pca_reconstruction_product_v6(gt: Array3D, pred: Array3D) -> dict:
    base = pca_reconstruction_transfer_v4(gt, pred).get("score", np.nan)
    if not np.isfinite(base):
        return {"score": np.nan}
    return {"score": float(np.clip(base ** 2, 0.0, 1.0))}


def autocorr_curve_match_v1(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    max_lag = min(40, gt.shape[1] // 8)
    region_scores = []
    for region in range(gt.shape[-1]):
        ac_g = np.nanmean(np.vstack([_autocorr_1d(gt[s, :, region], max_lag) for s in range(gt.shape[0])]), axis=0)
        ac_p = np.nanmean(np.vstack([_autocorr_1d(pred[s, :, region], max_lag) for s in range(pred.shape[0])]), axis=0)
        region_scores.append(_corr_score(ac_g, ac_p))
    return {"score": float(np.nanmean(region_scores)) if region_scores else np.nan}


def autocorr_feature_match_v2(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    max_lag = min(60, gt.shape[1] // 6)
    feats_g = []
    feats_p = []
    for region in range(gt.shape[-1]):
        ac_g = np.nanmean(np.vstack([_autocorr_1d(gt[s, :, region], max_lag) for s in range(gt.shape[0])]), axis=0)
        ac_p = np.nanmean(np.vstack([_autocorr_1d(pred[s, :, region], max_lag) for s in range(pred.shape[0])]), axis=0)
        if not np.isfinite(ac_g).any() or not np.isfinite(ac_p).any():
            continue
        def _features(ac):
            pos = np.maximum(ac, 0.0)
            zero_cross = np.argmax(ac <= 0.0)
            zero_cross = zero_cross / max_lag if np.any(ac <= 0.0) else 1.0
            return np.array([ac[0], ac[min(4, ac.size - 1)], np.sum(pos), zero_cross], dtype=np.float64)
        feats_g.append(_features(ac_g))
        feats_p.append(_features(ac_p))
    if not feats_g:
        return {"score": np.nan}
    fg = np.nanmean(np.vstack(feats_g), axis=0)
    fp = np.nanmean(np.vstack(feats_p), axis=0)
    distance = np.mean(np.abs(fp - fg) / (np.abs(fg) + 0.05))
    return {"score": _score_from_distance(distance)}


def autocorr_ar_match_v3(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    feats_g = []
    feats_p = []
    for region in range(gt.shape[-1]):
        xg = gt[:, :, region].reshape(-1)
        xp = pred[:, :, region].reshape(-1)
        mask = np.isfinite(xg) & np.isfinite(xp)
        xg = xg[mask]
        xp = xp[mask]
        if xg.size < 64 or xp.size < 64:
            continue
        def _ar1_features(x):
            x0 = x[:-1] - np.mean(x[:-1])
            x1 = x[1:] - np.mean(x[1:])
            phi = np.dot(x0, x1) / (np.dot(x0, x0) + EPS)
            resid = x1 - phi * x0
            noise_ratio = np.var(resid) / (np.var(x1) + EPS)
            return np.array([phi, noise_ratio], dtype=np.float64)
        feats_g.append(_ar1_features(xg))
        feats_p.append(_ar1_features(xp))
    if not feats_g:
        return {"score": np.nan}
    fg = np.nanmean(np.vstack(feats_g), axis=0)
    fp = np.nanmean(np.vstack(feats_p), axis=0)
    distance = np.mean(np.abs(fp - fg) / (np.abs(fg) + 0.05))
    return {"score": _score_from_distance(distance)}


def autocorr_weighted_rmse_v4(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    max_lag = min(48, gt.shape[1] // 6)
    weights = 1.0 / np.sqrt(np.arange(1, max_lag + 1, dtype=np.float64))
    region_scores = []
    for region in range(gt.shape[-1]):
        ac_g = np.nanmean(np.vstack([_autocorr_1d(gt[s, :, region], max_lag) for s in range(gt.shape[0])]), axis=0)
        ac_p = np.nanmean(np.vstack([_autocorr_1d(pred[s, :, region], max_lag) for s in range(pred.shape[0])]), axis=0)
        if not np.isfinite(ac_g).any() or not np.isfinite(ac_p).any():
            continue
        diff = (ac_p - ac_g) * weights
        denom = np.sqrt(np.mean((ac_g * weights) ** 2)) + 0.05
        region_scores.append(_score_from_distance(np.sqrt(np.mean(diff ** 2)) / denom))
    return {"score": float(np.nanmean(region_scores)) if region_scores else np.nan}


def autocorr_tau_distribution_v5(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    max_lag = min(40, gt.shape[1] // 8)
    tau_g = []
    tau_p = []
    lag1_g = []
    lag1_p = []
    for seq in range(gt.shape[0]):
        for region in range(gt.shape[-1]):
            ac_g = _autocorr_1d(gt[seq, :, region], max_lag)
            ac_p = _autocorr_1d(pred[seq, :, region], max_lag)
            if not np.isfinite(ac_g).any() or not np.isfinite(ac_p).any():
                continue
            pos_g = np.maximum(ac_g, 0.0)
            pos_p = np.maximum(ac_p, 0.0)
            tau_g.append(np.sum(pos_g))
            tau_p.append(np.sum(pos_p))
            lag1_g.append(ac_g[0])
            lag1_p.append(ac_p[0])
    score_tau = _score_from_distance(_quantile_distance(np.asarray(tau_g), np.asarray(tau_p)))
    score_lag1 = _score_from_distance(_quantile_distance(np.asarray(lag1_g), np.asarray(lag1_p)))
    return {"score": float(np.nanmean([score_tau, score_lag1]))}


def autocorr_weighted_rmse_power_v6(gt: Array3D, pred: Array3D) -> dict:
    base = autocorr_weighted_rmse_v4(gt, pred).get("score", np.nan)
    if not np.isfinite(base):
        return {"score": np.nan}
    return {"score": float(np.clip(base ** 2, 0.0, 1.0))}


def crosscorr_curve_match_v1(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    max_lag = min(20, gt.shape[1] // 10)
    pair_scores = []
    for i in range(gt.shape[-1]):
        for j in range(i + 1, gt.shape[-1]):
            cc_g = np.nanmean(np.vstack([_crosscorr_1d(gt[s, :, i], gt[s, :, j], max_lag) for s in range(gt.shape[0])]), axis=0)
            cc_p = np.nanmean(np.vstack([_crosscorr_1d(pred[s, :, i], pred[s, :, j], max_lag) for s in range(pred.shape[0])]), axis=0)
            pair_scores.append(_corr_score(cc_g, cc_p))
    return {"score": float(np.nanmean(pair_scores)) if pair_scores else np.nan}


def crosscorr_peaklag_amp_v2(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    max_lag = min(20, gt.shape[1] // 10)
    lag_d = []
    amp_d = []
    for i in range(gt.shape[-1]):
        for j in range(i + 1, gt.shape[-1]):
            cc_g = np.nanmean(np.vstack([_crosscorr_1d(gt[s, :, i], gt[s, :, j], max_lag) for s in range(gt.shape[0])]), axis=0)
            cc_p = np.nanmean(np.vstack([_crosscorr_1d(pred[s, :, i], pred[s, :, j], max_lag) for s in range(pred.shape[0])]), axis=0)
            if not np.isfinite(cc_g).any() or not np.isfinite(cc_p).any():
                continue
            lag_g = int(np.nanargmax(np.abs(cc_g))) - max_lag
            lag_p = int(np.nanargmax(np.abs(cc_p))) - max_lag
            amp_g = cc_g[lag_g + max_lag]
            amp_p = cc_p[lag_p + max_lag]
            lag_d.append(abs(lag_p - lag_g) / max_lag)
            amp_d.append(abs(amp_p - amp_g))
    if not lag_d:
        return {"score": np.nan}
    lag_score = 1.0 - np.clip(np.nanmean(lag_d), 0.0, 1.0)
    amp_score = _score_from_distance(np.nanmean(amp_d))
    return {"score": float(np.clip(0.5 * lag_score + 0.5 * amp_score, 0.0, 1.0))}


def crosscorr_leadlag_matrix_v3(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    max_lag = min(20, gt.shape[1] // 10)
    lead_g = []
    lead_p = []
    for i in range(gt.shape[-1]):
        for j in range(i + 1, gt.shape[-1]):
            cc_g = np.nanmean(np.vstack([_crosscorr_1d(gt[s, :, i], gt[s, :, j], max_lag) for s in range(gt.shape[0])]), axis=0)
            cc_p = np.nanmean(np.vstack([_crosscorr_1d(pred[s, :, i], pred[s, :, j], max_lag) for s in range(pred.shape[0])]), axis=0)
            if not np.isfinite(cc_g).any() or not np.isfinite(cc_p).any():
                continue
            lags = np.arange(-max_lag, max_lag + 1, dtype=np.float64)
            lead_g.append(np.sum(lags * np.abs(cc_g)) / (np.sum(np.abs(cc_g)) + EPS))
            lead_p.append(np.sum(lags * np.abs(cc_p)) / (np.sum(np.abs(cc_p)) + EPS))
    if not lead_g:
        return {"score": np.nan}
    corr = _corr_score(np.asarray(lead_g), np.asarray(lead_p))
    dist = _score_from_distance(np.mean(np.abs(np.asarray(lead_p) - np.asarray(lead_g)) / (max_lag + EPS)))
    return {"score": float(np.clip(0.5 * corr + 0.5 * dist, 0.0, 1.0))}


def crosscorr_lagged_matrix_v4(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    Cg0 = _safe_corrcoef(gt.reshape(-1, gt.shape[-1]))
    Cp0 = _safe_corrcoef(pred.reshape(-1, pred.shape[-1]))
    Cg1 = _lagged_corr_matrix(gt, lag=1)
    Cp1 = _lagged_corr_matrix(pred, lag=1)
    if Cg0 is None or Cp0 is None or Cg1 is None or Cp1 is None:
        return {"score": np.nan}
    e0g = _upper_triangle(Cg0)
    e0p = _upper_triangle(Cp0)
    e1g = Cg1.reshape(-1)
    e1p = Cp1.reshape(-1)
    score0 = 0.5 * _corr_score(e0g, e0p) + 0.5 * _score_from_distance(np.mean(np.abs(e0p - e0g)) / (_robust_scale(e0g) + EPS))
    score1 = 0.5 * _corr_score(e1g, e1p) + 0.5 * _score_from_distance(np.mean(np.abs(e1p - e1g)) / (_robust_scale(e1g) + EPS))
    return {"score": float(np.nanmean([score0, score1]))}


def crosscorr_topedge_profiles_v5(gt: Array3D, pred: Array3D) -> dict:
    gt, pred = _align_arrays(gt, pred)
    Cg0 = _safe_corrcoef(gt.reshape(-1, gt.shape[-1]))
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
    max_lag = min(16, gt.shape[1] // 12)
    scores = []
    for i, j in pairs:
        cc_g = np.nanmean(np.vstack([_crosscorr_1d(gt[s, :, i], gt[s, :, j], max_lag) for s in range(gt.shape[0])]), axis=0)
        cc_p = np.nanmean(np.vstack([_crosscorr_1d(pred[s, :, i], pred[s, :, j], max_lag) for s in range(pred.shape[0])]), axis=0)
        if not np.isfinite(cc_g).any() or not np.isfinite(cc_p).any():
            continue
        score_curve = _corr_score(cc_g, cc_p)
        lag_g = int(np.nanargmax(np.abs(cc_g))) - max_lag
        lag_p = int(np.nanargmax(np.abs(cc_p))) - max_lag
        lag_score = 1.0 - (abs(lag_p - lag_g) / max_lag)
        amp_g = cc_g[lag_g + max_lag]
        amp_p = cc_p[lag_p + max_lag]
        amp_score = _score_from_distance(abs(amp_p - amp_g) / (_robust_scale(cc_g) + EPS))
        scores.append(np.nanmean([score_curve, lag_score, amp_score]))
    return {"score": float(np.nanmean(scores)) if scores else np.nan}


def manifold_ph_lifetime_profile_v1(gt: Array3D, pred: Array3D) -> dict:
    Zg, Zp = _pooled_latent_clouds(gt, pred, k_max=3, n_points=96, seed=0)
    if Zg is None or Zp is None:
        return {"score": np.nan}
    dgms_g = _ripser_diagrams(Zg, maxdim=1, n_perm=64)
    dgms_p = _ripser_diagrams(Zp, maxdim=1, n_perm=64)
    if dgms_g is None or dgms_p is None:
        return {"score": np.nan}
    score_h0 = _lifetime_similarity(_lifetimes(dgms_g[0]), _lifetimes(dgms_p[0]))
    score_h1 = _lifetime_similarity(_lifetimes(dgms_g[1]), _lifetimes(dgms_p[1]))
    return {"score": float(np.nanmean([score_h0, score_h1]))}


def manifold_ph_betti_curve_v2(gt: Array3D, pred: Array3D) -> dict:
    Zg, Zp = _pooled_latent_clouds(gt, pred, k_max=3, n_points=96, seed=1)
    if Zg is None or Zp is None:
        return {"score": np.nan}
    dgms_g = _ripser_diagrams(Zg, maxdim=1, n_perm=64)
    dgms_p = _ripser_diagrams(Zp, maxdim=1, n_perm=64)
    if dgms_g is None or dgms_p is None:
        return {"score": np.nan}
    score_h0 = _betti_curve_similarity(dgms_g[0], dgms_p[0])
    score_h1 = _betti_curve_similarity(dgms_g[1], dgms_p[1])
    return {"score": float(np.nanmean([score_h0, score_h1]))}


def manifold_ph_hybrid_v3(gt: Array3D, pred: Array3D) -> dict:
    score_life = manifold_ph_lifetime_profile_v1(gt, pred).get("score", np.nan)
    score_betti = manifold_ph_betti_curve_v2(gt, pred).get("score", np.nan)
    if not np.isfinite(score_life) or not np.isfinite(score_betti):
        return {"score": np.nan}
    return {"score": float(np.sqrt(score_life * score_betti))}


def manifold_ph_pr_hybrid_v4(gt: Array3D, pred: Array3D) -> dict:
    Zg, Zp = _pooled_latent_clouds(gt, pred, k_max=3, n_points=128, seed=2)
    if Zg is None or Zp is None:
        return {"score": np.nan}
    score_life = manifold_ph_lifetime_profile_v1(gt, pred).get("score", np.nan)
    pr_g = _participation_ratio(Zg)
    pr_p = _participation_ratio(Zp)
    score_pr = _score_from_distance(abs(pr_p - pr_g) / (pr_g + 0.1))
    if not np.isfinite(score_life) or not np.isfinite(score_pr):
        return {"score": np.nan}
    return {"score": float(np.sqrt(score_life * score_pr))}


def manifold_ph_knn_profile_v5(gt: Array3D, pred: Array3D) -> dict:
    Zg, Zp = _pooled_latent_clouds(gt, pred, k_max=3, n_points=128, seed=3)
    if Zg is None or Zp is None:
        return {"score": np.nan}
    score_life = manifold_ph_lifetime_profile_v1(gt, pred).get("score", np.nan)
    knn_g = _knn_distance_profile(Zg, k=5)
    knn_p = _knn_distance_profile(Zp, k=5)
    score_knn = _score_from_distance(_quantile_distance(knn_g, knn_p, qs=(0.1, 0.25, 0.5, 0.75, 0.9)))
    if not np.isfinite(score_life) or not np.isfinite(score_knn):
        return {"score": np.nan}
    return {"score": float(np.sqrt(score_life * score_knn))}


def manifold_ph_density_hybrid_v6(gt: Array3D, pred: Array3D) -> dict:
    Zg, Zp = _pooled_latent_clouds(gt, pred, k_max=3, n_points=128, seed=4)
    if Zg is None or Zp is None:
        return {"score": np.nan}
    score_life = manifold_ph_lifetime_profile_v1(gt, pred).get("score", np.nan)
    score_betti = manifold_ph_betti_curve_v2(gt, pred).get("score", np.nan)
    rad_g = _radial_profile(Zg)
    rad_p = _radial_profile(Zp)
    knn_g = _knn_distance_profile(Zg, k=5)
    knn_p = _knn_distance_profile(Zp, k=5)
    score_rad = _score_from_distance(_quantile_distance(rad_g, rad_p, qs=(0.1, 0.25, 0.5, 0.75, 0.9)))
    score_knn = _score_from_distance(_quantile_distance(knn_g, knn_p, qs=(0.1, 0.25, 0.5, 0.75, 0.9)))
    parts = [score_life, score_betti, score_rad, score_knn]
    parts = [float(x) for x in parts if np.isfinite(x)]
    if not parts:
        return {"score": np.nan}
    return {"score": float(np.exp(np.mean(np.log(np.clip(parts, 1e-6, 1.0)))))}


def manifold_ph_stratified_lifetime_v7(gt: Array3D, pred: Array3D) -> dict:
    Zg, Zp = _stratified_latent_clouds(gt, pred, k_max=3, points_per_seq=2, max_sequences=48, seed=5)
    if Zg is None or Zp is None:
        return {"score": np.nan}
    dgms_g = _ripser_diagrams(Zg, maxdim=1, n_perm=64)
    dgms_p = _ripser_diagrams(Zp, maxdim=1, n_perm=64)
    if dgms_g is None or dgms_p is None:
        return {"score": np.nan}
    score_h0 = _lifetime_similarity(_lifetimes(dgms_g[0]), _lifetimes(dgms_p[0]))
    score_h1 = _lifetime_similarity(_lifetimes(dgms_g[1]), _lifetimes(dgms_p[1]))
    return {"score": float(np.nanmean([score_h0, score_h1]))}


def manifold_ph_stratified_hybrid_v8(gt: Array3D, pred: Array3D) -> dict:
    Zg, Zp = _stratified_latent_clouds(gt, pred, k_max=3, points_per_seq=2, max_sequences=48, seed=6)
    if Zg is None or Zp is None:
        return {"score": np.nan}
    dgms_g = _ripser_diagrams(Zg, maxdim=1, n_perm=64)
    dgms_p = _ripser_diagrams(Zp, maxdim=1, n_perm=64)
    if dgms_g is None or dgms_p is None:
        return {"score": np.nan}
    parts = [
        _lifetime_similarity(_lifetimes(dgms_g[0]), _lifetimes(dgms_p[0])),
        _lifetime_similarity(_lifetimes(dgms_g[1]), _lifetimes(dgms_p[1])),
        _betti_curve_similarity(dgms_g[0], dgms_p[0]),
        _betti_curve_similarity(dgms_g[1], dgms_p[1]),
    ]
    parts = [float(x) for x in parts if np.isfinite(x)]
    if not parts:
        return {"score": np.nan}
    return {"score": float(np.exp(np.mean(np.log(np.clip(parts, 1e-6, 1.0)))))}


def manifold_ph_stratified_density_v9(gt: Array3D, pred: Array3D) -> dict:
    Zg, Zp = _stratified_latent_clouds(gt, pred, k_max=3, points_per_seq=2, max_sequences=48, seed=7)
    if Zg is None or Zp is None:
        return {"score": np.nan}
    dgms_g = _ripser_diagrams(Zg, maxdim=1, n_perm=64)
    dgms_p = _ripser_diagrams(Zp, maxdim=1, n_perm=64)
    if dgms_g is None or dgms_p is None:
        return {"score": np.nan}
    parts = [
        _lifetime_similarity(_lifetimes(dgms_g[0]), _lifetimes(dgms_p[0])),
        _lifetime_similarity(_lifetimes(dgms_g[1]), _lifetimes(dgms_p[1])),
        _betti_curve_similarity(dgms_g[0], dgms_p[0]),
        _betti_curve_similarity(dgms_g[1], dgms_p[1]),
        _score_from_distance(_quantile_distance(_radial_profile(Zg), _radial_profile(Zp), qs=(0.1, 0.25, 0.5, 0.75, 0.9))),
        _score_from_distance(_quantile_distance(_knn_distance_profile(Zg, k=4), _knn_distance_profile(Zp, k=4), qs=(0.1, 0.25, 0.5, 0.75, 0.9))),
    ]
    parts = [float(x) for x in parts if np.isfinite(x)]
    if not parts:
        return {"score": np.nan}
    return {"score": float(np.exp(np.mean(np.log(np.clip(parts, 1e-6, 1.0)))))}


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


CANDIDATE_REGISTRY: dict[str, tuple[CandidateSpec, ...]] = {
    "trajectory": (
        CandidateSpec("trajectory", "trajectory_transition_matrix_v1", "GT-PCA latent transition matrix similarity.", trajectory_transition_matrix_v1),
        CandidateSpec("trajectory", "trajectory_speed_turn_v2", "Latent speed and turning-statistic agreement.", trajectory_speed_turn_v2),
        CandidateSpec("trajectory", "trajectory_path_features_v3", "Sequence-level latent path feature agreement.", trajectory_path_features_v3),
        CandidateSpec("trajectory", "trajectory_occupancy_velocity_v4", "Pooled latent occupancy and velocity-distribution agreement.", trajectory_occupancy_velocity_v4),
        CandidateSpec("trajectory", "trajectory_multistep_displacement_v5", "Multistep latent displacement and radius agreement.", trajectory_multistep_displacement_v5),
    ),
    "bandpower": (
        CandidateSpec("bandpower", "bandpower_mean_spectrum_v1", "Mean relative log-spectrum shape agreement.", bandpower_mean_spectrum_v1),
        CandidateSpec("bandpower", "bandpower_band_fraction_v2", "Data-driven band-fraction agreement.", bandpower_band_fraction_v2),
        CandidateSpec("bandpower", "bandpower_shape_features_v3", "Spectral centroid, spread, ratio, and slope agreement.", bandpower_shape_features_v3),
        CandidateSpec("bandpower", "bandpower_landmark_features_v4", "Spectral landmark, slope, and entropy agreement.", bandpower_landmark_features_v4),
        CandidateSpec("bandpower", "bandpower_region_distribution_v5", "Across-region spectral-fraction distribution agreement.", bandpower_region_distribution_v5),
    ),
    "pca": (
        CandidateSpec("pca", "pca_evr_profile_v1", "Explained-variance profile similarity.", pca_evr_profile_v1),
        CandidateSpec("pca", "pca_subspace_overlap_v2", "Weighted principal-angle subspace overlap.", pca_subspace_overlap_v2),
        CandidateSpec("pca", "pca_latent_covariance_v3", "GT-basis latent covariance agreement.", pca_latent_covariance_v3),
        CandidateSpec("pca", "pca_reconstruction_transfer_v4", "GT-basis reconstruction transfer and latent-variance agreement.", pca_reconstruction_transfer_v4),
        CandidateSpec("pca", "pca_variance_landmarks_v5", "Cumulative EVR landmark and participation-ratio agreement.", pca_variance_landmarks_v5),
        CandidateSpec("pca", "pca_reconstruction_product_v6", "Squared GT-basis reconstruction-transfer score for stronger dynamic range.", pca_reconstruction_product_v6),
    ),
    "autocorr": (
        CandidateSpec("autocorr", "autocorr_curve_match_v1", "Region-averaged autocorrelation curve similarity.", autocorr_curve_match_v1),
        CandidateSpec("autocorr", "autocorr_feature_match_v2", "Autocorrelation feature-vector agreement.", autocorr_feature_match_v2),
        CandidateSpec("autocorr", "autocorr_ar_match_v3", "AR(1) coefficient and noise-ratio agreement.", autocorr_ar_match_v3),
        CandidateSpec("autocorr", "autocorr_weighted_rmse_v4", "Early-lag weighted autocorrelation RMSE agreement.", autocorr_weighted_rmse_v4),
        CandidateSpec("autocorr", "autocorr_tau_distribution_v5", "Integrated-timescale and lag-1 distribution agreement.", autocorr_tau_distribution_v5),
        CandidateSpec("autocorr", "autocorr_weighted_rmse_power_v6", "Squared weighted-autocorrelation score to emphasize moderate mismatches.", autocorr_weighted_rmse_power_v6),
    ),
    "crosscorr": (
        CandidateSpec("crosscorr", "crosscorr_curve_match_v1", "Pairwise cross-correlation curve similarity.", crosscorr_curve_match_v1),
        CandidateSpec("crosscorr", "crosscorr_peaklag_amp_v2", "Peak-lag and peak-amplitude agreement.", crosscorr_peaklag_amp_v2),
        CandidateSpec("crosscorr", "crosscorr_leadlag_matrix_v3", "Lead-lag centroid matrix agreement.", crosscorr_leadlag_matrix_v3),
        CandidateSpec("crosscorr", "crosscorr_lagged_matrix_v4", "Lag-0 and lag-1 population correlation-matrix agreement.", crosscorr_lagged_matrix_v4),
        CandidateSpec("crosscorr", "crosscorr_topedge_profiles_v5", "Strong-edge cross-correlation profile agreement.", crosscorr_topedge_profiles_v5),
    ),
    "manifold": (
        CandidateSpec("manifold", "manifold_ph_lifetime_profile_v1", "Persistent-homology lifetime profile agreement in a pooled GT latent space.", manifold_ph_lifetime_profile_v1),
        CandidateSpec("manifold", "manifold_ph_betti_curve_v2", "Persistent-homology Betti-curve agreement in a pooled GT latent space.", manifold_ph_betti_curve_v2),
        CandidateSpec("manifold", "manifold_ph_hybrid_v3", "Hybrid persistent-homology lifetime and Betti-curve agreement.", manifold_ph_hybrid_v3),
        CandidateSpec("manifold", "manifold_ph_pr_hybrid_v4", "Persistent-homology lifetime score stabilized by participation-ratio agreement.", manifold_ph_pr_hybrid_v4),
        CandidateSpec("manifold", "manifold_ph_knn_profile_v5", "Persistent-homology lifetime score stabilized by local-neighborhood distance profiles.", manifold_ph_knn_profile_v5),
        CandidateSpec("manifold", "manifold_ph_density_hybrid_v6", "Persistent-homology, Betti, radial-shell, and local-density agreement.", manifold_ph_density_hybrid_v6),
        CandidateSpec("manifold", "manifold_ph_stratified_lifetime_v7", "Stratified per-sequence persistent-homology lifetime agreement.", manifold_ph_stratified_lifetime_v7),
        CandidateSpec("manifold", "manifold_ph_stratified_hybrid_v8", "Stratified per-sequence persistent-homology lifetime and Betti agreement.", manifold_ph_stratified_hybrid_v8),
        CandidateSpec("manifold", "manifold_ph_stratified_density_v9", "Stratified persistent topology with radial-shell and local-density agreement.", manifold_ph_stratified_density_v9),
    ),
}


CORRUPTION_REGISTRY: dict[str, CorruptionSpec] = {
    "trajectory": CorruptionSpec(
        "trajectory",
        "time_shuffle_blend",
        "Blend each trace with a within-sequence temporal shuffle.",
        (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        corruption_time_shuffle_blend,
    ),
    "bandpower": CorruptionSpec(
        "bandpower",
        "time_shuffle_blend",
        "Blend each trace with a within-sequence temporal shuffle.",
        (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        corruption_time_shuffle_blend,
    ),
    "pca": CorruptionSpec(
        "pca",
        "region_permute_blend",
        "Blend neural channels with a global region permutation.",
        (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        corruption_region_permute_blend,
    ),
    "autocorr": CorruptionSpec(
        "autocorr",
        "time_shuffle_blend",
        "Blend each trace with a within-sequence temporal shuffle.",
        (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        corruption_time_shuffle_blend,
    ),
    "crosscorr": CorruptionSpec(
        "crosscorr",
        "region_shift_desync",
        "Apply region-specific circular lag shifts to destroy lead-lag structure.",
        (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        corruption_region_shift_desync,
    ),
    "manifold": CorruptionSpec(
        "manifold",
        "region_permute_blend",
        "Blend neural channels with a global region permutation to distort latent topology.",
        (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        corruption_region_permute_blend,
    ),
}
