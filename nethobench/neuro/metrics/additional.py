from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.linalg import subspace_angles
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import LedoitWolf
from sklearn.feature_selection import mutual_info_regression
from nethobench.utils.calculation import _align_arrays, _rmse_similarity, weighted_mean_available, _corr_score
EPS = 1e-9



def _pooled_rows(arr: np.ndarray) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1])
    flat = flat[np.isfinite(flat).all(axis=1)]
    return flat


def _standardize_columns(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    med = np.nanmedian(data, axis=0)
    q25 = np.nanquantile(data, 0.25, axis=0)
    q75 = np.nanquantile(data, 0.75, axis=0)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, np.nanstd(data, axis=0))
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0)
    return (data - med) / scale


def _fit_reference_scaler(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=np.float64)
    med = np.nanmedian(data, axis=0)
    q25 = np.nanquantile(data, 0.25, axis=0)
    q75 = np.nanquantile(data, 0.75, axis=0)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, np.nanstd(data, axis=0))
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0)
    return med, scale


def _apply_reference_scaler(
    data: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    out = (data - center) / scale
    return out






def _upper_tri(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(mat, k=1)
    return np.asarray(mat[iu], dtype=np.float64)


def _covariance(flat: np.ndarray) -> np.ndarray | None:
    if flat.shape[0] < 5 or flat.shape[1] < 2:
        return None
    cov = np.cov(flat, rowvar=False)
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] < 2:
        return None
    return cov


def _ledoit_cov_precision(
    flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if flat.shape[0] < 5 or flat.shape[1] < 2:
        return None, None
    model = LedoitWolf().fit(flat)
    cov = np.asarray(model.covariance_, dtype=np.float64)
    prec = np.asarray(model.precision_, dtype=np.float64)
    return cov, prec


def _partial_corr_from_precision(prec: np.ndarray) -> np.ndarray | None:
    if prec is None:
        return None
    diag = np.sqrt(np.maximum(np.diag(prec), EPS))
    denom = np.outer(diag, diag)
    pcorr = -prec / denom
    np.fill_diagonal(pcorr, 1.0)
    return np.clip(pcorr, -1.0, 1.0)


def _normalized_spectrum(eigs: np.ndarray) -> np.ndarray:
    eigs = np.asarray(eigs, dtype=np.float64)
    eigs = eigs[np.isfinite(eigs) & (eigs > 0)]
    if eigs.size == 0:
        return np.asarray([], dtype=np.float64)
    eigs = np.sort(eigs)[::-1]
    total = float(np.sum(eigs))
    if total < EPS:
        return np.asarray([], dtype=np.float64)
    return eigs / total


def _spectrum_similarity(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return np.nan
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    return weighted_mean_available(
        dict(enumerate([_corr_score(np.log(x + EPS), np.log(y + EPS)),
        _rmse_similarity(x, y)])),
        weights={0:0.55, 1:0.45},
    )


def _matrix_similarity(gt_mat: np.ndarray | None, pred_mat: np.ndarray | None) -> float:
    if gt_mat is None or pred_mat is None:
        return np.nan
    gt_vec = _upper_tri(gt_mat)
    pred_vec = _upper_tri(pred_mat)
    return weighted_mean_available(
        dict(enumerate([_corr_score(gt_vec, pred_vec),
        _rmse_similarity(gt_vec, pred_vec)])),
        weights={0:0.6, 1:0.4},
    )


def _mean_region_psd(arr: np.ndarray) -> np.ndarray:
    pooled = []
    for region_idx in range(arr.shape[2]):
        region_series = np.asarray(arr[:, :, region_idx], dtype=np.float64).reshape(-1)
        region_series = region_series[np.isfinite(region_series)]
        if region_series.size < 32:
            continue
        _, psd = welch(region_series, nperseg=min(256, region_series.size))
        psd = np.asarray(psd, dtype=np.float64)
        if psd.size == 0 or not np.isfinite(psd).any():
            continue
        total = float(np.sum(psd))
        if total > EPS:
            pooled.append(psd / total)
    if not pooled:
        return np.asarray([], dtype=np.float64)
    min_len = min(len(p) for p in pooled)
    stack = np.vstack([p[:min_len] for p in pooled])
    return np.nanmean(stack, axis=0)


def _effective_dimension(cov: np.ndarray | None) -> float:
    if cov is None:
        return np.nan
    eigs = np.linalg.eigvalsh(cov)
    eigs = eigs[np.isfinite(eigs) & (eigs > EPS)]
    if eigs.size == 0:
        return np.nan
    pr = float((np.sum(eigs) ** 2) / np.sum(eigs**2))
    return pr / cov.shape[0]


def _principal_subspace(cov: np.ndarray | None, max_dim: int = 6) -> np.ndarray | None:
    if cov is None:
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = np.maximum(eigvals, 0.0)
    total = float(np.sum(eigvals))
    if total < EPS:
        return None
    cum = np.cumsum(eigvals) / total
    k = int(np.searchsorted(cum, 0.85) + 1)
    k = max(1, min(k, max_dim, cov.shape[0]))
    return eigvecs[:, :k]


def _subspace_angle_score(
    gt_cov: np.ndarray | None, pred_cov: np.ndarray | None
) -> float:
    gt_basis = _principal_subspace(gt_cov)
    pred_basis = _principal_subspace(pred_cov)
    if gt_basis is None or pred_basis is None:
        return np.nan
    k = min(gt_basis.shape[1], pred_basis.shape[1])
    gt_basis = gt_basis[:, :k]
    pred_basis = pred_basis[:, :k]
    angles = subspace_angles(gt_basis, pred_basis)
    if angles.size == 0:
        return np.nan
    return float(np.mean(np.cos(angles) ** 2))


def _lagged_covariance(flat: np.ndarray, lag: int) -> np.ndarray | None:
    if flat.shape[0] <= lag + 3 or flat.shape[1] < 2:
        return None
    x = flat[:-lag]
    y = flat[lag:]
    x = x - np.mean(x, axis=0, keepdims=True)
    y = y - np.mean(y, axis=0, keepdims=True)
    cov = (x.T @ y) / max(x.shape[0] - 1, 1)
    cov = np.asarray(cov, dtype=np.float64)
    return cov


def _var1_coefficients(flat: np.ndarray, ridge: float = 1e-2) -> np.ndarray | None:
    if flat.shape[0] < 6 or flat.shape[1] < 2:
        return None
    x = flat[:-1]
    y = flat[1:]
    xtx = x.T @ x
    reg = ridge * np.eye(xtx.shape[0], dtype=np.float64)
    try:
        coeff = np.linalg.solve(xtx + reg, x.T @ y)
    except np.linalg.LinAlgError:
        coeff = np.linalg.pinv(xtx + reg) @ (x.T @ y)
    coeff = np.asarray(coeff.T, dtype=np.float64)
    return coeff


def _mi_matrix(
    flat: np.ndarray, max_points: int = 1500, n_neighbors: int = 5
) -> np.ndarray | None:
    if flat.shape[0] < 40 or flat.shape[1] < 2:
        return None
    rng = np.random.default_rng(0)
    if flat.shape[0] > max_points:
        idx = rng.choice(flat.shape[0], size=max_points, replace=False)
        flat = flat[idx]
    flat = _standardize_columns(flat)
    n_reg = flat.shape[1]
    out = np.zeros((n_reg, n_reg), dtype=np.float64)
    for i in range(n_reg):
        for j in range(i + 1, n_reg):
            x = flat[:, i]
            y = flat[:, j]
            try:
                mi_xy = mutual_info_regression(
                    x.reshape(-1, 1), y, n_neighbors=n_neighbors, random_state=0
                )[0]
                mi_yx = mutual_info_regression(
                    y.reshape(-1, 1), x, n_neighbors=n_neighbors, random_state=0
                )[0]
                value = 0.5 * (max(float(mi_xy), 0.0) + max(float(mi_yx), 0.0))
            except Exception:
                value = np.nan
            out[i, j] = value
            out[j, i] = value
    return out


def _fit_pca_basis(
    flat: np.ndarray, n_components: int = 3
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if flat.shape[0] < 10 or flat.shape[1] < 2:
        return None, None
    mean = np.mean(flat, axis=0, keepdims=True)
    centered = flat - mean
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None
    k = min(n_components, vt.shape[0], flat.shape[1])
    if k < 1:
        return None, None
    return mean.ravel(), vt[:k].T


def _project_rows(
    flat: np.ndarray, mean: np.ndarray | None, basis: np.ndarray | None
) -> np.ndarray | None:
    if mean is None or basis is None:
        return None
    centered = flat - mean
    proj = centered @ basis
    proj = np.asarray(proj, dtype=np.float64)
    return proj


def _fit_kmeans_centers(
    features: np.ndarray,
    n_clusters: int,
    *,
    max_fit_points: int = 20000,
    random_state: int = 0,
) -> np.ndarray | None:
    features = np.asarray(features, dtype=np.float64)
    if (
        features.ndim != 2
        or features.shape[0] < max(12, n_clusters)
        or features.shape[1] < 1
    ):
        return None
    n_clusters = int(np.clip(n_clusters, 2, features.shape[0]))
    fit_features = features
    if features.shape[0] > max_fit_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(features.shape[0], size=max_fit_points, replace=False)
        fit_features = features[idx]
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=random_state,
        batch_size=min(2048, fit_features.shape[0]),
    )
    model.fit(fit_features)
    centers = np.asarray(model.cluster_centers_, dtype=np.float64)
    return centers


def _assign_to_centers(
    features: np.ndarray, centers: np.ndarray | None
) -> np.ndarray | None:
    if centers is None:
        return None
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < 1:
        return None
    feat_sq = np.sum(features * features, axis=1, keepdims=True)
    ctr_sq = np.sum(centers * centers, axis=1)[None, :]
    d2 = feat_sq + ctr_sq - 2.0 * (features @ centers.T)
    return np.argmin(d2, axis=1).astype(int)


def _hist_similarity(gt_hist: np.ndarray, pred_hist: np.ndarray) -> float:
    gt_hist = np.asarray(gt_hist, dtype=np.float64).ravel()
    pred_hist = np.asarray(pred_hist, dtype=np.float64).ravel()
    if gt_hist.size == 0 or pred_hist.size == 0:
        return np.nan
    gt_hist = np.clip(gt_hist, 0.0, None)
    pred_hist = np.clip(pred_hist, 0.0, None)
    if np.sum(gt_hist) < EPS or np.sum(pred_hist) < EPS:
        return np.nan
    gt_hist = gt_hist / np.sum(gt_hist)
    pred_hist = pred_hist / np.sum(pred_hist)
    overlap = float(np.sum(np.minimum(gt_hist, pred_hist)))
    return weighted_mean_available(
        dict(enumerate([overlap,
        _corr_score(gt_hist, pred_hist),
        _rmse_similarity(gt_hist, pred_hist)])),
        weights={0:0.40, 1:0.25, 2:0.35},
    )


def _occupancy_similarity(
    gt_assign: np.ndarray | None, pred_assign: np.ndarray | None, n_states: int
) -> float:
    if gt_assign is None or pred_assign is None or n_states < 2:
        return np.nan
    gt_hist = np.bincount(gt_assign, minlength=n_states).astype(np.float64)
    pred_hist = np.bincount(pred_assign, minlength=n_states).astype(np.float64)
    return _hist_similarity(gt_hist, pred_hist)


def _transition_similarity(
    gt_sequences: list[np.ndarray],
    pred_sequences: list[np.ndarray],
    n_states: int,
    lag: int,
) -> float:
    if (
        lag < 1
        or not gt_sequences
        or not pred_sequences
        or len(gt_sequences) != len(pred_sequences)
    ):
        return np.nan
    gt_mat = np.zeros((n_states, n_states), dtype=np.float64)
    pred_mat = np.zeros((n_states, n_states), dtype=np.float64)
    for gt_seq, pred_seq in zip(gt_sequences, pred_sequences):
        if gt_seq.size <= lag or pred_seq.size <= lag:
            continue
        g0 = gt_seq[:-lag]
        g1 = gt_seq[lag:]
        p0 = pred_seq[:-lag]
        p1 = pred_seq[lag:]
        np.add.at(gt_mat, (g0, g1), 1.0)
        np.add.at(pred_mat, (p0, p1), 1.0)
    if float(np.sum(gt_mat)) < EPS or float(np.sum(pred_mat)) < EPS:
        return np.nan
    return _hist_similarity(gt_mat.ravel(), pred_mat.ravel())


def _window_fc_features(
    arr: np.ndarray, window: int, step: int
) -> tuple[np.ndarray | None, list[int]]:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] < 2:
        return None, []
    features = []
    counts = []
    for seq in arr:
        seq_features = []
        t = seq.shape[0]
        if t < window:
            counts.append(0)
            continue
        for start in range(0, t - window + 1, step):
            win = seq[start : start + window]
            centered = win - np.mean(win, axis=0, keepdims=True)
            scale = np.std(centered, axis=0, ddof=1)
            valid = np.isfinite(scale) & (scale > EPS)
            corr = np.eye(win.shape[1], dtype=np.float64)
            if np.sum(valid) >= 2:
                z = centered[:, valid] / scale[valid]
                corr_valid = (z.T @ z) / max(z.shape[0] - 1, 1)
                corr_valid = np.clip(corr_valid, -1.0, 1.0)
                idx = np.flatnonzero(valid)
                corr[np.ix_(idx, idx)] = corr_valid
            seq_features.append(_upper_tri(corr))
        counts.append(len(seq_features))
        if seq_features:
            features.extend(seq_features)
    if not features:
        return None, counts
    return np.vstack(features), counts


def _split_assignments(
    assign: np.ndarray | None, counts: list[int]
) -> list[np.ndarray]:
    if assign is None:
        return []
    out = []
    cursor = 0
    for count in counts:
        count = int(count)
        out.append(assign[cursor : cursor + count])
        cursor += count
    return out


def _latent_state_occupancy_score(
    gt_flat_raw: np.ndarray, pred_flat_raw: np.ndarray
) -> float:
    center, scale = _fit_reference_scaler(gt_flat_raw)
    gt_scaled = _apply_reference_scaler(gt_flat_raw, center, scale)
    pred_scaled = _apply_reference_scaler(pred_flat_raw, center, scale)
    mean, basis = _fit_pca_basis(gt_scaled, n_components=3)
    gt_proj = _project_rows(gt_scaled, mean, basis)
    pred_proj = _project_rows(pred_scaled, mean, basis)
    if gt_proj is None or pred_proj is None:
        return np.nan
    if gt_proj.shape[0] > 30000:
        rng = np.random.default_rng(0)
        idx = rng.choice(gt_proj.shape[0], size=30000, replace=False)
        gt_proj = gt_proj[idx]
    if pred_proj.shape[0] > 30000:
        rng = np.random.default_rng(1)
        idx = rng.choice(pred_proj.shape[0], size=30000, replace=False)
        pred_proj = pred_proj[idx]
    per_k = []
    for n_states in (6, 8, 10):
        centers = _fit_kmeans_centers(
            gt_proj, n_states, max_fit_points=10000, random_state=n_states
        )
        gt_assign = _assign_to_centers(gt_proj, centers)
        pred_assign = _assign_to_centers(pred_proj, centers)
        per_k.append(_occupancy_similarity(gt_assign, pred_assign, n_states))
    vals = np.asarray(per_k, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else np.nan


def _prepare_latent_state_reference(
    gt: np.ndarray,
    pred: np.ndarray,
    gt_flat_raw: np.ndarray,
    pred_flat_raw: np.ndarray,
    *,
    n_components: int = 3,
) -> dict[str, object] | None:
    center, scale = _fit_reference_scaler(gt_flat_raw)
    gt_scaled = _apply_reference_scaler(gt_flat_raw, center, scale)
    pred_scaled = _apply_reference_scaler(pred_flat_raw, center, scale)
    mean, basis = _fit_pca_basis(gt_scaled, n_components=n_components)
    gt_proj_all = _project_rows(gt_scaled, mean, basis)
    pred_proj_all = _project_rows(pred_scaled, mean, basis)
    if gt_proj_all is None or pred_proj_all is None:
        return None
    if gt_proj_all.shape[0] > 30000:
        rng = np.random.default_rng(0)
        fit_idx = rng.choice(gt_proj_all.shape[0], size=30000, replace=False)
        gt_proj_fit = gt_proj_all[fit_idx]
        eval_idx = rng.choice(gt_proj_all.shape[0], size=30000, replace=False)
        gt_proj_eval = gt_proj_all[eval_idx]
    else:
        gt_proj_fit = gt_proj_all
        gt_proj_eval = gt_proj_all
    if pred_proj_all.shape[0] > 30000:
        rng = np.random.default_rng(1)
        pred_proj_eval = pred_proj_all[
            rng.choice(pred_proj_all.shape[0], size=30000, replace=False)
        ]
    else:
        pred_proj_eval = pred_proj_all
    gt_seq_proj = [
        _project_rows(_apply_reference_scaler(seq, center, scale), mean, basis)
        for seq in gt
    ]
    pred_seq_proj = [
        _project_rows(_apply_reference_scaler(seq, center, scale), mean, basis)
        for seq in pred
    ]
    return {
        "gt_proj_fit": gt_proj_fit,
        "gt_proj_eval": gt_proj_eval,
        "pred_proj_eval": pred_proj_eval,
        "gt_seq_proj": gt_seq_proj,
        "pred_seq_proj": pred_seq_proj,
    }


def _latent_state_score_bundle(
    gt: np.ndarray,
    pred: np.ndarray,
    gt_flat_raw: np.ndarray,
    pred_flat_raw: np.ndarray,
) -> dict[str, float]:
    ref = _prepare_latent_state_reference(gt, pred, gt_flat_raw, pred_flat_raw)
    if ref is None:
        return {
            "LatentStateOccupancyK11_score01": np.nan,
            "LatentStateOccupancyK12_score01": np.nan,
            "LatentStateTransitionLag1K11_score01": np.nan,
            "LatentStateTransitionLag2K11_score01": np.nan,
            "LatentStateTransitionLag3K11_score01": np.nan,
        }

    def _occupancy_for_k(k: int) -> float:
        centers = _fit_kmeans_centers(
            ref["gt_proj_fit"], k, max_fit_points=10000, random_state=k
        )
        gt_assign = _assign_to_centers(ref["gt_proj_eval"], centers)
        pred_assign = _assign_to_centers(ref["pred_proj_eval"], centers)
        return _occupancy_similarity(gt_assign, pred_assign, k)

    def _transition_for_k(k: int, lag: int) -> float:
        centers = _fit_kmeans_centers(
            ref["gt_proj_fit"], k, max_fit_points=10000, random_state=k
        )
        gt_seq_assign = [_assign_to_centers(seq, centers) for seq in ref["gt_seq_proj"]]
        pred_seq_assign = [
            _assign_to_centers(seq, centers) for seq in ref["pred_seq_proj"]
        ]
        return _transition_similarity(gt_seq_assign, pred_seq_assign, k, lag)

    return {
        "LatentStateOccupancyK11_score01": _occupancy_for_k(11),
        "LatentStateOccupancyK12_score01": _occupancy_for_k(12),
        "LatentStateTransitionLag1K11_score01": _transition_for_k(11, 1),
        "LatentStateTransitionLag2K11_score01": _transition_for_k(11, 2),
        "LatentStateTransitionLag3K11_score01": _transition_for_k(11, 3),
    }


def _dfc_state_occupancy_score(gt: np.ndarray, pred: np.ndarray) -> float:
    configs = ((45, 15, 8), (60, 30, 8), (90, 30, 8))
    per_cfg = []
    for window, step, n_states in configs:
        gt_feat, _ = _window_fc_features(gt, window=window, step=step)
        pred_feat, _ = _window_fc_features(pred, window=window, step=step)
        if gt_feat is None or pred_feat is None:
            continue
        centers = _fit_kmeans_centers(
            gt_feat,
            n_states,
            max_fit_points=3000,
            random_state=window + step + n_states,
        )
        gt_assign = _assign_to_centers(gt_feat, centers)
        pred_assign = _assign_to_centers(pred_feat, centers)
        per_cfg.append(_occupancy_similarity(gt_assign, pred_assign, n_states))
    vals = np.asarray(per_cfg, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else np.nan


def _dfc_state_transition_score(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_feat, gt_counts = _window_fc_features(gt, window=60, step=30)
    pred_feat, pred_counts = _window_fc_features(pred, window=60, step=30)
    if gt_feat is None or pred_feat is None:
        return np.nan
    centers = _fit_kmeans_centers(gt_feat, 8, max_fit_points=3000, random_state=602)
    gt_assign = _assign_to_centers(gt_feat, centers)
    pred_assign = _assign_to_centers(pred_feat, centers)
    gt_seq = _split_assignments(gt_assign, gt_counts)
    pred_seq = _split_assignments(pred_assign, pred_counts)
    lag2 = _transition_similarity(gt_seq, pred_seq, n_states=8, lag=2)
    lag3 = _transition_similarity(gt_seq, pred_seq, n_states=8, lag=3)
    return weighted_mean_available(dict(enumerate([lag2, lag3])), weights={0:0.55, 1:0.45})


def compute_additional_structural_metrics(
    gt_arr: np.ndarray, pred_arr: np.ndarray
) -> dict[str, object]:
    gt, pred = _align_arrays(gt_arr, pred_arr)
    gt_flat_raw = _pooled_rows(gt)
    pred_flat_raw = _pooled_rows(pred)
    gt_flat = _standardize_columns(gt_flat_raw)
    pred_flat = _standardize_columns(pred_flat_raw)

    gt_cov, gt_prec = _ledoit_cov_precision(gt_flat)
    pred_cov, pred_prec = _ledoit_cov_precision(pred_flat)

    gt_pcorr = _partial_corr_from_precision(gt_prec)
    pred_pcorr = _partial_corr_from_precision(pred_prec)

    gt_psd = _mean_region_psd(gt)
    pred_psd = _mean_region_psd(pred)

    gt_cov_spec = (
        _normalized_spectrum(np.linalg.eigvalsh(gt_cov))
        if gt_cov is not None
        else np.asarray([])
    )
    pred_cov_spec = (
        _normalized_spectrum(np.linalg.eigvalsh(pred_cov))
        if pred_cov is not None
        else np.asarray([])
    )
    gt_prec_spec = (
        _normalized_spectrum(np.linalg.eigvalsh(gt_prec))
        if gt_prec is not None
        else np.asarray([])
    )
    pred_prec_spec = (
        _normalized_spectrum(np.linalg.eigvalsh(pred_prec))
        if pred_prec is not None
        else np.asarray([])
    )

    partial_corr = _matrix_similarity(gt_pcorr, pred_pcorr)
    psd_shape = weighted_mean_available(
        dict(enumerate([_corr_score(np.log(gt_psd + EPS), np.log(pred_psd + EPS)),
        _rmse_similarity(gt_psd, pred_psd)])),
        weights={0:0.55, 1:0.45},
    )

    dim_gt = _effective_dimension(gt_cov)
    dim_pred = _effective_dimension(pred_cov)
    dimensionality = (
        float(1.0 / (1.0 + abs(dim_gt - dim_pred) / max(dim_gt, 1e-6)))
        if np.isfinite(dim_gt) and np.isfinite(dim_pred)
        else np.nan
    )

    cross_mi = _matrix_similarity(_mi_matrix(gt_flat), _mi_matrix(pred_flat))
    precision_spectrum = _spectrum_similarity(gt_prec_spec, pred_prec_spec)
    subspace_angle = _subspace_angle_score(gt_cov, pred_cov)
    eigenspectrum_shape = _spectrum_similarity(gt_cov_spec, pred_cov_spec)

    lag_scores = []
    for lag in (1, 2, 4):
        lag_scores.append(
            _matrix_similarity(
                _lagged_covariance(gt_flat, lag), _lagged_covariance(pred_flat, lag)
            )
        )
    lagged_covariance = (
        float(np.nanmean(lag_scores)) if np.isfinite(lag_scores).any() else np.nan
    )

    impulse_response = _matrix_similarity(
        _var1_coefficients(gt_flat), _var1_coefficients(pred_flat)
    )
    latent_state_scores = _latent_state_score_bundle(
        gt, pred, gt_flat_raw, pred_flat_raw
    )

    scores = {
        "PartialCorr_score01": partial_corr,
        "PSDShape_score01": psd_shape,
        "Dimensionality_score01": dimensionality,
        "CrossRegionMI_score01": cross_mi,
        "PrecisionMatrixSpectrum_score01": precision_spectrum,
        "SubspaceAngle_score01": subspace_angle,
        "EigenspectrumShape_score01": eigenspectrum_shape,
        "LaggedCovariance_score01": lagged_covariance,
        "ImpulseResponse_score01": impulse_response,
    }
    scores.update(latent_state_scores)
    return {"scores": scores}
