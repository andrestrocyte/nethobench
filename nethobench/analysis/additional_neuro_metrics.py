from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.linalg import subspace_angles
from sklearn.covariance import LedoitWolf
from sklearn.feature_selection import mutual_info_regression


EPS = 1e-9


def _align_arrays(gt_arr: np.ndarray, pred_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gt = np.asarray(gt_arr, dtype=np.float64)
    pred = np.asarray(pred_arr, dtype=np.float64)
    if gt.ndim != 3 or pred.ndim != 3:
        raise ValueError(f"Expected [n_seq, T, n_reg] arrays, got {gt.shape} and {pred.shape}")
    if gt.shape[0] != pred.shape[0] or gt.shape[2] != pred.shape[2]:
        raise ValueError(f"GT/pred sequence-region mismatch: {gt.shape} vs {pred.shape}")
    if pred.shape[1] != gt.shape[1] and pred.shape[1] % gt.shape[1] == 0:
        factor = pred.shape[1] // gt.shape[1]
        pred = pred.reshape(pred.shape[0], gt.shape[1], factor, pred.shape[2]).mean(axis=2)
    elif pred.shape[1] != gt.shape[1]:
        keep = min(gt.shape[1], pred.shape[1])
        gt = gt[:, :keep, :]
        pred = pred[:, :keep, :]
    return gt, pred


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


def _safe_corr01(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xx = x[mask]
    yy = y[mask]
    if np.nanstd(xx) < EPS or np.nanstd(yy) < EPS:
        return np.nan
    corr = np.corrcoef(xx, yy)[0, 1]
    if not np.isfinite(corr):
        return np.nan
    return float(np.clip(0.5 * (corr + 1.0), 0.0, 1.0))


def _rmse_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    err = float(np.sqrt(np.mean((x[mask] - y[mask]) ** 2)))
    scale = float(np.nanstd(x[mask]))
    if not np.isfinite(scale) or scale < EPS:
        scale = float(np.nanmean(np.abs(x[mask])))
    if not np.isfinite(scale) or scale < EPS:
        scale = 1.0
    return float(1.0 / (1.0 + err / scale))


def _blend_scores(*scores: float, weights: tuple[float, ...] | None = None) -> float:
    values = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(values)
    if not np.any(mask):
        return np.nan
    if weights is None:
        return float(np.nanmean(values[mask]))
    w = np.asarray(weights, dtype=np.float64)[mask]
    vals = values[mask]
    return float(np.sum(w * vals) / np.sum(w))


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
    cov[~np.isfinite(cov)] = 0.0
    return cov


def _ledoit_cov_precision(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
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
    pcorr[~np.isfinite(pcorr)] = 0.0
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
    return _blend_scores(
        _safe_corr01(np.log(x + EPS), np.log(y + EPS)),
        _rmse_similarity(x, y),
        weights=(0.55, 0.45),
    )


def _matrix_similarity(gt_mat: np.ndarray | None, pred_mat: np.ndarray | None) -> float:
    if gt_mat is None or pred_mat is None:
        return np.nan
    gt_vec = _upper_tri(gt_mat)
    pred_vec = _upper_tri(pred_mat)
    return _blend_scores(
        _safe_corr01(gt_vec, pred_vec),
        _rmse_similarity(gt_vec, pred_vec),
        weights=(0.6, 0.4),
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
    pr = float((np.sum(eigs) ** 2) / np.sum(eigs ** 2))
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


def _subspace_angle_score(gt_cov: np.ndarray | None, pred_cov: np.ndarray | None) -> float:
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
    cov[~np.isfinite(cov)] = 0.0
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
    coeff[~np.isfinite(coeff)] = 0.0
    return coeff


def _mi_matrix(flat: np.ndarray, max_points: int = 1500, n_neighbors: int = 5) -> np.ndarray | None:
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
                mi_xy = mutual_info_regression(x.reshape(-1, 1), y, n_neighbors=n_neighbors, random_state=0)[0]
                mi_yx = mutual_info_regression(y.reshape(-1, 1), x, n_neighbors=n_neighbors, random_state=0)[0]
                value = 0.5 * (max(float(mi_xy), 0.0) + max(float(mi_yx), 0.0))
            except Exception:
                value = np.nan
            out[i, j] = value
            out[j, i] = value
    return out


def compute_additional_structural_metrics(gt_arr: np.ndarray, pred_arr: np.ndarray) -> dict[str, object]:
    gt, pred = _align_arrays(gt_arr, pred_arr)
    gt_flat = _standardize_columns(_pooled_rows(gt))
    pred_flat = _standardize_columns(_pooled_rows(pred))

    gt_cov, gt_prec = _ledoit_cov_precision(gt_flat)
    pred_cov, pred_prec = _ledoit_cov_precision(pred_flat)

    gt_pcorr = _partial_corr_from_precision(gt_prec)
    pred_pcorr = _partial_corr_from_precision(pred_prec)

    gt_psd = _mean_region_psd(gt)
    pred_psd = _mean_region_psd(pred)

    gt_cov_spec = _normalized_spectrum(np.linalg.eigvalsh(gt_cov)) if gt_cov is not None else np.asarray([])
    pred_cov_spec = _normalized_spectrum(np.linalg.eigvalsh(pred_cov)) if pred_cov is not None else np.asarray([])
    gt_prec_spec = _normalized_spectrum(np.linalg.eigvalsh(gt_prec)) if gt_prec is not None else np.asarray([])
    pred_prec_spec = _normalized_spectrum(np.linalg.eigvalsh(pred_prec)) if pred_prec is not None else np.asarray([])

    partial_corr = _matrix_similarity(gt_pcorr, pred_pcorr)
    psd_shape = _blend_scores(
        _safe_corr01(np.log(gt_psd + EPS), np.log(pred_psd + EPS)),
        _rmse_similarity(gt_psd, pred_psd),
        weights=(0.55, 0.45),
    )

    dim_gt = _effective_dimension(gt_cov)
    dim_pred = _effective_dimension(pred_cov)
    dimensionality = float(1.0 / (1.0 + abs(dim_gt - dim_pred) / max(dim_gt, 1e-6))) if np.isfinite(dim_gt) and np.isfinite(dim_pred) else np.nan

    cross_mi = _matrix_similarity(_mi_matrix(gt_flat), _mi_matrix(pred_flat))
    precision_spectrum = _spectrum_similarity(gt_prec_spec, pred_prec_spec)
    subspace_angle = _subspace_angle_score(gt_cov, pred_cov)
    eigenspectrum_shape = _spectrum_similarity(gt_cov_spec, pred_cov_spec)

    lag_scores = []
    for lag in (1, 2, 4):
        lag_scores.append(_matrix_similarity(_lagged_covariance(gt_flat, lag), _lagged_covariance(pred_flat, lag)))
    lagged_covariance = float(np.nanmean(lag_scores)) if np.isfinite(lag_scores).any() else np.nan

    impulse_response = _matrix_similarity(_var1_coefficients(gt_flat), _var1_coefficients(pred_flat))

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
    return {"scores": scores}
