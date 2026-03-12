from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import runpy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import tempfile
from scipy import signal

# --------------------------------------------------------------------------- #
# Loading helpers                                                             #
# --------------------------------------------------------------------------- #

ACTIVE_NEURO_NOTEBOOK = Path(__file__).parent / "notebooks" / "neuro_metrics.ipynb"
ACTIVE_NEURO_CORE_SCRIPT = Path(__file__).parent / "analysis" / "neuro_metrics_core_script.py"
ACTIVE_NEURO_INSTANT_SCRIPT = Path(__file__).parent / "analysis" / "neuro_metrics_instant_script.py"


def _load_sequences(csv_path: Path, sequence_key: str = "sequenceId", time_key: str = "itemPosition") -> tuple[np.ndarray, list[str]]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if {sequence_key, time_key}.issubset(df.columns):
        df = df.sort_values([sequence_key, time_key]).reset_index(drop=True)
        region_cols = [c for c in df.columns if c not in {sequence_key, time_key}]
        if not region_cols:
            raise ValueError(f"No region columns found in {csv_path}")
        seq_lengths = df.groupby(sequence_key).size()
        if seq_lengths.nunique() != 1:
            raise ValueError(
                "Sequences must all have identical length. "
                f"Distribution:\n{seq_lengths.describe()}"
            )
        n_seq = seq_lengths.size
        n_time = int(seq_lengths.iloc[0])
        arr = (
            df[region_cols]
            .to_numpy(dtype=np.float64)
            .reshape(n_seq, n_time, len(region_cols))
        )
        return arr, region_cols

    df = pd.read_csv(csv_path, index_col=0)
    if df.index.dtype.kind not in {"i", "u"}:
        raise ValueError(
            f"Prediction CSV {csv_path} must have integral sequence ids in the index."
        )
    counts = df.index.value_counts()
    if counts.nunique() != 1:
        raise ValueError(
            "Prediction sequences must all have the same length. Counts:\n"
            f"{counts}"
        )
    n_seq = counts.shape[0]
    n_time = counts.iloc[0]
    region_cols = df.columns.tolist()
    arr = df.to_numpy(dtype=np.float64).reshape(n_seq, n_time, len(region_cols))
    return arr, region_cols


def _load_and_align(predictions_csv: Path, ground_truth_csv: Path, neuro_cols: Optional[list[str]] = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pred_arr, pred_regions = _load_sequences(predictions_csv)
    gt_arr, gt_regions = _load_sequences(ground_truth_csv)

    if neuro_cols:
        overlap = [r for r in neuro_cols if r in gt_regions and r in pred_regions]
    else:
        overlap = [r for r in gt_regions if r in pred_regions]
    if not overlap:
        raise ValueError("No overlapping neural regions between GT and predictions.")

    gt_idx = [gt_regions.index(r) for r in overlap]
    pred_idx = [pred_regions.index(r) for r in overlap]
    gt_arr = gt_arr[..., gt_idx]
    pred_arr = pred_arr[..., pred_idx]

    max_len = min(gt_arr.shape[1], pred_arr.shape[1])
    gt_arr = gt_arr[:, :max_len, :]
    pred_arr = pred_arr[:, :max_len, :]
    return gt_arr, pred_arr, overlap


# --------------------------------------------------------------------------- #
# Core scoring logic                                                          #
# --------------------------------------------------------------------------- #


def _ensure_ddconfig(ddconfig_path: Optional[Path]) -> tuple[Path, Optional[Path]]:
    if ddconfig_path is not None:
        return Path(ddconfig_path), None
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({"selected_columns_statistics": {}}, tmp)
    tmp.flush()
    tmp.close()
    return Path(tmp.name), Path(tmp.name)


def _flatten_scores(metrics: dict, buckets: dict, composite: float) -> Dict[str, float]:
    scores: Dict[str, float] = {k: float(v) for k, v in metrics.items()}
    for k, v in buckets.items():
        scores[f"bucket_{k}"] = float(v)
    scores["composite_score"] = float(composite)
    scores["FINAL_COMPOSITE_SCORE"] = float(composite)
    scores["FINAL_NEURO_COMPOSITE_SCORE"] = float(composite)
    return scores


def _flatten_active_notebook_scores(env: dict) -> Dict[str, float]:
    scores: Dict[str, float] = {}

    metrics_df = env.get("metrics_df")
    if isinstance(metrics_df, pd.DataFrame) and {"metric", "value"}.issubset(metrics_df.columns):
        for _, row in metrics_df.iterrows():
            key = str(row["metric"])
            val = float(row["value"]) if np.isfinite(row["value"]) else np.nan
            scores[key] = val

    families_df = env.get("families_df")
    if isinstance(families_df, pd.DataFrame) and {"family", "value"}.issubset(families_df.columns):
        for _, row in families_df.iterrows():
            key = f"family_{row['family']}"
            val = float(row["value"]) if np.isfinite(row["value"]) else np.nan
            scores[key] = val

    composite = env.get("FINAL_COMPOSITE_SCORE", np.nan)
    composite = float(composite) if np.isfinite(composite) else np.nan
    scores["composite_score"] = composite
    scores["FINAL_COMPOSITE_SCORE"] = composite
    scores["FINAL_NEURO_COMPOSITE_SCORE"] = composite
    return scores


def _run_active_neuro_notebook(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    ddconfig_path: Optional[Path] = None,
    save_plots_dir: Optional[Path] = None,
    save_wrapped_notebook: Optional[Path] = None,
) -> tuple[dict, Dict[str, float]]:
    if not ACTIVE_NEURO_CORE_SCRIPT.is_file():
        raise FileNotFoundError(f"Active neuro core script missing at {ACTIVE_NEURO_CORE_SCRIPT}")
    dd_path, cleanup_path = _ensure_ddconfig(ddconfig_path)
    try:
        init_globals = {
            "preds_fname": str(Path(predictions_csv)),
            "gt_fname": str(Path(ground_truth_csv)),
            "ddconfig_path": str(dd_path),
            "SAVE_PLOTS_DIR": str(Path(save_plots_dir)) if save_plots_dir is not None else None,
            "WRAPPED_NOTEBOOK_PATH": str(Path(save_wrapped_notebook)) if save_wrapped_notebook is not None else None,
            "ENABLE_PLOTS": bool(save_plots_dir is not None),
            "__name__": "__main__",
            "ACTIVE_NEURO_NOTEBOOK": str(ACTIVE_NEURO_NOTEBOOK),
        }
        env = runpy.run_path(str(ACTIVE_NEURO_CORE_SCRIPT), init_globals=init_globals)
        env["ACTIVE_NEURO_NOTEBOOK"] = str(ACTIVE_NEURO_NOTEBOOK)
        env["ACTIVE_NEURO_CORE_SCRIPT"] = str(ACTIVE_NEURO_CORE_SCRIPT)
        return env, _flatten_active_notebook_scores(env)
    finally:
        if cleanup_path and cleanup_path.exists():
            cleanup_path.unlink(missing_ok=True)


def _run_active_neuro_instant_notebook(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    ddconfig_path: Optional[Path] = None,
) -> Dict[str, float]:
    if not ACTIVE_NEURO_INSTANT_SCRIPT.is_file():
        raise FileNotFoundError(f"Active instant neuro script missing at {ACTIVE_NEURO_INSTANT_SCRIPT}")
    dd_path, cleanup_path = _ensure_ddconfig(ddconfig_path)
    try:
        init_globals = {
            "preds_fname": str(Path(predictions_csv)),
            "gt_fname": str(Path(ground_truth_csv)),
            "ddconfig_path": str(dd_path),
            "__name__": "__main__",
            "ACTIVE_NEURO_NOTEBOOK": str(ACTIVE_NEURO_NOTEBOOK),
        }
        env = runpy.run_path(str(ACTIVE_NEURO_INSTANT_SCRIPT), init_globals=init_globals)
        env["ACTIVE_NEURO_NOTEBOOK"] = str(ACTIVE_NEURO_NOTEBOOK)
        env["ACTIVE_NEURO_INSTANT_SCRIPT"] = str(ACTIVE_NEURO_INSTANT_SCRIPT)

        scores = _flatten_active_notebook_scores(env)
        exact = env.get("INSTANT_NOTEBOOK_SCORES", {})
        if isinstance(exact, dict):
            for key, value in exact.items():
                if np.isfinite(value):
                    scores[key] = float(value)

        compatibility_aliases = {
            "KL_geo_score01_avg": scores.get("KL_score01", np.nan),
            "MeanShiftZ_mean": scores.get("Mean_score01", np.nan),
            "MI_mean_score01_avg": scores.get("MI_score01", np.nan),
            "ERR_realism_score01_avg": scores.get("Error_score01", np.nan),
            "QNT_realism_score01_avg": scores.get("QNT_score01", np.nan),
            "FC_core_score01_avg": scores.get("FC_score01", np.nan),
            "PCA_comp_score01_avg": scores.get("PCA_score01", np.nan),
            "MOM_core_score01_avg": scores.get("MOM_score01", np.nan),
            "GRAPH_core_score01_avg": scores.get("GRAPH_score01", np.nan),
            "Bandpower_score01_avg": scores.get("BP_score01", np.nan),
        }
        for key, value in compatibility_aliases.items():
            if np.isfinite(value):
                scores[key] = float(value)
        return scores
    finally:
        if cleanup_path and cleanup_path.exists():
            cleanup_path.unlink(missing_ok=True)


def _run_neurobench_script(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    ddconfig_path: Optional[Path] = None,
    mode: str = "full",
    skip_crosscorr: bool = True,
    skip_cca: bool = True,
    enable_plots: bool = False,
    disable_runtime_limits: bool = True,
) -> dict:
    script_path = Path(__file__).parent / "analysis" / "neurobench_core_script.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Neuro script missing at {script_path}")
    dd_path, cleanup_path = _ensure_ddconfig(ddconfig_path)
    init_globals = {
        "preds_fname": str(predictions_csv),
        "gt_fname": str(ground_truth_csv),
        "ddconfig_path": str(dd_path) if dd_path else None,
        "MODE": mode,
        "SKIP_CROSSCORR": skip_crosscorr,
        "SKIP_CCA": skip_cca,
        "ENABLE_PLOTS": enable_plots,
        "DISABLE_RUNTIME_LIMITS": disable_runtime_limits,
    }
    env = runpy.run_path(str(script_path), init_globals=init_globals)
    if cleanup_path and cleanup_path.exists():
        cleanup_path.unlink(missing_ok=True)
    return env


def _avg_mean_strict(mean_val: float, strict_val: float) -> float:
    mean_val = float(mean_val) if np.isfinite(mean_val) else np.nan
    strict_val = float(strict_val) if np.isfinite(strict_val) else np.nan
    if np.isfinite(mean_val) and np.isfinite(strict_val):
        return 0.5 * (mean_val + strict_val)
    if np.isfinite(mean_val):
        return mean_val
    if np.isfinite(strict_val):
        return strict_val
    return np.nan


def _clip01(x: float, eps: float = 1e-6) -> float:
    return float(np.clip(x, eps, 1.0 - eps))


def _gmean(values) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    v = np.clip(v, 1e-6, 1.0 - 1e-6)
    return float(np.exp(np.mean(np.log(v))))


def _weighted_gmean(buckets: dict, weights: dict) -> float:
    keys = [k for k, v in buckets.items() if np.isfinite(v)]
    if not keys:
        return np.nan
    wsum = float(np.sum([weights[k] for k in keys]))
    if wsum <= 0:
        return np.nan
    acc = 0.0
    for k in keys:
        acc += weights[k] * np.log(_clip01(buckets[k]))
    return float(np.exp(acc / wsum))


def _get(d, path, default=np.nan):
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


MONOTONIC_V2_LEGACY_WEIGHT = 0.30

MONOTONIC_V2_FAMILY_WEIGHTS = {
    "Bandpower_score01_avg": 0.30,
    "FC_core_score01_avg": 0.30,
    "GRAPH_core_score01_avg": 0.20,
    "PCA_comp_score01_avg": 0.07,
    "MeanShiftZ_mean": 0.07,
    "QNT_realism_score01_avg": 0.04,
    "ERR_realism_score01_avg": 0.02,
}

MONOTONIC_V3_GAMMA = 1.5
MONOTONIC_V3_BOTTLENECK_WEIGHT = 0.02


def _blend_legacy_absolute(
    legacy_val: float,
    abs_val: float,
    w_legacy: float = MONOTONIC_V2_LEGACY_WEIGHT,
) -> float:
    legacy_val = float(legacy_val) if np.isfinite(legacy_val) else np.nan
    abs_val = float(abs_val) if np.isfinite(abs_val) else np.nan
    if np.isfinite(legacy_val) and np.isfinite(abs_val):
        wl = float(np.clip(w_legacy, 0.0, 1.0))
        wa = 1.0 - wl
        return float(np.exp(wl * np.log(_clip01(legacy_val)) + wa * np.log(_clip01(abs_val))))
    if np.isfinite(abs_val):
        return abs_val
    return legacy_val


def _strict_multiplicative_v3(
    family_scores: dict,
    weights: dict,
    *,
    gamma: float = MONOTONIC_V3_GAMMA,
    bottleneck_weight: float = MONOTONIC_V3_BOTTLENECK_WEIGHT,
) -> tuple[dict, float]:
    valid = {k: float(v) for k, v in family_scores.items() if np.isfinite(v)}
    if not valid:
        return {}, np.nan

    g = float(max(gamma, 1.0))
    sharpened = {k: float(_clip01(v) ** g) for k, v in valid.items()}
    gmean_score = _weighted_gmean(sharpened, weights)
    if not np.isfinite(gmean_score):
        return sharpened, np.nan
    bottleneck = float(np.min(list(sharpened.values())))
    bw = float(np.clip(bottleneck_weight, 0.0, 0.95))
    composite = float(
        np.exp(
            (1.0 - bw) * np.log(_clip01(gmean_score))
            + bw * np.log(_clip01(bottleneck))
        )
    )
    return sharpened, composite


def _compute_absolute_fidelity_channels(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    Compute monotonic absolute-fidelity channels (0-1, higher better) from aligned arrays.
    These channels do not rely on mismatch baselines and are used to stabilize monotonicity.
    """
    if gt.ndim != 3 or pred.ndim != 3 or gt.shape != pred.shape:
        return {}

    eps = 1e-12
    gt_flat = gt.reshape(-1, gt.shape[-1]).astype(np.float64, copy=False)
    pr_flat = pred.reshape(-1, pred.shape[-1]).astype(np.float64, copy=False)
    mask_rows = np.isfinite(gt_flat).all(axis=1) & np.isfinite(pr_flat).all(axis=1)
    gt_flat = gt_flat[mask_rows]
    pr_flat = pr_flat[mask_rows]
    if gt_flat.size == 0 or pr_flat.size == 0:
        return {}

    # GT-anchored scales prevent "self-forgiveness" when predictions get noisier.
    gt_l1_scale = float(np.mean(np.abs(gt_flat))) + eps
    pred_l1_scale = float(np.mean(np.abs(pr_flat))) + eps
    gt_l2_scale = float(np.sqrt(np.mean(gt_flat ** 2))) + eps
    gt_med = np.median(gt_flat, axis=0)
    gt_mad = np.median(np.abs(gt_flat - gt_med[None, :]), axis=0)
    gt_sigma = 1.4826 * gt_mad
    gt_sigma = np.where(np.isfinite(gt_sigma) & (gt_sigma > eps), gt_sigma, np.std(gt_flat, axis=0) + eps)
    gt_sigma = np.where(np.isfinite(gt_sigma) & (gt_sigma > eps), gt_sigma, 1.0)

    # 1) Mean-shift family absolute channel.
    # Use a small location term + dominant first-difference variance mismatch term.
    # This avoids variance-inflation loopholes and makes the channel respond to additive noise.
    d_mean = float(np.mean(np.abs(np.mean(gt_flat, axis=0) - np.mean(pr_flat, axis=0))))
    mean_anchor = float(min(gt_l1_scale, pred_l1_scale)) + eps
    mean_term = float(d_mean / mean_anchor)

    hf_terms = []
    for r in range(gt.shape[-1]):
        gx = gt[:, :, r].reshape(-1)
        px = pred[:, :, r].reshape(-1)
        m = np.isfinite(gx) & np.isfinite(px)
        gx = gx[m]
        px = px[m]
        if gx.size < 8 or px.size < 8:
            continue
        dg = np.diff(gx)
        dp = np.diff(px)
        if dg.size < 4 or dp.size < 4:
            continue
        vg = float(np.var(dg)) + eps
        vp = float(np.var(dp)) + eps
        hf_terms.append(float(np.abs(np.log(vp / vg))))
    hf_term = float(np.mean(hf_terms)) if hf_terms else 0.0
    raw_mean_shift_score = float(np.exp(-(0.05 * mean_term + 0.95 * hf_term)))
    mean_shift_score = float(np.clip(raw_mean_shift_score, 0.0, 1.0))

    # 2) Error absolute channel
    rmse = float(np.sqrt(np.mean((gt_flat - pr_flat) ** 2)))
    mae = float(np.mean(np.abs(gt_flat - pr_flat)))
    err_score = 1.0 / (1.0 + 0.5 * (rmse / gt_l2_scale + mae / gt_l1_scale))

    # 3) Quantile-shape absolute channel
    quantiles = np.linspace(0.05, 0.95, 19, dtype=np.float64)
    q_dists = []
    for r in range(gt_flat.shape[1]):
        g = gt_flat[:, r]
        p = pr_flat[:, r]
        if g.size < 20 or p.size < 20:
            continue
        qg = np.quantile(g, quantiles)
        qp = np.quantile(p, quantiles)
        scale = float(gt_sigma[r]) + eps
        q_dists.append(float(np.mean(np.abs(qg - qp)) / scale))
    qnt_score = 1.0 / (1.0 + (float(np.mean(q_dists)) if q_dists else np.nan))

    # 4) FC absolute channel (upper-triangle correlation agreement)
    Cg = np.corrcoef(gt_flat, rowvar=False)
    Cp = np.corrcoef(pr_flat, rowvar=False)
    iu = np.triu_indices_from(Cg, k=1)
    valid = np.isfinite(Cg[iu]) & np.isfinite(Cp[iu])
    if np.sum(valid) >= 3:
        fc_r = float(np.corrcoef(Cg[iu][valid], Cp[iu][valid])[0, 1])
        fc_score = (fc_r + 1.0) / 2.0
    else:
        fc_score = np.nan

    # 5) PCA-spectrum absolute channel
    Xg = gt_flat - np.mean(gt_flat, axis=0, keepdims=True)
    Xp = pr_flat - np.mean(pr_flat, axis=0, keepdims=True)
    sg = np.linalg.svd(Xg, full_matrices=False, compute_uv=False)
    sp = np.linalg.svd(Xp, full_matrices=False, compute_uv=False)
    eg = sg ** 2
    ep = sp ** 2
    eg = eg / (np.sum(eg) + eps)
    ep = ep / (np.sum(ep) + eps)
    k = min(20, eg.size, ep.size)
    pca_score = float(np.clip(1.0 - np.mean(np.abs(eg[:k] - ep[:k])), 0.0, 1.0)) if k > 0 else np.nan

    # 6) Graph absolute channel (top-k edge overlap)
    Cga = np.abs(Cg).astype(np.float64, copy=True)
    Cpa = np.abs(Cp).astype(np.float64, copy=True)
    np.fill_diagonal(Cga, 0.0)
    np.fill_diagonal(Cpa, 0.0)
    iu = np.triu_indices_from(Cga, k=1)
    n_edges = len(iu[0])
    if n_edges > 0:
        topk = min(30, n_edges)
        gi = np.argsort(Cga[iu])[-topk:]
        pi = np.argsort(Cpa[iu])[-topk:]
        gset = set(zip(iu[0][gi], iu[1][gi]))
        pset = set(zip(iu[0][pi], iu[1][pi]))
        graph_score = float(len(gset & pset) / max(1, len(gset | pset)))
    else:
        graph_score = np.nan

    # 7) Bandpower absolute channel (PSD cosine)
    bp_sims = []
    for r in range(gt.shape[-1]):
        gx = gt[:, :, r].reshape(-1)
        px = pred[:, :, r].reshape(-1)
        m = np.isfinite(gx) & np.isfinite(px)
        gx = gx[m]
        px = px[m]
        if gx.size < 32 or px.size < 32:
            continue
        nperseg = min(2048, gx.size, px.size)
        _, pxx_g = signal.welch(gx, nperseg=nperseg)
        _, pxx_p = signal.welch(px, nperseg=nperseg)
        denom = float(np.linalg.norm(pxx_g) * np.linalg.norm(pxx_p)) + eps
        bp_sims.append(float(np.dot(pxx_g, pxx_p) / denom))
    bandpower_score = float(np.clip(np.nanmean(bp_sims), 0.0, 1.0)) if bp_sims else np.nan

    out = {
        "MeanShiftZ_mean": float(np.clip(mean_shift_score, 0.0, 1.0)),
        "ERR_realism_score01_avg": float(np.clip(err_score, 0.0, 1.0)),
        "QNT_realism_score01_avg": float(np.clip(qnt_score, 0.0, 1.0)),
        "FC_core_score01_avg": float(np.clip(fc_score, 0.0, 1.0)) if np.isfinite(fc_score) else np.nan,
        "PCA_comp_score01_avg": float(np.clip(pca_score, 0.0, 1.0)) if np.isfinite(pca_score) else np.nan,
        "GRAPH_core_score01_avg": float(np.clip(graph_score, 0.0, 1.0)) if np.isfinite(graph_score) else np.nan,
        "Bandpower_score01_avg": float(np.clip(bandpower_score, 0.0, 1.0)) if np.isfinite(bandpower_score) else np.nan,
    }
    return out


def _build_full_metrics_from_env(
    env: dict,
    *,
    skip_crosscorr: bool,
    skip_cca: bool,
    abs_channels: Optional[dict] = None,
) -> tuple[dict, dict, float]:
    dist_out = env.get("dist_out", {})
    mean_results = env.get("mean_results", {})
    results_mi = env.get("results_mi", {})
    err_out = env.get("err_out", {})
    results_qnt = env.get("results_qnt", {})
    pca_realism_v1 = env.get("pca_realism_v1", {})
    fc_realism_summary = env.get("fc_realism_summary", {})
    autocorr_realism_summary = env.get("autocorr_realism_summary", {})
    crosscorr_realism_summary = env.get("crosscorr_realism_summary", {})
    moments_realism_summary = env.get("moments_realism_summary", {})
    graph_results = env.get("graph_results", {})
    cca_realism_summary = env.get("cca_realism_summary", {})
    mani_out = env.get("mani_out", {})
    bandpower_global_realism = env.get("bandpower_global_realism", {})

    _jsd_mean = _get(dist_out, ["summary", "JSD_worstq", "score01_mean"])
    _jsd_q10 = _get(dist_out, ["summary", "JSD_worstq", "score01_q10_strict"])
    JSD_score01 = _avg_mean_strict(_jsd_mean, _jsd_q10)

    _kl_mean = _get(dist_out, ["summary", "KL_geo", "mean"])
    _kl_q10 = _get(dist_out, ["summary", "KL_geo", "q10_strict"])
    KL_score01 = _avg_mean_strict(_kl_mean, _kl_q10)

    _w1_mean = _get(dist_out, ["summary", "W1_mean", "score01_mean"])
    _w1_q10 = _get(dist_out, ["summary", "W1_mean", "score01_q10_strict"])
    W1n_score01 = _avg_mean_strict(_w1_mean, _w1_q10)

    MeanShiftZ_mean = mean_results.get("final_scalar", np.nan)

    _qnt_scores = results_qnt.get("scores_for_composite", {})
    _qnt_tail_mean = _qnt_scores.get("QNT_tail_topq_z_score01_mean", np.nan)
    _qnt_tail_q10 = _qnt_scores.get("QNT_tail_topq_z_score01_q10", np.nan)
    QNT_tail_score01 = _avg_mean_strict(_qnt_tail_mean, _qnt_tail_q10)

    _qnt_full_mean = _qnt_scores.get("QNT_full_topq_z_score01_mean", np.nan)
    _qnt_full_q10 = _qnt_scores.get("QNT_full_topq_z_score01_q10", np.nan)
    QNT_full_score01 = _avg_mean_strict(_qnt_full_mean, _qnt_full_q10)

    _mi_scores = results_mi.get("scores_for_composite", {})
    _mi_mean = _mi_scores.get("MI_mean_z_score01_mean", np.nan)
    _mi_q10 = _mi_scores.get("MI_mean_z_score01_q10", np.nan)
    MI_score01 = _avg_mean_strict(_mi_mean, _mi_q10)

    _err_scores = err_out.get("scores_for_composite", {})
    _nrmse_mean = _err_scores.get("ERR_nRMSE_topq_z_score01_mean", np.nan)
    _nrmse_q10 = _err_scores.get("ERR_nRMSE_topq_z_score01_q10", np.nan)
    ERR_nRMSE_score01 = _avg_mean_strict(_nrmse_mean, _nrmse_q10)

    _nmae_mean = _err_scores.get("ERR_nMAE_topq_z_score01_mean", np.nan)
    _nmae_q10 = _err_scores.get("ERR_nMAE_topq_z_score01_q10", np.nan)
    ERR_nMAE_score01 = _avg_mean_strict(_nmae_mean, _nmae_q10)

    _pca_scores = pca_realism_v1.get("scores_for_composite", {})
    _pca_mean = _pca_scores.get("PCA_comp_z_score01_mean", np.nan)
    _pca_q10 = _pca_scores.get("PCA_comp_z_score01_q10", np.nan)
    PCA_score01 = _avg_mean_strict(_pca_mean, _pca_q10)

    _fc_scores = fc_realism_summary.get("scores_for_composite", {})
    _fc_mean = _fc_scores.get("FC_core_score01_mean", np.nan)
    _fc_q10 = _fc_scores.get("FC_core_score01_q10", np.nan)
    FC_score01 = _avg_mean_strict(_fc_mean, _fc_q10)

    _auto_scalars = autocorr_realism_summary.get("scalars", {})
    _auto_mean = _auto_scalars.get("AUTO_core_score01_mean", np.nan)
    _auto_q10 = _auto_scalars.get("AUTO_core_score01_q10", np.nan)
    AUTO_score01 = _avg_mean_strict(_auto_mean, _auto_q10)

    _cc_scalars = crosscorr_realism_summary.get("scalars", {})
    _cc_mean = _cc_scalars.get("CC_core_score01_mean", np.nan)
    _cc_q10 = _cc_scalars.get("CC_core_score01_q10", np.nan)
    CC_score01 = _avg_mean_strict(_cc_mean, _cc_q10) if not skip_crosscorr else np.nan

    _mom_scalars = moments_realism_summary.get("scalars", {})
    _mom_mean = _mom_scalars.get("MOM_core_score01_mean", np.nan)
    _mom_q10 = _mom_scalars.get("MOM_core_score01_q10", np.nan)
    MOM_score01 = _avg_mean_strict(_mom_mean, _mom_q10)

    _graph_scalars = graph_results.get("scalars", {})
    _graph_mean = _graph_scalars.get("GRAPH_core_score01_mean", np.nan)
    _graph_q10 = _graph_scalars.get("GRAPH_core_score01_q10", np.nan)
    GRAPH_score01 = _avg_mean_strict(_graph_mean, _graph_q10)

    _cca_scalars = cca_realism_summary.get("scalars", {})
    _cca_mean = _cca_scalars.get("CCA_core_score01_mean", np.nan)
    _cca_q10 = _cca_scalars.get("CCA_core_score01_q10", np.nan)
    CCA_score01 = _avg_mean_strict(_cca_mean, _cca_q10) if not skip_cca else np.nan

    _mani_scalars = mani_out.get("scalars", {})
    _mani_mean = _mani_scalars.get("MANI_core_score01_mean", np.nan)
    _mani_q10 = _mani_scalars.get("MANI_core_score01_q10", np.nan)
    MANI_score01 = _avg_mean_strict(_mani_mean, _mani_q10)

    _mani_df = mani_out.get("df", None)

    def _mani_component_avg(df, col):
        if df is None or col not in df:
            return np.nan
        vals = df[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.nan
        mean = float(np.mean(vals))
        q10 = float(np.quantile(vals, 0.10))
        return _avg_mean_strict(mean, q10)

    MANI_s_knn = _mani_component_avg(_mani_df, "s_knn")
    MANI_s_spec = _mani_component_avg(_mani_df, "s_spec")
    MANI_s_proc = _mani_component_avg(_mani_df, "s_proc")
    MANI_s_geo = _mani_component_avg(_mani_df, "s_geo")

    _bp_seq = bandpower_global_realism.get("per_seq_diag", {}).get("score01_seq", None)
    if _bp_seq is not None:
        _bp_seq = np.asarray(_bp_seq, dtype=np.float64)
        _bp_seq = _bp_seq[np.isfinite(_bp_seq)]
        _bp_mean = float(np.mean(_bp_seq)) if _bp_seq.size else np.nan
        _bp_q10 = float(np.quantile(_bp_seq, 0.10)) if _bp_seq.size else np.nan
    else:
        _bp_mean = np.nan
        _bp_q10 = np.nan
    Bandpower_score01 = _avg_mean_strict(_bp_mean, _bp_q10)

    metrics = {
        "JSD_worstq_score01_avg": float(JSD_score01),
        "KL_geo_score01_avg": float(KL_score01),
        "W1n_mean_score01_avg": float(W1n_score01),
        "MeanShiftZ_mean": float(MeanShiftZ_mean),
        "QNT_tail_score01_avg": float(QNT_tail_score01),
        "QNT_full_score01_avg": float(QNT_full_score01),
        "MI_mean_score01_avg": float(MI_score01),
        "ERR_nRMSE_score01_avg": float(ERR_nRMSE_score01),
        "ERR_nMAE_score01_avg": float(ERR_nMAE_score01),
        "AUTO_core_score01_avg": float(AUTO_score01),
        "CC_core_score01_avg": float(CC_score01),
        "Bandpower_score01_avg": float(Bandpower_score01),
        "FC_core_score01_avg": float(FC_score01),
        "GRAPH_core_score01_avg": float(GRAPH_score01),
        "PCA_comp_score01_avg": float(PCA_score01),
        "CCA_core_score01_avg": float(CCA_score01),
        "MANI_core_score01_avg": float(MANI_score01),
        "MOM_core_score01_avg": float(MOM_score01),
        "MANI_s_knn_score01_avg": float(MANI_s_knn),
        "MANI_s_spec_score01_avg": float(MANI_s_spec),
        "MANI_s_proc_score01_avg": float(MANI_s_proc),
        "MANI_s_geo_score01_avg": float(MANI_s_geo),
    }

    if skip_crosscorr:
        metrics.pop("CC_core_score01_avg", None)
    if skip_cca:
        metrics.pop("CCA_core_score01_avg", None)

    # Monotonic v2 blend: combine notebook realism (legacy) with absolute fidelity channels.
    abs_channels = abs_channels or {}
    legacy_family = {
        "MeanShiftZ_mean": float(MeanShiftZ_mean),
        "ERR_realism_score01_avg": _gmean([ERR_nRMSE_score01, ERR_nMAE_score01]),
        "QNT_realism_score01_avg": _gmean([QNT_tail_score01, QNT_full_score01]),
        "FC_core_score01_avg": float(FC_score01),
        "PCA_comp_score01_avg": float(PCA_score01),
        "GRAPH_core_score01_avg": float(GRAPH_score01),
        "Bandpower_score01_avg": float(Bandpower_score01),
    }
    v2_family = {
        k: _blend_legacy_absolute(legacy_family.get(k, np.nan), abs_channels.get(k, np.nan))
        for k in legacy_family
    }
    err_v2 = float(v2_family["ERR_realism_score01_avg"])
    qnt_v2 = float(v2_family["QNT_realism_score01_avg"])
    mean_v2 = float(v2_family["MeanShiftZ_mean"])
    fc_v2 = float(v2_family["FC_core_score01_avg"])
    pca_v2 = float(v2_family["PCA_comp_score01_avg"])
    graph_v2 = float(v2_family["GRAPH_core_score01_avg"])
    bandpower_v2 = float(v2_family["Bandpower_score01_avg"])

    metrics["MeanShiftZ_mean_legacy"] = float(MeanShiftZ_mean)
    metrics["ERR_realism_score01_avg_legacy"] = float(legacy_family["ERR_realism_score01_avg"])
    metrics["QNT_realism_score01_avg_legacy"] = float(legacy_family["QNT_realism_score01_avg"])
    metrics["FC_core_score01_avg_legacy"] = float(FC_score01)
    metrics["PCA_comp_score01_avg_legacy"] = float(PCA_score01)
    metrics["GRAPH_core_score01_avg_legacy"] = float(GRAPH_score01)
    metrics["Bandpower_score01_avg_legacy"] = float(Bandpower_score01)
    for k, v in abs_channels.items():
        metrics[f"{k}_absolute"] = float(v)

    # Expose v2 family channels.
    metrics["MeanShiftZ_mean"] = mean_v2
    metrics["ERR_realism_score01_avg"] = err_v2
    metrics["QNT_realism_score01_avg"] = qnt_v2
    metrics["FC_core_score01_avg"] = fc_v2
    metrics["PCA_comp_score01_avg"] = pca_v2
    metrics["GRAPH_core_score01_avg"] = graph_v2
    metrics["Bandpower_score01_avg"] = bandpower_v2
    # Apply v2 blend back to detailed keys used in legacy dashboards.
    metrics["ERR_nRMSE_score01_avg"] = err_v2
    metrics["ERR_nMAE_score01_avg"] = err_v2
    metrics["QNT_tail_score01_avg"] = qnt_v2
    metrics["QNT_full_score01_avg"] = qnt_v2

    buckets = {
        "distribution": _gmean([
            JSD_score01,
            KL_score01,
            W1n_score01,
            mean_v2,
            qnt_v2,
            qnt_v2,
        ]),
        "dependence": _gmean([
            MI_score01,
        ]),
        "point_error": _gmean([
            err_v2,
            err_v2,
        ]),
        "temporal_dynamics": _gmean([
            AUTO_score01,
            CC_score01,
            bandpower_v2,
            MANI_score01,
            CCA_score01,
        ]),
        "inter_region_structure": _gmean([
            fc_v2,
            graph_v2,
            pca_v2,
        ]),
        "higher_order": _gmean([
            MOM_score01,
        ]),
        "manifold_components": _gmean([
            MANI_s_knn,
            MANI_s_spec,
            MANI_s_proc,
            MANI_s_geo,
        ]),
    }

    weights = {
        "distribution": 0.22,
        "dependence": 0.10,
        "point_error": 0.10,
        "temporal_dynamics": 0.22,
        "inter_region_structure": 0.22,
        "higher_order": 0.07,
        "manifold_components": 0.07,
    }

    composite_legacy = _weighted_gmean(
        {
            "distribution": _gmean([JSD_score01, KL_score01, W1n_score01, MeanShiftZ_mean, QNT_tail_score01, QNT_full_score01]),
            "dependence": _gmean([MI_score01]),
            "point_error": _gmean([ERR_nRMSE_score01, ERR_nMAE_score01]),
            "temporal_dynamics": _gmean([AUTO_score01, CC_score01, Bandpower_score01, MANI_score01, CCA_score01]),
            "inter_region_structure": _gmean([FC_score01, GRAPH_score01, PCA_score01]),
            "higher_order": _gmean([MOM_score01]),
            "manifold_components": _gmean([MANI_s_knn, MANI_s_spec, MANI_s_proc, MANI_s_geo]),
        },
        weights,
    )
    v3_family, composite_v3 = _strict_multiplicative_v3(v2_family, MONOTONIC_V2_FAMILY_WEIGHTS)
    composite_v2 = _weighted_gmean(v2_family, MONOTONIC_V2_FAMILY_WEIGHTS)
    for k, v in v3_family.items():
        metrics[f"{k}_v3"] = float(v)
    metrics["legacy_composite_score"] = float(composite_legacy)
    metrics["monotonic_v2_composite_score"] = float(composite_v2)
    metrics["multiplicative_v3_composite_score"] = float(composite_v3)
    metrics["monotonic_v3_composite_score"] = float(composite_v3)
    return metrics, buckets, composite_v3


def _build_instant_metrics_from_env(env: dict, *, abs_channels: Optional[dict] = None) -> tuple[dict, float]:
    mean_results = env.get("mean_results", {})
    err_out = env.get("err_out", {})
    results_qnt = env.get("results_qnt", {})
    pca_realism_v1 = env.get("pca_realism_v1", {})
    fc_realism_summary = env.get("fc_realism_summary", {})
    graph_results = env.get("graph_results", {})
    bandpower_global_realism = env.get("bandpower_global_realism", {})

    MeanShiftZ_mean = mean_results.get("final_scalar", np.nan)

    _err_scores = err_out.get("scores_for_composite", {})
    _nrmse_mean = _err_scores.get("ERR_nRMSE_topq_z_score01_mean", np.nan)
    _nrmse_q10 = _err_scores.get("ERR_nRMSE_topq_z_score01_q10", np.nan)
    ERR_nRMSE_score01 = _avg_mean_strict(_nrmse_mean, _nrmse_q10)

    _nmae_mean = _err_scores.get("ERR_nMAE_topq_z_score01_mean", np.nan)
    _nmae_q10 = _err_scores.get("ERR_nMAE_topq_z_score01_q10", np.nan)
    ERR_nMAE_score01 = _avg_mean_strict(_nmae_mean, _nmae_q10)

    ERR_score01 = _gmean([ERR_nRMSE_score01, ERR_nMAE_score01])

    _qnt_scores = results_qnt.get("scores_for_composite", {})
    _qnt_tail_mean = _qnt_scores.get("QNT_tail_topq_z_score01_mean", np.nan)
    _qnt_tail_q10 = _qnt_scores.get("QNT_tail_topq_z_score01_q10", np.nan)
    QNT_tail_score01 = _avg_mean_strict(_qnt_tail_mean, _qnt_tail_q10)

    _qnt_full_mean = _qnt_scores.get("QNT_full_topq_z_score01_mean", np.nan)
    _qnt_full_q10 = _qnt_scores.get("QNT_full_topq_z_score01_q10", np.nan)
    QNT_full_score01 = _avg_mean_strict(_qnt_full_mean, _qnt_full_q10)

    QNT_score01 = _gmean([QNT_tail_score01, QNT_full_score01])

    _pca_scores = pca_realism_v1.get("scores_for_composite", {})
    _pca_mean = _pca_scores.get("PCA_comp_z_score01_mean", np.nan)
    _pca_q10 = _pca_scores.get("PCA_comp_z_score01_q10", np.nan)
    PCA_score01 = _avg_mean_strict(_pca_mean, _pca_q10)

    _fc_scores = fc_realism_summary.get("scores_for_composite", {})
    _fc_mean = _fc_scores.get("FC_core_score01_mean", np.nan)
    _fc_q10 = _fc_scores.get("FC_core_score01_q10", np.nan)
    FC_score01 = _avg_mean_strict(_fc_mean, _fc_q10)

    _graph_scalars = graph_results.get("scalars", {})
    _graph_mean = _graph_scalars.get("GRAPH_core_score01_mean", np.nan)
    _graph_q10 = _graph_scalars.get("GRAPH_core_score01_q10", np.nan)
    GRAPH_score01 = _avg_mean_strict(_graph_mean, _graph_q10)

    _bp_seq = bandpower_global_realism.get("per_seq_diag", {}).get("score01_seq", None)
    if _bp_seq is not None:
        _bp_seq = np.asarray(_bp_seq, dtype=np.float64)
        _bp_seq = _bp_seq[np.isfinite(_bp_seq)]
        _bp_mean = float(np.mean(_bp_seq)) if _bp_seq.size else np.nan
        _bp_q10 = float(np.quantile(_bp_seq, 0.10)) if _bp_seq.size else np.nan
    else:
        _bp_mean = np.nan
        _bp_q10 = np.nan
    Bandpower_score01 = _avg_mean_strict(_bp_mean, _bp_q10)

    legacy = {
        "MeanShiftZ_mean": float(MeanShiftZ_mean),
        "ERR_realism_score01_avg": float(ERR_score01),
        "QNT_realism_score01_avg": float(QNT_score01),
        "FC_core_score01_avg": float(FC_score01),
        "PCA_comp_score01_avg": float(PCA_score01),
        "GRAPH_core_score01_avg": float(GRAPH_score01),
        "Bandpower_score01_avg": float(Bandpower_score01),
    }
    abs_channels = abs_channels or {}
    metrics = {
        k: _blend_legacy_absolute(legacy.get(k, np.nan), abs_channels.get(k, np.nan))
        for k in legacy
    }
    for k, v in legacy.items():
        metrics[f"{k}_legacy"] = float(v)
    for k, v in abs_channels.items():
        metrics[f"{k}_absolute"] = float(v)

    composite_legacy = _weighted_gmean(legacy, {k: 1.0 for k in legacy})
    family_v2 = {k: metrics[k] for k in legacy}
    v3_family, composite_v3 = _strict_multiplicative_v3(family_v2, MONOTONIC_V2_FAMILY_WEIGHTS)
    composite_v2 = _weighted_gmean(family_v2, MONOTONIC_V2_FAMILY_WEIGHTS)
    for k, v in v3_family.items():
        metrics[f"{k}_v3"] = float(v)
    metrics["legacy_composite_score"] = float(composite_legacy)
    metrics["monotonic_v2_composite_score"] = float(composite_v2)
    metrics["multiplicative_v3_composite_score"] = float(composite_v3)
    metrics["monotonic_v3_composite_score"] = float(composite_v3)
    return metrics, composite_v3


def _compute_scores_from_arrays(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    region_names: Optional[list[str]] = None,
    ddconfig_path: Optional[Path] = None,
) -> Dict[str, float]:
    if gt.ndim != 3 or pred.ndim != 3:
        raise ValueError("Expected gt/pred arrays with shape [n_seq, n_time, n_reg].")
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} vs {pred.shape}")
    n_seq, n_time, n_reg = gt.shape
    region_names = region_names or [f"R{i}" for i in range(n_reg)]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        gt_path = tmpdir_path / "gt.csv"
        pr_path = tmpdir_path / "pred.csv"

        seq_ids = np.repeat(np.arange(n_seq), n_time)
        item_pos = np.tile(np.arange(n_time), n_seq)
        gt_df = pd.DataFrame(gt.reshape(-1, n_reg), columns=region_names)
        gt_df.insert(0, "itemPosition", item_pos)
        gt_df.insert(0, "sequenceId", seq_ids)
        gt_df.to_csv(gt_path, index=False)

        pr_df = pd.DataFrame(pred.reshape(-1, n_reg), columns=region_names)
        pr_df.index = seq_ids
        pr_df.to_csv(pr_path)

        _, scores = _run_active_neuro_notebook(pr_path, gt_path, ddconfig_path=ddconfig_path)
        return scores


def _compute_neuro_scores_from_script(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    ddconfig_path: Optional[Path] = None,
    mode: str = "full",
    skip_crosscorr: bool = True,
    skip_cca: bool = True,
    disable_runtime_limits: bool = True,
) -> Dict[str, float]:
    try:
        gt_arr, pred_arr, _ = _load_and_align(Path(predictions_csv), Path(ground_truth_csv))
        abs_channels = _compute_absolute_fidelity_channels(gt_arr, pred_arr)
    except Exception:
        abs_channels = {}

    env = _run_neurobench_script(
        Path(predictions_csv),
        Path(ground_truth_csv),
        ddconfig_path=ddconfig_path,
        mode=mode,
        skip_crosscorr=skip_crosscorr,
        skip_cca=skip_cca,
        enable_plots=False,
        disable_runtime_limits=disable_runtime_limits,
    )
    if mode == "instant":
        metrics, composite = _build_instant_metrics_from_env(env, abs_channels=abs_channels)
        return _flatten_scores(metrics, {}, composite)
    metrics, buckets, composite = _build_full_metrics_from_env(
        env,
        skip_crosscorr=skip_crosscorr,
        skip_cca=skip_cca,
        abs_channels=abs_channels,
    )
    return _flatten_scores(metrics, buckets, composite)


def compute_neuro_scores(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    per_sequence_stats: bool = False,
    neuro_cols: Optional[list[str]] = None,
    ddconfig_path: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Compute neural plausibility scores using the bundled benchmark notebook logic.
    """
    if per_sequence_stats:
        raise ValueError("per_sequence_stats is not supported for notebook-based core scores.")

    if neuro_cols:
        gt, pred, overlap = _load_and_align(
            Path(predictions_csv),
            Path(ground_truth_csv),
            neuro_cols=neuro_cols,
        )
        return _compute_scores_from_arrays(gt, pred, region_names=overlap, ddconfig_path=ddconfig_path)

    _, scores = _run_active_neuro_notebook(
        Path(predictions_csv),
        Path(ground_truth_csv),
        ddconfig_path=ddconfig_path,
    )
    return scores


def compute_instant_neuro_scores(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    ddconfig_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Compute a trimmed notebook-derived neuro score subset.
    """
    return _run_active_neuro_instant_notebook(
        Path(predictions_csv),
        Path(ground_truth_csv),
        ddconfig_path=ddconfig_path,
    )


def _timestamped_outdir(base: Optional[Path] = None, stem: Optional[str] = None) -> Path:
    base = Path(base) if base is not None else Path.cwd() / "outputs"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if stem:
        outdir = base / f"{stem}-analysis-{ts}"
    else:
        outdir = base / f"neuro-analysis-{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def run_neuro_full_analysis(
    predictions_csv: Path,
    ground_truth_csv: Path,
    ddconfig_path: Path,
    *,
    output_root: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Execute the active neuro notebook headlessly, save figures, and export notebook-derived scores.
    """
    preds_path = Path(predictions_csv)
    outdir = _timestamped_outdir(output_root, stem=preds_path.stem)
    wrapped_notebook = outdir / "wrapped_neuro_metrics.ipynb"
    _, scores = _run_active_neuro_notebook(
        Path(predictions_csv),
        Path(ground_truth_csv),
        ddconfig_path=ddconfig_path,
        save_plots_dir=outdir,
        save_wrapped_notebook=wrapped_notebook,
    )

    scores_path = outdir / "scores.json"
    scores_path.write_text(json.dumps(scores, indent=2))
    return {
        "output_dir": outdir,
        "scores": scores,
        "scores_path": scores_path,
        "wrapped_notebook": wrapped_notebook,
    }
