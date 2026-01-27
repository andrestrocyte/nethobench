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
import nbformat
from scipy import signal, stats

# --------------------------------------------------------------------------- #
# Loading helpers                                                             #
# --------------------------------------------------------------------------- #


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


def _exec_notebook_metrics(
    notebook_path: Path,
    predictions_csv: Path,
    ground_truth_csv: Path,
    ddconfig_path: Path,
    *,
    skip_heavy_diagnostics: bool = True,
) -> dict:
    nb = nbformat.read(notebook_path, as_version=4)
    env: dict = {"__name__": "__main__"}

    env["display"] = lambda *args, **kwargs: None
    exec(
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "plt.show = lambda *args, **kwargs: None\n",
        env,
    )

    pred_str = str(predictions_csv)
    gt_str = str(ground_truth_csv)
    dd_str = str(ddconfig_path)

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        if skip_heavy_diagnostics:
            src = cell.source
            # These cells are expensive and do not contribute to METRICS/BUCKETS.
            if (
                "UMAP baseline embedding" in src
                or "Trajectory similarity in embedding space" in src
                or "Advanced dynamical similarity metrics" in src
                or "Animated trajectory visualizer" in src
                or "Trajectory similarity permutation test" in src
            ):
                continue
        src_lines = []
        for line in cell.source.splitlines():
            stripped = line.strip()
            if stripped.startswith("%") or stripped.startswith("!"):
                continue
            if stripped.startswith("preds_fname ="):
                line = f'preds_fname = r\"{pred_str}\"'
            elif stripped.startswith("gt_fname ="):
                line = f'gt_fname = r\"{gt_str}\"'
            elif stripped.startswith("ddconfig_path ="):
                line = f'ddconfig_path = r\"{dd_str}\"'
            src_lines.append(line)
        if not src_lines:
            continue
        exec("\n".join(src_lines), env)

    return env


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


def _run_neurobench_script(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    ddconfig_path: Optional[Path] = None,
    mode: str = "full",
    skip_crosscorr: bool = True,
    skip_cca: bool = True,
    enable_plots: bool = False,
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


def _build_full_metrics_from_env(env: dict, *, skip_crosscorr: bool, skip_cca: bool) -> tuple[dict, dict, float]:
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

    buckets = {
        "distribution": _gmean([
            JSD_score01,
            KL_score01,
            W1n_score01,
            MeanShiftZ_mean,
            QNT_tail_score01,
            QNT_full_score01,
        ]),
        "dependence": _gmean([
            MI_score01,
        ]),
        "point_error": _gmean([
            ERR_nRMSE_score01,
            ERR_nMAE_score01,
        ]),
        "temporal_dynamics": _gmean([
            AUTO_score01,
            CC_score01,
            Bandpower_score01,
            MANI_score01,
            CCA_score01,
        ]),
        "inter_region_structure": _gmean([
            FC_score01,
            GRAPH_score01,
            PCA_score01,
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

    composite = _weighted_gmean(buckets, weights)
    return metrics, buckets, composite


def _build_instant_metrics_from_env(env: dict) -> tuple[dict, float]:
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

    metrics = {
        "MeanShiftZ_mean": float(MeanShiftZ_mean),
        "ERR_realism_score01_avg": float(ERR_score01),
        "QNT_realism_score01_avg": float(QNT_score01),
        "FC_core_score01_avg": float(FC_score01),
        "PCA_comp_score01_avg": float(PCA_score01),
        "GRAPH_core_score01_avg": float(GRAPH_score01),
        "Bandpower_score01_avg": float(Bandpower_score01),
    }

    composite = _weighted_gmean(metrics, {k: 1.0 for k in metrics})
    return metrics, composite


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

        return _compute_neuro_scores_from_script(pr_path, gt_path, ddconfig_path=ddconfig_path)


def _compute_neuro_scores_from_script(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    ddconfig_path: Optional[Path] = None,
    mode: str = "full",
    skip_crosscorr: bool = True,
    skip_cca: bool = True,
) -> Dict[str, float]:
    env = _run_neurobench_script(
        Path(predictions_csv),
        Path(ground_truth_csv),
        ddconfig_path=ddconfig_path,
        mode=mode,
        skip_crosscorr=skip_crosscorr,
        skip_cca=skip_cca,
        enable_plots=False,
    )
    if mode == "instant":
        metrics, composite = _build_instant_metrics_from_env(env)
        return _flatten_scores(metrics, {}, composite)
    metrics, buckets, composite = _build_full_metrics_from_env(
        env,
        skip_crosscorr=skip_crosscorr,
        skip_cca=skip_cca,
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

    return _compute_neuro_scores_from_script(
        Path(predictions_csv),
        Path(ground_truth_csv),
        ddconfig_path=ddconfig_path,
        mode="full",
        skip_crosscorr=True,
        skip_cca=True,
    )


def compute_instant_neuro_scores(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    ddconfig_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Compute a fast neuro composite using mean diff, error realism, quantile, FC, PCA, Jaccard, and Power.
    """
    return _compute_neuro_scores_from_script(
        Path(predictions_csv),
        Path(ground_truth_csv),
        ddconfig_path=ddconfig_path,
        mode="instant",
        skip_crosscorr=True,
        skip_cca=True,
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
    Execute the bundled NeuroBench full analysis (notebook-derived script) and save figures.
    """
    from nbconvert.preprocessors import ExecutePreprocessor
    import nbformat

    nb_path = Path(__file__).parent / "notebooks" / "final_implementation_benchmark.ipynb"
    if not nb_path.is_file():
        raise FileNotFoundError(f"Neuro notebook missing at {nb_path}")

    preds_path = Path(predictions_csv)
    outdir = _timestamped_outdir(output_root, stem=preds_path.stem)

    nb = nbformat.read(nb_path, as_version=4)

    # Patch notebook cell(s) that hardcode file paths.
    pred_str = str(predictions_csv)
    gt_str = str(ground_truth_csv)
    dd_str = str(ddconfig_path)
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        lines = cell.source.splitlines()
        changed = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("preds_fname ="):
                lines[i] = f'preds_fname = r"{pred_str}"'
                changed = True
            elif stripped.startswith("gt_fname ="):
                lines[i] = f'gt_fname = r"{gt_str}"'
                changed = True
            elif stripped.startswith("ddconfig_path ="):
                lines[i] = f'ddconfig_path = r"{dd_str}"'
                changed = True
        if changed:
            cell.source = "\n".join(lines)

    patch = f"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
outdir = Path(r\"{outdir}\")
plot_counter = {{'n': 0}}
orig_show = plt.show
def saving_show(*args, **kwargs):
    figs = [plt.figure(num) for num in plt.get_fignums()]
    for fig in figs:
        plot_counter['n'] += 1
        fig.savefig(outdir / f\"figure_{{plot_counter['n']:03d}}.png\", dpi=200, bbox_inches=\"tight\")
    plt.close('all')
plt.show = saving_show
"""
    nb.cells.insert(0, nbformat.v4.new_code_cell(patch))

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": nb_path.parent}})
    executed_path = outdir / "executed_neurobench.ipynb"
    with executed_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    return {"output_dir": outdir}
