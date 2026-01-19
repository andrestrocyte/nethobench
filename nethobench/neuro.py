from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def _compute_scores_from_arrays(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    rng_seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(rng_seed)
    gt_flat = gt.reshape(-1, gt.shape[-1])
    pred_flat = pred.reshape(-1, pred.shape[-1])

    # Mean difference of means (raw)
    gt_mean = np.nanmean(gt_flat, axis=0)
    pr_mean = np.nanmean(pred_flat, axis=0)
    valid = np.isfinite(gt_mean) & np.isfinite(pr_mean)
    mean_diff = float(np.mean(np.abs(gt_mean[valid] - pr_mean[valid]))) if np.any(valid) else 0.0
    mean_diff_score = 1.0 / (1.0 + mean_diff * 0.1)

    # KL similarity per region
    kl_vals = []
    for idx in range(gt_flat.shape[1]):
        g = gt_flat[:, idx]
        p = pred_flat[:, idx]
        g = g[np.isfinite(g)]
        p = p[np.isfinite(p)]
        if g.size < 10 or p.size < 10:
            continue
        lo = min(g.min(), p.min())
        hi = max(g.max(), p.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            continue
        edges = np.linspace(lo, hi, 60)
        Pg, _ = np.histogram(g, bins=edges, density=True)
        Pp, _ = np.histogram(p, bins=edges, density=True)
        eps = 1e-12
        Pg = (Pg + eps) / (Pg + eps).sum()
        Pp = (Pp + eps) / (Pp + eps).sum()
        kl = 0.5 * (stats.entropy(Pg, Pp) + stats.entropy(Pp, Pg))
        kl_vals.append(kl)
    kl_score = 1.0 / (1.0 + float(np.mean(kl_vals))) if kl_vals else 1.0

    # Correlation structure similarity (corr-of-corrs)
    corr_scores = []
    for seq_idx in range(gt.shape[0]):
        gt_seq = gt[seq_idx]
        pr_seq = pred[seq_idx]
        if np.allclose(np.nanstd(gt_seq, axis=0), 0) or np.allclose(np.nanstd(pr_seq, axis=0), 0):
            continue
        Cg = np.corrcoef(gt_seq, rowvar=False)
        Cp = np.corrcoef(pr_seq, rowvar=False)
        iu = np.triu_indices_from(Cg, k=1)
        valid = np.isfinite(Cg[iu]) & np.isfinite(Cp[iu])
        if np.sum(valid) < 3:
            continue
        r = np.corrcoef(Cg[iu][valid], Cp[iu][valid])[0, 1]
        corr_scores.append(r)
    corr_score = ((float(np.nanmean(corr_scores)) + 1.0) / 2.0) if corr_scores else 0.5

    # Dimensionality via PCA spectra Wasserstein
    def spectrum(X: np.ndarray, n: int = 20) -> np.ndarray:
        idx = rng.choice(X.shape[0], size=min(X.shape[0], 20000), replace=False)
        Xs = X[idx] - X[idx].mean(axis=0)
        U, S, _ = np.linalg.svd(Xs, full_matrices=False)
        eig = (S ** 2) / max(1, len(Xs) - 1)
        spec = eig / (eig.sum() + 1e-12)
        return spec[:n]

    spec_gt = spectrum(gt_flat)
    spec_pr = spectrum(pred_flat)
    spec_dist = float(stats.wasserstein_distance(spec_gt, spec_pr))
    dim_score = 1.0 - min(spec_dist / 0.5, 1.0)

    # Graph overlap (correlation threshold)
    thresh = 0.5
    adj_gt = (np.abs(np.corrcoef(gt_flat, rowvar=False)) > thresh).astype(int)
    adj_pr = (np.abs(np.corrcoef(pred_flat, rowvar=False)) > thresh).astype(int)
    np.fill_diagonal(adj_gt, 0)
    np.fill_diagonal(adj_pr, 0)
    overlap = np.logical_and(adj_gt, adj_pr).sum()
    union = np.logical_or(adj_gt, adj_pr).sum()
    graph_score = float(overlap / union) if union else 1.0

    # Autocorrelation similarity
    def mean_autocorr(arr: np.ndarray, max_lag: int = 60) -> np.ndarray:
        curves = []
        for region_idx in range(arr.shape[-1]):
            series = arr[:, :, region_idx].reshape(-1)
            if np.allclose(series.std(), 0):
                continue
            series = (series - series.mean()) / (series.std() + 1e-9)
            corr = np.correlate(series, series, mode="full") / len(series)
            mid = len(corr) // 2
            curves.append(corr[mid: mid + max_lag])
        return np.mean(curves, axis=0) if curves else np.zeros(max_lag)

    ac_gt = mean_autocorr(gt)
    ac_pr = mean_autocorr(pred)
    ac_score = (float(np.corrcoef(ac_gt, ac_pr)[0, 1]) + 1.0) / 2.0 if ac_gt.size and ac_pr.size else 0.5

    # Welch PSD similarity
    def welch_similarity(gt_arr: np.ndarray, pr_arr: np.ndarray, fs: float = 1.0) -> float:
        sims = []
        for region_idx in range(gt_arr.shape[-1]):
            gt_vals = gt_arr[:, :, region_idx].reshape(-1)
            pr_vals = pr_arr[:, :, region_idx].reshape(-1)
            mask = np.isfinite(gt_vals) & np.isfinite(pr_vals)
            gt_vals = gt_vals[mask]
            pr_vals = pr_vals[mask]
            if gt_vals.size == 0 or pr_vals.size == 0:
                continue
            nperseg = min(256, gt_vals.size, pr_vals.size)
            freqs, Pxx_gt = signal.welch(gt_vals, fs=fs, nperseg=nperseg)
            _, Pxx_pr = signal.welch(pr_vals, fs=fs, nperseg=nperseg)
            denom = np.linalg.norm(Pxx_gt) * np.linalg.norm(Pxx_pr) + 1e-12
            cos_sim = float(np.dot(Pxx_gt, Pxx_pr) / denom)
            sims.append(np.clip(cos_sim, 0.0, 1.0))
        return float(np.nanmean(sims)) if sims else 0.5

    psd_similarity_score = welch_similarity(gt, pred)

    scores = {
        "mean_diff_score": mean_diff_score,
        "kl_score": kl_score,
        "correlation_score": corr_score,
        "dimensionality_score": dim_score,
        "graph_score": graph_score,
        "autocorr_score": ac_score,
        "psd_similarity_score": psd_similarity_score,
    }

    composite = 1.0
    for value in scores.values():
        composite *= max(0.0, min(1.0, float(value)))
    scores["composite_score"] = float(composite)
    return scores


def compute_neuro_scores(
    predictions_csv: Path,
    ground_truth_csv: Path,
    *,
    per_sequence_stats: bool = False,
    neuro_cols: Optional[list[str]] = None,
) -> Dict[str, object]:
    """
    Compute condensed neural plausibility scores (NeuroBench-style).
    """
    gt, pred, _ = _load_and_align(Path(predictions_csv), Path(ground_truth_csv), neuro_cols=neuro_cols)
    pooled_scores = _compute_scores_from_arrays(gt, pred)

    if not per_sequence_stats:
        return pooled_scores

    per_seq_scores = []
    for seq_idx in range(gt.shape[0]):
        seq_scores = _compute_scores_from_arrays(
            gt[seq_idx : seq_idx + 1],
            pred[seq_idx : seq_idx + 1],
        )
        per_seq_scores.append({"sequence_index": int(seq_idx), **seq_scores})

    metric_names = pooled_scores.keys()
    mean_scores = {}
    std_scores = {}
    for name in metric_names:
        vals = np.array([seq[name] for seq in per_seq_scores], dtype=float)
        mean_scores[name] = float(np.nanmean(vals))
        std_scores[name] = float(np.nanstd(vals))

    return {
        "pooled_scores": pooled_scores,
        "per_sequence_mean": mean_scores,
        "per_sequence_std": std_scores,
        "per_sequence_scores": per_seq_scores,
    }


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
    preds_path = Path(predictions_csv)
    outdir = _timestamped_outdir(output_root, stem=preds_path.stem)

    # Monkey-patch plt.show to save and close instead of displaying.
    plot_counter = {"n": 0}
    orig_show = plt.show

    def saving_show(*args, **kwargs):
        figs = [plt.figure(num) for num in plt.get_fignums()]
        for fig in figs:
            plot_counter["n"] += 1
            fig.savefig(outdir / f"figure_{plot_counter['n']:03d}.png", dpi=200, bbox_inches="tight")
        plt.close("all")

    plt.show = saving_show

    from .analysis.final_implementation_benchmark import neurobench_analysis

    neurobench_analysis(str(predictions_csv), str(ground_truth_csv), str(ddconfig_path))

    # Restore show (best effort)
    plt.show = orig_show
    return {"output_dir": outdir}
