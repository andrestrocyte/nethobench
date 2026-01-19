from IPython.display import display, HTML

def neurobench_analysis(preds_fname, gt_fname, ddconfig_path):
    
    
    
    
    import pandas as pd
    import numpy as np
    
    
    from matplotlib import pyplot as plt
    
    
    
    
    import pandas as pd
    import numpy as np
    import json
    
    
    preds_df = pd.read_csv(preds_fname, index_col=0)
    pred_region_names = preds_df.columns.tolist()
    pred_arr = preds_df.to_numpy()
    n_pred_trials = preds_df.index.max() + 1
    n_pred_time = pred_arr.shape[0] // n_pred_trials
    n_pred_dims = pred_arr.shape[1]
    data_predicted = pred_arr.reshape(n_pred_trials, n_pred_time, n_pred_dims)
    
    print("Pred region names:", pred_region_names)
    print("Pred shape:", data_predicted.shape)
    
    gt_df = pd.read_csv(gt_fname)
    gt_df = gt_df.sort_values(["sequenceId", "itemPosition"]).reset_index(drop=True)
    gt_regions = [c for c in gt_df.columns if c not in {"sequenceId", "itemPosition"}]
    seq_lengths = gt_df.groupby("sequenceId").size()
    if seq_lengths.nunique() != 1:
        raise ValueError(f"Secuencias con distinta longitud: {seq_lengths.describe()}")
    n_seq = seq_lengths.size
    n_time = int(seq_lengths.iloc[0])
    n_dims = len(gt_regions)
    gt_array = gt_df[gt_regions].to_numpy(dtype=np.float32).reshape(n_seq, n_time, n_dims)
    
    with open(ddconfig_path, "r") as fh:
        stats = json.load(fh)["selected_columns_statistics"]
    
    gt_array_denorm = gt_array.astype(np.float64, copy=True)
    
    print("GT region names:", gt_regions)
    print("GT shape (raw):", gt_array.shape)
    print("GT copy shape:", gt_array_denorm.shape)
    
    aligned_time = min(data_predicted.shape[1], gt_array.shape[1])
    if aligned_time < gt_array.shape[1]:
        gt_array = gt_array[:, :aligned_time, :]
        gt_array_denorm = gt_array_denorm[:, :aligned_time, :]
    if aligned_time < data_predicted.shape[1]:
        data_predicted = data_predicted[:, :aligned_time, :]
    
    print("Aligned timesteps:", aligned_time)
    print("Pred shape (aligned):", data_predicted.shape)
    print("GT shape (aligned):", gt_array.shape)
    print("GT raw copy shape (aligned):", gt_array_denorm.shape)
    
    
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    num_trials = 4
    pred_trials = np.linspace(0, data_predicted.shape[0] - 1, num_trials, dtype=int)
    gt_trials = np.linspace(0, gt_array_denorm.shape[0] - 1, num_trials, dtype=int)
    
    fig_pred, axes_pred = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes_pred = axes_pred.flatten()
    for ax, seq in zip(axes_pred, pred_trials):
        for r in range(data_predicted.shape[2]):
            ax.plot(data_predicted[seq, :, r], label=pred_region_names[r])
        ax.set_title(f"Predicted trial {seq}")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
    axes_pred[-1].set_xlabel("Step index")
    axes_pred[-2].set_xlabel("Step index")
    handles, labels = axes_pred[0].get_legend_handles_labels()
    fig_pred.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, 1.03))
    fig_pred.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    
    fig_gt, axes_gt = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes_gt = axes_gt.flatten()
    for ax, seq in zip(axes_gt, gt_trials):
        for r in range(gt_array_denorm.shape[2]):
            ax.plot(gt_array_denorm[seq, :, r], label=gt_regions[r])
        ax.set_title(f"GT trial {seq}")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
    axes_gt[-1].set_xlabel("Step index")
    axes_gt[-2].set_xlabel("Step index")
    handles_gt, labels_gt = axes_gt[0].get_legend_handles_labels()
    fig_gt.legend(handles_gt, labels_gt, loc="upper center", ncol=min(len(labels_gt), 4), bbox_to_anchor=(0.5, 1.03))
    fig_gt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    
    
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import entropy
    
    def _resample_pred_to_gt(gt_arr, pred_arr):
        gt_len = gt_arr.shape[1]
        pred_len = pred_arr.shape[1]
        if pred_len != gt_len and pred_len % gt_len == 0:
            factor = pred_len // gt_len
            pred_res = pred_arr.reshape(pred_arr.shape[0], gt_len, factor, pred_arr.shape[2]).mean(axis=2)
            gt_cmp = gt_arr
        elif pred_len != gt_len:
            min_len = min(gt_len, pred_len)
            gt_cmp = gt_arr[:, :min_len, :]
            pred_res = pred_arr[:, :min_len, :]
        else:
            gt_cmp = gt_arr
            pred_res = pred_arr
        return gt_cmp, pred_res
    
    gt_mom, pred_mom = _resample_pred_to_gt(gt_array_denorm, data_predicted)
    
    def plot_distributions_and_kl(gt_denorm, predictions, region_names,
                                  bins=60, clip_q=(0.001, 0.999)):
        """
        Compare marginal distributions of GT vs predictions.
        Subplot grid with one region per subplot.
        Returns KL values per region.
        """
        assert gt_denorm.ndim == 3
        assert predictions.ndim == 3
        assert gt_denorm.shape[2] == predictions.shape[2] == len(region_names)
    
        n_seq, n_sub, n_reg = gt_denorm.shape
        kl_per_region = []
    
        ncols = 4
        nrows = int(np.ceil(n_reg / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
        axes = axes.ravel()
    
        for r, name in enumerate(region_names):
            gt_vals = gt_denorm[:, :, r].ravel().astype(np.float64)
            pr_vals = predictions[:, :, r].ravel().astype(np.float64)
    
            lo = min(np.quantile(gt_vals, clip_q[0]), np.quantile(pr_vals, clip_q[0]))
            hi = max(np.quantile(gt_vals, clip_q[1]), np.quantile(pr_vals, clip_q[1]))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = gt_vals.min(), gt_vals.max()
                if lo == hi: hi = lo + 1e-6
    
            edges = np.linspace(lo, hi, bins)
    
            gt_hist, _ = np.histogram(gt_vals, bins=edges, density=True)
            pr_hist, _ = np.histogram(pr_vals, bins=edges, density=True)
    
            eps = 1e-12
            gt_hist = (gt_hist + eps) / (gt_hist + eps).sum()
            pr_hist = (pr_hist + eps) / (pr_hist + eps).sum()
    
            kl_sym = 0.5 * (entropy(gt_hist, pr_hist) + entropy(pr_hist, gt_hist))
            kl_per_region.append(float(kl_sym))
    
            ax = axes[r]
            ax.hist(gt_vals, bins=edges, alpha=0.5, label="GT (denorm)", density=True)
            ax.hist(pr_vals, bins=edges, alpha=0.5, label="Pred", density=True)
            ax.set_title(f"{name}\nKL={kl_sym:.3f}", fontsize=9)
            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
    
        for ax in axes[n_reg:]:
            ax.axis("off")
    
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle("GT vs Predicted Distributions per Region (Trimmed)", fontsize=14)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.show()
    
        print("Symmetric KL divergences per region:")
        for name, kl in zip(region_names, kl_per_region):
            print(f"  {name}: {kl:.3f}")
        print("Mean KL divergence:", float(np.mean(kl_per_region)))
    
        return {"kl_per_region": kl_per_region, "mean_kl": float(np.mean(kl_per_region))}
    
    
    
    res = plot_distributions_and_kl(gt_array_denorm, data_predicted, pred_region_names)
    
    
    
    
    import numpy as np
    
    def compute_region_means(gt_arr, pred_arr, region_names):
        """Compute and print mean values (20 decimals) per region."""
        assert gt_arr.shape[-1] == pred_arr.shape[-1] == len(region_names)
    
        gt_means, pred_means = [], []
        print("Per-region mean values (denormalized scale):")
        print("-" * 70)
        for r, name in enumerate(region_names):
            gt_vals = gt_arr[:, :, r].ravel().astype(np.float64)
            pr_vals = pred_arr[:, :, r].ravel().astype(np.float64)
    
            gt_mean = np.mean(gt_vals[np.isfinite(gt_vals)])
            pr_mean = np.mean(pr_vals[np.isfinite(pr_vals)])
            gt_means.append(gt_mean)
            pred_means.append(pr_mean)
    
            print(f"{name:30s} | GT μ = {gt_mean:.20f} | Pred μ = {pr_mean:.20f}")
    
        print("-" * 70)
        print(f"Global GT mean   : {np.mean(gt_means):.20f}")
        print(f"Global Pred mean : {np.mean(pred_means):.20f}")
    
        return np.array(gt_means), np.array(pred_means)
    
    gt_means, pred_means = compute_region_means(gt_array_denorm, data_predicted, pred_region_names)
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    diff = gt_means - pred_means
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(pred_region_names, diff, color='steelblue', alpha=0.8, edgecolor='black')
    
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    plt.xticks(rotation=90)
    plt.ylabel('Inference μ − Predicted μ', fontsize=12)
    plt.title('Mean Difference per Brain Area', fontsize=14)
    plt.tight_layout()
    
    for bar in bars:
        height = bar.get_height()
        if abs(height) > 1e-6:  # skip very small differences
            plt.text(bar.get_x() + bar.get_width()/2, height,
                     f"{height:.3f}", ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
    
    plt.show()
    
    
    
    
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.feature_selection import mutual_info_regression
    
    _gt_aligned, _pred_aligned = _resample_pred_to_gt(gt_array_denorm, data_predicted)
    num_seq, num_time, num_regions = _gt_aligned.shape
    
    mi_all = np.full((num_seq, num_regions), np.nan, dtype=np.float64)
    
    for seq_idx in range(num_seq):
        X_seq = _gt_aligned[seq_idx]
        Y_seq = _pred_aligned[seq_idx]
    
        mask = np.isfinite(X_seq).all(axis=1) & np.isfinite(Y_seq).all(axis=1)
        X_seq = X_seq[mask]
        Y_seq = Y_seq[mask]
    
        if X_seq.shape[0] < 20:  # too few samples
            continue
    
        for r in range(num_regions):
            try:
                mi_val = mutual_info_regression(
                    X_seq[:, [r]],  # GT for region r
                    Y_seq[:, r],    # Prediction for region r
                    discrete_features=False
                )[0]
                mi_all[seq_idx, r] = mi_val
            except Exception:
                mi_all[seq_idx, r] = np.nan
    
    mi_mean = np.nanmean(mi_all, axis=0)
    mi_std  = np.nanstd(mi_all, axis=0)
    mi_df = pd.DataFrame({
        'region': gt_regions,
        'mi_mean': mi_mean,
        'mi_std': mi_std
    }).set_index('region')
    
    display(mi_df.round(4).sort_values('mi_mean', ascending=False))
    
    plt.figure(figsize=(10, 4))
    plt.bar(mi_df.index, mi_df['mi_mean'], yerr=mi_df['mi_std'], 
            color='tab:purple', alpha=0.8, ecolor='black', capsize=3)
    plt.xticks(rotation=90)
    plt.ylabel('Mutual information (nats)')
    plt.title('GT vs prediction mutual information per region (per-sequence averaged)')
    plt.grid(alpha=0.2, axis='y')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(6, 3))
    plt.hist(np.nanmean(mi_all, axis=1), bins=20, color='tab:blue', alpha=0.7, edgecolor='k')
    plt.xlabel('Mean MI across regions')
    plt.ylabel('Count')
    plt.title('Distribution of per-sequence MI means')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("Interpretation:")
    print("- MI is now computed independently per sequence, avoiding mixed distributions.")
    print("- Bars show the average MI per region ± across-sequence variability.")
    print("- The histogram shows how much information each sequence carries overall.")
    
    
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    _gt_aligned, _pred_aligned = _resample_pred_to_gt(gt_array_denorm, data_predicted)
    residuals = _pred_aligned - _gt_aligned   # shape: (sequences, time, regions)
    n_regions = residuals.shape[2]
    
    residuals_flat = residuals.reshape(-1, n_regions)
    
    global_min, global_max = np.nanpercentile(residuals_flat, [1, 99])
    bins = np.linspace(global_min, global_max, 40)
    
    fig, axes = plt.subplots(
        nrows=n_regions, ncols=1,
        figsize=(8, 1.2 * n_regions),
        sharex=True
    )
    
    if n_regions == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        region_name = gt_regions[i]
        valid = np.isfinite(residuals_flat[:, i])
        ax.hist(
            residuals_flat[valid, i],
            bins=bins,
            color='steelblue',
            alpha=0.7,
            density=True
        )
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.set_ylabel(region_name, rotation=0, labelpad=40, ha='right', va='center')
        ax.grid(alpha=0.2)
    
    axes[-1].set_xlabel('Residual (Pred − GT)')
    fig.suptitle('Distribution of prediction residuals per region', y=0.995, fontsize=12)
    fig.tight_layout(h_pad=0.3)
    plt.show()
    
    
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    _gt_aligned, _pred_aligned = _resample_pred_to_gt(gt_array_denorm, data_predicted)
    residuals = _pred_aligned - _gt_aligned
    
    rmse_per_region = np.sqrt(np.nanmean(residuals ** 2, axis=(0, 1)))
    mae_per_region = np.nanmean(np.abs(residuals), axis=(0, 1))
    rmse_per_sequence = np.sqrt(np.nanmean(residuals ** 2, axis=(1, 2)))
    mae_per_sequence = np.nanmean(np.abs(residuals), axis=(1, 2))
    rmse_per_time = np.sqrt(np.nanmean(residuals ** 2, axis=(0, 2)))
    mae_per_time = np.nanmean(np.abs(residuals), axis=(0, 2))
    
    error_summary = pd.DataFrame({'rmse': rmse_per_region, 'mae': mae_per_region}, index=gt_regions)
    error_summary = error_summary.sort_values('rmse', ascending=False)
    display(error_summary.round(4))
    
    seq_summary = pd.DataFrame({'sequence_id': np.arange(len(rmse_per_sequence)), 'rmse': rmse_per_sequence, 'mae': mae_per_sequence})
    display(seq_summary.round(4))
    
    global_rmse = float(np.sqrt(np.nanmean(residuals ** 2)))
    global_mae = float(np.nanmean(np.abs(residuals)))
    print(f'Global RMSE: {global_rmse:.4f} | Global MAE: {global_mae:.4f}')
    
    fig, ax = plt.subplots(figsize=(10, 4))
    positions = np.arange(error_summary.shape[0])
    ax.bar(positions, error_summary['rmse'], alpha=0.8, label='RMSE')
    ax.plot(positions, error_summary['mae'], 'o-', color='tab:red', label='MAE')
    ax.set_xticks(positions)
    ax.set_xticklabels(error_summary.index, rotation=90)
    ax.set_ylabel('Error')
    ax.set_title('Per-region RMSE and MAE')
    ax.grid(alpha=0.2, axis='y')
    ax.legend()
    fig.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(seq_summary['sequence_id'], seq_summary['rmse'], 'o-', label='RMSE')
    ax.plot(seq_summary['sequence_id'], seq_summary['mae'], 'o-', label='MAE')
    ax.set_xlabel('Sequence index')
    ax.set_ylabel('Error')
    ax.set_title('Per-sequence error summary')
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(8, 3.5))
    time_axis = np.arange(rmse_per_time.shape[0])
    ax.plot(time_axis, rmse_per_time, label='RMSE')
    ax.plot(time_axis, mae_per_time, label='MAE')
    ax.set_xlabel('Time index')
    ax.set_ylabel('Error')
    ax.set_title('Temporal error profile (averaged over regions)')
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plt.show()
    
    
    
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    _gt_aligned, _pred_aligned = _resample_pred_to_gt(gt_array_denorm, data_predicted)
    quantiles = np.linspace(0.01, 0.99, 99)
    q_diff = []
    for idx, region in enumerate(gt_regions):
        gt_vals = _gt_aligned[:, :, idx].reshape(-1)
        pred_vals = _pred_aligned[:, :, idx].reshape(-1)
        gt_vals = gt_vals[np.isfinite(gt_vals)]
        pred_vals = pred_vals[np.isfinite(pred_vals)]
        if gt_vals.size < 10 or pred_vals.size < 10:
            q_diff.append(np.full_like(quantiles, np.nan))
            continue
        q_gt = np.quantile(gt_vals, quantiles)
        q_pr = np.quantile(pred_vals, quantiles)
        q_diff.append(q_pr - q_gt)
    q_diff = np.array(q_diff)
    
    plt.figure(figsize=(8, 4))
    plt.imshow(q_diff, aspect='auto', cmap='coolwarm', extent=[quantiles[0], quantiles[-1], 0, len(gt_regions)])
    plt.colorbar(label='Quantile difference (Pred - GT)')
    plt.yticks(np.arange(len(gt_regions)) + 0.5, gt_regions)
    plt.xlabel('Quantile level')
    plt.ylabel('Region')
    plt.title('Tail discrepancies across quantiles')
    plt.tight_layout()
    plt.show()
    
    region_example = gt_regions[0]
    idx = gt_regions.index(region_example)
    vals_gt = _gt_aligned[:, :, idx].reshape(-1)
    vals_pr = _pred_aligned[:, :, idx].reshape(-1)
    vals_gt = np.sort(vals_gt[np.isfinite(vals_gt)])
    vals_pr = np.sort(vals_pr[np.isfinite(vals_pr)])
    min_len = min(len(vals_gt), len(vals_pr))
    plt.figure(figsize=(4, 4))
    plt.plot(vals_gt[:min_len], vals_pr[:min_len], 'o', alpha=0.4)
    lims = [min(vals_gt[0], vals_pr[0]), max(vals_gt[-1], vals_pr[-1])]
    plt.plot(lims, lims, 'k--')
    plt.xlabel(f'GT sorted values ({region_example})')
    plt.ylabel('Pred sorted values')
    plt.title('Empirical QQ comparison')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    _gt_aligned, _pred_aligned = _resample_pred_to_gt(gt_array_denorm, data_predicted)
    residuals = _pred_aligned - _gt_aligned
    residuals = residuals.reshape(-1, residuals.shape[-1])
    residuals = residuals[np.isfinite(residuals).all(axis=1)]
    residual_norm = np.linalg.norm(residuals, axis=1)
    percentiles = np.linspace(0.1, 0.9, 9)
    thresholds = np.quantile(residual_norm, percentiles)
    coverage = []
    for q, thresh in zip(percentiles, thresholds):
        frac = np.mean(residual_norm <= thresh)
        coverage.append((q, frac))
    coverage = np.array(coverage)
    
    plt.figure(figsize=(4, 4))
    plt.plot(percentiles, coverage[:, 1], 'o-', label='Empirical coverage')
    plt.plot([0, 1], [0, 1], 'k--', label='Ideal')
    plt.xlabel('Nominal quantile of residual norm')
    plt.ylabel('Empirical coverage')
    plt.title('Residual calibration proxy')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print('Interpretation: if predictions were perfectly calibrated, empirical coverage would follow the diagonal. Deviations indicate under/over dispersion in residual magnitudes.')
    
    
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def _corr_per_sequence(arr, eps=1e-9):
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr - np.nanmean(arr, axis=0, keepdims=True)
        std = np.nanstd(arr, axis=0, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        z = arr / std
        C = np.corrcoef(z, rowvar=False)
        return np.where(np.isfinite(C), C, np.nan)
    
    gt_corrs = []
    pred_corrs = []
    for s_idx in range(gt_array_denorm.shape[0]):
        gt_corrs.append(_corr_per_sequence(gt_array_denorm[s_idx]))
        pred_corrs.append(_corr_per_sequence(data_predicted[s_idx]))
    
    Cg_mean = np.nanmean(np.stack(gt_corrs), axis=0)
    Cp_mean = np.nanmean(np.stack(pred_corrs), axis=0)
    Cdiff = Cp_mean - Cg_mean
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    heatmaps = [
        (Cg_mean, "GT mean correlation", "coolwarm", -1.0, 1.0),
        (Cp_mean, "Pred mean correlation", "coolwarm", -1.0, 1.0),
        (Cdiff, "Pred - GT correlation", "bwr", -0.5, 0.5),
    ]
    for ax, (mat, title, cmap, vmin, vmax) in zip(axes, heatmaps):
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(range(len(gt_regions)))
        ax.set_yticks(range(len(gt_regions)))
        ax.set_xticklabels(gt_regions, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(gt_regions, fontsize=7)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    
    iu = np.triu_indices(len(gt_regions), k=1)
    gt_vec = np.concatenate([mat[iu] for mat in gt_corrs])
    pred_vec = np.concatenate([mat[iu] for mat in pred_corrs])
    mask = np.isfinite(gt_vec) & np.isfinite(pred_vec)
    gt_vec = gt_vec[mask]
    pred_vec = pred_vec[mask]
    
    from scipy.stats import pearsonr
    global_r = pearsonr(gt_vec, pred_vec)[0] if gt_vec.size > 1 else np.nan
    
    plt.figure(figsize=(5.4, 4.6))
    hb = plt.hexbin(gt_vec, pred_vec, gridsize=60, cmap="inferno", mincnt=1)
    plt.plot([-1, 1], [-1, 1], 'w--', lw=1, alpha=0.6, label="ideal y=x")
    plt.xlabel("GT pairwise corr")
    plt.ylabel("Pred pairwise corr")
    plt.title(f"Regional connectivity alignment (r={global_r:.3f})")
    plt.legend(loc='upper left', frameon=False)
    cbar = plt.colorbar(hb)
    cbar.set_label("count")
    plt.tight_layout()
    plt.show()
    print("Pearson correlation across all upper-triangle entries:", global_r)
    
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    
    window_size = 60   # timesteps per window
    window_step = 10   # stride between consecutive windows
    eps = 1e-8
    
    n_seq, n_time, n_reg = gt_array_denorm.shape
    starts = np.arange(0, n_time - window_size + 1, window_step, dtype=int)
    centers = starts + window_size / 2.0
    
    def _corr_matrix(chunk):
        flat = chunk.reshape(-1, n_reg).astype(np.float64)
        flat -= np.nanmean(flat, axis=0, keepdims=True)
        std = np.nanstd(flat, axis=0, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        flat /= std
        C = np.corrcoef(flat, rowvar=False)
        return np.where(np.isfinite(C), C, np.nan)
    
    Cg_windows, Cp_windows, window_scores = [], [], []
    for start in starts:
        end = start + window_size
        gt_chunk = gt_array_denorm[:, start:end, :]
        pred_chunk = data_predicted[:, start:end, :]
    
        Cg = _corr_matrix(gt_chunk)
        Cp = _corr_matrix(pred_chunk)
        Cg_windows.append(Cg)
        Cp_windows.append(Cp)
    
        iu = np.triu_indices(n_reg, k=1)
        vg = Cg[iu]
        vp = Cp[iu]
        mask = np.isfinite(vg) & np.isfinite(vp)
        if mask.sum() >= 2:
            window_scores.append(pearsonr(vg[mask], vp[mask])[0])
        else:
            window_scores.append(np.nan)
    
    window_scores = np.array(window_scores)
    
    plt.figure(figsize=(6.5, 3.8))
    plt.plot(centers, window_scores, color='tab:purple', marker='o', ms=3)
    plt.axhline(np.nanmean(window_scores), color='gray', linestyle='--', linewidth=1, label='mean')
    plt.xlabel('window center (timestep)')
    plt.ylabel('corr(upper-tri GT, Pred)')
    plt.ylim(-0.2, 1.0)
    plt.title('Connectivity similarity over sliding windows')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print('Mean windowed connectivity correlation:', float(np.nanmean(window_scores)))
    print('Median windowed connectivity correlation:', float(np.nanmedian(window_scores)))
    
    if len(starts) >= 3:
        sample_indices = [0, len(starts)//2, len(starts)-1]
    else:
        sample_indices = list(range(len(starts)))
    
    if sample_indices:
        fig, axes = plt.subplots(len(sample_indices), 3, figsize=(14, 4.2 * len(sample_indices)))
        if len(sample_indices) == 1:
            axes = np.expand_dims(axes, axis=0)
    
        for row, idx in enumerate(sample_indices):
            center = centers[idx]
            labels = [f'GT window@{center:.0f}', f'Pred window@{center:.0f}', 'Pred - GT']
            mats = [Cg_windows[idx], Cp_windows[idx], Cp_windows[idx] - Cg_windows[idx]]
            cmaps = ['coolwarm', 'coolwarm', 'bwr']
            ranges = [(-1, 1), (-1, 1), (-0.5, 0.5)]
            for col in range(3):
                vmin, vmax = ranges[col]
                ax = axes[row, col]
                im = ax.imshow(mats[col], vmin=vmin, vmax=vmax, cmap=cmaps[col])
                ax.set_xticks(range(n_reg))
                ax.set_yticks(range(n_reg))
                if row == len(sample_indices) - 1:
                    ax.set_xticklabels(gt_regions, rotation=45, ha='right', fontsize=7)
                else:
                    ax.set_xticklabels([])
                ax.set_yticklabels(gt_regions, fontsize=7)
                ax.set_title(labels[col])
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        plt.show()
    
    
    
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    window_size = 80   # number of frames per window
    window_step = 20   # step between windows
    
    _gt_aligned, _pred_aligned = _resample_pred_to_gt(gt_array_denorm, data_predicted)
    num_seq, num_time, num_regions = _gt_aligned.shape
    starts = np.arange(0, num_time - window_size + 1, window_step)
    
    fc_metrics = []
    fc_timecourses = []   # store D_t per sequence for plotting later
    
    for seq_idx in range(num_seq):
        gt_seq = _gt_aligned[seq_idx]
        pr_seq = _pred_aligned[seq_idx]
        seq_residuals = []
    
        for start in starts:
            end = start + window_size
            gt_win = gt_seq[start:end]
            pr_win = pr_seq[start:end]
    
            gt_corr = np.corrcoef(gt_win, rowvar=False)
            pr_corr = np.corrcoef(pr_win, rowvar=False)
    
            diff = gt_corr - pr_corr
            frob_diff = np.linalg.norm(diff, ord='fro')
            seq_residuals.append(frob_diff)
    
        seq_residuals = np.array(seq_residuals)
        fc_timecourses.append(seq_residuals)
    
        fc_metrics.append({
            'sequence_id': seq_idx,
            'mean_diff': float(np.mean(seq_residuals)),
            'std_diff': float(np.std(seq_residuals)),
            'max_diff': float(np.max(seq_residuals)),
            'min_diff': float(np.min(seq_residuals))
        })
    
    fc_df = pd.DataFrame(fc_metrics)
    display(fc_df.round(4))
    
    plt.figure(figsize=(6, 4))
    plt.hist(fc_df['mean_diff'], bins=20, color='tab:blue', alpha=0.7, edgecolor='k')
    plt.xlabel('Mean windowed FC Frobenius diff')
    plt.ylabel('Count')
    plt.title('Distribution of functional connectivity deviations')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    n_examples = min(5, num_seq)
    example_ids = np.linspace(0, num_seq - 1, n_examples, dtype=int)
    time_axis = starts + window_size / 2  # center of each window
    
    plt.figure(figsize=(8, 4))
    for seq_idx in example_ids:
        plt.plot(time_axis, fc_timecourses[seq_idx], label=f'Seq {seq_idx}', alpha=0.8)
    plt.xlabel('Time (frames)')
    plt.ylabel('Frobenius diff (GT vs Pred FC)')
    plt.title('Temporal evolution of FC alignment per sequence')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    fc_timecourses = np.array(fc_timecourses)  # shape (num_seq, num_windows)
    mean_frob = fc_timecourses.mean(axis=0)
    sem_frob = fc_timecourses.std(axis=0) / np.sqrt(num_seq)
    
    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, mean_frob, color='tab:red', lw=2, label='Mean across sequences')
    plt.fill_between(time_axis, mean_frob - sem_frob, mean_frob + sem_frob,
                     color='tab:red', alpha=0.3, label='± SEM')
    plt.xlabel('Time (frames)')
    plt.ylabel('Mean Frobenius diff (GT vs Pred FC)')
    plt.title('Group-level evolution of FC prediction mismatch')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("Interpretation:")
    print("- X-axis: time (center of each sliding window).")
    print("- Y-axis: mean Frobenius distance between GT and predicted FC matrices across sequences.")
    print("- The shaded area shows variability (SEM). Peaks = larger mismatch, Valleys = better FC alignment.")
    
    
    
    
    import numpy as np
    from scipy.stats import pearsonr
    
    def _zscore_cols(mat, eps=1e-8):
        mat = np.asarray(mat, dtype=np.float64)
        mat -= np.nanmean(mat, axis=0, keepdims=True)
        std = np.nanstd(mat, axis=0, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        return mat / std
    
    def _corr_upper_triangle(mat3d):
        mats = []
        for seq in mat3d:
            Z = _zscore_cols(seq)
            C = np.corrcoef(Z, rowvar=False)
            iu = np.triu_indices(C.shape[0], k=1)
            mats.append(C[iu])
        vec = np.concatenate(mats)
        mask = np.isfinite(vec)
        return vec[mask]
    
    gt_vec = _corr_upper_triangle(gt_array_denorm)
    pred_vec = _corr_upper_triangle(data_predicted)
    min_len = min(gt_vec.size, pred_vec.size)
    if min_len == 0:
        global_corr_score = np.nan
    else:
        mask = np.isfinite(gt_vec[:min_len]) & np.isfinite(pred_vec[:min_len])
        if mask.sum() >= 2:
            global_corr_score = pearsonr(gt_vec[:min_len][mask], pred_vec[:min_len][mask])[0]
        else:
            global_corr_score = np.nan
    
    def _window_connectivity_scores(window_size=60, window_step=10, eps=1e-8):
        n_seq, n_time, n_reg = gt_array_denorm.shape
        starts = np.arange(0, n_time - window_size + 1, window_step, dtype=int)
        scores = []
        for start in starts:
            end = start + window_size
            gt_chunk = gt_array_denorm[:, start:end, :]
            pred_chunk = data_predicted[:, start:end, :]
            def _chunk_corr(chunk):
                flat = chunk.reshape(-1, n_reg)
                flat = _zscore_cols(flat, eps=eps)
                C = np.corrcoef(flat, rowvar=False)
                return C
            Cg = _chunk_corr(gt_chunk)
            Cp = _chunk_corr(pred_chunk)
            iu = np.triu_indices(n_reg, k=1)
            vg, vp = Cg[iu], Cp[iu]
            mask = np.isfinite(vg) & np.isfinite(vp)
            if mask.sum() >= 2:
                scores.append(pearsonr(vg[mask], vp[mask])[0])
            else:
                scores.append(np.nan)
        return np.array(scores)
    
    window_scores = _window_connectivity_scores(window_size=60, window_step=10)
    window_mean = float(np.nanmean(window_scores)) if window_scores.size else np.nan
    window_median = float(np.nanmedian(window_scores)) if window_scores.size else np.nan
    window_coverage = float(np.mean(np.isfinite(window_scores))) if window_scores.size else 0.0
    
    alpha = 0.6
    composite_score = float(alpha * global_corr_score + (1 - alpha) * window_mean)
    
    print(f'Global connectivity correlation score: {global_corr_score:.3f}')
    print(f'Mean sliding-window correlation: {window_mean:.3f}')
    print(f'Median sliding-window correlation: {window_median:.3f}')
    print(f'Valid window fraction: {window_coverage:.2%}')
    print(f'Composite score (alpha={alpha:.2f}): {composite_score:.3f}')
    
    
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from scipy.linalg import svd
    
    max_components = min(10, gt_array_denorm.shape[1], gt_array_denorm.shape[2])
    var_gt_list, var_pr_list, subspace_sims = [], [], []
    
    for seq_idx in range(gt_array_denorm.shape[0]):
        Xg = _zscore_cols(gt_array_denorm[seq_idx])
        Xp = _zscore_cols(data_predicted[seq_idx])
        n_comp = min(max_components, Xg.shape[0], Xp.shape[0], Xg.shape[1])
        if n_comp < 2:
            continue
        pca_gt = PCA(n_components=n_comp).fit(Xg)
        pca_pr = PCA(n_components=n_comp).fit(Xp)
    
        var_gt_list.append(pca_gt.explained_variance_ratio_)
        var_pr_list.append(pca_pr.explained_variance_ratio_)
    
        M = pca_gt.components_ @ pca_pr.components_.T
        _, sing_vals, _ = svd(M, full_matrices=False)
        sing_vals = np.clip(sing_vals, 0.0, 1.0)
        subspace_sims.append(float(np.mean(sing_vals)))
    
    if not var_gt_list:
        raise RuntimeError('PCA comparison skipped: not enough valid sequences.')
    
    var_gt_arr = np.array(var_gt_list)
    var_pr_arr = np.array(var_pr_list)
    subspace_sims = np.array(subspace_sims)
    
    mean_abs_diff = float(np.nanmean(np.abs(var_gt_arr - var_pr_arr)))
    spec_score = float(np.clip(1.0 - mean_abs_diff, 0.0, 1.0))
    subspace_score = float(np.nanmean(subspace_sims)) if subspace_sims.size else np.nan
    alpha_pca = 0.5
    pca_composite_score = float(alpha_pca * subspace_score + (1 - alpha_pca) * spec_score)
    
    print(f'Mean absolute diff (explained variance ratios): {mean_abs_diff:.3f}')
    print(f'Spectrum similarity score (1 - diff): {spec_score:.3f}')
    print(f'Mean subspace similarity (singular values): {subspace_score:.3f}')
    print(f'Composite PCA alignment score (alpha={alpha_pca:.2f}): {pca_composite_score:.3f}')
    
    components = np.arange(1, var_gt_arr.shape[1] + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(components, var_gt_arr.mean(axis=0), '-o', label='GT')
    plt.plot(components, var_pr_arr.mean(axis=0), '-o', label='Pred')
    plt.fill_between(components,
                     var_gt_arr.mean(axis=0) - var_gt_arr.std(axis=0),
                     var_gt_arr.mean(axis=0) + var_gt_arr.std(axis=0),
                     alpha=0.2, label='GT ± std')
    plt.fill_between(components,
                     var_pr_arr.mean(axis=0) - var_pr_arr.std(axis=0),
                     var_pr_arr.mean(axis=0) + var_pr_arr.std(axis=0),
                     alpha=0.2, label='Pred ± std')
    plt.xlabel('PCA component')
    plt.ylabel('Explained variance ratio')
    plt.title('PCA spectrum across sequences (repeat)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(6, 4))
    plt.hist(subspace_sims, bins=30, color='steelblue', edgecolor='k', alpha=0.7)
    plt.xlabel('Subspace similarity (mean singular value)')
    plt.ylabel('Count')
    plt.title('Distribution of PCA subspace alignment (repeat)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    
    
    
    import numpy as np
    from scipy.signal import correlate
    from scipy.stats import pearsonr
    
    max_lag_short = 60
    max_lag_long = 600
    
    def _autocorr_1d(x, max_lag, eps=1e-9):
        x = np.asarray(x, dtype=np.float64)
        x = (x - np.nanmean(x)) / (np.nanstd(x) + eps)
        corr = correlate(x, x, mode="full", method="auto") / len(x)
        mid = len(corr) // 2
        lag_len = min(max_lag, len(corr) - mid)
        return corr[mid:mid + lag_len]
    
    def _autocorr_tensor(data, max_lag):
        n_seq, n_time, n_reg = data.shape
        lag_len = min(max_lag, n_time)
        out = np.empty((n_seq, n_reg, lag_len), dtype=np.float64)
        for s in range(n_seq):
            for r in range(n_reg):
                out[s, r] = _autocorr_1d(data[s, :, r], lag_len)
        return out
    
    gt_ac_short = _autocorr_tensor(gt_array_denorm, max_lag_short)
    pred_ac_short = _autocorr_tensor(data_predicted, max_lag_short)
    gt_ac_long = _autocorr_tensor(gt_array_denorm, max_lag_long)
    pred_ac_long = _autocorr_tensor(data_predicted, max_lag_long)
    
    mean_gt_short = gt_ac_short.mean(axis=(0, 1))
    mean_pred_short = pred_ac_short.mean(axis=(0, 1))
    lags_short = np.arange(gt_ac_short.shape[2])
    
    plt.figure(figsize=(6, 4))
    plt.plot(lags_short, mean_gt_short, label="GT")
    plt.plot(lags_short, mean_pred_short, label="Pred")
    plt.xlabel("Retardo")
    plt.ylabel("Autocorrelación normalizada")
    plt.title(f"Autocorrelación media (0 a {int(lags_short[-1])})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    for r_idx, region in enumerate(gt_regions):
        ax = axes[r_idx]
        ax.plot(lags_short, gt_ac_short[:, r_idx, :].mean(axis=0), label="GT", color="tab:green")
        ax.plot(lags_short, pred_ac_short[:, r_idx, :].mean(axis=0), label="Pred", color="tab:blue")
        ax.set_title(region, fontsize=9)
        if r_idx % 4 == 0:
            ax.set_ylabel("Autocorr")
        if r_idx >= len(gt_regions) - 4:
            ax.set_xlabel(f"Retardo (0-{int(lags_short[-1])})")
        ax.grid(alpha=0.2)
    for ax in axes[len(gt_regions):]:
        ax.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle('Autocorrelación media por región (ventana corta)', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    lags_long = np.arange(gt_ac_long.shape[2])
    fig, axes = plt.subplots(4, 4, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    for r_idx, region in enumerate(gt_regions):
        ax = axes[r_idx]
        ax.plot(lags_long, gt_ac_long[:, r_idx, :].mean(axis=0), label="GT", color="tab:green")
        ax.plot(lags_long, pred_ac_long[:, r_idx, :].mean(axis=0), label="Pred", color="tab:blue")
        ax.set_title(region, fontsize=9)
        if r_idx % 4 == 0:
            ax.set_ylabel("Autocorr")
        if r_idx >= len(gt_regions) - 4:
            ax.set_xlabel(f"Retardo (0-{int(lags_long[-1])})")
        ax.grid(alpha=0.2)
    for ax in axes[len(gt_regions):]:
        ax.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle('Autocorrelación media por región (ventana larga)', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    gt_flat = gt_ac_short.reshape(-1)
    pred_flat = pred_ac_short.reshape(-1)
    mask = np.isfinite(gt_flat) & np.isfinite(pred_flat)
    if mask.sum() >= 2:
        autocorr_corr_score = pearsonr(gt_flat[mask], pred_flat[mask])[0]
    else:
        autocorr_corr_score = np.nan
    rmse_autocorr = float(np.sqrt(np.nanmean((gt_flat - pred_flat) ** 2)))
    
    print(f'Score de autocorrelación (lag corto): {autocorr_corr_score:.3f}')
    print(f'RMSE de autocorrelación GT vs Pred (lag corto): {rmse_autocorr:.4f}')
    
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import correlate
    from scipy.stats import pearsonr
    
    def _cross_corr_1d(x, y, max_lag, eps=1e-9):
        x = (x - np.nanmean(x)) / (np.nanstd(x) + eps)
        y = (y - np.nanmean(y)) / (np.nanstd(y) + eps)
        corr = correlate(x, y, mode='full', method='auto') / len(x)
        mid = len(corr) // 2
        span = min(max_lag, mid)
        return corr[mid - span: mid + span + 1]
    
    def _cross_corr_pairs(data_a, data_b, max_lag):
        n_seq, n_time, n_reg = data_a.shape
        pairs = [(i, j) for i in range(n_reg) for j in range(i + 1, n_reg)]
        n_pairs = len(pairs)
        n_lag = 2 * min(max_lag, n_time // 2) + 1
        cc_a = np.empty((n_seq, n_pairs, n_lag), dtype=np.float64)
        cc_b = np.empty_like(cc_a)
        for s in range(n_seq):
            for p_idx, (i, j) in enumerate(pairs):
                cc_a[s, p_idx] = _cross_corr_1d(data_a[s, :, i], data_a[s, :, j], max_lag)
                cc_b[s, p_idx] = _cross_corr_1d(data_b[s, :, i], data_b[s, :, j], max_lag)
        return pairs, cc_a, cc_b
    
    gt_len = gt_array_denorm.shape[1]
    pred_len = data_predicted.shape[1]
    gt_cc = gt_array_denorm
    if pred_len != gt_len and pred_len % gt_len == 0:
        factor = pred_len // gt_len
        pred_cc = data_predicted.reshape(data_predicted.shape[0], gt_len, factor, data_predicted.shape[2]).mean(axis=2)
    elif pred_len != gt_len:
        min_len = min(gt_len, pred_len)
        gt_cc = gt_array_denorm[:, :min_len, :]
        pred_cc = data_predicted[:, :min_len, :]
    else:
        pred_cc = data_predicted
    
    max_lag = 12
    pairs, cc_gt, cc_pred = _cross_corr_pairs(gt_cc, pred_cc, max_lag=max_lag)
    lag_axis = np.arange(-min(max_lag, gt_cc.shape[1] // 2), min(max_lag, gt_cc.shape[1] // 2) + 1)
    
    flat_gt = cc_gt.reshape(-1)
    flat_pred = cc_pred.reshape(-1)
    mask = np.isfinite(flat_gt) & np.isfinite(flat_pred)
    if mask.sum() >= 2:
        crosscorr_global_score = pearsonr(flat_gt[mask], flat_pred[mask])[0]
    else:
        crosscorr_global_score = np.nan
    crosscorr_rmse = float(np.sqrt(np.nanmean((flat_gt - flat_pred) ** 2)))
    
    print(f'Score global de cross-correlación (Pearson): {crosscorr_global_score:.3f}')
    print(f'RMSE de cross-correlaciones: {crosscorr_rmse:.4f}')
    
    mean_gt = cc_gt.mean(axis=(0, 1))
    std_gt = cc_gt.std(axis=(0, 1))
    mean_pred = cc_pred.mean(axis=(0, 1))
    std_pred = cc_pred.std(axis=(0, 1))
    plt.figure(figsize=(6, 4))
    plt.plot(lag_axis, mean_gt, label='GT media', color='tab:blue')
    plt.fill_between(lag_axis, mean_gt - std_gt, mean_gt + std_gt, color='tab:blue', alpha=0.2)
    plt.plot(lag_axis, mean_pred, label='Pred media', color='tab:orange')
    plt.fill_between(lag_axis, mean_pred - std_pred, mean_pred + std_pred, color='tab:orange', alpha=0.2)
    plt.axvline(0, color='k', linestyle='--', lw=1)
    plt.title('Cross-correlación media entre pares de regiones (±1 std)')
    plt.xlabel('Retardo (pasos)')
    plt.ylabel('Correlación normalizada')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    n_reg = gt_cc.shape[2]
    region_mean_gt = np.zeros((n_reg, lag_axis.size))
    region_mean_pred = np.zeros_like(region_mean_gt)
    counts = np.zeros(n_reg, dtype=int)
    for idx, (i, j) in enumerate(pairs):
        region_mean_gt[i] += cc_gt[:, idx, :].mean(axis=0)
        region_mean_gt[j] += cc_gt[:, idx, :].mean(axis=0)
        region_mean_pred[i] += cc_pred[:, idx, :].mean(axis=0)
        region_mean_pred[j] += cc_pred[:, idx, :].mean(axis=0)
        counts[i] += 1
        counts[j] += 1
    region_mean_gt /= counts[:, None]
    region_mean_pred /= counts[:, None]
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    for r_idx, region in enumerate(gt_regions):
        ax = axes[r_idx]
        ax.plot(lag_axis, region_mean_gt[r_idx], label='GT', color='tab:green')
        ax.plot(lag_axis, region_mean_pred[r_idx], label='Pred', color='tab:purple')
        ax.axvline(0, color='k', linestyle='--', lw=0.8)
        ax.set_title(region, fontsize=9)
        if r_idx % 4 == 0:
            ax.set_ylabel('Correlación')
        if r_idx >= n_reg - 4:
            ax.set_xlabel('Retardo')
        ax.grid(alpha=0.2)
    for ax in axes[n_reg:]:
        ax.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle('Cross-correlación media por región (pares que la involucran)', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    def _peak_lags(cc_array, lags):
        peaks = []
        values = []
        for curve in cc_array.reshape(-1, cc_array.shape[-1]):
            idx = int(np.nanargmax(np.abs(curve)))
            peaks.append(lags[idx])
            values.append(curve[idx])
        return np.array(peaks), np.array(values)
    
    peaks_gt, vals_gt = _peak_lags(cc_gt, lag_axis)
    peaks_pred, vals_pred = _peak_lags(cc_pred, lag_axis)
    plt.figure(figsize=(6, 4))
    bins = np.arange(lag_axis.min() - 0.5, lag_axis.max() + 1.5, 1)
    plt.hist(peaks_gt, bins=bins, alpha=0.5, label='GT', color='tab:blue')
    plt.hist(peaks_pred, bins=bins, alpha=0.5, label='Pred', color='tab:orange')
    plt.xlabel('Retardo del pico de cross-correlación (pasos)')
    plt.ylabel('Conteo')
    plt.title('Distribución de desfases pico (GT vs Pred)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f'Desfase pico medio GT: {np.nanmean(peaks_gt):.2f} pasos | Pred: {np.nanmean(peaks_pred):.2f} pasos')
    
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import skew, kurtosis, pearsonr
    
    
    def _higher_order_moments(gt, pred):
        n_seq, _, n_reg = gt.shape
        GT_skew = np.empty((n_seq, n_reg), dtype=np.float64)
        PR_skew = np.empty_like(GT_skew)
        GT_kurt = np.empty_like(GT_skew)
        PR_kurt = np.empty_like(GT_skew)
        for s in range(n_seq):
            GT_skew[s] = skew(gt[s], axis=0, nan_policy='omit')
            PR_skew[s] = skew(pred[s], axis=0, nan_policy='omit')
            GT_kurt[s] = kurtosis(gt[s], axis=0, nan_policy='omit')
            PR_kurt[s] = kurtosis(pred[s], axis=0, nan_policy='omit')
        return GT_skew, PR_skew, GT_kurt, PR_kurt
    
    skew_gt, skew_pred, kurt_gt, kurt_pred = _higher_order_moments(gt_mom, pred_mom)
    
    flat_skew_gt = skew_gt.ravel()
    flat_skew_pred = skew_pred.ravel()
    flat_kurt_gt = kurt_gt.ravel()
    flat_kurt_pred = kurt_pred.ravel()
    mask_skew = np.isfinite(flat_skew_gt) & np.isfinite(flat_skew_pred)
    mask_kurt = np.isfinite(flat_kurt_gt) & np.isfinite(flat_kurt_pred)
    skew_score = pearsonr(flat_skew_gt[mask_skew], flat_skew_pred[mask_skew])[0] if mask_skew.sum() >= 2 else np.nan
    kurt_score = pearsonr(flat_kurt_gt[mask_kurt], flat_kurt_pred[mask_kurt])[0] if mask_kurt.sum() >= 2 else np.nan
    skew_rmse = float(np.sqrt(np.nanmean((flat_skew_gt - flat_skew_pred) ** 2)))
    kurt_rmse = float(np.sqrt(np.nanmean((flat_kurt_gt - flat_kurt_pred) ** 2)))
    alpha_moment = 0.5
    moment_composite_score = float(alpha_moment * skew_score + (1 - alpha_moment) * kurt_score)
    
    print(f'Score de asimetría (Pearson): {skew_score:.3f}')
    print(f'RMSE de asimetría: {skew_rmse:.4f}')
    print(f'Score de curtosis (Pearson): {kurt_score:.3f}')
    print(f'RMSE de curtosis: {kurt_rmse:.4f}')
    print(f'Score compuesto (alpha={alpha_moment:.2f}): {moment_composite_score:.3f}')
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(flat_skew_gt, flat_skew_pred, alpha=0.35, color='tab:blue', edgecolor='none', s=28)
    plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Skewness GT')
    plt.ylabel('Skewness Pred')
    plt.title('Asimetría por región y secuencia')
    plt.subplot(1, 2, 2)
    plt.scatter(flat_kurt_gt, flat_kurt_pred, alpha=0.35, color='tab:orange', edgecolor='none', s=28)
    plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Kurtosis GT')
    plt.ylabel('Kurtosis Pred')
    plt.title('Curtosis por región y secuencia')
    plt.tight_layout()
    plt.show()
    
    seq_idx = np.arange(gt_mom.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(16, 10), sharex=True, sharey=False)
    axes = axes.ravel()
    for idx, region in enumerate(gt_regions):
        ax = axes[idx]
        ax.plot(seq_idx, skew_gt[:, idx], 'o-', label='GT', alpha=0.7)
        ax.plot(seq_idx, skew_pred[:, idx], 'x--', label='Pred', alpha=0.7)
        ax.set_title(region, fontsize=9)
        if idx % 4 == 0:
            ax.set_ylabel('Skewness')
        if idx >= len(gt_regions) - 4:
            ax.set_xlabel('Secuencia')
        ax.grid(alpha=0.2)
    for ax in axes[len(gt_regions):]:
        ax.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle('Skewness GT vs Pred por región')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 10), sharex=True, sharey=False)
    axes = axes.ravel()
    for idx, region in enumerate(gt_regions):
        ax = axes[idx]
        ax.plot(seq_idx, kurt_gt[:, idx], 'o-', label='GT', alpha=0.7)
        ax.plot(seq_idx, kurt_pred[:, idx], 'x--', label='Pred', alpha=0.7)
        ax.set_title(region, fontsize=9)
        if idx % 4 == 0:
            ax.set_ylabel('Kurtosis')
        if idx >= len(gt_regions) - 4:
            ax.set_xlabel('Secuencia')
        ax.grid(alpha=0.2)
    for ax in axes[len(gt_regions):]:
        ax.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle('Kurtosis GT vs Pred por región')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import OrderedDict
    
    _required = []
    _missing = [name for name in _required if name not in globals()]
    if _missing:
        raise RuntimeError(f'Missing variables from higher-order moment cell: {_missing}. Run that cell first.')
    
    def _region_to_area(region_name):
        if '-layer' in region_name:
            return region_name.split('-layer')[0]
        return region_name
    
    area_to_indices = OrderedDict()
    for idx, region in enumerate(gt_regions):
        area = _region_to_area(region)
        area_to_indices.setdefault(area, []).append(idx)
    
    area_labels = list(area_to_indices.keys())
    
    def _aggregate_stats(moment_array):
        means, stds, counts = [], [], []
        for indices in area_to_indices.values():
            sub = moment_array[:, indices]
            valid = np.isfinite(sub)
            if not np.any(valid):
                means.append(np.nan)
                stds.append(np.nan)
                counts.append(0)
                continue
            vals = sub[valid]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
            counts.append(int(valid.sum()))
        return np.asarray(means), np.asarray(stds), np.asarray(counts)
    
    skew_gt_mean, skew_gt_std, skew_counts = _aggregate_stats(skew_gt)
    skew_pred_mean, skew_pred_std, _ = _aggregate_stats(skew_pred)
    kurt_gt_mean, kurt_gt_std, kurt_counts = _aggregate_stats(kurt_gt)
    kurt_pred_mean, kurt_pred_std, _ = _aggregate_stats(kurt_pred)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(skew_gt_mean, skew_pred_mean, alpha=0.35, color='tab:blue', edgecolor='none', s=60)
    for area, x, y in zip(area_labels, skew_gt_mean, skew_pred_mean):
        if np.isfinite(x) and np.isfinite(y):
            plt.text(x, y, area, fontsize=7, alpha=0.7)
    plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Area-averaged skewness GT')
    plt.ylabel('Area-averaged skewness Pred')
    plt.title('Asimetría agregada por área (promedio secuencias)')
    plt.subplot(1, 2, 2)
    plt.scatter(kurt_gt_mean, kurt_pred_mean, alpha=0.35, color='tab:orange', edgecolor='none', s=60)
    for area, x, y in zip(area_labels, kurt_gt_mean, kurt_pred_mean):
        if np.isfinite(x) and np.isfinite(y):
            plt.text(x, y, area, fontsize=7, alpha=0.7)
    plt.axline((0, 0), slope=1, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Area-averaged kurtosis GT')
    plt.ylabel('Area-averaged kurtosis Pred')
    plt.title('Curtosis agregada por área (promedio secuencias)')
    plt.tight_layout()
    plt.show()
    
    indices = np.arange(len(area_labels))
    bar_width = 0.35
    
    fig, ax = plt.subplots(figsize=(max(8, len(area_labels) * 0.6), 4.5))
    ax.bar(indices - bar_width/2, skew_gt_mean, width=bar_width, yerr=skew_gt_std, label='GT', capsize=4, alpha=0.8)
    ax.bar(indices + bar_width/2, skew_pred_mean, width=bar_width, yerr=skew_pred_std, label='Pred', capsize=4, alpha=0.8)
    ax.set_xticks(indices)
    ax.set_xticklabels(area_labels, rotation=90)
    ax.set_ylabel('Skewness (area mean ± std)')
    ax.set_title('Skewness promedio por área (sobre todas las secuencias)')
    ax.grid(axis='y', alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(max(8, len(area_labels) * 0.6), 4.5))
    ax.bar(indices - bar_width/2, kurt_gt_mean, width=bar_width, yerr=kurt_gt_std, label='GT', capsize=4, alpha=0.8)
    ax.bar(indices + bar_width/2, kurt_pred_mean, width=bar_width, yerr=kurt_pred_std, label='Pred', capsize=4, alpha=0.8)
    ax.set_xticks(indices)
    ax.set_xticklabels(area_labels, rotation=90)
    ax.set_ylabel('Kurtosis (area mean ± std)')
    ax.set_title('Curtosis promedio por área (sobre todas las secuencias)')
    ax.grid(axis='y', alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plt.show()
    
    
    
    
    from scipy import stats
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    def _flatten_sequences(arr):
        return arr.reshape(-1, arr.shape[-1])
    
    region_labels = gt_regions
    if pred_region_names != gt_regions:
        region_labels = [r for r in gt_regions if r in pred_region_names]
        if not region_labels:
            raise ValueError('No overlapping regions between GT and predictions for distribution analysis.')
        gt_idx = [gt_regions.index(r) for r in region_labels]
        pred_idx = [pred_region_names.index(r) for r in region_labels]
        gt_flat = _flatten_sequences(gt_array_denorm)[:, gt_idx]
        pred_flat = _flatten_sequences(data_predicted)[:, pred_idx]
    else:
        gt_flat = _flatten_sequences(gt_array_denorm)
        pred_flat = _flatten_sequences(data_predicted)
    
    def _distribution_summary(flat, labels):
        rows = []
        for col_idx, region in enumerate(labels):
            values = flat[:, col_idx]
            values = values[np.isfinite(values)]
            if values.size < 8:
                rows.append((region, np.nan, np.nan, np.nan, np.nan, np.nan))
                continue
            mean_val = float(np.mean(values))
            k2_stat, p_value = stats.normaltest(values)
            skew_val = stats.skew(values, bias=False)
            kurt_val = stats.kurtosis(values, fisher=True, bias=False)
            rows.append((region, mean_val, k2_stat, p_value, skew_val, kurt_val))
        return pd.DataFrame(rows, columns=['region', 'mean', 'k2_stat', 'p_value', 'skewness', 'excess_kurtosis'])
    
    summary_gt = _distribution_summary(gt_flat, region_labels).assign(dataset='ground_truth')
    summary_pred = _distribution_summary(pred_flat, region_labels).assign(dataset='predicted')
    summary_all = pd.concat([summary_gt, summary_pred], ignore_index=True)
    
    display(summary_all.set_index(['dataset', 'region']))
    print("Scipy skew/kurtosis utilities: stats.skew, stats.skewtest, stats.kurtosis, stats.kurtosistest. D'Agostino-Pearson normality via stats.normaltest.")
    
    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(region_labels) * 0.6), 4), sharey=True)
    for ax, (dataset, df) in zip(axes, summary_all.groupby('dataset', sort=False)):
        means = df.set_index('region').loc[region_labels, 'mean']
        bars = ax.bar(range(len(region_labels)), means.to_numpy())
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{bar.get_height():.10f}",
                    rotation=90, va='bottom', ha='center', fontsize=8)
        ax.set_xticks(range(len(region_labels)))
        ax.set_xticklabels(region_labels, rotation=90)
        ax.set_title(f"{dataset.replace('_', ' ').title()} mean per region")
        ax.set_ylabel('Mean value')
    fig.tight_layout()
    
    n_regions = len(region_labels)
    n_cols = 4
    n_rows = int(np.ceil(n_regions / n_cols))
    fig_box, axes_box = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.0, n_rows * 3.5))
    axes_box = np.atleast_1d(axes_box).ravel()
    for idx, region in enumerate(region_labels):
        ax = axes_box[idx]
        gt_vals = gt_flat[:, idx]
        pred_vals = pred_flat[:, idx]
        gt_vals = gt_vals[np.isfinite(gt_vals)]
        pred_vals = pred_vals[np.isfinite(pred_vals)]
    
        data = []
        labels = []
        stats_lines = []
    
        if gt_vals.size > 0:
            data.append(gt_vals)
            labels.append('GT')
            stats_lines.append(f"GT μ={np.mean(gt_vals):.20f} | σ={np.std(gt_vals, ddof=1):.20f}")
        if pred_vals.size > 0:
            data.append(pred_vals)
            labels.append('Pred')
            stats_lines.append(f"Pred μ={np.mean(pred_vals):.20f} | σ={np.std(pred_vals, ddof=1):.20f}")
    
        if data:
            ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
            ax.set_title(region)
            ax.set_ylabel('Value')
            ax.text(0.5, 0.95, "\n".join(stats_lines), transform=ax.transAxes, ha='center', va='top', fontsize=8)
        else:
            ax.set_title(region)
            ax.text(0.5, 0.5, 'No finite data', transform=ax.transAxes, ha='center', va='center', fontsize=8)
    
    for ax in axes_box[n_regions:]:
        ax.axis('off')
    fig_box.tight_layout()
    
    plt.show()
    
    
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    
    _required = []
    _missing = [name for name in _required if name not in globals()]
    if _missing:
        raise RuntimeError(f'Missing prerequisites from previous cells: {_missing}. Run the distribution + moment analyses first.')
    
    summary_indexed = summary_all.set_index(['dataset', 'region'])
    try:
        gauss_gt = summary_indexed.xs('ground_truth', level='dataset')
        gauss_pred = summary_indexed.xs('predicted', level='dataset')
    except KeyError as exc:
        raise RuntimeError('Expected datasets `ground_truth` and `predicted` in summary_all.') from exc
    
    region_labels = list(gt_regions)
    
    def _seq_stats(arr):
        return pd.DataFrame({
            'mean': np.nanmean(arr, axis=0),
            'std': np.nanstd(arr, axis=0),
            'median': np.nanmedian(arr, axis=0)
        }, index=region_labels)
    
    skew_stats_gt = _seq_stats(skew_gt)
    skew_stats_pred = _seq_stats(skew_pred)
    kurt_stats_gt = _seq_stats(kurt_gt)
    kurt_stats_pred = _seq_stats(kurt_pred)
    
    combined = pd.DataFrame({
        'mean_gt': gauss_gt.reindex(region_labels)['mean'],
        'mean_pred': gauss_pred.reindex(region_labels)['mean'],
        'mean_delta': gauss_pred.reindex(region_labels)['mean'] - gauss_gt.reindex(region_labels)['mean'],
        'normaltest_p_gt': gauss_gt.reindex(region_labels)['p_value'],
        'normaltest_p_pred': gauss_pred.reindex(region_labels)['p_value'],
        'normaltest_p_delta': gauss_pred.reindex(region_labels)['p_value'] - gauss_gt.reindex(region_labels)['p_value'],
        'global_skew_gt': gauss_gt.reindex(region_labels)['skewness'],
        'global_skew_pred': gauss_pred.reindex(region_labels)['skewness'],
        'seq_skew_mean_gt': skew_stats_gt['mean'],
        'seq_skew_mean_pred': skew_stats_pred['mean'],
        'seq_skew_std_gt': skew_stats_gt['std'],
        'seq_skew_std_pred': skew_stats_pred['std'],
        'seq_skew_delta': skew_stats_pred['mean'] - skew_stats_gt['mean'],
        'global_kurt_gt': gauss_gt.reindex(region_labels)['excess_kurtosis'],
        'global_kurt_pred': gauss_pred.reindex(region_labels)['excess_kurtosis'],
        'seq_kurt_mean_gt': kurt_stats_gt['mean'],
        'seq_kurt_mean_pred': kurt_stats_pred['mean'],
        'seq_kurt_std_gt': kurt_stats_gt['std'],
        'seq_kurt_std_pred': kurt_stats_pred['std'],
        'seq_kurt_delta': kurt_stats_pred['mean'] - kurt_stats_gt['mean']
    }).round(6)
    
    combined_display = combined[['mean_gt', 'mean_pred', 'mean_delta',
                                  'normaltest_p_gt', 'normaltest_p_pred', 'normaltest_p_delta',
                                  'seq_skew_mean_gt', 'seq_skew_mean_pred', 'seq_skew_delta',
                                  'seq_kurt_mean_gt', 'seq_kurt_mean_pred', 'seq_kurt_delta']]
    display(combined_display)
    
    metrics_delta = combined[['mean_delta', 'normaltest_p_delta', 'seq_skew_delta', 'seq_kurt_delta']]
    summary_stats = metrics_delta.describe(percentiles=[0.5, 0.9]).T.round(4)
    summary_stats.rename(columns={'50%': 'median', '90%': 'p90'}, inplace=True)
    print('Delta distribution across regions (pred - GT):')
    display(summary_stats)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    mat = metrics_delta.to_numpy().T
    norm = TwoSlopeNorm(vmin=np.nanmin(mat), vcenter=0.0, vmax=np.nanmax(mat))
    im = ax.imshow(mat, aspect='auto', cmap='coolwarm', norm=norm)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(['ΔMean', 'ΔNormality p', 'ΔSkew', 'ΔKurtosis'])
    ax.set_xticks(range(len(region_labels)))
    ax.set_xticklabels(region_labels, rotation=90)
    ax.set_title('Prediction minus GT deviations (per region)')
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label='Pred - GT')
    plt.tight_layout()
    plt.show()
    
    print('Combined gaussianity and higher-moment summary per region; heatmap visualizes prediction deviations from GT across metrics.')
    
    
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    from scipy.stats import pearsonr
    
    def _match_lengths_for_graph(gt_arr, pred_arr):
        gt_len = gt_arr.shape[1]
        pred_len = pred_arr.shape[1]
        if pred_len != gt_len and pred_len % gt_len == 0:
            factor = pred_len // gt_len
            pred_eq = pred_arr.reshape(pred_arr.shape[0], gt_len, factor, pred_arr.shape[2]).mean(axis=2)
            gt_eq = gt_arr
        elif pred_len != gt_len:
            min_len = min(gt_len, pred_len)
            gt_eq = gt_arr[:, :min_len, :]
            pred_eq = pred_arr[:, :min_len, :]
        else:
            gt_eq = gt_arr
            pred_eq = pred_arr
        return gt_eq, pred_eq
    
    graph_thresh = 0.5
    gt_eq, pred_eq = _match_lengths_for_graph(gt_array_denorm, data_predicted)
    
    def _corr_to_graphs(gt_eq, pred_eq, thresh):
        gt_flat = gt_eq.reshape(-1, gt_eq.shape[-1]).astype(np.float64)
        pred_flat = pred_eq.reshape(-1, pred_eq.shape[-1]).astype(np.float64)
        Cg = np.corrcoef(gt_flat, rowvar=False)
        Cp = np.corrcoef(pred_flat, rowvar=False)
        n_reg = Cg.shape[0]
        mask_triu = np.triu(np.ones_like(Cg, dtype=bool), k=1)
        valid = mask_triu & np.isfinite(Cg) & np.isfinite(Cp)
        adj_gt = (np.abs(Cg) >= thresh) & valid
        adj_pred = (np.abs(Cp) >= thresh) & valid
        Gg = nx.Graph()
        Gp = nx.Graph()
        for i in range(n_reg):
            Gg.add_node(i)
            Gp.add_node(i)
        for i in range(n_reg):
            for j in range(i + 1, n_reg):
                if adj_gt[i, j]:
                    Gg.add_edge(i, j, weight=float(Cg[i, j]))
                if adj_pred[i, j]:
                    Gp.add_edge(i, j, weight=float(Cp[i, j]))
        return Cg, Cp, adj_gt, adj_pred, Gg, Gp
    
    Cg, Cp, adj_gt, adj_pred, Gg, Gp = _corr_to_graphs(gt_eq, pred_eq, graph_thresh)
    
    def _graph_metrics(G):
        if len(G) == 0:
            return {
                'clustering': np.nan,
                'density': 0.0,
                'n_edges': 0,
                'n_components': 0,
                'avg_path_length': np.nan
            }
        metrics = {
            'clustering': nx.average_clustering(G) if G.number_of_edges() > 0 else 0.0,
            'density': nx.density(G),
            'n_edges': G.number_of_edges(),
            'n_components': nx.number_connected_components(G)
        }
        try:
            metrics['avg_path_length'] = nx.average_shortest_path_length(G)
        except nx.NetworkXError:
            metrics['avg_path_length'] = np.nan
        return metrics
    
    metrics_gt = _graph_metrics(Gg)
    metrics_pred = _graph_metrics(Gp)
    
    edge_intersection = np.logical_and(adj_gt, adj_pred).sum()
    edge_union = np.logical_or(adj_gt, adj_pred).sum()
    edge_jaccard = float(edge_intersection / edge_union) if edge_union else 1.0
    deg_gt = adj_gt.sum(axis=1)
    deg_pred = adj_pred.sum(axis=1)
    mask_deg = np.isfinite(deg_gt) & np.isfinite(deg_pred)
    if mask_deg.sum() >= 2 and np.std(deg_gt[mask_deg]) > 0 and np.std(deg_pred[mask_deg]) > 0:
        degree_score = pearsonr(deg_gt[mask_deg], deg_pred[mask_deg])[0]
    else:
        degree_score = np.nan
    graph_score = float(0.5 * edge_jaccard + 0.5 * (degree_score if np.isfinite(degree_score) else 0.0))
    
    print(f'Score de superposición de aristas (Jaccard): {edge_jaccard:.3f}')
    print(f'Score de correlación de grados: {degree_score:.3f}')
    print(f'Score compuesto de grafo: {graph_score:.3f}')
    print('Métricas GT:', metrics_gt)
    print('Métricas Pred:', metrics_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    labels = {idx: name for idx, name in enumerate(gt_regions)}
    pos = nx.spring_layout(Gg, seed=42)
    nx.draw_networkx(Gg, pos=pos, ax=axes[0], with_labels=True, labels=labels,
                     node_color='lightblue', edge_color='gray', font_size=7, node_size=650)
    axes[0].set_title(f'Grafo GT (|corr| >= {graph_thresh})')
    nx.draw_networkx(Gp, pos=pos, ax=axes[1], with_labels=True, labels=labels,
                     node_color='lightgreen', edge_color='gray', font_size=7, node_size=650)
    axes[1].set_title(f'Grafo Pred (|corr| >= {graph_thresh})')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    
    import numpy as np
    import pandas as pd
    from sklearn.cross_decomposition import CCA
    
    _gt_aligned, _pred_aligned = _resample_pred_to_gt(gt_array_denorm, data_predicted)
    num_seq = _gt_aligned.shape[0]
    cca_results = []
    for seq_idx in range(num_seq):
        X = _gt_aligned[seq_idx]
        Y = _pred_aligned[seq_idx]
        mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        X = X[mask]
        Y = Y[mask]
        if X.shape[0] < 10:
            cca_results.append({'sequence_id': seq_idx, 'first_corr': np.nan, 'mean_corr': np.nan})
            continue
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
        Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-9)
        n_components = min(5, X.shape[1], Y.shape[1], X.shape[0]-1)
        cca = CCA(n_components=n_components, max_iter=1000)
        X_c, Y_c = cca.fit_transform(X, Y)
        corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
        cca_results.append({
            'sequence_id': seq_idx,
            'first_corr': float(corrs[0]),
            'mean_corr': float(np.mean(corrs))
        })
    cca_df = pd.DataFrame(cca_results)
    display(cca_df.round(3))
    
    plt.figure(figsize=(6, 3))
    plt.hist(cca_df['first_corr'].dropna(), bins=20, color='tab:green', edgecolor='k', alpha=0.7)
    plt.xlabel('First canonical correlation')
    plt.ylabel('Count')
    plt.title('Per-sequence CCA alignment')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import CCA
    from numpy.linalg import svd
    
    _gt_aligned, _pred_aligned = _resample_pred_to_gt(gt_array_denorm, data_predicted)
    num_seq = _gt_aligned.shape[0]
    
    cca_subspaces = []
    for seq_idx in range(num_seq):
        X = _gt_aligned[seq_idx]
        Y = _pred_aligned[seq_idx]
        mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        X, Y = X[mask], Y[mask]
        if X.shape[0] < 10:
            cca_subspaces.append(None)
            continue
    
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
        Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-9)
    
        n_components = min(5, X.shape[1], Y.shape[1], X.shape[0]-1)
        cca = CCA(n_components=n_components, max_iter=1000)
        cca.fit(X, Y)
    
        cca_subspaces.append(cca.x_weights_)
    
    def subspace_similarity(A, B):
        """Return mean cos(theta) between two subspaces."""
        Ux, _, _ = svd(A, full_matrices=False)
        Uy, _, _ = svd(B, full_matrices=False)
        k = min(Ux.shape[1], Uy.shape[1])
        _, s, _ = svd(Ux[:, :k].T @ Uy[:, :k], full_matrices=False)
        return np.mean(s)  # mean cos(theta_i)
    
    valid_idx = [i for i, sub in enumerate(cca_subspaces) if sub is not None]
    n_valid = len(valid_idx)
    S = np.zeros((n_valid, n_valid))
    for i, idx_i in enumerate(valid_idx):
        for j, idx_j in enumerate(valid_idx):
            if i <= j:
                sim = subspace_similarity(cca_subspaces[idx_i], cca_subspaces[idx_j])
                S[i, j] = S[j, i] = sim
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(S, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Subspace similarity (mean cos θ)')
    ax.set_title('Pairwise alignment of per-sequence CCA subspaces')
    ax.set_xlabel('Sequence index')
    ax.set_ylabel('Sequence index')
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    def _prepare_flat(gt_arr, pred_arr, zscore=True):
        """
        Flatten GT and Pred arrays across sequences and time, ensuring matching shape and valid samples.
    
        Parameters
        ----------
        gt_arr : np.ndarray
            Ground-truth data of shape (n_seq, n_time, n_regions).
        pred_arr : np.ndarray
            Predicted data of shape (n_seq, n_time, n_regions).
        zscore : bool, optional
            Whether to z-score features (columns) after flattening. Default True.
    
        Returns
        -------
        X_flat : np.ndarray
            Flattened GT data of shape (n_seq*n_time_valid, n_regions).
        Y_flat : np.ndarray
            Flattened Pred data of shape (n_seq*n_time_valid, n_regions).
        """
        assert gt_arr.shape == pred_arr.shape, \
            f"Shape mismatch: GT {gt_arr.shape} vs Pred {pred_arr.shape}"
    
        X_flat = gt_arr.reshape(-1, gt_arr.shape[-1])
        Y_flat = pred_arr.reshape(-1, pred_arr.shape[-1])
    
        mask = np.isfinite(X_flat).all(axis=1) & np.isfinite(Y_flat).all(axis=1)
        X_flat, Y_flat = X_flat[mask], Y_flat[mask]
    
        if zscore:
            X_flat = (X_flat - X_flat.mean(axis=0)) / (X_flat.std(axis=0) + 1e-9)
            Y_flat = (Y_flat - Y_flat.mean(axis=0)) / (Y_flat.std(axis=0) + 1e-9)
    
        print(f"Aplanado y filtrado → {X_flat.shape[0]} muestras válidas, {X_flat.shape[1]} regiones.")
        return X_flat, Y_flat
    
    
    def subsample_time(gt_arr, pred_arr, step=80):
        gt_ss = gt_arr[:, ::step, :]
        pred_ss = pred_arr[:, ::step, :]
        print(f'Submuestreo cada {step} frames → GT {gt_ss.shape}, Pred {pred_ss.shape}')
        return gt_ss, pred_ss
    
    gt_subsampled, pred_subsampled = subsample_time(gt_array_denorm, data_predicted, step=80)
    X_flat_sub, Y_flat_sub = _prepare_flat(gt_subsampled, pred_subsampled)
    print(f'Datos aplanados tras submuestreo: GT {X_flat_sub.shape}, Pred {Y_flat_sub.shape}')
    
    
    
    
    
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csgraph
    
    def _knn_graph(data, k):
        nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
        nn.fit(data)
        _, indices = nn.kneighbors(data)
        n = data.shape[0]
        adj = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in indices[i, 1:]:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        return adj
    
    def _laplacian_eigs(adj, top=12):
        L = csgraph.laplacian(adj, normed=True)
        evals, _ = np.linalg.eigh(L)
        idx = np.argsort(evals)[:top]
        return evals[idx]
    
    k_neighbors = 8
    adj_gt_sub = _knn_graph(X_flat_sub, k_neighbors)
    adj_pred_sub = _knn_graph(Y_flat_sub, k_neighbors)
    overlap = np.logical_and(adj_gt_sub, adj_pred_sub).sum()
    union = np.logical_or(adj_gt_sub, adj_pred_sub).sum()
    knn_jaccard_sub = float(overlap / union) if union else 1.0
    evals_gt_sub = _laplacian_eigs(adj_gt_sub)
    evals_pred_sub = _laplacian_eigs(adj_pred_sub)
    spectral_rmse_sub = float(np.sqrt(np.mean((evals_gt_sub - evals_pred_sub) ** 2)))
    spectral_score_sub = float(1.0 / (1.0 + spectral_rmse_sub))
    
    print(f'[Submuestreo] k-NN edge Jaccard: {knn_jaccard_sub:.3f}')
    print('[Submuestreo] Eigenvalues GT:', np.round(evals_gt_sub, 4))
    print('[Submuestreo] Eigenvalues Pred:', np.round(evals_pred_sub, 4))
    print(f'[Submuestreo] Spectral RMSE: {spectral_rmse_sub:.4f} -> Score 1/(1+RMSE): {spectral_score_sub:.3f}')
    
    plt.figure(figsize=(6,4))
    plt.plot(evals_gt_sub, label='GT (sub)', marker='o')
    plt.plot(evals_pred_sub, label='Pred (sub)', marker='x')
    plt.title('Eigenvalores del Laplaciano con submuestreo step=4')
    plt.xlabel('Índice ordenado')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csgraph
    import pandas as pd
    
    def _knn_graph(data, k):
        """Binary symmetric k-NN adjacency."""
        nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
        nn.fit(data)
        _, indices = nn.kneighbors(data)
        n = data.shape[0]
        adj = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in indices[i, 1:]:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        return adj
    
    def _laplacian_eigs(adj, top=12):
        """Return smallest eigenvalues of normalized Laplacian."""
        L = csgraph.laplacian(adj, normed=True)
        evals, _ = np.linalg.eigh(L)
        idx = np.argsort(evals)[:top]
        return evals[idx]
    
    _gt_aligned, _pred_aligned = _resample_pred_to_gt(gt_array_denorm, data_predicted)
    num_seq = _gt_aligned.shape[0]
    k_neighbors = 8
    top_eigs = 12
    
    results = []
    
    for seq_idx in range(num_seq):
        X = _gt_aligned[seq_idx]
        Y = _pred_aligned[seq_idx]
        mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        X, Y = X[mask], Y[mask]
        if X.shape[0] < k_neighbors + 2:
            results.append({'sequence_id': seq_idx, 'knn_jaccard': np.nan,
                            'spectral_rmse': np.nan, 'spectral_score': np.nan})
            continue
    
        adj_gt = _knn_graph(X, k_neighbors)
        adj_pr = _knn_graph(Y, k_neighbors)
    
        overlap = np.logical_and(adj_gt, adj_pr).sum()
        union = np.logical_or(adj_gt, adj_pr).sum()
        knn_jaccard = float(overlap / union) if union else np.nan
    
        evals_gt = _laplacian_eigs(adj_gt, top=top_eigs)
        evals_pr = _laplacian_eigs(adj_pr, top=top_eigs)
        spectral_rmse = float(np.sqrt(np.mean((evals_gt - evals_pr) ** 2)))
        spectral_score = float(1.0 / (1.0 + spectral_rmse))
    
        results.append({
            'sequence_id': seq_idx,
            'knn_jaccard': knn_jaccard,
            'spectral_rmse': spectral_rmse,
            'spectral_score': spectral_score
        })
    
    df = pd.DataFrame(results)
    display(df.describe().round(4))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].hist(df['knn_jaccard'].dropna(), bins=20, color='tab:blue', edgecolor='k', alpha=0.7)
    axes[0].set_xlabel('k-NN edge Jaccard (GT vs Pred)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Local topology alignment across sequences')
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(df['spectral_rmse'].dropna(), bins=20, color='tab:orange', edgecolor='k', alpha=0.7)
    axes[1].set_xlabel('Spectral RMSE')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Laplacian spectrum difference')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(6,3))
    plt.plot(df['sequence_id'], df['spectral_score'], 'o-', label='Spectral score')
    plt.plot(df['sequence_id'], df['knn_jaccard'], 'o-', label='kNN Jaccard')
    plt.xlabel('Sequence index')
    plt.ylabel('Score')
    plt.title('Per-sequence manifold alignment')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    
    
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA
    from scipy.sparse import csgraph
    from scipy.linalg import orthogonal_procrustes
    from scipy.stats import wasserstein_distance
    
    rng = np.random.default_rng(42)
    
    def _knn_graph(data, k=8):
        nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(data)
        _, idx = nn.kneighbors(data)
        n = data.shape[0]
        A = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in idx[i, 1:]:
                A[i, j] = 1.0
                A[j, i] = 1.0
        return A
    
    def _local_topology_metrics(X, Y, k=8):
        A1, A2 = _knn_graph(X, k), _knn_graph(Y, k)
        overlap = np.logical_and(A1, A2).sum()
        union = np.logical_or(A1, A2).sum()
        jaccard = float(overlap / union) if union else np.nan
        deg1, deg2 = A1.sum(1), A2.sum(1)
        deg_corr = np.corrcoef(deg1, deg2)[0,1]
        return jaccard, deg_corr, A1, A2
    
    def _covariance_similarity(X, Y):
        cov_gt, cov_pr = np.cov(X.T), np.cov(Y.T)
        eval_gt = np.linalg.eigvalsh(cov_gt)
        eval_pr = np.linalg.eigvalsh(cov_pr)
        eval_gt /= eval_gt.sum() + 1e-9
        eval_pr /= eval_pr.sum() + 1e-9
        rmse = np.sqrt(np.mean((eval_gt - eval_pr)**2))
        score = 1 / (1 + rmse)
        return rmse, score, eval_gt, eval_pr
    
    def _geodesic_distance_distribution(X, k=10):
        nn = NearestNeighbors(n_neighbors=k+1).fit(X)
        A = nn.kneighbors_graph(X).toarray()
        D = csgraph.dijkstra(A, directed=False)
        d = D[np.triu_indices_from(D, 1)]
        return d[np.isfinite(d)]
    
    def _intrinsic_curvature_metrics(X, Y, k=10):
        d1, d2 = _geodesic_distance_distribution(X, k), _geodesic_distance_distribution(Y, k)
        min_len = min(len(d1), len(d2))
        corr = np.corrcoef(d1[:min_len], d2[:min_len])[0,1]
        emd = wasserstein_distance(d1, d2)
        score = 1 / (1 + emd)
        return corr, score, d1, d2
    
    def _all_metrics(X, Y, label="Raw", k=8):
        print(f"\n=== {label} ===")
        jaccard, deg_corr, A1, A2 = _local_topology_metrics(X, Y, k)
        cov_rmse, cov_score, eval_gt, eval_pr = _covariance_similarity(X, Y)
        geo_corr, geo_score, d1, d2 = _intrinsic_curvature_metrics(X, Y, k)
        print(f"Local topology:")
        print(f"   k-NN Jaccard:                 {jaccard:8.4f}")
        print(f"   Degree distribution corr:     {deg_corr:8.4f}")
        print(f"Linear geometry:")
        print(f"   Covariance spectrum RMSE:     {cov_rmse:8.4f}")
        print(f"   Covariance spectral score:    {cov_score:8.4f}")
        print(f"Intrinsic curvature:")
        print(f"   Geodesic distance corr:       {geo_corr:8.4f}")
        print(f"   Geodesic EMD score:           {geo_score:8.4f}")
        return dict(jaccard=jaccard, deg_corr=deg_corr, cov_rmse=cov_rmse,
                    cov_score=cov_score, geo_corr=geo_corr, geo_score=geo_score)
    
    raw_metrics = _all_metrics(X_flat_sub, Y_flat_sub, label="Raw (Unaligned)", k=8)
    
    pca = PCA(n_components=min(8, X_flat_sub.shape[1]), random_state=42)
    Zx = pca.fit_transform(X_flat_sub)
    Zy = pca.transform(Y_flat_sub)
    R, s = orthogonal_procrustes(Zy, Zx)
    Y_aligned = Zy @ R * s
    
    aligned_metrics = _all_metrics(Zx, Y_aligned, label="After PCA+Procrustes Alignment", k=8)
    
    import pandas as pd
    summary = pd.DataFrame([raw_metrics, aligned_metrics], index=["Raw", "Aligned"])
    print("\n=== Summary ===")
    display(summary)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,3, figsize=(13,4))
    
    ax[0].plot(np.sort(aligned_metrics['cov_score'] and np.linspace(0,1,10)), 'w') # dummy to keep spacing
    eval_gt = np.linalg.eigvalsh(np.cov(Zx.T))
    eval_al = np.linalg.eigvalsh(np.cov(Y_aligned.T))
    eval_gt /= eval_gt.sum(); eval_al /= eval_al.sum()
    ax[0].plot(np.sort(eval_gt)[::-1], '-o', label='GT')
    ax[0].plot(np.sort(eval_al)[::-1], '-o', label='Pred aligned')
    ax[0].set_title(f'Covariance spectra\n(Raw score={raw_metrics["cov_score"]:.2f}, Aligned={aligned_metrics["cov_score"]:.2f})')
    ax[0].legend(); ax[0].set_xlabel('Component'); ax[0].set_ylabel('Norm var')
    
    A_gt = _knn_graph(Zx); A_pr = _knn_graph(Y_aligned)
    ax[1].hist(A_gt.sum(1), bins=20, alpha=0.5, label='GT')
    ax[1].hist(A_pr.sum(1), bins=20, alpha=0.5, label='Pred aligned')
    ax[1].set_title('k-NN degree distributions'); ax[1].set_xlabel('Degree'); ax[1].set_ylabel('Count'); ax[1].legend()
    
    d_gt = _geodesic_distance_distribution(Zx)
    d_pr = _geodesic_distance_distribution(Y_aligned)
    ax[2].hist(d_gt, bins=50, alpha=0.5, label='GT', density=True)
    ax[2].hist(d_pr, bins=50, alpha=0.5, label='Pred aligned', density=True)
    ax[2].set_title('Geodesic distance distributions'); ax[2].legend(); ax[2].set_xlabel('Distance'); ax[2].set_ylabel('Density')
    
    plt.tight_layout()
    plt.show()
    
    
    
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap
    from scipy.linalg import orthogonal_procrustes
    
    rng = np.random.default_rng(42)
    try:
        X_base = X_flat_sub
        Y_base = Y_flat_sub
        print('Usando datos submuestreados para el análisis (X_flat_sub / Y_flat_sub).')
    except NameError:
        X_base, Y_base = _prepare_flat(gt_array_denorm, data_predicted)
        print('No hay submuestreo disponible; usando todos los frames con muestreo aleatorio.')
    
    max_pca_samples = 8000
    idx = rng.choice(X_base.shape[0], size=min(max_pca_samples, X_base.shape[0]), replace=False)
    X_p = X_base[idx]
    Y_p = Y_base[idx]
    pca_dim = min(8, X_p.shape[1])
    pca = PCA(n_components=pca_dim, random_state=42)
    Z_gt = pca.fit_transform(X_p)
    Z_pred = pca.transform(Y_p)
    R, scale = orthogonal_procrustes(Z_pred, Z_gt)
    Z_pred_aligned = Z_pred @ R * scale
    residual = np.linalg.norm(Z_gt - Z_pred_aligned, ord='fro')
    baseline = np.linalg.norm(Z_gt, ord='fro') + 1e-9
    procrustes_score = float(1.0 - residual / baseline)
    print(f'Procrustes residual Frobenius (subset): {residual:.2f} / {baseline:.2f} -> Score {procrustes_score:.3f}')
    
    max_isomap_samples = 2000
    idx_iso = rng.choice(X_base.shape[0], size=min(max_isomap_samples, X_base.shape[0]), replace=False)
    X_iso = X_base[idx_iso]
    Y_iso = Y_base[idx_iso]
    isomap = Isomap(n_neighbors=10, n_components=4)
    iso_gt = isomap.fit_transform(X_iso)
    D_gt = isomap.dist_matrix_.copy()
    iso_pred = isomap.fit_transform(Y_iso)
    D_pred = isomap.dist_matrix_.copy()
    mask = np.isfinite(D_gt) & np.isfinite(D_pred)
    flat_D_gt = D_gt[mask]
    flat_D_pred = D_pred[mask]
    if flat_D_gt.size >= 2:
        geo_corr = pearsonr(flat_D_gt, flat_D_pred)[0]
        geo_rmse = float(np.sqrt(np.nanmean((flat_D_gt - flat_D_pred) ** 2)))
    else:
        geo_corr = np.nan
        geo_rmse = np.nan
    print(f'Geodesic distance correlation (Isomap subset): {geo_corr:.3f}')
    print(f'Geodesic distance RMSE (subset): {geo_rmse:.2f}')
    
    
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    try:
        from umap import UMAP
        umap_constructor = UMAP
    except ImportError:
        from sklearn.manifold import TSNE
        umap_constructor = lambda **kwargs: TSNE(n_components=kwargs.get('n_components', 2), random_state=kwargs.get('random_state', 42))
        print('umap-learn not installed; falling back to t-SNE for embedding.')
    
    
    def _align_regions_for_embedding(gt_arr, pred_arr, gt_regions, pred_regions):
        labels = gt_regions if pred_regions == gt_regions else [r for r in gt_regions if r in pred_regions]
        if not labels:
            raise ValueError('No overlapping regions between GT and predictions for embedding analysis.')
        gt_idx = [gt_regions.index(r) for r in labels]
        pred_idx = [pred_regions.index(r) for r in labels]
        return labels, gt_arr[..., gt_idx], pred_arr[..., pred_idx]
    
    def _circular_shuffle_per_region(arr, rng):
        arr = np.asarray(arr)
        shuffled = arr.copy()
        n_time = arr.shape[1]
        shifts = rng.integers(0, n_time, size=arr.shape[2])
        for region_idx, shift in enumerate(shifts):
            if shift == 0:
                continue
            shuffled[..., region_idx] = np.roll(shuffled[..., region_idx], shift=shift, axis=1)
        return shuffled
    
    region_labels, gt_aligned, pred_aligned = _align_regions_for_embedding(
        gt_array_denorm, data_predicted, gt_regions, pred_region_names
    )
    
    rng = np.random.default_rng(42)
    gt_shuffled = _circular_shuffle_per_region(gt_aligned, rng)
    
    gt_flat = _flatten_sequences(gt_aligned)
    pred_flat = _flatten_sequences(pred_aligned)
    shuffled_flat = _flatten_sequences(gt_shuffled)
    
    reducer = umap_constructor(n_components=2, random_state=42)
    embedding_ref = reducer.fit_transform(shuffled_flat)
    embedding_gt = reducer.transform(gt_flat) if hasattr(reducer, 'transform') else reducer.fit_transform(gt_flat)
    embedding_pred = reducer.transform(pred_flat) if hasattr(reducer, 'transform') else reducer.fit_transform(pred_flat)
    
    def _maybe_subsample(points, labels, max_points=5000):
        if points.shape[0] <= max_points:
            return points, labels
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        return points[idx], labels[idx]
    
    plot_data = []
    plot_data.append((embedding_ref, np.full(embedding_ref.shape[0], 'Shuffled GT', dtype=object)))
    plot_data.append((embedding_gt, np.full(embedding_gt.shape[0], 'Ground Truth', dtype=object)))
    plot_data.append((embedding_pred, np.full(embedding_pred.shape[0], 'Predicted', dtype=object)))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'Shuffled GT': '#bbbbbb', 'Ground Truth': '#1f77b4', 'Predicted': '#ff7f0e'}
    for emb, label_arr in plot_data:
        emb, label_arr = _maybe_subsample(emb, label_arr)
        ax.scatter(emb[:, 0], emb[:, 1], s=8, alpha=0.3, label=label_arr[0], c=colors.get(label_arr[0], None))
    
    ax.set_title('Embedding comparison (UMAP trained on circularly shuffled GT)')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    
    print('UMAP fitted on circularly shuffled GT; lower separation between GT and predictions suggests closer distributional alignment.')
    
    
    
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    _required = []
    _missing = [name for name in _required if name not in globals()]
    if _missing:
        raise RuntimeError(f'Missing variables from previous cell: {_missing}. Run the UMAP cell first.')
    if not hasattr(reducer, 'transform'):
        raise RuntimeError('Reducer lacks transform(). Install umap-learn to compare trajectories in a shared embedding space.')
    
    n_seq, n_time = gt_aligned.shape[:2]
    embed_dim = embedding_gt.shape[1]
    embedding_gt_seq = embedding_gt.reshape(n_seq, n_time, embed_dim)
    embedding_pred_seq = embedding_pred.reshape(n_seq, n_time, embed_dim)
    embedding_ref_seq = embedding_ref.reshape(n_seq, n_time, embed_dim)
    
    def _path_length(traj):
        steps = np.diff(traj, axis=0)
        return np.linalg.norm(steps, axis=1).sum()
    
    def _path_dispersion(traj):
        centroid = np.mean(traj, axis=0)
        return np.linalg.norm(traj - centroid, axis=1).mean()
    
    def _step_cosine(gt_traj, pred_traj, eps=1e-9):
        gt_steps = np.diff(gt_traj, axis=0)
        pred_steps = np.diff(pred_traj, axis=0)
        if gt_steps.size == 0 or pred_steps.size == 0:
            return np.nan
        dot = np.sum(gt_steps * pred_steps, axis=1)
        norms = np.linalg.norm(gt_steps, axis=1) * np.linalg.norm(pred_steps, axis=1) + eps
        return np.mean(dot / norms)
    
    def _per_time_distance(gt_traj, pred_traj):
        deltas = np.linalg.norm(gt_traj - pred_traj, axis=1)
        return deltas.mean(), deltas.std(), deltas
    
    metrics = []
    all_deltas = []
    for seq_idx in range(n_seq):
        gt_traj = embedding_gt_seq[seq_idx]
        pred_traj = embedding_pred_seq[seq_idx]
        ref_traj = embedding_ref_seq[seq_idx]
        mean_dist, std_dist, deltas = _per_time_distance(gt_traj, pred_traj)
        all_deltas.append(deltas)
        metrics.append({
            'sequence_id': seq_idx,
            'path_length_gt': _path_length(gt_traj),
            'path_length_pred': _path_length(pred_traj),
            'path_length_shuffled': _path_length(ref_traj),
            'mean_gt_pred_distance': mean_dist,
            'std_gt_pred_distance': std_dist,
            'gt_path_dispersion': _path_dispersion(gt_traj),
            'pred_path_dispersion': _path_dispersion(pred_traj),
            'shuffled_path_dispersion': _path_dispersion(ref_traj),
            'step_direction_cosine': _step_cosine(gt_traj, pred_traj)
        })
    metrics_df = pd.DataFrame(metrics)
    display(metrics_df.round(4))
    
    all_deltas = np.concatenate(all_deltas)
    print(
        f'Overall mean distance: {np.mean(all_deltas):.4f} | median: {np.median(all_deltas):.4f} | 95th pct: {np.percentile(all_deltas, 95):.4f}'
    )
    
    n_plot = min(4, n_seq)
    fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 4), sharey=True)
    if n_plot == 1:
        axes = [axes]
    for ax, seq_idx in zip(axes, range(n_plot)):
        gt_traj = embedding_gt_seq[seq_idx]
        pred_traj = embedding_pred_seq[seq_idx]
        ref_traj = embedding_ref_seq[seq_idx]
        ax.plot(ref_traj[:, 0], ref_traj[:, 1], color='#bbbbbb', linewidth=1, alpha=0.6, label='Shuffled')
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], '-o', markersize=3, linewidth=1.5, label='GT')
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], '-o', markersize=3, linewidth=1.5, label='Pred', color='#ff7f0e')
        ax.scatter(gt_traj[0, 0], gt_traj[0, 1], color='green', s=25, marker='s', label='GT start' if seq_idx == 0 else None)
        ax.scatter(pred_traj[0, 0], pred_traj[0, 1], color='purple', s=25, marker='s', label='Pred start' if seq_idx == 0 else None)
        ax.set_title(f'Sequence {seq_idx}')
        ax.set_xlabel('Component 1')
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel('Component 2')
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=4)
    fig.suptitle('Trajectory comparison in embedding space', y=1.05)
    fig.tight_layout()
    
    plt.show()
    print('Trajectory metrics compare GT vs predictions relative to the shuffled baseline; lower distances and similar path lengths suggest closer dynamical behaviour.')
    
    
    
    
    import numpy as np
    import pandas as pd
    
    _required = []
    _missing = [name for name in _required if name not in globals()]
    if _missing:
        raise RuntimeError(f'Required trajectory tensors not found: {_missing}. Run the trajectory cell first.')
    
    def _dtw_distance(traj_a, traj_b):
        n_a, n_b = len(traj_a), len(traj_b)
        cost = np.full((n_a + 1, n_b + 1), np.inf, dtype=np.float64)
        cost[0, 0] = 0.0
        for i in range(1, n_a + 1):
            for j in range(1, n_b + 1):
                dist = np.linalg.norm(traj_a[i - 1] - traj_b[j - 1])
                cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
        return cost[n_a, n_b] / (n_a + n_b)
    
    def _discrete_frechet(traj_a, traj_b):
        n_a, n_b = len(traj_a), len(traj_b)
        ca = np.zeros((n_a, n_b), dtype=np.float64)
        ca[0, 0] = np.linalg.norm(traj_a[0] - traj_b[0])
        for i in range(1, n_a):
            ca[i, 0] = max(ca[i - 1, 0], np.linalg.norm(traj_a[i] - traj_b[0]))
        for j in range(1, n_b):
            ca[0, j] = max(ca[0, j - 1], np.linalg.norm(traj_a[0] - traj_b[j]))
        for i in range(1, n_a):
            for j in range(1, n_b):
                dist = np.linalg.norm(traj_a[i] - traj_b[j])
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), dist)
        return ca[-1, -1]
    
    def _velocity_alignment(traj_a, traj_b, eps=1e-9):
        vel_a = np.diff(traj_a, axis=0)
        vel_b = np.diff(traj_b, axis=0)
        min_len = min(len(vel_a), len(vel_b))
        if min_len == 0:
            return np.nan
        vel_a = vel_a[:min_len]
        vel_b = vel_b[:min_len]
        dot = np.sum(vel_a * vel_b, axis=1)
        norms = np.linalg.norm(vel_a, axis=1) * np.linalg.norm(vel_b, axis=1) + eps
        return np.mean(dot / norms)
    
    def _accel_alignment(traj_a, traj_b, eps=1e-9):
        acc_a = np.diff(np.diff(traj_a, axis=0), axis=0)
        acc_b = np.diff(np.diff(traj_b, axis=0), axis=0)
        min_len = min(len(acc_a), len(acc_b))
        if min_len == 0:
            return np.nan
        acc_a = acc_a[:min_len]
        acc_b = acc_b[:min_len]
        dot = np.sum(acc_a * acc_b, axis=1)
        norms = np.linalg.norm(acc_a, axis=1) * np.linalg.norm(acc_b, axis=1) + eps
        return np.mean(dot / norms)
    
    def _directed_hausdorff(traj_a, traj_b):
        from scipy.spatial.distance import cdist
        dists = cdist(traj_a, traj_b)
        return float(np.max(np.min(dists, axis=1)))
    
    records = []
    for seq_idx in range(embedding_gt_seq.shape[0]):
        gt_traj = embedding_gt_seq[seq_idx]
        pred_traj = embedding_pred_seq[seq_idx]
        shuffled_traj = embedding_ref_seq[seq_idx]
        records.append({
            'sequence_id': seq_idx,
            'dtw_gt_pred': _dtw_distance(gt_traj, pred_traj),
            'dtw_gt_shuffled': _dtw_distance(gt_traj, shuffled_traj),
            'frechet_gt_pred': _discrete_frechet(gt_traj, pred_traj),
            'frechet_gt_shuffled': _discrete_frechet(gt_traj, shuffled_traj),
            'directed_hausdorff_gt_pred': _directed_hausdorff(gt_traj, pred_traj),
            'directed_hausdorff_pred_gt': _directed_hausdorff(pred_traj, gt_traj),
            'velocity_alignment': _velocity_alignment(gt_traj, pred_traj),
            'acceleration_alignment': _accel_alignment(gt_traj, pred_traj)
        })
    advanced_metrics_df = pd.DataFrame(records)
    display(advanced_metrics_df)
    
    summary = advanced_metrics_df.describe(percentiles=[0.5, 0.9]).round(4)
    print('Metric summary (per column):')
    display(summary)
    
    print('Lower DTW/Frechet/Hausdorff vs GT indicate trajectories closer to GT than shuffled; alignment scores closer to 1 imply similar dynamics.')
    
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    
    _required = []
    _missing = [name for name in _required if name not in globals()]
    if _missing:
        raise RuntimeError(f'Required trajectory tensors not found: {_missing}. Run the trajectory cell first.')
    
    def animate_sequence(seq_idx=0, include_shuffled=True, fps=12, tail=25, save_path=None):
        if seq_idx >= embedding_gt_seq.shape[0]:
            raise IndexError(f'seq_idx {seq_idx} out of range (n_seq={embedding_gt_seq.shape[0]}).')
        gt_traj = embedding_gt_seq[seq_idx]
        pred_traj = embedding_pred_seq[seq_idx]
        shuffled_traj = embedding_ref_seq[seq_idx]
        n_frames = gt_traj.shape[0]
    
        if include_shuffled:
            all_points = np.concatenate([gt_traj, pred_traj, shuffled_traj], axis=0)
        else:
            all_points = np.concatenate([gt_traj, pred_traj], axis=0)
        padding = 0.05 * (all_points.max(axis=0) - all_points.min(axis=0) + 1e-9)
        mins = all_points.min(axis=0) - padding
        maxs = all_points.max(axis=0) + padding
    
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_title(f'Embedding trajectory animation (sequence {seq_idx})')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.grid(True, alpha=0.2)
    
        line_gt, = ax.plot([], [], color='#1f77b4', linewidth=2, label='GT path')
        point_gt, = ax.plot([], [], marker='o', color='#1f77b4', markersize=6)
        line_pred, = ax.plot([], [], color='#ff7f0e', linewidth=2, label='Pred path')
        point_pred, = ax.plot([], [], marker='o', color='#ff7f0e', markersize=6)
    
        if include_shuffled:
            line_shuffled, = ax.plot([], [], color='#bbbbbb', linewidth=1.5, linestyle='--', alpha=0.6, label='Shuffled path')
        else:
            line_shuffled = None
    
        tail = max(1, tail)
    
        def _update(frame):
            start = max(0, frame - tail)
            gt_segment = gt_traj[start:frame + 1]
            pred_segment = pred_traj[start:frame + 1]
            line_gt.set_data(gt_segment[:, 0], gt_segment[:, 1])
            line_pred.set_data(pred_segment[:, 0], pred_segment[:, 1])
            point_gt.set_data([gt_traj[frame, 0]], [gt_traj[frame, 1]])
            point_pred.set_data([pred_traj[frame, 0]], [pred_traj[frame, 1]])
            artists = [line_gt, point_gt, line_pred, point_pred]
            if include_shuffled and line_shuffled is not None:
                shuffled_segment = shuffled_traj[start:frame + 1]
                line_shuffled.set_data(shuffled_segment[:, 0], shuffled_segment[:, 1])
                artists.append(line_shuffled)
            return artists
    
        anim = FuncAnimation(fig, _update, frames=n_frames, interval=1000 / fps, blit=True)
    
        try:
            html = HTML(anim.to_jshtml())
        finally:
            plt.close(fig)
    
        if save_path:
            try:
                anim.save(save_path, writer='ffmpeg', dpi=120)
                print(f'Animation saved to {save_path}')
            except Exception as exc:
                print(f'Could not save animation to {save_path}: {exc}')
    
        return html
    
    preview_animation = animate_sequence(seq_idx=0, include_shuffled=True, tail=40)
    display(preview_animation)
    print('Use animate_sequence(seq_idx, include_shuffled, fps, tail, save_path) for custom previews or exports.')
    
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.random import default_rng
    
    _required = []
    _missing = [name for name in _required if name not in globals()]
    if _missing:
        raise RuntimeError(f'Missing trajectory embeddings: {_missing}. Run the UMAP trajectory cell first.')
    
    def _downsample_traj(traj, max_len=200):
        if traj.shape[0] <= max_len:
            return traj
        idx = np.linspace(0, traj.shape[0] - 1, max_len).astype(int)
        return traj[idx]
    
    def _dtw_distance(traj_a, traj_b):
        n_a, n_b = len(traj_a), len(traj_b)
        cost = np.full((n_a + 1, n_b + 1), np.inf, dtype=np.float64)
        cost[0, 0] = 0.0
        for i in range(1, n_a + 1):
            ta = traj_a[i - 1]
            row_prev = cost[i - 1]
            row_curr = cost[i]
            for j in range(1, n_b + 1):
                tb = traj_b[j - 1]
                dist = np.linalg.norm(ta - tb)
                prev = row_prev[j]
                if row_curr[j - 1] < prev:
                    prev = row_curr[j - 1]
                if row_prev[j - 1] < prev:
                    prev = row_prev[j - 1]
                row_curr[j] = dist + prev
        return cost[n_a, n_b] / (n_a + n_b)
    
    rng = default_rng(123)
    emb_gt_full = embedding_gt_seq
    emb_pred_full = embedding_pred_seq
    n_seq = emb_gt_full.shape[0]
    
    max_seq = min(n_seq, 12)
    seq_indices = rng.choice(n_seq, size=max_seq, replace=False)
    max_len = 200
    emb_gt = [ _downsample_traj(emb_gt_full[i], max_len) for i in seq_indices ]
    emb_pred = [ _downsample_traj(emb_pred_full[i], max_len) for i in seq_indices ]
    
    actual_distances = np.array([_dtw_distance(emb_gt[i], emb_pred[i]) for i in range(max_seq)])
    
    num_permutations = 6
    null_samples = np.empty((num_permutations, max_seq), dtype=np.float64)
    for p in range(num_permutations):
        perm_seq = rng.choice(n_seq, size=max_seq, replace=True)
        perm_time = rng.permutation(emb_pred_full.shape[1])[:max_len]
        for i, seq_id in enumerate(seq_indices):
            shuffled = _downsample_traj(emb_pred_full[perm_seq[i], perm_time], max_len)
            null_samples[p, i] = _dtw_distance(emb_gt[i], shuffled)
    
    p_values = np.mean(null_samples <= actual_distances, axis=0)
    print('DTW distances (mean ± std):', float(actual_distances.mean()), float(actual_distances.std()))
    print('Permutation-based p-values (fraction null <= actual):')
    print(np.round(p_values, 3))
    
    plt.figure(figsize=(6, 4))
    plt.hist(null_samples.flatten(), bins=20, alpha=0.6, label='Shuffled null')
    plt.axvline(actual_distances.mean(), color='red', linewidth=2, label='Actual mean')
    plt.xlabel('DTW distance')
    plt.ylabel('Frequency')
    plt.title('Trajectory DTW comparison vs shuffled baseline')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal
    
    fs = 1.0  # sampling frequency (adjust if known)
    _gt_aligned, _pred_aligned = _resample_pred_to_gt(gt_array_denorm, data_predicted)
    regions_to_plot = gt_regions[:min(4, len(gt_regions))]
    psd_results = []
    for region in regions_to_plot:
        idx = gt_regions.index(region)
        gt_vals = _gt_aligned[:, :, idx].reshape(-1)
        pred_vals = _pred_aligned[:, :, idx].reshape(-1)
        gt_vals = gt_vals[np.isfinite(gt_vals)]
        pred_vals = pred_vals[np.isfinite(pred_vals)]
        f_gt, Pxx_gt = signal.welch(gt_vals, fs=fs, nperseg=min(256, gt_vals.size))
        f_pr, Pxx_pr = signal.welch(pred_vals, fs=fs, nperseg=min(256, pred_vals.size))
        psd_results.append((region, f_gt, Pxx_gt, Pxx_pr))
    
    fig, axes = plt.subplots(len(psd_results), 1, figsize=(6, 3 * len(psd_results)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (region, freqs, gt_psd, pr_psd) in zip(axes, psd_results):
        ax.semilogy(freqs, gt_psd, label='GT')
        ax.semilogy(freqs, pr_psd, label='Pred')
        ax.set_ylabel('PSD')
        ax.set_title(f'Welch PSD: {region}')
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Frequency (Hz)')
    axes[0].legend()
    plt.tight_layout()
    plt.show()
    
    if len(gt_regions) >= 2:
        i0, i1 = 0, 1
        gt_x = _gt_aligned[:, :, i0].reshape(-1)
        gt_y = _gt_aligned[:, :, i1].reshape(-1)
        pr_x = _pred_aligned[:, :, i0].reshape(-1)
        pr_y = _pred_aligned[:, :, i1].reshape(-1)
        f_c, coh_gt = signal.coherence(gt_x, gt_y, fs=fs, nperseg=min(256, gt_x.size))
        _, coh_pr = signal.coherence(pr_x, pr_y, fs=fs, nperseg=min(256, pr_x.size))
        plt.figure(figsize=(6, 3))
        plt.plot(f_c, coh_gt, label='GT coherence')
        plt.plot(f_c, coh_pr, label='Pred coherence')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(f'Coherence between {gt_regions[i0]} and {gt_regions[i1]}')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
