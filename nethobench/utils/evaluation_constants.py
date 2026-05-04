"""
Centralized evaluation constants for NethoBench.

All heuristic thresholds, statistical quantiles, ML hyperparameters,
and metric-composition weights live here so they can be audited and
adjusted in one place rather than being buried in individual modules.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Data Size & Minimum Thresholds
# ---------------------------------------------------------------------------

# Minimum sequence length for cross-modal lead-lag metrics.
MIN_SAMPLES_CORR_WINDOW: int = 33

# Minimum number of finite items before _iqr_robust falls back to std.
MIN_ITEMS_IQR_ROBUST_FALLBACK: int = 10

# Minimum array size for lead-lag correlation calculations.
LEAD_LAG_MIN_ARRAY_SIZE: int = 5

# Default maximum lag for lead-lag peak detection.
LEAD_LAG_DEFAULT_MAX_LAG: int = 30

# Minimum sample sizes for KL-based histograms, Error (NRMSE), and MI scores.
MIN_SAMPLES_KL: int = 10
MIN_SAMPLES_ERROR: int = 50
MIN_SAMPLES_MI: int = 80

# Maximum time steps to subsample in MI and quantile calculations.
MAX_TIME_STEPS_SUBSAMPLE: int = 1200

# Minimum row thresholds for PCA, UMAP, Ripser, and Autocorrelation.
MIN_ROWS_PCA: int = 32
MIN_SAMPLES_MANIFOLD: int = 64
CROSS_CORR_MAX_LAG_SMALL: int = 16
MIN_ROWS_CORRCOEF: int = 16
AUTOCORR_MAX_LAG: int = 48

# Point subsampling limits for latent-cloud and manifold calculations.
LATENT_SUBSAMPLE_SIZE: int = 96
N_POINTS_MANIFOLD_PH_KNN: int = 128

# Maximum sequences used in stratified latent-cloud subsampling.
MAX_SEQUENCES_STRATIFIED: int = 48

# Maximum points subsampled in manifold alignment to prevent O(N²) memory blow-up.
MAX_POINTS_MANIFOLD_ALIGNMENT: int = 2000

# ---------------------------------------------------------------------------
# 2. Statistical Quantiles & Cutoffs
# ---------------------------------------------------------------------------

# Quantile used repeatedly for top-K ratios and tail selections.
QUANTILE_10P: float = 0.10

# IQR boundaries used across robust scaling and outlier-robust statistics.
QUANTILE_IQR_LO: float = 0.25
QUANTILE_IQR_HI: float = 0.75

# Extreme-outlier trimming bounds for histogram support calculation.
TAIL_TRIM_QUANTILE_LO: float = 0.001
TAIL_TRIM_QUANTILE_HI: float = 0.999

# Default quantile grid for distribution-distance metrics.
DISTRIBUTION_GRID_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

# Constant to convert Median Absolute Deviation (MAD) to standard deviation.
MAD_TO_STD_CONSTANT: float = 1.4826

# Percentile threshold determining whether an animal is "stationary".
STATIONARY_THRESHOLD_PERCENTILE: int = 10

# Percentile used to filter extreme speed outliers for histogram plotting.
SPEED_OUTLIER_PERCENTILE: int = 99

# ---------------------------------------------------------------------------
# 3. Machine Learning & Signal Processing Hyperparameters
# ---------------------------------------------------------------------------

# Cumulative variance threshold for selecting PCA components.
PCA_VARIANCE_THRESHOLD: float = 0.85

# Subsampling and neighbourhood hyperparameters for mutual-information matrices.
MI_MAX_POINTS: int = 1500
MI_N_NEIGHBORS: int = 5

# Dynamic Functional Connectivity state configurations:
# (window_size, step_size, n_clusters)
DFC_STATE_CONFIGS: tuple[tuple[int, int, int], ...] = (
    (45, 15, 8),
    (60, 30, 8),
    (90, 30, 8),
)

# Welch PSD parameters.
WELCH_SAMPLING_FREQUENCY: float = 30.0
WELCH_NPERSEG: int = 256

# Default number of histogram bins for scoring functions.
HISTOGRAM_BINS_DEFAULT: int = 12

# Default KMeans cluster counts for ethology trajectory and syllable scores.
KMEANS_K_TRAJECTORY: int = 6
KMEANS_K_SYLLABLE: int = 8

# Chunk size (in frames) for movement-embedding diagnostics.
CHUNK_SIZE_EMBEDDINGS: int = 5

# ---------------------------------------------------------------------------
# 4. Metric Weighting and Composition
# ---------------------------------------------------------------------------

# Weights for spectrum similarity (correlation vs. RMSE).
WEIGHT_SPECTRUM_CORR: float = 0.55
WEIGHT_SPECTRUM_RMSE: float = 0.45

# Weights for matrix similarity (correlation vs. RMSE).
WEIGHT_MATRIX_CORR: float = 0.60
WEIGHT_MATRIX_RMSE: float = 0.40

# Weights for histogram similarity (overlap, correlation, RMSE).
WEIGHT_HIST_OVERLAP: float = 0.40
WEIGHT_HIST_CORR: float = 0.25
WEIGHT_HIST_RMSE: float = 0.35

# Weights for moment-feature blending (variance, skewness, kurtosis).
WEIGHT_MOMENT_VAR: float = 0.40
WEIGHT_MOMENT_SKEW: float = 0.30
WEIGHT_MOMENT_KURT: float = 0.30

# Weights for graph-metric blending (Jaccard, weight, degree, clustering).
WEIGHT_GRAPH_JACCARD: float = 0.35
WEIGHT_GRAPH_WEIGHT: float = 0.30
WEIGHT_GRAPH_DEGREE: float = 0.20
WEIGHT_GRAPH_CLUSTER: float = 0.15

# Weights blending CrossCorr sub-metrics (lagged vs. top-edge).
WEIGHT_CROSSCORR_LAGGED: float = 0.75
WEIGHT_CROSSCORR_TOPEDGE: float = 0.25

# Weights blending Trajectory sub-metrics (occupancy vs. path).
WEIGHT_TRAJECTORY_OCCUPANCY: float = 0.60
WEIGHT_TRAJECTORY_PATH: float = 0.40

# Weights blending Manifold sub-metrics (topology vs. local geometry).
WEIGHT_MANIFOLD_TOPOLOGY: float = 0.75
WEIGHT_MANIFOLD_LOCAL: float = 0.25
