"""
Centralized evaluation constants for NethoBench.

All heuristic thresholds, statistical quantiles, ML hyperparameters,
and metric-composition weights live here so they can be audited and
adjusted in one place rather than being buried in individual modules.

Constants are split into two categories:
1. Immutable Benchmark Standards – mathematical bedrock that ensures
   scores are comparable across users and runs.
2. Data-Dependent Hyperparameters – values sensitive to frame rate,
   resolution, sequence length, and memory. These are wrapped in a
   global ``NethoConfig`` singleton so they can be overridden once at
   pipeline startup without rewriting every function signature.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. IMMUTABLE BENCHMARK STANDARDS
# ---------------------------------------------------------------------------
# These define statistical stability limits and composite weights. They
# remain hardcoded so that a NethoBench score from User A is directly
# comparable to a score from User B.

# -- Statistical Minimums & Quantiles --
MIN_ITEMS_IQR_ROBUST_FALLBACK: int = 10
MIN_SAMPLES_KL: int = 10
MIN_SAMPLES_ERROR: int = 50
MIN_SAMPLES_MI: int = 80
MIN_ROWS_PCA: int = 32
MIN_SAMPLES_MANIFOLD: int = 64
MIN_ROWS_CORRCOEF: int = 16
QUANTILE_10P: float = 0.10
QUANTILE_IQR_LO: float = 0.25
QUANTILE_IQR_HI: float = 0.75
TAIL_TRIM_QUANTILE_LO: float = 0.001
TAIL_TRIM_QUANTILE_HI: float = 0.999
DISTRIBUTION_GRID_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
MAD_TO_STD_CONSTANT: float = 1.4826

# -- Composite Weightings --
WEIGHT_SPECTRUM_CORR: float = 0.55
WEIGHT_SPECTRUM_RMSE: float = 0.45
WEIGHT_MATRIX_CORR: float = 0.60
WEIGHT_MATRIX_RMSE: float = 0.40
WEIGHT_HIST_OVERLAP: float = 0.40
WEIGHT_HIST_CORR: float = 0.25
WEIGHT_HIST_RMSE: float = 0.35
WEIGHT_MOMENT_VAR: float = 0.40
WEIGHT_MOMENT_SKEW: float = 0.30
WEIGHT_MOMENT_KURT: float = 0.30
WEIGHT_GRAPH_JACCARD: float = 0.35
WEIGHT_GRAPH_WEIGHT: float = 0.30
WEIGHT_GRAPH_DEGREE: float = 0.20
WEIGHT_GRAPH_CLUSTER: float = 0.15
WEIGHT_CROSSCORR_LAGGED: float = 0.75
WEIGHT_CROSSCORR_TOPEDGE: float = 0.25
WEIGHT_TRAJECTORY_OCCUPANCY: float = 0.60
WEIGHT_TRAJECTORY_PATH: float = 0.40
WEIGHT_MANIFOLD_TOPOLOGY: float = 0.75
WEIGHT_MANIFOLD_LOCAL: float = 0.25

# ---------------------------------------------------------------------------
# 2. DATA-DEPENDENT HYPERPARAMETERS (Global Config Singleton)
# ---------------------------------------------------------------------------
# These values are sensitive to dataset frame rate, spatial resolution,
# sequence length, and available memory. Override them at pipeline start
# via ``config.update_from_dict(cfg_dict)``.


class NethoConfig:
    """Global configuration state for data-dependent parameters."""

    def __init__(self):
        self._settings: dict[str, object] = {
            # Time / Frame-Rate Dependent
            "MIN_SAMPLES_CORR_WINDOW": 33,
            "LEAD_LAG_DEFAULT_MAX_LAG": 30,
            "CROSS_CORR_MAX_LAG_SMALL": 16,
            "AUTOCORR_MAX_LAG": 48,
            "DFC_STATE_CONFIGS": ((45, 15, 8), (60, 30, 8), (90, 30, 8)),
            "WELCH_SAMPLING_FREQUENCY": 30.0,
            "WELCH_NPERSEG": 256,
            "CHUNK_SIZE_EMBEDDINGS": 5,
            # Compute / Subsampling Limits
            "MAX_TIME_STEPS_SUBSAMPLE": 1200,
            "LATENT_SUBSAMPLE_SIZE": 96,
            "N_POINTS_MANIFOLD_PH_KNN": 128,
            "MAX_SEQUENCES_STRATIFIED": 48,
            "MAX_POINTS_MANIFOLD_ALIGNMENT": 2000,
            "MI_MAX_POINTS": 1500,
            # Algorithm / Heuristic Parameters
            "LEAD_LAG_MIN_ARRAY_SIZE": 5,
            "STATIONARY_THRESHOLD_PERCENTILE": 10,
            "SPEED_OUTLIER_PERCENTILE": 99,
            "PCA_VARIANCE_THRESHOLD": 0.85,
            "MI_N_NEIGHBORS": 5,
            "HISTOGRAM_BINS_DEFAULT": 12,
            "KMEANS_K_TRAJECTORY": 6,
            "KMEANS_K_SYLLABLE": 8,
        }

    def update_from_dict(self, cfg_dict: dict | None) -> None:
        """Update configurations if they are provided in the JSON/dict."""
        if not cfg_dict:
            return
        for key, value in cfg_dict.items():
            if key in self._settings:
                self._settings[key] = value

    def __getattr__(self, name: str) -> object:
        """Allow dot notation access: config.WELCH_SAMPLING_FREQUENCY"""
        if name in self._settings:
            return self._settings[name]
        raise AttributeError(f"'NethoConfig' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: object) -> None:
        """Allow direct attribute assignment for known keys."""
        if name.startswith("_"):
            super().__setattr__(name, value)
        elif hasattr(self, "_settings") and name in self._settings:
            self._settings[name] = value
        else:
            super().__setattr__(name, value)


# Instantiate the global singleton
config = NethoConfig()
