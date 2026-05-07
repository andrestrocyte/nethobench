"""Compatibility wrapper for the pre-refactor sensitive metric module path."""

from nethobench.neuro.metrics.sensitive import *  # noqa: F401,F403
from nethobench.neuro.metrics.sensitive import (
    align_arrays as _align_arrays,
    autocorr_weighted_rmse_power,
    bandpower_band_fraction,
    correlation_score,
    crosscorr_lagged_matrix,
    crosscorr_topedge_profiles,
    finite_rows as _finite_rows,
    manifold_ph_knn_profile,
    manifold_ph_stratified_lifetime,
    pca_reconstruction_product,
    robust_scale,
    safe_corrcoef as _safe_corrcoef,
    score_from_distance as _score_from_distance,
    trajectory_occupancy_velocity,
    trajectory_path_features,
)

_corr_score = correlation_score
_robust_scale = robust_scale
autocorr_weighted_rmse_power_v6 = autocorr_weighted_rmse_power
bandpower_band_fraction_v2 = bandpower_band_fraction
crosscorr_lagged_matrix_v4 = crosscorr_lagged_matrix
crosscorr_topedge_profiles_v5 = crosscorr_topedge_profiles
manifold_ph_knn_profile_v5 = manifold_ph_knn_profile
manifold_ph_stratified_lifetime_v7 = manifold_ph_stratified_lifetime
pca_reconstruction_product_v6 = pca_reconstruction_product
trajectory_occupancy_velocity_v4 = trajectory_occupancy_velocity
trajectory_path_features_v3 = trajectory_path_features
