from __future__ import annotations

import numpy as np

from nethobench.neuro.metrics.refined import (
    _extract_score,
    _final_manifold_score,
    _final_trajectory_score,
    _perfected_moment_score_legacy,
    _perfected_graph_score_legacy,
)


def compute_moment_score(
    gt_arr: np.ndarray, pred_arr: np.ndarray
) -> dict[str, object]:
    description = "Variance, skewness, and kurtosis agreement."
    result = _perfected_moment_score_legacy(gt_arr, pred_arr)
    score = _extract_score(result)
    return {
        "scores": {
            "MOM_score": float(score) if np.isfinite(score) else np.nan,
        },
        "candidate_name": "perfected_moment_score",
        "description": description,
    }


def compute_graph_score(
    gt_arr: np.ndarray, pred_arr: np.ndarray
) -> dict[str, object]:
    description = "Top-edge topology, weighted degree, and clustering agreement."
    result = _perfected_graph_score_legacy(gt_arr, pred_arr)
    score = _extract_score(result)
    return {
        "scores": {
            "GRAPH_score": float(score) if np.isfinite(score) else np.nan,
            "GRAPH_jacc_mean": float(result.get("jaccard", np.nan)),
            "GRAPH_deg_mean": float(result.get("degree_score", np.nan)),
            "GRAPH_cluster_mean": float(result.get("clustering_score", np.nan)),
        },
        "candidate_name": "perfected_graph_score",
        "description": description,
    }


def compute_manifold_score(
    gt_arr: np.ndarray, pred_arr: np.ndarray
) -> dict[str, object]:
    description = (
        "Persistent-homology lifetime agreement stabilized by local-neighborhood geometry. "
        "It combines a topology term with a local geometry term in a fixed GT latent space."
    )
    score = _extract_score(_final_manifold_score(gt_arr, pred_arr))
    return {
        "scores": {
            "MANI_score": float(score) if np.isfinite(score) else np.nan,
        },
        "candidate_name": "final_manifold_score",
        "description": description,
    }


def compute_trajectory_score(
    gt_arr: np.ndarray, pred_arr: np.ndarray
) -> dict[str, object]:
    description = (
        "Pooled latent occupancy and velocity-distribution agreement stabilized by latent path features. "
        "It scores occupancy, speed, turning, and sequence-level path structure in a shared GT PCA space."
    )
    score = _extract_score(_final_trajectory_score(gt_arr, pred_arr))
    return {
        "scores": {
            "TRJDIST_score": float(score) if np.isfinite(score) else np.nan,
        },
        "candidate_name": "final_trajectory_score",
        "description": description,
    }
