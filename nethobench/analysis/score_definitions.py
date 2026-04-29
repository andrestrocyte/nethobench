from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import numpy as np
import pandas as pd

NEURO_FAMILY_METRICS: "OrderedDict[str, OrderedDict[str, float]]" = OrderedDict(
    [
        (
            "distribution",
            OrderedDict(
                [
                    ("KL_or_JSD_score01", 0.25),
                    ("QNT_score01", 0.25),
                    ("MOM_score01", 0.25),
                    ("Mean_score01", 0.25),
                ]
            ),
        ),
        (
            "temporal_spectral",
            OrderedDict(
                [
                    ("TRJDIST_score01", 1.0),
                ]
            ),
        ),
        (
            "relational",
            OrderedDict(
                [
                    ("GRAPH_score01", 0.25),
                    ("CrossRegionMI_score01", 0.25),
                    ("LaggedCovariance_score01", 0.25),
                    ("ImpulseResponse_score01", 0.25),
                ]
            ),
        ),
        (
            "geometry",
            OrderedDict(
                [
                    ("MANI_score01", 0.50),
                    ("SubspaceAngle_score01", 0.50),
                ]
            ),
        ),
        (
            "state_dynamics",
            OrderedDict(
                [
                    ("LatentStateOccupancyK11_score01", 0.20),
                    ("LatentStateOccupancyK12_score01", 0.20),
                    ("LatentStateTransitionLag1K11_score01", 0.20),
                    ("LatentStateTransitionLag2K11_score01", 0.20),
                    ("LatentStateTransitionLag3K11_score01", 0.20),
                ]
            ),
        ),
    ]
)

NEURO_FAMILY_WEIGHTS: "OrderedDict[str, float]" = OrderedDict(
    [
        ("distribution", 0.22),
        ("temporal_spectral", 0.18),
        ("relational", 0.24),
        ("geometry", 0.18),
        ("state_dynamics", 0.18),
    ]
)

FIDELITY_METRICS: "OrderedDict[str, float]" = OrderedDict(
    [
        ("Error_score01", 0.65),
        ("MI_score01", 0.35),
    ]
)


def weighted_mean_available(
    values: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    keys = [
        k for k, v in values.items() if np.isfinite(v) and weights.get(k, 0.0) > 0.0
    ]
    if not keys:
        return np.nan
    denom = float(np.sum([weights[k] for k in keys]))
    if denom <= 0.0:
        return np.nan
    return float(np.sum([weights[k] * float(values[k]) for k in keys]) / denom)


def compute_neuro_family_scores(metric_scores: Mapping[str, float]) -> dict[str, float]:
    families: dict[str, float] = {}
    for family_name, metric_weights in NEURO_FAMILY_METRICS.items():
        values = {
            metric: float(metric_scores.get(metric, np.nan))
            for metric in metric_weights
        }
        families[f"family_{family_name}"] = weighted_mean_available(
            values, metric_weights
        )
    return families


def compute_neuro_composite(metric_scores: Mapping[str, float]) -> float:
    families = compute_neuro_family_scores(metric_scores)
    family_values = {
        family_name: float(families.get(f"family_{family_name}", np.nan))
        for family_name in NEURO_FAMILY_WEIGHTS
    }
    return weighted_mean_available(family_values, NEURO_FAMILY_WEIGHTS)


def build_neuro_metrics_df(metric_scores: Mapping[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family_name, metric_weights in NEURO_FAMILY_METRICS.items():
        for metric_name, weight in metric_weights.items():
            rows.append(
                {
                    "family": family_name,
                    "metric": metric_name,
                    "weight": float(weight),
                    "value": float(metric_scores.get(metric_name, np.nan)),
                }
            )
    return pd.DataFrame(rows)


def build_neuro_families_df(metric_scores: Mapping[str, float]) -> pd.DataFrame:
    family_scores = compute_neuro_family_scores(metric_scores)
    rows: list[dict[str, object]] = []
    for family_name, weight in NEURO_FAMILY_WEIGHTS.items():
        rows.append(
            {
                "family": family_name,
                "weight": float(weight),
                "value": float(family_scores.get(f"family_{family_name}", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def compute_fidelity_composite(metric_scores: Mapping[str, float]) -> float:
    values = {
        metric: float(metric_scores.get(metric, np.nan)) for metric in FIDELITY_METRICS
    }
    return weighted_mean_available(values, FIDELITY_METRICS)


def build_fidelity_metrics_df(metric_scores: Mapping[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric_name, weight in FIDELITY_METRICS.items():
        rows.append(
            {
                "family": "fidelity",
                "metric": metric_name,
                "weight": float(weight),
                "value": float(metric_scores.get(metric_name, np.nan)),
            }
        )
    return pd.DataFrame(rows)
