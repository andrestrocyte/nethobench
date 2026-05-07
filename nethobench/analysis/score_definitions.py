from __future__ import annotations

from collections import OrderedDict

from nethobench.neuro.metrics.definitions import (
    NEURO_FAMILY_WEIGHTS,
    build_fidelity_metrics_df,
    build_neuro_families_df,
    build_neuro_metrics_df,
    compute_fidelity_composite,
    compute_neuro_composite,
    compute_neuro_family_scores,
    weighted_mean_available,
)

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
        ("temporal_spectral", OrderedDict([("TRJDIST_score01", 1.0)])),
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
            OrderedDict([("MANI_score01", 0.50), ("SubspaceAngle_score01", 0.50)]),
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

FIDELITY_METRICS: "OrderedDict[str, float]" = OrderedDict(
    [
        ("Error_score01", 0.65),
        ("MI_score01", 0.35),
    ]
)

__all__ = [
    "NEURO_FAMILY_METRICS",
    "NEURO_FAMILY_WEIGHTS",
    "FIDELITY_METRICS",
    "weighted_mean_available",
    "compute_neuro_family_scores",
    "compute_neuro_composite",
    "build_neuro_metrics_df",
    "build_neuro_families_df",
    "compute_fidelity_composite",
    "build_fidelity_metrics_df",
]
