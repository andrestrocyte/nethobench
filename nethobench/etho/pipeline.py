"""Versioned behavioral evaluation with an explicit eight-component legacy path."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from nethobench.etho import legacy_pipeline
from nethobench.etho.extended import compute_extended_etho_metrics, fit_etho_calibration
from nethobench.utils.calculation import merge_aligned
from nethobench.utils.helpers import geometric_mean_scores, load_gt_and_preds

CORE_V1 = ("position_kl_score", "stationary_score", "velocity_score", "acceleration_score",
           "direction_score", "quadrant_score", "syllable_score", "trajectory_shape_score")
CORE_V2 = CORE_V1 + ("inter_limb_distance_score", "bout_duration_score")


def _version(cfg):
    version = cfg.get("etho_score_version", "v2")
    if version not in ("v2", "legacy_v1"):
        raise ValueError("etho_score_version must be 'v2' or 'legacy_v1'")
    return version


def _extend(scores, paired_df, cfg):
    extra, details = compute_extended_etho_metrics(paired_df, cfg)
    scores = dict(scores)
    scores["legacy_composite_score"] = scores["composite_score"]
    scores.update(extra)
    values = [scores[k] for k in CORE_V2]
    available = np.isfinite(values)
    # v2 must not look better simply because a component could not be measured.
    composite = geometric_mean_scores(values) if available.all() else float("nan")
    scores.update(composite_score=composite, composite_score_v2=composite,
                  component_availability_fraction=float(available.mean()),
                  duration_reference_coverage=details["duration"]["reference_coverage"],
                  duration_comparison_coverage=details["duration"]["comparison_coverage"])
    details["components"] = list(CORE_V2)
    details["missing_components"] = [k for k in CORE_V2 if not np.isfinite(scores[k])]
    if cfg.get("etho_details_path"):
        path = Path(cfg["etho_details_path"])
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(details,indent=2)+"\n")
    return scores, details


def compute_etho_scores(gt_dir=None, inf_dir=None, *, paired_df=None, cfg=None):
    """Return global scores and the unchanged legacy per-sequence summaries.

    Default ``cfg['etho_score_version']='v2'`` adds calibrated inter-part distance
    distributions and censored, state-conditional bout duration to the geometric
    mean. ``legacy_composite_score`` retains the previous eight-component value.
    New components are population-distribution scores, not per-sequence scores.

    Set ``etho_score_version='legacy_v1'`` for the exact previous four-tuple,
    including the previous dictionary keys and missing-component policy.
    Supply ``etho_calibration`` (dictionary or JSON path) to reuse training-fitted
    geometry scales and kinematic states; otherwise reference poses calibrate
    these additions. ``etho_details_path`` optionally saves support/calibration.
    """
    cfg = dict(cfg or {})
    if _version(cfg) == "legacy_v1":
        return legacy_pipeline.compute_etho_scores(gt_dir,inf_dir,paired_df=paired_df,cfg=cfg)
    if paired_df is None:
        if gt_dir is None or inf_dir is None:
            raise ValueError("Must provide paired_df or both gt_dir and inf_dir")
        gt, pred = load_gt_and_preds(gt_dir,inf_dir,sequence_key=cfg.get("sequence_key","sequenceId"))
        paired_df = merge_aligned(gt,pred,cfg)
    old, sequence, means, stds = legacy_pipeline.compute_etho_scores(paired_df=paired_df,cfg=cfg)
    scores, _ = _extend(old,paired_df,cfg)
    return scores, sequence, means, stds


def run_etho_full_analysis(gt_dir, inf_dir, *, output_root=None, cfg=None):
    """Retain existing diagnostic plots and save the versioned score and support."""
    cfg = dict(cfg or {})
    version = _version(cfg)
    outdir = legacy_pipeline.run_etho_full_analysis(gt_dir,inf_dir,output_root=output_root,cfg=cfg)
    if version == "legacy_v1":
        return outdir
    gt, pred = load_gt_and_preds(gt_dir,inf_dir,sequence_key=cfg.get("sequence_key","sequenceId"))
    paired = merge_aligned(gt,pred,cfg)
    path = outdir/"scores.json"
    payload = json.loads(path.read_text())
    payload["global_scores"], details = _extend(payload["global_scores"],paired,cfg)
    payload["etho_score_version"] = "v2"
    path.write_text(json.dumps(payload,indent=2)+"\n")
    (outdir/"etho_metadata.json").write_text(json.dumps(details,indent=2)+"\n")
    return outdir
