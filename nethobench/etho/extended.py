"""Calibrated distance-distribution and censored bout-duration metrics.

These functions do not change the eight legacy behavioral components.
Calibration is fitted to training poses, or to reference poses when omitted.
"""
from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans

PROTOCOL = "etho-v2"


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _segments(frame, cfg):
    """Sort within sequences; never differentiate or form bouts across gaps."""
    seq = cfg.get("sequence_key", "sequenceId")
    time = cfg.get("time_key", "itemPosition")
    if frame.empty:
        raise ValueError("Pose data must not be empty")
    if frame[[seq, time]].isna().any().any() or frame.duplicated([seq, time]).any():
        raise ValueError("Sequence/time identifiers must be unique and nonmissing")
    step = _positive(cfg.get("time_step", 1), "time_step")
    for _, group in frame.sort_values([seq, time]).groupby(seq, sort=False):
        times = group[time].to_numpy(dtype=float)
        if not np.isfinite(times).all():
            raise ValueError("Time identifiers must be finite numeric values")
        cuts = np.flatnonzero(~np.isclose(np.diff(times), step, rtol=1e-7, atol=0)) + 1
        for part in np.split(np.arange(len(group)), cuts):
            yield group.iloc[part]


def _features(frame, cfg):
    center = cfg.get("center_part", "CENTER")
    dt = _positive(cfg.get("sampling_interval", 1), "sampling_interval")
    result = []
    for segment in _segments(frame, cfg):
        xy = segment[[f"{center}_X", f"{center}_Y"]].to_numpy(float)
        if not np.isfinite(xy).all():
            raise ValueError("Centroid coordinates must be finite; segment tracking gaps explicitly")
        if len(xy) >= 3:
            velocity = np.diff(xy, axis=0) / dt
            result.append(np.column_stack([np.linalg.norm(velocity[:-1], axis=1),
                                           np.linalg.norm(np.diff(velocity, axis=0) / dt, axis=1)]))
    return result


def _distances(frame, pair):
    a, b = pair
    x = frame[[f"{a}_X", f"{a}_Y"]].to_numpy(float)
    y = frame[[f"{b}_X", f"{b}_Y"]].to_numpy(float)
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Finite coordinates are required for distance pair {pair}")
    return np.linalg.norm(x-y, axis=1)


def bout_records(sequences):
    """Run lengths in samples with explicit left/right window censoring."""
    records = []
    for sequence, raw in enumerate(sequences):
        x = np.asarray(raw)
        if x.ndim != 1 or not np.isfinite(x).all() or np.any(x < 0) or np.any(x != np.floor(x)):
            raise ValueError("State labels must be one-dimensional nonnegative integers")
        if not len(x):
            continue
        edges = np.r_[0, np.flatnonzero(x[1:] != x[:-1])+1, len(x)]
        for start, end in zip(edges[:-1], edges[1:]):
            records.append({"sequence":sequence, "state":int(x[start]), "frames":int(end-start),
                            "left_censored":start == 0, "right_censored":end == len(x)})
    return pd.DataFrame(records, columns=["sequence","state","frames","left_censored","right_censored"])


def _survival(records, horizon):
    rows = records.loc[~records.left_censored.astype(bool)]
    if rows.empty:
        return None
    duration = rows.frames.to_numpy(int)
    event = ~rows.right_censored.to_numpy(bool)
    survival = 1.0
    curve = np.ones(horizon+1)
    for t in np.unique(duration):
        survival *= 1 - np.sum((duration == t) & event) / np.sum(duration >= t)
        if t <= horizon:
            curve[t:] = survival
    # A flat, positive tail after the last observation is not identified.
    if duration.max() < horizon and survival > 1e-12:
        return None
    return curve


def duration_from_labels(reference, prediction, scales, *, horizon=20):
    """State-conditional restricted survival distance, with support diagnostics.

    Scales are fixed calibration median complete-bout lengths in samples.
    A missing predicted state/onset scores zero by explicit convention. A
    partially observed positive survival tail is unavailable, not extrapolated.
    """
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer number of samples")
    scales = np.asarray(scales, float)
    if scales.ndim != 1 or not len(scales) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Duration scales must be finite positive values")
    a, b = bout_records(reference), bout_records(prediction)
    if (not a.empty and a.state.max() >= len(scales)) or (not b.empty and b.state.max() >= len(scales)):
        raise ValueError("State label exceeds calibration state count")
    support = []
    total = float(a.frames.sum())
    for state, scale in enumerate(scales):
        ra, rb = a[a.state == state], b[b.state == state]
        sa, sb = _survival(ra, horizon), _survival(rb, horizon)
        mass = float(ra.frames.sum()) / total if total else 0.0
        if sa is None:
            value, status = np.nan, "reference_duration_unavailable"
        elif rb.loc[~rb.left_censored.astype(bool)].empty:
            value, status = 0.0, "no_predicted_onsets_penalty"
        elif sb is None:
            value, status = np.nan, "prediction_tail_unavailable"
        else:
            value = 1 / (1 + np.sum(np.abs(sa[:-1]-sb[:-1])) / scale)
            status = "supported"
        support.append(dict(state=state, reference_occupancy=mass,
                            reference_bouts=len(ra), prediction_bouts=len(rb),
                            reference_onsets=int((~ra.left_censored.astype(bool)).sum()),
                            prediction_onsets=int((~rb.left_censored.astype(bool)).sum()),
                            score=float(value), status=status))
    table = pd.DataFrame(support)
    reference_ok = table.status != "reference_duration_unavailable"
    usable = np.isfinite(table.score) & reference_ok
    reference_mass = float(table.loc[reference_ok,"reference_occupancy"].sum())
    coverage = float(table.loc[usable,"reference_occupancy"].sum())
    # Reference support is fixed across candidate models. Never silently remove
    # an unidentifiable candidate tail and renormalize in that candidate's favor.
    score = (float(np.average(table.loc[usable,"score"], weights=table.loc[usable,"reference_occupancy"]))
             if coverage > 0 and np.isclose(coverage,reference_mass,atol=1e-12,rtol=0) else np.nan)
    return {"score":score, "reference_coverage":reference_mass, "comparison_coverage":coverage,
            "horizon_frames":int(horizon), "states":support}


def _labels(features, calibration):
    mean = np.asarray(calibration["feature_mean"])
    scale = np.asarray(calibration["feature_scale"])
    centers = np.asarray(calibration["state_centers"])
    return [np.argmin(np.sum((((f-mean)/scale)[:,None,:] - centers[None,:,:])**2, axis=2), axis=1)
            for f in features]


def fit_etho_calibration(reference: pd.DataFrame, cfg=None):
    """Fit reusable calibration on training/reference poses, never predictions.

    Input columns are unsuffixed pose coordinates plus sequence/time identifiers.
    The returned dictionary is JSON serializable and contains no raw poses.
    """
    cfg = dict(cfg or {})
    options = {k:cfg.get(k,v) for k,v in {"sequence_key":"sequenceId", "time_key":"itemPosition",
               "center_part":"CENTER", "sampling_interval":1.0, "time_step":1.0}.items()}
    parts = sorted(c[:-2] for c in reference if c.endswith("_X") and c[:-2]+"_Y" in reference)
    pairs = cfg.get("inter_limb_pairs", list(combinations(parts, 2)))
    canonical = []
    for pair in pairs:
        if len(pair) != 2 or pair[0] == pair[1] or any(p not in parts for p in pair):
            raise ValueError(f"Invalid inter_limb_pairs entry: {pair}")
        pair = sorted(pair)
        if pair in canonical:
            raise ValueError(f"Duplicate distance pair: {pair}")
        canonical.append(pair)
    if not canonical:
        raise ValueError("At least two body parts are required for distance scoring")
    scales = []
    for pair in canonical:
        distance = _distances(reference, pair)
        positive = distance[distance > 0]
        scales.append(float(np.median(positive)) if positive.size else 0.0)
    features = _features(reference, options)
    k = cfg.get("duration_states", 8)
    horizon = cfg.get("duration_horizon_frames", 20)
    if not isinstance(k,int) or k < 1 or not isinstance(horizon,int) or horizon < 1:
        raise ValueError("duration_states and duration_horizon_frames must be positive integers")
    calibration = dict(protocol=PROTOCOL, options=options, pairs=canonical, distance_scales=scales,
                       requested_duration_states=k,
                       duration_horizon_frames=horizon, feature_mean=[0,0], feature_scale=[1,1],
                       state_centers=[], duration_scales=[])
    if features:
        joined = np.concatenate(features)
        mean, scale = joined.mean(axis=0), joined.std(axis=0)
        scale[scale == 0] = 1.0
        standardized = (joined-mean)/scale
        rng = np.random.default_rng(401)
        sample = standardized[rng.choice(len(joined),min(80000,len(joined)),replace=False)]
        k = min(k,len(np.unique(sample,axis=0)))
        centers = KMeans(n_clusters=k,n_init=10,random_state=401).fit(sample).cluster_centers_
        calibration.update(feature_mean=mean.tolist(),feature_scale=scale.tolist(),state_centers=centers.tolist())
        runs = bout_records(_labels(features, calibration))
        for state in range(k):
            complete = runs[(runs.state == state) & ~runs.left_censored & ~runs.right_censored]
            calibration["duration_scales"].append(float(complete.frames.median()) if len(complete) else 1.0)
    return calibration


def inter_limb_distance_score(reference, prediction, calibration):
    """Compare full pair-distance marginals using calibrated Wasserstein-1."""
    losses, details = [], []
    for pair, scale in zip(calibration["pairs"],calibration["distance_scales"]):
        error = float(wasserstein_distance(_distances(reference,pair),_distances(prediction,pair)))
        loss = error/scale if scale > 0 else (0.0 if error == 0 else np.inf)
        losses.append(loss)
        details.append(dict(pair=pair, wasserstein_distance=error, calibration_scale=scale, normalized_distance=loss))
    return {"score":float(1/(1+np.mean(losses))), "pairs":details}


def _validate_calibration(calibration):
    if calibration.get("protocol") != PROTOCOL:
        raise ValueError("Expected an etho-v2 calibration")
    pairs, scales = calibration["pairs"], np.asarray(calibration["distance_scales"],float)
    if not pairs or scales.shape != (len(pairs),) or not np.isfinite(scales).all() or np.any(scales < 0):
        raise ValueError("Invalid distance calibration scales")
    if any(len(p) != 2 or p[0] == p[1] for p in pairs) or len({tuple(sorted(p)) for p in pairs}) != len(pairs):
        raise ValueError("Invalid or duplicate calibrated distance pairs")
    centers=np.asarray(calibration["state_centers"],float)
    mean=np.asarray(calibration["feature_mean"],float)
    scale=np.asarray(calibration["feature_scale"],float)
    durations=np.asarray(calibration["duration_scales"],float)
    if mean.shape != (2,) or scale.shape != (2,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid kinematic feature calibration")
    if centers.size and (centers.ndim != 2 or centers.shape[1] != 2 or not np.isfinite(centers).all()):
        raise ValueError("Invalid state centers")
    if durations.shape != (len(centers),) or not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("Invalid duration calibration scales")
    horizon=calibration["duration_horizon_frames"]
    if not isinstance(horizon,int) or horizon < 1:
        raise ValueError("Invalid duration horizon")


def compute_extended_etho_metrics(paired_df, cfg=None):
    cfg = dict(cfg or {})
    keys = [cfg.get("sequence_key","sequenceId"),cfg.get("time_key","itemPosition")]
    def extract(suffix):
        cols = [c for c in paired_df if c.endswith(suffix)]
        return paired_df[keys+cols].rename(columns={c:c[:-len(suffix)] for c in cols})
    reference, prediction = extract("_gt"), extract("_inf")
    calibration = cfg.get("etho_calibration")
    if isinstance(calibration,(str,Path)):
        calibration = json.loads(Path(calibration).read_text())
    source = "supplied" if calibration is not None else "reference"
    if calibration is None:
        calibration = fit_etho_calibration(reference,cfg)
    _validate_calibration(calibration)
    if "duration_horizon_frames" in cfg and cfg["duration_horizon_frames"] != calibration["duration_horizon_frames"]:
        raise ValueError("Configuration conflicts with calibration: duration_horizon_frames")
    if "duration_states" in cfg and cfg["duration_states"] != calibration["requested_duration_states"]:
        raise ValueError("Configuration conflicts with calibration: duration_states")
    if "inter_limb_pairs" in cfg and sorted(sorted(p) for p in cfg["inter_limb_pairs"]) != sorted(calibration["pairs"]):
        raise ValueError("Configuration conflicts with calibration: inter_limb_pairs")
    options = dict(calibration["options"])
    for key in ["sequence_key","time_key","center_part","sampling_interval","time_step"]:
        if key in cfg and key in options and cfg[key] != options[key]:
            raise ValueError(f"Configuration conflicts with calibration: {key}")
        if key in cfg:
            options[key] = cfg[key]
    geometry = inter_limb_distance_score(reference,prediction,calibration)
    if calibration["state_centers"]:
        a, b = _labels(_features(reference,options),calibration), _labels(_features(prediction,options),calibration)
        duration = duration_from_labels(a,b,calibration["duration_scales"],horizon=calibration["duration_horizon_frames"])
    else:
        duration = dict(score=np.nan, reference_coverage=0.0, comparison_coverage=0.0,
                        horizon_frames=calibration["duration_horizon_frames"],states=[])
    return {"inter_limb_distance_score":geometry["score"],"bout_duration_score":duration["score"]}, {
        "protocol":PROTOCOL,"calibration_source":source,"calibration":calibration,
        "geometry":geometry,"duration":duration}
