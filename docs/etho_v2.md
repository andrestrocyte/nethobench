# Etho v2: distance distributions and bout duration

Etho v2 extends the behavioral composite from eight to ten equally weighted
components. It does not change neural scores, the cross-modal pipeline, or any
of the original eight behavioral functions. Package version: 0.3.0; calibration
protocol: `etho-v2`. Select `etho_score_version="legacy_v1"` for the original
API, dictionary keys, missing-value policy and numerical results.

## Why these definitions

The repository's `inter_limb_distances` is a useful descriptive table of means
and standard deviations. Two distance distributions can have the same mean and
variance but different shapes. It was not itself a scalar composite component.
The standalone MuJoCo study compared whole distance distributions, which is the
better starting point for scoring, but its absolute normalization floor of 0.05
made the score depend on coordinate units.

The separate `analysis/behavior_crossmodal_supplement.py` duration diagnostic
pooled runs across states and included truncated boundary runs as complete
bouts. It was not one of the standard eight Etho components. The MuJoCo study
used state-specific survival curves and handled boundary censoring. V2 retains
that approach but does not extrapolate unidentified positive survival tails.
Neither of these existing functions has been removed or redefined.

## Pair-distance agreement

For tracked parts a and b, at each observed frame define

    d_ab(t) = ||x_a(t) - x_b(t)||_2.

Let P_ab and Q_ab be the empirical reference and forecast distance distributions.
The fixed calibration scale s_ab is the median strictly positive distance in the
training/calibration data. Then

    L_ab = W1(P_ab, Q_ab) / s_ab
    S_distance = 1 / (1 + mean_ab L_ab).

[Wasserstein-1](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)
compares full empirical distributions. All unique pairs of tracked 2D parts are
used by default; `inter_limb_pairs` can specify a fixed anatomical subset. This
includes torso/center pairs unless excluded explicitly. Pairs receive equal
weight. Calibration must be held fixed across candidate models. For a pair that
is always coincident in calibration, zero discrepancy contributes zero loss;
any nonzero discrepancy contributes infinite loss and the distance score is zero.
There is no absolute coordinate-unit floor.

The metric is invariant to rigid transforms and to uniform unit changes when
calibration coordinates use those same units. It detects marginal geometry
changes but does not establish joint pose feasibility, contact mechanics,
left/right identity correctness, or conditional trajectory accuracy.

## State-conditional restricted bout duration

Speed and acceleration magnitude are derived from successive CENTER positions
(or `center_part`) within sequences. No differences cross sequence boundaries
or missing time steps. Both features are standardized using calibration means
and standard deviations, then clustered into up to eight kinematic states.
K-means uses random seed 401, 10 initializations and at most 80,000 calibration
frames. The number of clusters is capped by the number of unique sampled feature
vectors. These are kinematic clusters, not validated ethological labels. Features
use three consecutive positions; the derived label sequence is two frames shorter.

Runs of each state define durations in samples. The first run of each sequence
or contiguous segment has an unknown onset and is excluded from duration fitting.
The final run is right-censored and remains in the risk set. A run touching both
boundaries is excluded because its onset is unknown. Occupancy weights include
all labeled frames. This convention assumes window endpoints provide ordinary
noninformative right censoring; event-selected windows need separate scrutiny.

For each state k, a discrete Kaplan–Meier estimate uses

    S_k(t) = product over u <= t of (1 - events_k(u) / at_risk_k(u)).

This is the usual right-censoring construction; see the
[Kaplan–Meier documentation](https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html).
The implementation uses NumPy rather than adding a survival-library dependency.
The fixed normalization m_k is the median complete calibration bout in that
state, or one sample if no complete calibration bout exists. For a prespecified
integer horizon H,

    L_k = sum_(t=0)^(H-1) |S_ref,k(t) - S_pred,k(t)| / m_k
    s_k = 1 / (1 + L_k).

The component is the reference-occupancy-weighted mean of s_k over states whose
reference survival curve is supported through H. This is a restricted duration
comparison, not a comparison of unbounded lifetime distributions. Defaults:
`duration_states=8`, `duration_horizon_frames=20`, `sampling_interval=1` and
`time_step=1`. With `sampling_interval=0.05`, H=20 corresponds to one second.
Choose H before looking at comparative results, and retain it across models.
`time_step` describes spacing in the time identifier; `sampling_interval`
describes seconds per observed sample. Calibration stores both.

A survival curve that reaches zero can be extended as zero. A positive tail
beyond the last observed follow-up cannot be inferred and is marked unavailable.
A reference-supported state with no observed predicted onset receives zero by
an explicit penalty convention; this is not an estimated survival curve. If a
predicted state has onsets but an unidentified positive tail, the duration
component is unavailable. It is not renormalized after dropping that state.

`duration_reference_coverage` is the reference occupancy mass of states with
identifiable reference duration curves; `duration_comparison_coverage` is the
mass for which the comparison or explicit missing-onset penalty is available.
Both may be below one. A finite duration score can describe only a subset of
reference states: always report coverage, and do not claim support for excluded
states. A constant reference with no observable onsets has no duration score.
More windows, longer observation, or a prospectively shorter horizon may improve
support. Do not shorten the horizon independently for individual models.

## Composite and compatibility

The ten components are position KL, stationarity, velocity, acceleration,
direction, quadrant occupancy, syllable occupancy, trajectory shape, pair-distance
distributions and bout duration. The geometric mean retains the existing numeric
clipping convention:

    composite_v2 = exp(mean_i log(clip(score_i, 1e-6, 1 - 1e-6))).

Consequently a zero component yields a very low, but nonzero, composite. If any
of the ten components is nonfinite, the v2 composite is NaN. No missing component
is silently discarded. `legacy_composite_score` always preserves the old
aggregation, including its original missing-component behavior. Equal weights
are a transparent default, not an empirically optimized claim of importance.

The return tuple and original component values are unchanged. Per-sequence
outputs continue to contain only the original summaries; the new components
compare distributions pooled across sequences. Full analysis saves detailed
calibration, pair losses, per-state support and missing components in
`etho_metadata.json`. Scores-only users can request the same detail through
`etho_details_path`. A supplied calibration can be a dictionary or JSON path.
Without one, only reference data fit calibration. See the README for a training
calibration example. Record score version, calibration, sampling interval,
horizon, pair list and coverage with every result.

## Comparative checks and saved-forecast rescore

The [controlled checks](validation/etho_v2/controlled_checks.csv) test five
specific properties per metric. Each passed check contributes two points:

| Implementation | Distance checks | Duration checks |
| --- | ---: | ---: |
| Repository diagnostics | 6/10 | 4/10 |
| Standalone MuJoCo definitions | 8/10 | 8/10 |
| Etho v2 | 10/10 | 10/10 |

These are narrow functional test counts, not overall scientific quality ratings.
The repository distance helper was designed as a descriptive table; lacking a
unit-free scalar is a missing scoring capability rather than a coding error.
The comparison includes identity, distribution/state-specific changes, boundary
censoring, unit changes and unavailable duration tails. Run the self-contained
comparison with:

```bash
python scripts/compare_etho_metric_implementations.py --output-root outputs/etho-comparison
```

Unmodified study functions needed for this comparison are included in
`scripts/etho_study_reference.py`. Their source provenance and the numerical
results are recorded in [the validation directory](validation/etho_v2/).

A separate rescore used all **318 best-checkpoint saved evaluations** from the
MuJoCo study: three policy populations, two evaluation splits, and the available
model seeds and stochastic draws. No model was retrained. The existing eight
components were taken from the saved evaluation table. Each population's training
poses calibrated the additions; the comparison used all 36 pairs of nine points,
a 20-sample horizon and 0.05-second sampling. The study distance diagnostic used
12 selected pairs, so its values are not expected to be identical to v2.

The distance component was finite in 318/318 records. The complete v2 score was
available in **265/318**: 134/159 in-distribution and 131/159 shifted evaluations.
The remaining **53** had insufficient predicted duration-tail support. Reference
duration coverage ranged from 0.911 to 1.000 in-distribution and 0.996 to 1.000
under shift. This is a substantive support limitation, not a missing-data
cosmetic issue. It prevents claiming an unchanged complete ranking of all models.

For comparisons fully supported across all seeds/populations, in-distribution
analog retrieval changed from 0.876 (legacy) to 0.896 (v2), simulator-mean control
from 0.680 to 0.699, and persistence from 0.093 to 0.038. These examples illustrate
the changed definition; they do not establish conditional forecasting superiority.
Stochastic-oracle and several learned-model aggregate v2 results are withheld
when any underlying duration score is unavailable. Aggregation averages draws,
then seeds, then populations, preserving rather than skipping missing values.

[Per-record scores](validation/etho_v2/rescored_rows.csv),
[state support](validation/etho_v2/duration_support.csv), calibration files and
summaries are provided. `scripts/rescore_etho_v2_archive.py` reproduces that rescore
from the external raw study archive and old score CSV; input SHA-256 hashes are
in `provenance.json`. The large raw archive is not bundled with the package.
No existing manuscript, figure or eight-component result is redefined by this
rescore. Structural agreement remains complementary to predictive accuracy,
physics checks and task-specific behavioral validation.
