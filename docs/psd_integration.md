# PSD-integrated temporal scoring

NethoBench 0.2.0 defines the neural temporal/spectral family as

\[
T = \frac{1}{3}
\left(
s_{\mathrm{trajectory}} +
s_{\mathrm{ACF}} +
s_{\mathrm{PSD}}
\right).
\]

The family remains weighted by 0.18 in the full neural composite. Adding PSD
therefore does not increase the temporal family's influence; it partitions the
existing temporal weight among three prespecified components.

## Sequence boundary and unit of analysis

All temporal metrics receive arrays with shape
`[sequence, time, region]`. Autocorrelation and PSD are estimated separately
for every sequence-region trace. Samples from different trials are never
concatenated into an artificial time series.

The reported score for a component is

\[
\frac{1}{2}\operatorname{mean}(s_i)
+ \frac{1}{2}Q_{0.10}(s_i),
\]

over finite sequence-region scores \(s_i\). This retains average performance
while penalizing a poor lower tail.

## Autocorrelation

`ACF_score` compares Pearson autocorrelation curves at fixed physical lags. By
default the lags correspond to 1, 2, 4, 8, 16, 32, and 48 frames at 30 Hz and
are rescaled from seconds when the sampling frequency changes.

The raw-trace curve similarity is averaged with the same comparison on first
differences. The increment term increases sensitivity to short-timescale
jitter that may be obscured by slowly varying activity.

## Power spectral density

Welch PSDs are computed with the configured
`WELCH_SAMPLING_FREQUENCY` and `WELCH_NPERSEG`. The DC bin is excluded.
For each sequence-region pair:

- `PSDShape_score` is the Bhattacharyya coefficient between unit-mass PSDs.
- `PSDPower_score` is the ratio of the smaller to the larger total spectral
  power.
- `PSD_score` is the geometric mean of shape and power agreement.

The power term prevents a prediction with the correct normalized spectral shape
but strongly contracted variance from receiving a near-perfect PSD score.

## Diagnostics and weighting

`PSDShape_score`, `PSDPower_score`, and `DirectionalDynamics_score` are emitted
for auditing. They are not additional members of the family composite.
Directional dynamics compares lagged cross-region covariance matrices and is
kept as a diagnostic because it overlaps conceptually with relational metrics.

## Missing values

A trace is scored only if at least 80% of its samples are finite and at least
eight finite samples are available. Sparse gaps are linearly interpolated on
the original time grid. Scores with no valid unit are returned as `NaN` and are
handled by the existing available-metric aggregation policy.

## Reproducibility checks

Run:

```bash
pytest -q tests/unit/test_temporal_metrics.py
pytest -q tests/integration/test_cli.py
```

The temporal tests cover identity, temporal shuffle, short-timescale jitter,
variance contraction, time reversal, sparse missing values, fixed weights, and
sampling-rate-aware lag conversion.
