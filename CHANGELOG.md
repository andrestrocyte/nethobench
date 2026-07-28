# Changelog

## 0.2.0 — 2026-07-28

- Integrate sequence-aware autocorrelation and Welch PSD agreement into the
  temporal/spectral family.
- Fix the temporal family to an equal average of `TRJDIST_score`, `ACF_score`,
  and `PSD_score`.
- Report PSD shape, PSD power, and directional-dynamics diagnostics without
  double-weighting them in the composite.
- Convert physical lag durations to frame offsets using the configured sampling
  frequency.
- Preserve legacy `*_score01` aliases for downstream compatibility.
- Add formula-level, perturbation-sensitivity, missing-data, frame-rate, and CLI
  regression tests.
- Clarify pooled moment and correlation-graph implementations while preserving
  their legacy function aliases.
- Allow IBL dataset dry-runs to report available storage without enforcing the
  minimum-free-space gate used by real downloads and training.

This release changes neural composite values relative to 0.1.0 because the
temporal family is no longer trajectory-only.
