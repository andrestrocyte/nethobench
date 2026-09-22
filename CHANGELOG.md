# Changelog

## 0.2.1 — 2026-09-22

- Preserve all 52 existing package source files from the ICLR V2 scoring commit.
- Add hashed Python 3.10 scoring/test/build locks and pinned build bootstrap tools.
- Add CI for Linux/macOS, Python 3.12 compatibility, installed-wheel validation,
  and numerical comparisons with the published CLI.
- Restrict installed packages to `nethobench`; retain tests and fixtures in source
  distributions. Restore the documented fixtures from the original public repo.
- Add explicitly opt-in sequence-aware directed relational and row-conditional
  state-transition sensitivity diagnostics, separately named from default scores.
- Document unresolved scientific limitations and the preserved score definition.

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
