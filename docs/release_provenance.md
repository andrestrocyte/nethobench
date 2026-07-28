# Release provenance

This repository is a clean code snapshot prepared on 2026-07-28 for the
PSD-integrated NethoBench 0.2.0 release.

- Baseline public commit: `652d931`
  (`Add bundled quickstart data and compatibility shims`).
- Neural scoring uses the same PSD-integrated implementation used for the
  rebuttal evidence reruns: the fixed temporal family is
  `TRJDIST_score`, `ACF_score`, and `PSD_score` with equal weights.
- Public-study adapters and their unit tests are included because they call the
  same scoring pipeline.
- One release-only hygiene fix moves the IBL minimum-free-space assertion below
  its no-op dry-run return. This does not change any numerical scoring,
  aggregation, model, or real-run storage behavior.
- Training data, generated predictions, result tables, model checkpoints,
  manuscript sources, reviews, and rebuttal-only scripts are intentionally not
  included.
- Large example datasets from the baseline repository are replaced by compact
  test fixtures under `tests/resources/data/`.

The implementation and formula audit are documented in
[`psd_integration.md`](psd_integration.md). The release should be accepted only
if the full included test suite passes and a built wheel contains the temporal
module and associated package files.
