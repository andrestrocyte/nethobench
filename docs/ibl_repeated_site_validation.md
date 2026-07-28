# IBL Repeated-Site Neuropixels Validation

This module adds a focused IBL Reproducible Ephys / repeated-site validation for the NethoBench neural realism score. It is intended as evidence that the benchmark can be applied beyond slow region-level calcium traces, using public high-temporal-resolution Neuropixels spiking data.

The first target is the IBL repeated-site release rather than the full Brainwide Map. The repeated-site release is more controlled for a fast validation because sessions repeatedly target a common location that includes posterior parietal cortex, hippocampus, and thalamus, while still spanning multiple labs.

## Setup

The core `nethobench` install does not require IBL packages. Install optional IBL dependencies only when running this module:

```bash
python -m pip install ONE-api ibllib
```

## Commands

Initialize a manifest:

```bash
nethobench ibl-init-manifest --output docs/manifests/ibl_repeated_site.example.json
```

Check the configuration and optional dependencies without downloading data:

```bash
nethobench ibl-export --manifest docs/manifests/ibl_repeated_site.example.json --dry-run
```

Discover repeated-site sessions and write an inventory:

```bash
nethobench ibl-discover --manifest docs/manifests/ibl_repeated_site.example.json
```

Run the lightweight validation export:

```bash
nethobench ibl-export --manifest docs/manifests/ibl_repeated_site.example.json
```

Run the expanded 24-session modelling study:

```bash
nethobench ibl-run-study \
  --manifest docs/manifests/ibl_repeated_site.example.json \
  --output-root outputs/ibl-repeated-site-24
```

The expanded study can also be run in stages:

```bash
nethobench ibl-build-dataset --manifest docs/manifests/ibl_repeated_site.example.json --output-root outputs/ibl-repeated-site-24
nethobench ibl-train-models --manifest docs/manifests/ibl_repeated_site.example.json --output-root outputs/ibl-repeated-site-24
nethobench ibl-score-study --manifest docs/manifests/ibl_repeated_site.example.json --output-root outputs/ibl-repeated-site-24
```

The direct script entry point is also available:

```bash
ibl_export_nethobench --manifest docs/manifests/ibl_repeated_site.example.json
```

## Outputs

The export writes:

- `gt_region_rates.csv`
- `pred_poisson_region_rates.csv`
- `pred_var_region_rates.csv`
- `ibl_region_config.json`
- `ibl_split_half_ceiling_scores.json`
- `ibl_model_scores.json`
- `ibl_corruption_ladder_scores.json`
- `ibl_repeated_site_report.json`
- `ibl_family_comparison.png`
- `ibl_corruption_ladder.png`
- `unit_level/ibl_unit_level_supplement_scores.json` when enough high-quality units are available

The expanded study additionally writes:

- `ibl_dataset_inventory.csv`
- `ibl_feature_manifest.json`
- `ibl_model_predictions_manifest.json`
- `ibl_model_scores_long.csv`
- `ibl_bootstrap_ci.json`
- `ibl_study_report.json`
- `ibl_dataset_coverage.png`
- `ibl_family_comparison_expanded.png`
- `ibl_model_family_heatmap.png`
- `ibl_task_model_stripplot.png`

The CSV files use the standard NethoBench contract: `sequenceId`, `itemPosition`, and one neural channel column per selected region.

## Default Protocol

- Access data through OpenAlyx at `https://openalyx.internationalbrainlab.org`.
- Use the public OpenAlyx password `international` unless a local ONE token is already configured.
- Search repeated-site sessions using the `RepeatedSite` release tag.
- Load spike sorting and trials from ALF collections.
- Align trials to `stimOn_times`.
- Bin spikes from `-0.5s` to `1.5s` around the alignment event using `20ms` bins.
- Filter clusters to good labels, `0.2 <= firing rate <= 100 Hz`.
- Aggregate units into stable region-level population rates.
- Transform rates with `log1p(rate_hz)`.
- Fit z-scoring only on train-sequence context bins.

## Baselines And Models

The lightweight export intentionally avoids a large training burden:

- Split-half ceiling: compares held-out trial halves.
- Poisson/PSTH baseline: repeats the train-set time-bin x region mean activity.
- VAR baseline: fits a ridge-regularized linear autoregressive model and rolls out over target bins.
- Corruption ladder: evaluates time shuffle, channel permutation, lag jitter, dropout, and gain scaling.
- Unit-level supplement: selects the top firing-rate good units from the best few sessions, applies `sqrt(count)`, and scores only within session because unit identities do not align across animals.

The expanded study adds:

- GLM-Poisson with lagged activity and time covariates.
- Lag-3 ridge VAR with validation-selected ridge.
- LDS-VAR using a PCA latent state capped at 8 dimensions.
- GRU sequence forecaster.
- Compact causal Transformer.
- Transformer with NethoBench-aligned moment, quantile, and autocorrelation regularizers.

The expanded claim is that NethoBench generalizes from widefield calcium to public high-temporal-resolution Neuropixels spiking, where it separates statistical, linear dynamical, recurrent, and Transformer models by interpretable structural-realism families. The full Brainwide Map remains out of scope.

## Disk Policy

For larger runs, use `disk_policy="clean_raw_after_binning"` and keep `min_free_disk_gb >= 8`. The builder saves compact binned features under the study output root, then removes newly downloaded large spike arrays owned by that session. Existing cached raw arrays are left alone.
