# TRIBE v2 Realism Benchmark

This benchmark treats TRIBE v2 as a cortical, average-subject, stimulus-to-fMRI encoder and evaluates only the neural realism axis in NethoBench v1.

## Commands

Initialize a manifest template:

```bash
nethobench tribev2-init-manifest --dataset lahner2024bold --output docs/manifests/tribev2_lahner2024bold.example.json
nethobench tribev2-init-manifest --dataset hcp --output docs/manifests/tribev2_hcp.example.json
nethobench tribev2-init-manifest --dataset narratives --output docs/manifests/tribev2_narratives.example.json
```

Run the export pipeline:

```bash
nethobench tribev2-export --manifest /abs/path/to/manifest.json
```

Or with the direct script:

```bash
tribev2_export_nethobench --manifest /abs/path/to/manifest.json
```

## Outputs

The export writes:

- `pred_parcels.csv`
- `gt_group_parcels.csv`
- `tribev2_parcel_config.json`
- `human_subject_vs_group_scores.json`
- `split_half_ceiling_scores.json`
- `corruption_ladder_scores.json`
- `tribev2_benchmark_report.json`
- `tribe_vs_human_vs_ceiling.png`
- `corruption_sensitivity_ladder.png`

## Manifest Shape

The manifest is JSON with:

- `name`, `dataset`, `track`
- TRIBE settings: `tribev2_root`, `tribev2_checkpoint`, `tribev2_cache_folder`, `tribev2_device`
- preprocessing settings: `resample_frequency_hz`, `hemodynamic_lag_seconds`, `apply_hemodynamic_shift`, `zscore`, `detrend`
- `stimuli`
- `subjects`

Each stimulus item must specify exactly one source:

- `events_csv`
- `video_path`
- `audio_path`
- `text_path`

Each subject item must specify exactly one ground-truth source:

- `parcel_csv`
- `surface_left_path` + `surface_right_path`
- `volume_path`

For raw surfaces or volumes, provide `frequency_hz` or `tr_seconds`.

## Notes

- Parcelization is fixed to `360` HCP parcels on `fsaverage5`, with all left-hemisphere parcels first and all right-hemisphere parcels second.
- Raw-vertex mode is intentionally not supported.
- Volume-to-surface projection requires the `tribev2` stack plus its optional neuroimaging dependencies. Surface and parcel CSV inputs avoid that requirement.
- The benchmark report compares:
  - `TRIBE vs cohort mean`
  - `subject vs leave-one-out cohort mean`
  - `split-half cohort vs split-half cohort`
  - corrupted TRIBE predictions vs cohort mean
