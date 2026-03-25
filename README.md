# NethoBench

NethoBench is a benchmark for multimodal brain-behavior models that evaluates neural realism, behavioral realism, and cross-modal plausibility within a single framework. It combines population-level neural metrics spanning distributional structure, temporal dynamics, inter-regional interactions, low-dimensional geometry, and latent state dynamics with complementary behavioral metrics on pose trajectories, kinematics, motifs, and trajectory statistics. Crucially, it adds an explicit cross-modal axis through neural-to-behavior decoding, behavior-to-neural encoding, latent alignment, and temporal-consistency measures, exposing failure modes that unimodal metrics or task losses alone can miss.

![NethoBench Logo](assets/nethobench.png)

NethoBench outputs:
1. Neuro score (default neural realism benchmark)
2. Fidelity score (separate direct trace-alignment family)
3. Behavior scores (pose / kinematics realism)
4. Cross-modal scores (neural <-> behavior coupling)
5. A final multimodal composite (average over available neuro / behavior / cross axes)

## Install
```bash
pip install -e .
```

Recommended (clean environment):
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Data expectations
- Files must include `sequenceId` and `itemPosition` for alignment.
- Neural: region columns (one column per region).
- Behavior: keypoint columns with `_X/_Y` pairs (e.g. `CENTER_X`, `CENTER_Y`).
- For multimodal runs, GT and prediction CSVs should contain both neural and behavior columns, or you should provide a config that lists them.

## CLI
- Neuro score:
  - `nethobench neuro-scores --gt gt_neural.csv --preds pred_neural.csv`
  - Uses the active score cells from `nethobench/notebooks/neuro_metrics.ipynb`.
- Fidelity score:
  - `nethobench fidelity-scores --gt gt_neural.csv --preds pred_neural.csv`
  - Reports the separate direct-fidelity family only.
- Neuro full analysis:
  - `nethobench neuro-analysis --gt gt_neural.csv --preds pred_neural.csv --ddconfig configs/data-clean-all.json`
  - Executes the full active notebook and saves plots + executed notebook + scores.
- Synthetic validation:
  - `nethobench synthetic-validation --output-root outputs/synthetic_validation`
- Harder biophysical synthetic validation:
  - `nethobench synthetic-validation-biophysical --output-root outputs/biophysical_validation`
- Behavior-only:
  - `nethobench etho-scores --gt-dir /path/to/gt_dir --inf-dir /path/to/inf_dir`
  - Add `--run-notebook` to also execute the bundled ethology notebook headlessly.
- Multimodal scores:
  - `nethobench cross-scores --gt gt_multimodal.csv --preds pred_multimodal.csv --config config.json`
  - Uses the default neuro score internally.
- Cross full analysis:
  - `nethobench cross-analysis --gt gt_multimodal.csv --preds pred_multimodal.csv --config config.json`

### Config schema (JSON)
```json
{
  "sequence_key": "sequenceId",
  "time_key": "itemPosition",
  "neuro_cols": ["region1", "region2", "region3"],
  "behavior_parts": ["CENTER", "NOSE", "TAIL_BASE"],
  "body_axis": ["NOSE", "TAIL_BASE"]
}
```

## Neuro scoring
`nethobench neuro-scores` is driven by the active notebook implementation in `nethobench/notebooks/neuro_metrics.ipynb` via the extracted baseline-cell runner in `nethobench/analysis/neuro_metrics_core_script.py`.

`nethobench neuro-analysis` executes that same notebook end to end and exports the full figure set, including corruption ladders, family polar summaries, and the unified dashboard.

The default neuro score is a curated structural neural realism benchmark. Every metric is bounded to `[0, 1]`, where higher means the predicted neural population matches the reference population more closely.

The active neuro families are:
- `distribution`: whether marginal activity statistics, tails, and moments are realistic.
- `temporal_spectral`: whether neural trajectories evolve through time like the reference data.
- `relational`: whether inter-regional interactions and lagged dependencies are realistic.
- `geometry`: whether the dominant latent organization and manifold geometry are realistic.
- `state_dynamics`: whether the model occupies and transitions between GT-defined latent states correctly.

### Active metric list

- `distribution`
  - `KL_or_JSD_score01`: histogram-based similarity of per-region activity-value distributions.
  - `QNT_score01`: agreement of quantile structure and tail behavior.
  - `MOM_score01`: agreement of variance, skewness, and kurtosis structure.
  - `Mean_score01`: agreement of first-order location statistics.

- `temporal_spectral`
  - `TRJDIST_score01`: low-dimensional trajectory realism, combining occupancy, speed, turning, and path-shape features in a shared GT latent space.

- `relational`
  - `GRAPH_score01`: agreement of graph-level interaction structure derived from inter-regional coupling.
  - `CrossRegionMI_score01`: agreement of region-by-region mutual-information structure within each dataset.
  - `LaggedCovariance_score01`: agreement of covariance structure at nonzero temporal lags.
  - `ImpulseResponse_score01`: agreement of simple directed temporal influence kernels between regions.

- `geometry`
  - `MANI_score01`: agreement of low-dimensional geometry, combining persistent-topology lifetimes with local-neighborhood structure.
  - `SubspaceAngle_score01`: agreement of dominant latent subspaces via principal angles.

- `state_dynamics`
  - `LatentStateOccupancyK11_score01`: agreement of GT-derived latent-state occupancy using an 11-state partition.
  - `LatentStateOccupancyK12_score01`: same occupancy test with a 12-state partition.
  - `LatentStateTransitionLag1K11_score01`: agreement of one-step transition structure between GT-derived latent states.
  - `LatentStateTransitionLag2K11_score01`: agreement of two-step transition structure between GT-derived latent states.
  - `LatentStateTransitionLag3K11_score01`: agreement of three-step transition structure between GT-derived latent states.

### Family and composite definition
The default neuro score is a weighted arithmetic mean over family composites:

$$
D = \frac{s_{KL} + s_{QNT} + s_{MOM} + s_{Mean}}{4}
$$

$$
T = s_{TRJDIST}
$$

$$
R = \frac{s_{GRAPH} + s_{CrossRegionMI} + s_{LaggedCovariance} + s_{ImpulseResponse}}{4}
$$

$$
G = \frac{s_{MANI} + s_{SubspaceAngle}}{2}
$$

$$
S = \frac{s_{Occ11} + s_{Occ12} + s_{Trans1} + s_{Trans2} + s_{Trans3}}{5}
$$

$$
\mathrm{Composite} = 0.22D + 0.18T + 0.24R + 0.18G + 0.18S
$$

The CLI reports these as:
- `family_distribution`
- `family_temporal_spectral`
- `family_relational`
- `family_geometry`
- `family_state_dynamics`
- `FINAL_COMPOSITE_SCORE`

### Separate fidelity score
The direct trace-alignment family is now reported separately:

- `Error_score01`
- `MI_score01`
- `family_fidelity`
- `FIDELITY_SCORE`

with:

$$
S_{fidelity} = 0.65 \, s_{Error} + 0.35 \, s_{MI}
$$

Run it with:
```bash
nethobench fidelity-scores --gt gt_neural.csv --preds pred_neural.csv
```

### Legacy metrics
The previous broader metric inventory is preserved under:
- `nethobench/legacy/legacy_metrics.py`
- `nethobench/legacy/legacy_neuro_core_script.py`
- `nethobench/legacy/notebooks/neuro_metrics_legacy.ipynb`

The public CLI now defaults to the curated neuro score above. The legacy folder is retained for historical comparison, ablations, and migration.

## Synthetic validation
`nethobench synthetic-validation` builds a synthetic neural benchmark with fully known structure and validates the active neuro score against it. The generator uses low-dimensional latent dynamics with region loadings and calcium-like observations. The validation:

- estimates empirical oracle ceilings from independent draws of the same generator
- applies targeted perturbations in generator parameter space for the active neuro score families
- writes family-level and metric-level dose-response and selectivity tables
- exports synthetic CSV datasets with `sequenceId` and `itemPosition`, so the same data can be reused downstream in Sequifier or other pipelines

`nethobench synthetic-validation-biophysical` repeats the same logic on a harder spike/event-driven calcium world with heterogeneous kinetics and state switches.

These validation commands audit the active neuro score. Fidelity is intentionally separate and can be computed independently with `fidelity-scores` if needed.

## Behavior and cross-modal scoring
- **Behavior** (from EthoBench): position KL, quadrant KL, stationary fraction, velocity/acceleration KL, direction alignment, syllable distribution similarity, and trajectory-shape similarity.
- **Cross-modal**: neural-behavior CCA alignment, bidirectional predictive `R^2`, lead-lag agreement, and a cross-modal composite over available terms.
- Bundled notebooks live under `nethobench/notebooks/`.
- The active neural benchmark notebook is `nethobench/notebooks/neuro_metrics.ipynb`.
- Headless analysis commands save figures, notebook wrappers, and score JSON under `./outputs/...`.

### Composite logic
- If only neuro: composite = neuro composite.
- If only behavior: composite = behavior composite.
- If multimodal: composite = average(neuro composite, behavior composite, cross composite) over available finite axes.

## Python API
```python
from nethobench import (
    compute_neuro_scores,
    compute_fidelity_scores,
    run_neuro_full_analysis,
    compute_etho_scores,
    compute_cross_scores,
    generate_synthetic_neural_dataset,
    run_synthetic_neuro_validation,
    run_biophysical_synthetic_neuro_validation,
    run_ethobench_notebook,
    run_cross_full_analysis,
)
```

## Outputs
- `neuro-scores` prints the active neuro score metrics with colored arrow bars and always saves a JSON payload. `--json-out` lets you choose the path.
- `fidelity-scores` does the same for the separate fidelity family.
- `neuro-analysis`, `etho-scores --run-notebook`, and `cross-analysis` save figures + executed notebook under `./outputs/.../`.
- `cross-scores` reports per-axis composites and the final multimodal composite.

## Example neuro-score output (CLI)
The terminal CLI uses ANSI colors; the example below shows the same structure in plain text so it renders cleanly on GitHub.

```text
Neuro scores:
  KL_or_JSD_score01                   : 0.742  [medium]  0 ━━━━━━━━━━━▶──── 1
  QNT_score01                         : 0.766  [medium]  0 ━━━━━━━━━━━▶──── 1
  MOM_score01                         : 0.641  [medium]  0 ━━━━━━━━━▶────── 1
  Mean_score01                        : 0.781  [medium]  0 ━━━━━━━━━━━━▶─── 1
  TRJDIST_score01                     : 0.989  [high]    0 ━━━━━━━━━━━━━━━▶ 1
  GRAPH_score01                       : 0.996  [high]    0 ━━━━━━━━━━━━━━━▶ 1
  CrossRegionMI_score01               : 0.911  [high]    0 ━━━━━━━━━━━━━▶── 1
  LaggedCovariance_score01            : 0.969  [high]    0 ━━━━━━━━━━━━━━▶─ 1
  ImpulseResponse_score01             : 0.922  [high]    0 ━━━━━━━━━━━━━▶── 1
  MANI_score01                        : 0.884  [high]    0 ━━━━━━━━━━━━▶─── 1
  SubspaceAngle_score01               : 0.946  [high]    0 ━━━━━━━━━━━━━━▶─ 1
  LatentStateOccupancyK11_score01     : 0.910  [high]    0 ━━━━━━━━━━━━━▶── 1
  LatentStateOccupancyK12_score01     : 0.903  [high]    0 ━━━━━━━━━━━━━▶── 1
  LatentStateTransitionLag1K11_score01: 0.861  [high]    0 ━━━━━━━━━━━━▶── 1
  LatentStateTransitionLag2K11_score01: 0.844  [high]    0 ━━━━━━━━━━━━▶── 1
  LatentStateTransitionLag3K11_score01: 0.829  [high]    0 ━━━━━━━━━━━▶─── 1
  family_distribution                 : 0.733  [medium]  0 ━━━━━━━━━━━▶──── 1
  family_temporal_spectral            : 0.989  [high]    0 ━━━━━━━━━━━━━━━▶ 1
  family_relational                   : 0.950  [high]    0 ━━━━━━━━━━━━━━▶─ 1
  family_geometry                     : 0.915  [high]    0 ━━━━━━━━━━━━━▶── 1
  family_state_dynamics               : 0.869  [high]    0 ━━━━━━━━━━━━▶── 1
  FINAL_COMPOSITE_SCORE               : 0.891  [high]    0 ━━━━━━━━━━━━━▶── 1
```

## Reproducibility
The notebook-backed neuro score, the separate fidelity command, and the downstream synthetic / Sequifier / CalciumGAN workspaces are all aligned to the same active neuro metric inventory.

Full rerun command list:
- [final_score_rerun_commands.md](/Users/deviandr/nethobench/docs/final_score_rerun_commands.md)

Single orchestration entry point:
```bash
cd /Users/deviandr/nethobench
bash scripts/run_full_project_rerun.sh all
```

## Status
- The default neural benchmark is now the curated structural neuro score described above.
- The previous broad metric inventory remains available under `legacy/`.
- Cross-modal metrics remain lightweight and interpretable; extend them in `nethobench/cross.py` if needed.

## License
This project is released under the MIT License. See `LICENSE`.
