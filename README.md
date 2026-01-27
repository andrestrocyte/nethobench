# NethoBench

Unified evaluation of neural realism (NeuroBench-style), behavioral realism (EthoBench-style), and cross-modal coupling.

![NethoBench Logo](assets/nethobench.png)

NethoBench outputs:
1) Neuro scores (neural realism)
2) Behavior scores (pose / kinematics realism)
3) Cross-modal scores (neural <-> behavior coupling)
4) A final composite (average over available axes)

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
- Behavior: keypoint columns with `_X/_Y` pairs (e.g., `CENTER_X`, `CENTER_Y`).
- For multimodal runs, GT and prediction CSVs should contain both neural and behavior columns (or provide a config that lists them).

## CLI
- Neuro scores (core metrics + neuro composite):
  - `nethobench neuro-scores --gt gt_neural.csv --preds pred_neural.csv`
- Neuro full analysis (runs the bundled notebook headlessly; saves plots + executed notebook):
  - `nethobench neuro-analysis --gt gt_neural.csv --preds pred_neural.csv --ddconfig configs/data-clean-all.json`
- Behavior-only:
  - `nethobench etho-scores --gt-dir /path/to/gt_dir --inf-dir /path/to/inf_dir`
  - add `--run-notebook` to also execute the bundled ethology notebook headlessly
- Multimodal scores (neuro + behavior + coupling):
  - `nethobench cross-scores --gt gt_multimodal.csv --preds pred_multimodal.csv --config config.json`
- Cross full analysis (headless notebook):
  - `nethobench cross-analysis --gt gt_multimodal.csv --preds pred_multimodal.csv --config config.json`

### Config schema (JSON)
```json
{
  "sequence_key": "sequenceId",
  "time_key": "itemPosition",
  "neuro_cols": ["region1", "region2", "region3"],
  "behavior_parts": ["CENTER", "NOSE", "TAIL_BASE"],
  "body_axis": ["NOSE", "TAIL_BASE"]  // optional override for direction metric
}
```

## Neuro scoring (benchmark v1)
Neuro uses mismatch-corrected realism scores grouped into buckets, then computes a weighted geometric-mean composite.

Core metric families:
- Distribution / marginals: tail-binned JSD (worst-q), symmetric KL (geo mean of `1/(1+KL)`), normalized W1 (`W1/IQR`), mean-shift realism (MeanShiftZ), and quantile-shape realism (tail + full).
- Dependence: kNN mutual information realism (mismatch-corrected).
- Point error: nRMSE and nMAE realism (IQR-normalized, mismatch-corrected).
- Temporal dynamics: autocorr realism, crosscorr realism, bandpower realism, CV-CCA realism.
- Inter-region structure: FC core realism, correlation-graph realism, PCA realism.
- Higher-order: skew/kurtosis realism.
- Manifold: kNN overlap, spectral similarity, Procrustes alignment, geodesic W1.

See `nethobench/notebooks/final_implementation_benchmark.ipynb` for exact definitions, parameters, and the final composite formula.
- **Behavior** (from EthoBench): position KL, quadrant KL, stationary fraction, velocity/acceleration KL, direction alignment (velocity vs body axis), syllable (k-means) distribution similarity, trajectory-shape similarity, multiplicative composite.
- **Cross-modal** (new):  
  - `cca_alignment_score`: gap between GT vs Pred **neural–behavior canonical correlations** (0–1).  
  - `neural_to_behavior_r2` / `behavior_to_neural_r2`: linear predictive R² in GT and Pred, with similarity scores.  
  - `lead_lag_score`: agreement of peak lead/lag between neural PCs and behavior speed.  
  - `cross_composite`: geometric mean of the above (ignoring NaNs).
- Bundled notebooks:
  - `notebooks/final_implementation_benchmark.ipynb` (neuro)
  - `notebooks/full_comprehensive_behavioral_analysis.ipynb` (etho)
  - `notebooks/cross_modal_full_analysis.ipynb` (cross)
  Headless runners save figures + executed notebooks under `./outputs/...`.

### Composite logic
- If only neuro: composite = neuro composite.  
- If only behavior: composite = behavior composite.  
- If multimodal: composite = average(neuro composite, behavior composite, cross composite) over available (finite) axes.

## Python API
```python
from nethobench import (
    compute_neuro_scores,
    compute_etho_scores,
    compute_cross_scores,
    run_ethobench_notebook,   # optional behavior notebook capture
)
```

## Outputs
- CLI prints scores and saves JSON when `--json-out` is passed.
- `neuro-analysis`, `etho-scores --run-notebook`, and `cross-analysis` save figures + executed notebook under `./outputs/.../`.
- `cross-scores` reports per-axis composites and the final multimodal composite.

## Example core-score output (CLI)
```
Neuro scores:
  JSD_worstq_score01_avg        : 0.xxx
  KL_geo_score01_avg            : 0.xxx
  W1n_mean_score01_avg          : 0.xxx
  MeanShiftZ_mean               : 0.xxx
  QNT_tail_score01_avg          : 0.xxx
  QNT_full_score01_avg          : 0.xxx
  MI_mean_score01_avg           : 0.xxx
  ERR_nRMSE_score01_avg         : 0.xxx
  ERR_nMAE_score01_avg          : 0.xxx
  AUTO_core_score01_avg         : 0.xxx
  CC_core_score01_avg           : 0.xxx
  Bandpower_score01_avg         : 0.xxx
  FC_core_score01_avg           : 0.xxx
  GRAPH_core_score01_avg        : 0.xxx
  PCA_comp_score01_avg          : 0.xxx
  CCA_core_score01_avg          : 0.xxx
  MOM_core_score01_avg          : 0.xxx
  MANI_core_score01_avg         : 0.xxx
  MANI_s_knn_score01_avg        : 0.xxx
  MANI_s_spec_score01_avg       : 0.xxx
  MANI_s_proc_score01_avg       : 0.xxx
  MANI_s_geo_score01_avg        : 0.xxx
  bucket_distribution           : 0.xxx
  bucket_dependence             : 0.xxx
  bucket_point_error            : 0.xxx
  bucket_temporal_dynamics      : 0.xxx
  bucket_inter_region_structure : 0.xxx
  bucket_higher_order           : 0.xxx
  bucket_manifold_components    : 0.xxx
  FINAL_NEURO_COMPOSITE_SCORE   : 0.xxx
```

## Status
- Focused on reproducible metrics; exploratory notebooks (ethology) can still be run via `ethobench`-style capture if you place the notebook under `nethobench/notebooks/`.  
- Cross-modal metrics are light-weight and interpretable; extend with your own in `nethobench/cross.py`.
