# NethoBench: Unified Neural & Behavioral Benchmark

![NethoBench Logo](assets/nethobench.png)

NethoBench merges **neural** (NeuroBench) and **behavioral** (EthoBench) plausibility checks with a new **cross-modal axis** that tests whether brain and body stay coupled. It produces:

1) **Neuro scores** – distributional / temporal / network / geometric fidelity of neural activity.  
2) **Behavior scores** – kinematics / geometry / syllables / trajectory plausibility of pose tracks.  
3) **Cross-modal scores** – consistency of neural ↔ behavior coupling (encoding/decoding alignment).  
4) A **composite** that averages available axes (single-modality if only one axis is present).

## Quick install
```bash
pip install -e .
```

## Data expectations
- Files must include `sequenceId` and `itemPosition` for alignment.  
- **Neural**: numeric columns listed under `neuro_cols` in the config (or all non-index columns for the neuro-only command).  
- **Behavior**: keypoint columns with `_X/_Y` pairs (e.g., `CENTER_X`, `CENTER_Y`, `NOSE_X`, `NOSE_Y`, `TAIL_BASE_X`, `TAIL_BASE_Y`). List bases under `behavior_parts` in the config.  
- For multimodal runs, GT and prediction CSVs should contain both neural and behavior columns.

## CLI
- Neuro-only:  
  `nethobench neuro-scores --gt gt_neural.csv --preds pred_neural.csv`
- Neuro full analysis (saves figures):  
  `nethobench neuro-analysis --gt gt_neural.csv --preds pred_neural.csv --ddconfig configs/data-clean-all.json`
- Behavior-only:  
  `nethobench etho-scores --gt-dir /path/to/gt_dir --inf-dir /path/to/inf_dir`
  - Add `--run-notebook` to also execute the bundled ethology notebook headlessly.
- Multimodal (neural + behavior + coupling):  
  `nethobench cross-scores --gt gt_multimodal.csv --preds pred_multimodal.csv --config config.json`
 - Cross full analysis (headless notebook):  
  `nethobench cross-analysis --gt gt_multimodal.csv --preds pred_multimodal.csv --config config.json`

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

## What gets scored
- **Neuro** (from NeuroBench): mean-diff, symmetric-KL, correlation-graph overlap, PCA spectrum, autocorr, PSD similarity, multiplicative composite.
- **Behavior** (from EthoBench): position KL, quadrant KL, stationary fraction, velocity/acceleration KL, direction alignment (velocity vs body axis), syllable (k-means) distribution similarity, trajectory-shape similarity, multiplicative composite.
- **Cross-modal** (new):  
  - `cca_alignment_score`: gap between GT vs Pred **neural–behavior canonical correlations** (0–1).  
  - `neural_to_behavior_r2` / `behavior_to_neural_r2`: linear predictive R² in GT and Pred, with similarity scores.  
  - `lead_lag_score`: agreement of peak lead/lag between neural PCs and behavior speed.  
  - `cross_composite`: geometric mean of the above (ignoring NaNs).
- Bundled notebooks: `notebooks/final_implementation_benchmark.ipynb` (neuro), `notebooks/full_comprehensive_behavioral_analysis.ipynb` (etho), and `notebooks/cross_modal_full_analysis.ipynb` (cross). Headless runners save figures + executed notebooks under `./outputs/...`.

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
- CLI prints scores and saves JSON when `--json-out` is passed (per-sequence stats included).  
- `neuro-analysis`, `etho-scores --run-notebook`, and `cross-analysis` save figures + executed notebook under `./outputs/…/`.  
- `cross-scores` also reports the per-axis composites and the final multimodal composite.

## Status
- Focused on reproducible metrics; exploratory notebooks (ethology) can still be run via `ethobench`-style capture if you place the notebook under `nethobench/notebooks/`.  
- Cross-modal metrics are light-weight and interpretable; extend with your own in `nethobench/cross.py`.
