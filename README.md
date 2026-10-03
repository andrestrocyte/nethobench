# NethoBench

![NethoBench logo](assets/nethobench.png)

**Compare the structure of generated neural activity, behavior, and their interaction.**

A forecast can have a small pointwise error while losing variability, temporal structure, or relationships between channels. Conversely, a stochastic sample can preserve those properties without matching the observed trajectory point by point. NethoBench measures structural agreement and predictive fidelity separately, so you can see what a model preserves and where it fails.

Use it to compare forecasting models, inspect long rollouts, or characterize the effect of a model or input change on a fixed evaluation set. The package provides command-line scoring, a Python interface, and diagnostic plots.

## Choose an evaluation

| Your generated output | Command | What it measures |
| --- | --- | --- |
| Neural activity | `neuro-scores` | Five families of structural agreement and a weighted composite |
| Neural activity aligned to an observed trajectory | `fidelity-scores` | Pointwise error and mutual-information agreement, reported separately |
| Tracked body positions | `etho-scores` | Spatial, kinematic, trajectory, and movement-geometry agreement |
| Neural activity and behavior together | `cross-scores` | Neural and behavioral scores plus cross-modal coupling |

Choose the mode by **what the model produces**. A neural model conditioned on behavior still uses neural scoring; behavioral scoring evaluates generated behavior itself. Cross-modal scoring requires both neural and behavioral output streams.

## Install

Python **3.10 or newer** is required. From the root of a downloaded or cloned source checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
nethobench --help
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

For a pinned CPU scoring environment, use the [locked installation instructions](docs/reproducibility_and_compatibility.md#locked-scoring-and-test-environment). A normal installation resolves compatible dependency versions; the lock fixes the dependency set for repeatable evaluations.

## Quickstart with included data

The source checkout contains small datasets in [`tests/resources/data/`](tests/resources/data/). No external download or model training is needed. These fixtures exercise the scoring workflow; they are not a model-performance benchmark. Fixtures are not installed with the wheel.

### Neural structure

```bash
mkdir -p outputs/quickstart

nethobench neuro-scores \
  --gt tests/resources/data/neural/ground-truth/gt-ba4.csv \
  --preds tests/resources/data/neural/predictions/preds-ba4.csv \
  --json-out outputs/quickstart/neural.json
```

Scores are printed in the terminal and saved under the JSON `scores` key. Inspect the family profile before relying on its composite:

```python
import json
from pathlib import Path

scores = json.loads(Path("outputs/quickstart/neural.json").read_text())["scores"]
for family in ("distribution", "temporal_spectral", "relational", "geometry", "state_dynamics"):
    print(f"{family}: {scores[f'family_{family}']:.3f}")
print(f"composite: {scores['FINAL_COMPOSITE_SCORE']:.3f}")
```

### Pointwise fidelity

Evaluate the same prediction against the observed trajectory:

```bash
nethobench fidelity-scores \
  --gt tests/resources/data/neural/ground-truth/gt-ba4.csv \
  --preds tests/resources/data/neural/predictions/preds-ba4.csv \
  --json-out outputs/quickstart/fidelity.json
```

### Behavior and cross-modal coupling

```bash
nethobench etho-scores \
  --gt-dir tests/resources/data/behavioural/behav-ground-truth.csv \
  --inf-dir tests/resources/data/behavioural/behav-predictions.csv \
  --json-out outputs/quickstart/behavior

nethobench cross-scores \
  --gt tests/resources/data/cross/cross-ground-truth.csv \
  --preds tests/resources/data/cross/cross-predictions.csv \
  --json-out outputs/quickstart/cross
```

For `neuro-scores` and `fidelity-scores`, `--json-out` is a **file path**. For `etho-scores` and `cross-scores`, it is a **directory**, with results written to `scores.json` inside it. Behavioral inputs can be individual CSV/Parquet files or directories containing them.

## Behavioral score versions

Version 0.3.0 makes **Etho v2** the default for `etho-scores` and the behavioral
Python pipeline. Its equal-weight geometric mean includes the original eight
components plus **inter-part distance distributions** and **state-conditional
bout duration**. The new duration comparison handles bouts cut off by recording
boundaries and reports whether the data support the requested duration range.

Use one training-fitted calibration for every compared model. It fixes distance
scales, speed/acceleration states and duration normalization:

```python
import json
import pandas as pd
from nethobench.etho.pipeline import fit_etho_calibration, compute_etho_scores

calibration = fit_etho_calibration(
    pd.read_csv("training_poses.csv"),
    {"sampling_interval": 0.05, "duration_horizon_frames": 20},
)
with open("etho_calibration.json", "w") as handle:
    json.dump(calibration, handle, indent=2)
scores, per_sequence, means, stds = compute_etho_scores(
    "reference_poses.csv", "forecast_poses.csv",
    cfg={"etho_calibration": "etho_calibration.json",
         "etho_details_path": "etho_details.json"},
)
```

The example requests a one-second duration horizon at 20 Hz. Choose that horizon
before comparing models. Without supplied calibration, v2 fits only the reference
poses. It requires at least two tracked points and enough temporal support;
**an unavailable component makes the v2 composite unavailable**, rather than
silently dropping that component. Small smoke-test fixtures need not support all
components. Inspect the two duration coverage values and the detail file.

For exact reproduction of the original API, keys and eight-component composite,
pass `cfg={"etho_score_version": "legacy_v1"}`. For the CLI, put
`{"etho_score_version": "legacy_v1"}` in a JSON file and pass `--config`.
The default v2 result also retains `legacy_composite_score`. Existing per-sequence
summaries are unchanged; the two additions compare pooled distributions.
`cross-scores` retains its existing behavioral definition, and neural scoring is
unchanged. Do not mix score versions in a comparison.

See [definitions, limitations and validation](docs/etho_v2.md) for the formulas,
configuration and comparisons with the earlier implementations.

## Use your own data

### Neural tables

Provide reference and prediction CSVs with sequence IDs, time indices, and matching neural channels:

```csv
sequenceId,itemPosition,region_1,region_2
0,0,0.12,0.41
0,1,0.18,0.39
1,0,0.09,0.44
1,1,0.15,0.40
```

This illustrates the column layout, not a sufficient dataset for all metrics. Use sufficiently long sequences and multiple windows for meaningful structural estimates. Keep time steps ordered, use a consistent sampling rate, and preserve channel correspondence between reference and prediction.

By default, the keys are `sequenceId` and `itemPosition`. Neural scoring treats other columns as neural features; use an explicit channel list when tables contain metadata.

### Behavioral and multimodal tables

Behavioral coordinates use paired `_X` and `_Y` suffixes, such as `CENTER_X`, `CENTER_Y`, `NOSE_X`, and `NOSE_Y`. The default center is `CENTER`, and the default body axis is `NOSE` to `TAIL_BASE`. Multimodal tables contain neural channels and these coordinates on the same sequence/time grid.

### Configuration

Pass a JSON file to override column names, select features, or set data-dependent parameters:

```json
{
  "sequence_key": "trial_id",
  "time_key": "frame_idx",
  "neuro_cols": ["vis_ctx_1", "vis_ctx_2", "motor_ctx"],
  "WELCH_SAMPLING_FREQUENCY": 60.0
}
```

```bash
nethobench neuro-scores \
  --gt reference.csv --preds forecast.csv \
  --config config.json --json-out outputs/neural.json
```

Set `WELCH_SAMPLING_FREQUENCY` to the **actual sampling rate in Hz**; its default is 30.0. The setting affects physical-time autocorrelation lags and spectral calculations. For behavioral data, `behavior_parts`, `center_part`, and `body_axis` configure the tracked landmarks. Keep evaluation settings fixed across models.

See [temporal and spectral definitions](docs/psd_integration.md) and [configuration defaults](nethobench/utils/evaluation_constants.py) for available parameters.

## Python interface

Neural scoring also accepts NumPy arrays with shape **`(sequence, time, channel)`**. Reference and prediction must describe corresponding channels and evaluation windows.

```python
import numpy as np

from nethobench.neuro.metrics.composites import calculate_neuro_composites
from nethobench.utils.evaluation_constants import config

# Illustrative arrays; replace these with aligned reference and forecast data.
rng = np.random.default_rng(42)
reference = rng.normal(size=(8, 128, 4))
prediction = 0.8 * reference + rng.normal(scale=0.25, size=reference.shape)

config.update_from_dict({"WELCH_SAMPLING_FREQUENCY": 30.0})
scores = calculate_neuro_composites(reference, prediction)

print(scores["family_relational"])
print(scores["family_temporal_spectral"])
print(scores["FINAL_COMPOSITE_SCORE"])
```

The configuration object is shared within the Python process. Set it explicitly before an evaluation when switching between datasets or sampling rates.

## Understand the neural scores

The neural structural score uses **18 components grouped into five families**. Finite similarity scores lie in `[0, 1]`, with larger values indicating closer agreement under that metric.

| Family | Question | Components | Composite weight |
| --- | --- | --- | ---: |
| Distributional | Are activity values distributed similarly? | Histogram divergence, quantiles, moments, means | 0.22 |
| Temporal-spectral | Are temporal organization and spectral content preserved? | Trajectory distribution, autocorrelation, power spectrum | 0.18 |
| Relational | Are relationships between channels preserved? | Graph agreement, cross-region mutual information, lagged covariance, VAR(1) response proxy | 0.24 |
| Geometric | Is population geometry preserved? | Persistent-homology/local-neighborhood agreement, subspace angles | 0.18 |
| State-dynamical | Are state occupancy and transitions similar? | Two occupancy and three transition comparisons | 0.18 |

Components have equal weight **within** each family. With all families available:

```text
structural composite = 0.22 × distributional
                     + 0.18 × temporal-spectral
                     + 0.24 × relational
                     + 0.18 × geometric
                     + 0.18 × state-dynamical
```

Pointwise fidelity is separate: `0.65 × Error_score + 0.35 × MI_score`. It does not enter the neural structural composite. Cross-modal output also includes a separate aggregate over the available evaluation domains.

Short or degenerate inputs can leave components undefined. Aggregation renormalizes weights over available components or families, so inspect missing values before comparing composites. Diagnostic outputs such as `PSDShape_score`, `PSDPower_score`, and `DirectionalDynamics_score` do not receive extra composite weight. Keys ending in `_score01` are compatibility aliases, not additional metrics.

The [metric definitions](nethobench/neuro/metrics/definitions.py) specify membership and weights; the [temporal guide](docs/psd_integration.md) explains ACF and PSD scoring.

## Generate diagnostic figures

The analysis commands produce plots and reports for the same input domains:

```bash
nethobench neuro-analysis \
  --gt tests/resources/data/neural/ground-truth/gt-ba4.csv \
  --preds tests/resources/data/neural/predictions/preds-ba4.csv \
  --output-root outputs/neural-analysis
```

Use `etho-analysis` or `cross-analysis` for behavioral or multimodal inputs. Each accepts `--output-root`; run `nethobench <command> --help` for its input arguments.

## Compare models carefully

- **Score the same support.** Match channels, preprocessing, sampling rate, sequence lengths, and evaluation windows. When evaluating forecasts, exclude the observed context unless including it is an explicit part of the protocol.
- **Inspect families alongside the composite.** Similar totals can hide different structural failures. Report component availability as well as score values.
- **Separate structure from conditional accuracy.** A realistic independent trajectory need not predict the observed future. Pair structural metrics with MAE/MSE, proper probabilistic scores for stochastic predictions, and task-specific endpoints where relevant.
- **Distinguish samples from their average.** The mean score of stochastic draws is generally different from the score of their mean trajectory. Record which quantity you evaluate.
- **Keep the scoring definition fixed.** Default lagged-covariance and VAR comparisons pool sequence rows and compare one matrix triangle. Explicit within-sequence, directed alternatives are available through the [implementation-sensitivity diagnostics](docs/reproducibility_and_compatibility.md#opt-in-implementation-sensitivity); they do not replace default CLI outputs. Do not mix definitions in one comparison.

Structural agreement is a descriptive measurement, not evidence of causal connectivity, biological mechanism, or downstream usefulness. Split-half reference comparisons and clearly specified perturbations help interpret the scores on a particular dataset; they do not define universal ceilings or floors.

## Reproducibility and development

Record the package version, dependency environment, input files, preprocessing, evaluation configuration, and stochastic seeds with every run. Version 0.3.0 versions the behavioral composite explicitly; neural definitions and optional neural diagnostics remain unchanged. Select `legacy_v1` to reproduce earlier behavioral scores.

To install the test dependencies and run the suite:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Automated checks cover unit and integration tests, installed-package behavior, and numerical compatibility. See [reproducibility and compatibility](docs/reproducibility_and_compatibility.md), the [dependency locks](requirements/), and the [changelog](CHANGELOG.md) for details.

## License

NethoBench is distributed under the [MIT License](LICENSE).
