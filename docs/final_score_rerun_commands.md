# Final Score Rerun Commands

This document collects the commands needed to rerun the project against the current default neuro score.

Important:
- `nethobench neuro-scores` now uses the curated structural neuro score.
- `nethobench fidelity-scores` is separate.
- Some external workspaces write into fixed `results/` directories. If you need to preserve previous outputs, copy those workspaces first.
- The full rerun script now also computes fidelity sidecars wherever aligned prediction-vs-GT evaluation exists. These are saved separately and never folded into the neuro composite.

## One-command orchestration

You can also run the whole pipeline sequentially with:

```bash
cd /Users/deviandr/nethobench
bash scripts/run_full_project_rerun.sh all
```

Or only selected sections, for example:

```bash
cd /Users/deviandr/nethobench
bash scripts/run_full_project_rerun.sh scaling synthetic bio-oracle
```

Available sections:
- `install`
- `real`
- `scaling`
- `synthetic`
- `bio-oracle`
- `seq-synth`
- `seq-bio`
- `calciumgan`

## 1. Install the current local package

```bash
/Users/deviandr/miniforge3/bin/python -m pip install -e /Users/deviandr/nethobench
```

## 2. Single-run neuro and fidelity scores

```bash
cd /Users/deviandr/nethobench
/Users/deviandr/miniforge3/bin/python -m nethobench.cli neuro-scores --gt gt_neural.csv --preds pred_neural.csv
```

```bash
cd /Users/deviandr/nethobench
/Users/deviandr/miniforge3/bin/python -m nethobench.cli fidelity-scores --gt gt_neural.csv --preds pred_neural.csv
```

## 3. Real benchmark model comparisons

### 3.1 `organized_csv`

```bash
cd /Users/deviandr/nethobench
for m in OS 1_step AR TF AR_KV; do
  /Users/deviandr/miniforge3/bin/python -m nethobench.cli neuro-scores \
    --gt "/Users/deviandr/Desktop/organized_csv/$m/gt.csv" \
    --preds "/Users/deviandr/Desktop/organized_csv/$m/pred.csv" \
    --json-out "/Users/deviandr/Desktop/organized_csv/$m/neuro_scores_final.json"
done
```

### 3.2 `csv_90_360`

```bash
cd /Users/deviandr/nethobench
for m in OS 1_step AR TF AR_KV; do
  /Users/deviandr/miniforge3/bin/python -m nethobench.cli neuro-scores \
    --gt "/Users/deviandr/Desktop/csv_90_360/$m/gt.csv" \
    --preds "/Users/deviandr/Desktop/csv_90_360/$m/pred.csv" \
    --json-out "/Users/deviandr/Desktop/csv_90_360/$m/neuro_scores_final.json"
done
```

### 3.3 `csv_90_630`

```bash
cd /Users/deviandr/nethobench
for m in OS 1_step AR TF AR_KV; do
  /Users/deviandr/miniforge3/bin/python -m nethobench.cli neuro-scores \
    --gt "/Users/deviandr/Desktop/csv_90_630/$m/gt.csv" \
    --preds "/Users/deviandr/Desktop/csv_90_630/$m/pred.csv" \
    --json-out "/Users/deviandr/Desktop/csv_90_630/$m/neuro_scores_final.json"
done
```

### 3.4 `csv_90_1260`

```bash
cd /Users/deviandr/nethobench
for m in OS 1_step AR TF AR_KV; do
  /Users/deviandr/miniforge3/bin/python -m nethobench.cli neuro-scores \
    --gt "/Users/deviandr/Desktop/csv_90_1260/$m/gt.csv" \
    --preds "/Users/deviandr/Desktop/csv_90_1260/$m/pred.csv" \
    --json-out "/Users/deviandr/Desktop/csv_90_1260/$m/neuro_scores_final.json"
done
```

### 3.5 `csv_90_2070`

```bash
cd /Users/deviandr/nethobench
for m in TF AR_KV; do
  /Users/deviandr/miniforge3/bin/python -m nethobench.cli neuro-scores \
    --gt "/Users/deviandr/Desktop/csv_90_2070/$m/gt.csv" \
    --preds "/Users/deviandr/Desktop/csv_90_2070/$m/pred.csv" \
    --json-out "/Users/deviandr/Desktop/csv_90_2070/$m/neuro_scores_final.json"
done
```

### 3.6 `csv_90_4050`

```bash
cd /Users/deviandr/nethobench
for m in TF AR_KV; do
  /Users/deviandr/miniforge3/bin/python -m nethobench.cli neuro-scores \
    --gt "/Users/deviandr/Desktop/csv_90_4050/$m/gt.csv" \
    --preds "/Users/deviandr/Desktop/csv_90_4050/$m/pred.csv" \
    --json-out "/Users/deviandr/Desktop/csv_90_4050/$m/neuro_scores_final.json"
done
```

## 4. Scaling-law rerun

These commands rerun the scaling-law benchmark and rebuild the tables and dashboards using the new default neuro score.

```bash
/Users/deviandr/miniforge3/bin/python -m pip install altair
```

```bash
cd /Users/deviandr/Desktop/netho-seq-scaling-rerun
/Users/deviandr/miniforge3/bin/python analysis/run_neurobench_batch.py --mode full --overwrite
```

```bash
cd /Users/deviandr/Desktop/netho-seq-scaling-rerun
/Users/deviandr/miniforge3/bin/python analysis/build_scaling_visualizations.py
```

## 5. Original synthetic oracle validation

```bash
cd /Users/deviandr/nethobench
/Users/deviandr/miniforge3/bin/python -m nethobench.cli synthetic-validation \
  --output-root /Users/deviandr/Desktop/nethobench_synthetic_validation_final \
  --n-sequences 100 \
  --seq-length 768 \
  --n-regions 16 \
  --latent-dim 6 \
  --burn-in 128 \
  --oracle-replicates 3
```

## 6. Hard biophysical oracle validation

Recommended large run:

```bash
cd /Users/deviandr/nethobench
nohup /Users/deviandr/miniforge3/bin/python -u -m nethobench.cli synthetic-validation-biophysical \
  --output-root /Users/deviandr/Desktop/nethobench_biophysical_validation_final \
  --n-sequences 128 \
  --seq-length 1024 \
  --n-regions 16 \
  --latent-dim 6 \
  --burn-in 192 \
  --oracle-replicates 4 \
  > /Users/deviandr/Desktop/nethobench_biophysical_validation_final.log 2>&1 &
```

Watch it:

```bash
tail -f /Users/deviandr/Desktop/nethobench_biophysical_validation_final.log
```

## 7. Sequifier convergence on the original synthetic world

```bash
cd /Users/deviandr/Desktop/nethobench-sequifier-convergence
/Users/deviandr/miniforge3/bin/python scripts/prepare_convergence_project.py --force
```

```bash
cd /Users/deviandr/Desktop/nethobench-sequifier-convergence
PYTHONPATH=/Users/deviandr/sequifier/src /Users/deviandr/miniforge3/bin/python -m sequifier.sequifier preprocess --config-path configs/preprocess.yaml
```

```bash
cd /Users/deviandr/Desktop/nethobench-sequifier-convergence
PYTHONPATH=/Users/deviandr/sequifier/src /Users/deviandr/miniforge3/bin/python -m sequifier.sequifier preprocess --config-path configs/preprocess_underfit.yaml
```

```bash
cd /Users/deviandr/Desktop/nethobench-sequifier-convergence
PYTHONPATH=/Users/deviandr/sequifier/src /Users/deviandr/miniforge3/bin/python -m sequifier.sequifier train --config-path configs/train_converged.yaml
```

```bash
cd /Users/deviandr/Desktop/nethobench-sequifier-convergence
PYTHONPATH=/Users/deviandr/sequifier/src /Users/deviandr/miniforge3/bin/python -m sequifier.sequifier train --config-path configs/train_underfit.yaml
```

```bash
cd /Users/deviandr/Desktop/nethobench-sequifier-convergence
/Users/deviandr/miniforge3/bin/python scripts/run_inference_and_evaluate.py --run-name converged
```

```bash
cd /Users/deviandr/Desktop/nethobench-sequifier-convergence
/Users/deviandr/miniforge3/bin/python scripts/run_inference_and_evaluate.py --run-name underfit
```

```bash
cd /Users/deviandr/Desktop/nethobench-sequifier-convergence
/Users/deviandr/miniforge3/bin/python scripts/run_inference_and_evaluate.py --run-name converged --eval-mode rollout --rollout-steps 1024
```

```bash
cd /Users/deviandr/Desktop/nethobench-sequifier-convergence
/Users/deviandr/miniforge3/bin/python scripts/run_inference_and_evaluate.py --run-name underfit --eval-mode rollout --rollout-steps 1024
```

## 8. Sequifier convergence on the harder biophysical world

```bash
cd /Users/deviandr/Desktop/nethobench-sequifier-convergence-biophysical
/Users/deviandr/miniforge3/bin/python scripts/prepare_convergence_project.py --force
```

```bash
cd /Users/deviandr/Desktop/nethobench-sequifier-convergence-biophysical
PYTHONPATH=/Users/deviandr/sequifier/src /Users/deviandr/miniforge3/bin/python -m sequifier.sequifier preprocess --config-path configs/preprocess.yaml
```

```bash
cd /Users/deviandr/Desktop/nethobench-sequifier-convergence-biophysical
PYTHONPATH=/Users/deviandr/sequifier/src /Users/deviandr/miniforge3/bin/python -m sequifier.sequifier preprocess --config-path configs/preprocess_weakest.yaml
```

```bash
cd /Users/deviandr/Desktop/nethobench-sequifier-convergence-biophysical
PYTHONPATH=/Users/deviandr/sequifier/src /Users/deviandr/miniforge3/bin/python -m sequifier.sequifier train --config-path configs/train_converged.yaml
```

```bash
cd /Users/deviandr/Desktop/nethobench-sequifier-convergence-biophysical
PYTHONPATH=/Users/deviandr/sequifier/src /Users/deviandr/miniforge3/bin/python -m sequifier.sequifier train --config-path configs/train_weakest.yaml
```

```bash
cd /Users/deviandr/Desktop/nethobench-sequifier-convergence-biophysical
/Users/deviandr/miniforge3/bin/python scripts/run_inference_and_evaluate.py --run-name converged
```

```bash
cd /Users/deviandr/Desktop/nethobench-sequifier-convergence-biophysical
/Users/deviandr/miniforge3/bin/python scripts/run_inference_and_evaluate.py --run-name weakest
```

```bash
cd /Users/deviandr/Desktop/nethobench-sequifier-convergence-biophysical
/Users/deviandr/miniforge3/bin/python scripts/run_inference_and_evaluate.py --run-name converged --eval-mode rollout --rollout-steps 1024
```

```bash
cd /Users/deviandr/Desktop/nethobench-sequifier-convergence-biophysical
/Users/deviandr/miniforge3/bin/python scripts/run_inference_and_evaluate.py --run-name weakest --eval-mode rollout --rollout-steps 1024
```

## 9. CalciumGAN robustness and transfer

### 9.1 Prepare workspace and training data

```bash
cd /Users/deviandr/Desktop/nethobench-calciumgan-biophysical
/Users/deviandr/miniforge3/bin/python scripts/prepare_calciumgan_project.py --force \
  --n-sequences 128 \
  --seq-length 1024 \
  --n-regions 16 \
  --latent-dim 6 \
  --burn-in 192
```

```bash
/Users/deviandr/miniforge3/bin/python -m pip install "tensorflow>=2.15,<2.16" tensorboard scipy
```

```bash
cd /Users/deviandr/Desktop/nethobench-calciumgan-biophysical
/Users/deviandr/miniforge3/bin/python scripts/build_tfrecords.py
```

### 9.2 Train CalciumGAN

```bash
cd /Users/deviandr/Desktop/nethobench-calciumgan-biophysical
nohup /Users/deviandr/miniforge3/bin/python scripts/train_calciumgan.py \
  > /Users/deviandr/Desktop/nethobench-calciumgan-biophysical/calciumgan_train.log 2>&1 &
```

```bash
tail -f /Users/deviandr/Desktop/nethobench-calciumgan-biophysical/calciumgan_train.log
```

### 9.3 Export GAN samples and DG baseline

```bash
cd /Users/deviandr/Desktop/nethobench-calciumgan-biophysical
/Users/deviandr/miniforge3/bin/python scripts/export_calciumgan_generated_dataset.py \
  --num-samples 128 \
  --seed 1234 \
  --output-name generated_reference
```

```bash
cd /Users/deviandr/Desktop/nethobench-calciumgan-biophysical
/Users/deviandr/miniforge3/bin/python scripts/export_calciumgan_generated_dataset.py \
  --num-samples 128 \
  --seed 4321 \
  --output-name generated_oracle
```

```bash
cd /Users/deviandr/Desktop/nethobench-calciumgan-biophysical
/Users/deviandr/miniforge3/bin/python scripts/generate_dg_baseline.py --num-samples 128
```

### 9.4 Evaluate unconditional generators

```bash
cd /Users/deviandr/Desktop/nethobench-calciumgan-biophysical
/Users/deviandr/miniforge3/bin/python scripts/evaluate_calciumgan_generator.py --run-name calciumgan
```

```bash
cd /Users/deviandr/Desktop/nethobench-calciumgan-biophysical
/Users/deviandr/miniforge3/bin/python scripts/evaluate_calciumgan_generator.py --run-name dg_baseline
```

### 9.5 Transfer evaluation on CalciumGAN source

```bash
cd /Users/deviandr/Desktop/nethobench-calciumgan-biophysical
/Users/deviandr/miniforge3/bin/python scripts/prepare_transfer_project.py --source calciumgan --force
```

```bash
cd /Users/deviandr/Desktop/nethobench-calciumgan-biophysical
PYTHONPATH=/Users/deviandr/sequifier/src:/Users/deviandr/nethobench /Users/deviandr/miniforge3/bin/python -m sequifier.sequifier preprocess \
  --config-path configs/transfer/calciumgan/preprocess_converged.yaml
```

```bash
cd /Users/deviandr/Desktop/nethobench-calciumgan-biophysical
PYTHONPATH=/Users/deviandr/sequifier/src:/Users/deviandr/nethobench /Users/deviandr/miniforge3/bin/python -m sequifier.sequifier preprocess \
  --config-path configs/transfer/calciumgan/preprocess_weakest.yaml
```

```bash
cd /Users/deviandr/Desktop/nethobench-calciumgan-biophysical
/Users/deviandr/miniforge3/bin/python scripts/run_transfer_inference_and_evaluate.py \
  --source calciumgan \
  --run-name converged \
  --eval-mode rollout \
  --rollout-steps 1024
```

```bash
cd /Users/deviandr/Desktop/nethobench-calciumgan-biophysical
/Users/deviandr/miniforge3/bin/python scripts/run_transfer_inference_and_evaluate.py \
  --source calciumgan \
  --run-name weakest \
  --eval-mode rollout \
  --rollout-steps 1024
```

### 9.6 Transfer evaluation on DG source

```bash
cd /Users/deviandr/Desktop/nethobench-calciumgan-biophysical
/Users/deviandr/miniforge3/bin/python scripts/prepare_transfer_project.py --source dg_baseline --force
```

```bash
cd /Users/deviandr/Desktop/nethobench-calciumgan-biophysical
PYTHONPATH=/Users/deviandr/sequifier/src:/Users/deviandr/nethobench /Users/deviandr/miniforge3/bin/python -m sequifier.sequifier preprocess \
  --config-path configs/transfer/dg_baseline/preprocess_converged.yaml
```

```bash
cd /Users/deviandr/Desktop/nethobench-calciumgan-biophysical
PYTHONPATH=/Users/deviandr/sequifier/src:/Users/deviandr/nethobench /Users/deviandr/miniforge3/bin/python -m sequifier.sequifier preprocess \
  --config-path configs/transfer/dg_baseline/preprocess_weakest.yaml
```

```bash
cd /Users/deviandr/Desktop/nethobench-calciumgan-biophysical
/Users/deviandr/miniforge3/bin/python scripts/run_transfer_inference_and_evaluate.py \
  --source dg_baseline \
  --run-name converged \
  --eval-mode rollout \
  --rollout-steps 1024
```

```bash
cd /Users/deviandr/Desktop/nethobench-calciumgan-biophysical
/Users/deviandr/miniforge3/bin/python scripts/run_transfer_inference_and_evaluate.py \
  --source dg_baseline \
  --run-name weakest \
  --eval-mode rollout \
  --rollout-steps 1024
```
