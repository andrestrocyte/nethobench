#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/Users/deviandr/miniforge3/bin/python}"
NETHOBENCH_ROOT="${NETHOBENCH_ROOT:-/Users/deviandr/nethobench}"
SEQUIFIER_ROOT="${SEQUIFIER_ROOT:-/Users/deviandr/sequifier}"
DESKTOP_ROOT="${DESKTOP_ROOT:-/Users/deviandr/Desktop}"

SCALING_ROOT="${SCALING_ROOT:-$DESKTOP_ROOT/netho-seq-scaling-rerun}"
SEQ_SYNTH_ROOT="${SEQ_SYNTH_ROOT:-$DESKTOP_ROOT/nethobench-sequifier-convergence}"
SEQ_BIO_ROOT="${SEQ_BIO_ROOT:-$DESKTOP_ROOT/nethobench-sequifier-convergence-biophysical}"
CALCIUMGAN_ROOT="${CALCIUMGAN_ROOT:-$DESKTOP_ROOT/nethobench-calciumgan-biophysical}"

function usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_full_project_rerun.sh all
  bash scripts/run_full_project_rerun.sh install real scaling synthetic bio-oracle seq-synth seq-bio calciumgan

Sections:
  install     Install local nethobench + extra deps
  real        Recompute real CSV benchmark neuro scores
  scaling     Recompute scaling-law scores and visualizations
  synthetic   Recompute the original synthetic oracle validation
  bio-oracle  Recompute the harder biophysical oracle validation
  seq-synth   Recompute Sequifier convergence on the original synthetic world
  seq-bio     Recompute Sequifier convergence on the harder biophysical world
  calciumgan  Recompute CalciumGAN robustness and transfer analyses

Notes:
  - `all` runs every section above in the order listed.
  - This is a long-running foreground pipeline. The full `all` path can take many hours.
  - The script writes into the same external workspaces used earlier in the project.
EOF
}

function run_cmd() {
  echo
  echo ">>> $*"
  "$@"
}

function score_folder() {
  local folder="$1"
  shift
  cd "$NETHOBENCH_ROOT"
  for m in "$@"; do
    run_cmd "$PYTHON_BIN" -m nethobench.cli neuro-scores \
      --gt "$folder/$m/gt.csv" \
      --preds "$folder/$m/pred.csv" \
      --json-out "$folder/$m/neuro_scores_final.json"
    run_cmd "$PYTHON_BIN" -m nethobench.cli fidelity-scores \
      --gt "$folder/$m/gt.csv" \
      --preds "$folder/$m/pred.csv" \
      --json-out "$folder/$m/fidelity_scores_final.json"
  done
}

function section_install() {
  run_cmd "$PYTHON_BIN" -m pip install -e "$NETHOBENCH_ROOT"
  run_cmd "$PYTHON_BIN" -m pip install altair
  run_cmd "$PYTHON_BIN" -m pip install "tensorflow>=2.15,<2.16" tensorboard scipy
}

function section_real() {
  score_folder "$DESKTOP_ROOT/organized_csv" OS 1_step AR TF AR_KV
  score_folder "$DESKTOP_ROOT/csv_90_360" OS 1_step AR TF AR_KV
  score_folder "$DESKTOP_ROOT/csv_90_630" OS 1_step AR TF AR_KV
  score_folder "$DESKTOP_ROOT/csv_90_1260" OS 1_step AR TF AR_KV
  score_folder "$DESKTOP_ROOT/csv_90_2070" TF AR_KV
  score_folder "$DESKTOP_ROOT/csv_90_4050" TF AR_KV
}

function section_scaling() {
  cd "$SCALING_ROOT"
  run_cmd "$PYTHON_BIN" analysis/run_neurobench_batch.py --mode full --overwrite
  run_cmd "$PYTHON_BIN" analysis/build_scaling_visualizations.py
}

function section_synthetic() {
  cd "$NETHOBENCH_ROOT"
  run_cmd "$PYTHON_BIN" -m nethobench.cli synthetic-validation \
    --output-root "$DESKTOP_ROOT/nethobench_synthetic_validation_final" \
    --n-sequences 100 \
    --seq-length 768 \
    --n-regions 16 \
    --latent-dim 6 \
    --burn-in 128 \
    --oracle-replicates 3
}

function section_bio_oracle() {
  cd "$NETHOBENCH_ROOT"
  run_cmd "$PYTHON_BIN" -u -m nethobench.cli synthetic-validation-biophysical \
    --output-root "$DESKTOP_ROOT/nethobench_biophysical_validation_final" \
    --n-sequences 128 \
    --seq-length 1024 \
    --n-regions 16 \
    --latent-dim 6 \
    --burn-in 192 \
    --oracle-replicates 4
}

function section_seq_synth() {
  cd "$SEQ_SYNTH_ROOT"
  run_cmd "$PYTHON_BIN" scripts/prepare_convergence_project.py --force
  run_cmd env PYTHONPATH="$SEQUIFIER_ROOT/src" "$PYTHON_BIN" -m sequifier.sequifier preprocess --config-path configs/preprocess.yaml
  run_cmd env PYTHONPATH="$SEQUIFIER_ROOT/src" "$PYTHON_BIN" -m sequifier.sequifier preprocess --config-path configs/preprocess_underfit.yaml
  run_cmd env PYTHONPATH="$SEQUIFIER_ROOT/src" "$PYTHON_BIN" -m sequifier.sequifier train --config-path configs/train_converged.yaml
  run_cmd env PYTHONPATH="$SEQUIFIER_ROOT/src" "$PYTHON_BIN" -m sequifier.sequifier train --config-path configs/train_underfit.yaml
  run_cmd "$PYTHON_BIN" scripts/run_inference_and_evaluate.py --run-name converged
  run_cmd "$PYTHON_BIN" scripts/run_inference_and_evaluate.py --run-name underfit
  run_cmd "$PYTHON_BIN" scripts/run_inference_and_evaluate.py --run-name converged --eval-mode rollout --rollout-steps 1024
  run_cmd "$PYTHON_BIN" scripts/run_inference_and_evaluate.py --run-name underfit --eval-mode rollout --rollout-steps 1024
}

function section_seq_bio() {
  cd "$SEQ_BIO_ROOT"
  run_cmd "$PYTHON_BIN" scripts/prepare_convergence_project.py --force
  run_cmd env PYTHONPATH="$SEQUIFIER_ROOT/src" "$PYTHON_BIN" -m sequifier.sequifier preprocess --config-path configs/preprocess.yaml
  run_cmd env PYTHONPATH="$SEQUIFIER_ROOT/src" "$PYTHON_BIN" -m sequifier.sequifier preprocess --config-path configs/preprocess_weakest.yaml
  run_cmd env PYTHONPATH="$SEQUIFIER_ROOT/src" "$PYTHON_BIN" -m sequifier.sequifier train --config-path configs/train_converged.yaml
  run_cmd env PYTHONPATH="$SEQUIFIER_ROOT/src" "$PYTHON_BIN" -m sequifier.sequifier train --config-path configs/train_weakest.yaml
  run_cmd "$PYTHON_BIN" scripts/run_inference_and_evaluate.py --run-name converged
  run_cmd "$PYTHON_BIN" scripts/run_inference_and_evaluate.py --run-name weakest
  run_cmd "$PYTHON_BIN" scripts/run_inference_and_evaluate.py --run-name converged --eval-mode rollout --rollout-steps 1024
  run_cmd "$PYTHON_BIN" scripts/run_inference_and_evaluate.py --run-name weakest --eval-mode rollout --rollout-steps 1024
}

function section_calciumgan() {
  cd "$CALCIUMGAN_ROOT"
  run_cmd "$PYTHON_BIN" scripts/prepare_calciumgan_project.py --force \
    --n-sequences 128 \
    --seq-length 1024 \
    --n-regions 16 \
    --latent-dim 6 \
    --burn-in 192
  run_cmd "$PYTHON_BIN" scripts/build_tfrecords.py
  run_cmd "$PYTHON_BIN" scripts/train_calciumgan.py
  run_cmd "$PYTHON_BIN" scripts/export_calciumgan_generated_dataset.py --num-samples 128 --seed 1234 --output-name generated_reference
  run_cmd "$PYTHON_BIN" scripts/export_calciumgan_generated_dataset.py --num-samples 128 --seed 4321 --output-name generated_oracle
  run_cmd "$PYTHON_BIN" scripts/generate_dg_baseline.py --num-samples 128
  run_cmd "$PYTHON_BIN" scripts/evaluate_calciumgan_generator.py --run-name calciumgan
  run_cmd "$PYTHON_BIN" scripts/evaluate_calciumgan_generator.py --run-name dg_baseline
  run_cmd "$PYTHON_BIN" scripts/prepare_transfer_project.py --source calciumgan --force
  run_cmd env PYTHONPATH="$SEQUIFIER_ROOT/src:$NETHOBENCH_ROOT" "$PYTHON_BIN" -m sequifier.sequifier preprocess --config-path configs/transfer/calciumgan/preprocess_converged.yaml
  run_cmd env PYTHONPATH="$SEQUIFIER_ROOT/src:$NETHOBENCH_ROOT" "$PYTHON_BIN" -m sequifier.sequifier preprocess --config-path configs/transfer/calciumgan/preprocess_weakest.yaml
  run_cmd "$PYTHON_BIN" scripts/run_transfer_inference_and_evaluate.py --source calciumgan --run-name converged --eval-mode rollout --rollout-steps 1024
  run_cmd "$PYTHON_BIN" scripts/run_transfer_inference_and_evaluate.py --source calciumgan --run-name weakest --eval-mode rollout --rollout-steps 1024
  run_cmd "$PYTHON_BIN" scripts/prepare_transfer_project.py --source dg_baseline --force
  run_cmd env PYTHONPATH="$SEQUIFIER_ROOT/src:$NETHOBENCH_ROOT" "$PYTHON_BIN" -m sequifier.sequifier preprocess --config-path configs/transfer/dg_baseline/preprocess_converged.yaml
  run_cmd env PYTHONPATH="$SEQUIFIER_ROOT/src:$NETHOBENCH_ROOT" "$PYTHON_BIN" -m sequifier.sequifier preprocess --config-path configs/transfer/dg_baseline/preprocess_weakest.yaml
  run_cmd "$PYTHON_BIN" scripts/run_transfer_inference_and_evaluate.py --source dg_baseline --run-name converged --eval-mode rollout --rollout-steps 1024
  run_cmd "$PYTHON_BIN" scripts/run_transfer_inference_and_evaluate.py --source dg_baseline --run-name weakest --eval-mode rollout --rollout-steps 1024
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

declare -a sections=("$@")
if [[ "${sections[0]}" == "all" ]]; then
  sections=(install real scaling synthetic bio-oracle seq-synth seq-bio calciumgan)
fi

for section in "${sections[@]}"; do
  case "$section" in
    install) section_install ;;
    real) section_real ;;
    scaling) section_scaling ;;
    synthetic) section_synthetic ;;
    bio-oracle) section_bio_oracle ;;
    seq-synth) section_seq_synth ;;
    seq-bio) section_seq_bio ;;
    calciumgan) section_calciumgan ;;
    -h|--help|help) usage; exit 0 ;;
    *) echo "Unknown section: $section" >&2; usage; exit 1 ;;
  esac
done

echo
echo "All requested sections completed."
