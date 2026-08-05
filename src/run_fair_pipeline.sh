#!/usr/bin/env bash
# Fair-CCTC pipeline (default: credit).
# Fair-CCTC includes orthogonal projection (OP) by default (USE_OP=1).
# USE_OP=0 is only for ablation (no OP).
#
# Usage:
#   ./run_fair_pipeline.sh
#   DATASET=credit DEVICE=0 ./run_fair_pipeline.sh
#   DATASET=ACSIncome REDUCTION_RATE=0.001 ./run_fair_pipeline.sh
#
# Default paths (credit, USE_OP=1):
#   download:     dataset/download_datasets/credit/credit.csv
#   preprocess:   dataset/preprocessed_datasets_fair/credit/
#   after OP:     dataset/preprocessed_datasets_fair_op/credit/
#   syn:          Results/cctc_datasets/credit/op/<rr>/<gamma>/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHON="${PYTHON:-python3}"
export DATASET="${DATASET:-credit}"
export DATASETS="${DATASETS:-${DATASET}}"
export REDUCTION_RATE="${REDUCTION_RATE:-0.001}"
export GAMMA="${GAMMA:-0.25}"
export DEVICE="${DEVICE:-0}"
export NUM_EXP="${NUM_EXP:-5}"
export FAIR_RHO="${FAIR_RHO:-1.0}"
# Fair-CCTC includes OP by default.
export USE_OP="${USE_OP:-1}"
export EPOCH_EVAL_TRAIN="${EPOCH_EVAL_TRAIN:-100}"
export LR_NET="${LR_NET:-0.001}"
export BATCH_TRAIN="${BATCH_TRAIN:-512}"

# Propagate fair flags to eval scripts.
export FAIR=1

FAIR_PRE_DIR="${ROOT}/dataset/preprocessed_datasets_fair/${DATASET}"
FAIR_OP_DIR="${ROOT}/dataset/preprocessed_datasets_fair_op/${DATASET}"

echo "=== Fair-CCTC pipeline: DATASET=${DATASET} USE_OP=${USE_OP} REDUCTION_RATE=${REDUCTION_RATE} GAMMA=${GAMMA} DEVICE=${DEVICE} ==="

echo "=== [1/5] Download fair dataset ==="
"${PYTHON}" dataset/download_dataset_fair.py --datasets "${DATASETS}"

echo "=== [2/5] Fair preprocess ==="
"${PYTHON}" dataset/fair_preprocessor.py --dataset "${DATASET}"

if [[ "${USE_OP}" == "1" ]]; then
  echo "=== [3/5] Orthogonal projection (OP) ==="
  "${PYTHON}" dataset/fair_orthogonal_projection.py \
    --input_dir "${FAIR_PRE_DIR}" \
    --output_dir "${FAIR_OP_DIR}" \
    --dataset_name "${DATASET}" \
    --op_method op
else
  echo "=== [3/5] Orthogonal projection skipped (USE_OP=0, ablation) ==="
fi

echo "=== [4/5] Fair-CCTC condensation ==="
FAIR_CCTC_ARGS=(
  --dataset "${DATASET}"
  --fair
  --reduction_rate "${REDUCTION_RATE}"
  --gamma "${GAMMA}"
  --num_exp "${NUM_EXP}"
  --fair_rho "${FAIR_RHO}"
  --device "${DEVICE}"
)
if [[ "${USE_OP}" == "1" ]]; then
  FAIR_CCTC_ARGS+=(--use_op)
fi
"${PYTHON}" ours/fair_CCTC.py "${FAIR_CCTC_ARGS[@]}"

echo "=== [5/5] Eval synthetic (Fair-CCTC) ==="
"${ROOT}/run_eval_syn.sh"
# Whole baseline needs standard preprocessed_datasets/<method>/<dataset>/;
# skip here until that tree exists for the fair dataset.
# "${ROOT}/run_eval_whole.sh"

echo "=== Fair-CCTC pipeline finished ==="
if [[ "${USE_OP}" == "1" ]]; then
  echo "  preprocess: dataset/preprocessed_datasets_fair_op/${DATASET}/"
  echo "  syn:        Results/cctc_datasets/${DATASET}/op/${REDUCTION_RATE}/${GAMMA}/"
else
  echo "  preprocess: dataset/preprocessed_datasets_fair/${DATASET}/"
  echo "  syn:        Results/cctc_datasets/${DATASET}/fair/${REDUCTION_RATE}/${GAMMA}/"
fi
