#!/usr/bin/env bash
# Evaluate on CCTC / Fair-CCTC synthetic CSVs.
# Standard: run run_cctc.sh first.
# Fair-CCTC: FAIR=1 (USE_OP defaults to 1; OP is part of Fair-CCTC).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
: "${PYTHON:=python3}"

: "${DATASET:=Adult}"
: "${REDUCTION_RATE:=0.001}"
: "${GAMMA:=0.25}"
: "${DEVICE:=0}"
: "${NUM_EXP:=5}"
: "${EPOCH_EVAL_TRAIN:=100}"
: "${LR_NET:=0.001}"
: "${BATCH_TRAIN:=512}"
: "${FAIR:=0}"
# When FAIR=1, OP is on by default (Fair-CCTC); USE_OP=0 only for ablation.
if [[ "${FAIR}" == "1" ]]; then
  : "${USE_OP:=1}"
else
  : "${USE_OP:=0}"
fi

EXTRA_ARGS=()
if [[ "${FAIR}" == "1" || "${USE_OP}" == "1" ]]; then
  EXTRA_ARGS+=(--fair)
fi
if [[ "${USE_OP}" == "1" ]]; then
  EXTRA_ARGS+=(--use_op)
fi

echo "[run_eval_syn] DATASET=${DATASET} REDUCTION_RATE=${REDUCTION_RATE} GAMMA=${GAMMA} FAIR=${FAIR} USE_OP=${USE_OP}"
"${PYTHON}" evaluation/eval_syn.py \
  --dataset "${DATASET}" \
  --method cctc \
  --reduction_rate "${REDUCTION_RATE}" \
  --gamma "${GAMMA}" \
  --eval_model MLP \
  --epoch_eval_train "${EPOCH_EVAL_TRAIN}" \
  --lr_net "${LR_NET}" \
  --num_exp "${NUM_EXP}" \
  --batch_train "${BATCH_TRAIN}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}"
