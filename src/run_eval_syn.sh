#!/usr/bin/env bash
# Evaluate on CCTC synthetic CSVs (run run_cctc.sh first).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
: "${PYTHON:=python3}"

: "${DATASET:=Adult}"
: "${REDUCTION_RATE:=0.001}"
: "${GAMMA:=0.25}"
: "${DEVICE:=0}"
: "${NUM_EXP:=5}"
: "${EPOCH_EVAL_TRAIN:=500}"
: "${LR_NET:=0.001}"
: "${BATCH_TRAIN:=512}"

echo "[run_eval_syn] DATASET=${DATASET} REDUCTION_RATE=${REDUCTION_RATE} GAMMA=${GAMMA}"
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
  --device "${DEVICE}"
