#!/usr/bin/env bash
# Train MLP on full real train split; evaluate on test (same preprocessed tree as CCTC).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
: "${PYTHON:=python3}"

: "${DATASET:=Adult}"
: "${DEVICE:=0}"
: "${NUM_EXP:=5}"
: "${EPOCH_EVAL_TRAIN:=500}"
: "${LR_NET:=0.001}"
: "${BATCH_TRAIN:=512}"

echo "[run_eval_whole] DATASET=${DATASET}"
"${PYTHON}" evaluation/eval_whole.py \
  --dataset "${DATASET}" \
  --eval_model MLP \
  --epoch_eval_train "${EPOCH_EVAL_TRAIN}" \
  --lr_net "${LR_NET}" \
  --num_exp "${NUM_EXP}" \
  --batch_train "${BATCH_TRAIN}" \
  --device "${DEVICE}"
