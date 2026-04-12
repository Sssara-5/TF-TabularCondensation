#!/usr/bin/env bash
# Run CCTC condensation (GPU + faiss-gpu required).
# Env: DATASET, REDUCTION_RATE, GAMMA, NUM_EXP, DEVICE (defaults match config.py).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
: "${PYTHON:=python3}"

: "${DATASET:=Adult}"
: "${REDUCTION_RATE:=0.001}"
: "${GAMMA:=0.25}"
: "${NUM_EXP:=5}"
: "${DEVICE:=0}"

echo "[run_cctc] DATASET=${DATASET} REDUCTION_RATE=${REDUCTION_RATE} GAMMA=${GAMMA}"
"${PYTHON}" CCTC.py \
  --dataset "${DATASET}" \
  --reduction_rate "${REDUCTION_RATE}" \
  --gamma "${GAMMA}" \
  --num_exp "${NUM_EXP}" \
  --device "${DEVICE}"
