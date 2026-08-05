#!/usr/bin/env bash
# Full pipeline: prepare data -> CCTC -> eval syn -> eval whole.

# Override with env vars, e.g. DATASET=Adult REDUCTION_RATE=0.001 ./run_pipeline.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHON="${PYTHON:-python3}"
export DATASET="${DATASET:-Adult}"
export DATASETS="${DATASETS:-${DATASET}}"
export REDUCTION_RATE="${REDUCTION_RATE:-0.001}"
export GAMMA="${GAMMA:-0.25}"
export DEVICE="${DEVICE:-0}"
export NUM_EXP="${NUM_EXP:-5}"
# Avoid leaking fair-eval flags into the standard pipeline.
export FAIR=0
export USE_OP=0

echo "=== Pipeline: DATASET=${DATASET} REDUCTION_RATE=${REDUCTION_RATE} GAMMA=${GAMMA} DEVICE=${DEVICE} ==="

echo "=== [1/4] Prepare data ==="
"${ROOT}/run_prepare_data.sh"

echo "=== [2/4] CCTC ==="
"${ROOT}/run_cctc.sh"

echo "=== [3/4] Eval synthetic ==="
"${ROOT}/run_eval_syn.sh"

echo "=== [4/4] Eval whole ==="
"${ROOT}/run_eval_whole.sh"

echo "=== Pipeline finished ==="
