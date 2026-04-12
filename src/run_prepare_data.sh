#!/usr/bin/env bash
# Download OpenML CSVs and preprocess (categorical_method follows dataset defaults in config.py).
# Env: DATASET (default Adult), DATASETS (comma list for download; default = DATASET), DEVICE unused here.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
: "${PYTHON:=python3}"

: "${DATASET:=Adult}"
: "${DATASETS:=${DATASET}}"

echo "[run_prepare_data] DATASET=${DATASET} DATASETS=${DATASETS}"
"${PYTHON}" dataset/download_dataset.py --datasets "${DATASETS}"
"${PYTHON}" dataset/pre_processing.py --dataset "${DATASET}"
echo "[run_prepare_data] Done. Preprocessed under dataset/preprocessed_datasets/<method>/${DATASET}/"
