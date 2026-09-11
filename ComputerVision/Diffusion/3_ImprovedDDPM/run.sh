#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU="${GPU:--1}"
DATASETS="${DATASETS:-cifar10 cifar100 celeba anime_faces}"
MODE="${MODE:-diffusion}"

IFS=' ' read -r -a dataset_choices <<< "$DATASETS"
for dataset in "${dataset_choices[@]}"; do
    [[ -z "$dataset" ]] && continue
    echo "Running dataset: $dataset mode: $MODE"
    "$PYTHON_BIN" "$SCRIPT_DIR/train_and_test.py" --type "$dataset" --gpu "$GPU" --mode "$MODE" --continue "$@"
done
