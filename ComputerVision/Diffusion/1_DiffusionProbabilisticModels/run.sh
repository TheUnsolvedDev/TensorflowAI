#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU="${GPU:--1}"
DATASETS="${DATASETS:-mnist fashion_mnist cifar10}"
RESUME="${RESUME:-0}"

extra_args=()
if [[ "$RESUME" == "1" ]]; then
    extra_args+=(--continue)
fi

IFS=' ' read -r -a dataset_choices <<< "$DATASETS"
for dataset in "${dataset_choices[@]}"; do
    [[ -z "$dataset" ]] && continue
    echo "Running dataset: $dataset"
    "$PYTHON_BIN" "$SCRIPT_DIR/train_and_test.py" --type "$dataset" --gpu "$GPU" "${extra_args[@]}" "$@"
done
