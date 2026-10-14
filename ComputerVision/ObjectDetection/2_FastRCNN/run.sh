#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/shuvrajeet/miniconda3/envs/research/bin/python}"

for dataset in coco voc; do
    echo "Training Fast R-CNN on $dataset"
    "$PYTHON_BIN" "$SCRIPT_DIR/train.py" --detector "$dataset" "$@"
done
