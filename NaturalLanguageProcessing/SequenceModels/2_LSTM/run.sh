#!/usr/bin/env bash
set -euo pipefail
datasets=("ag_news" "dbpedia" "imdb")
for dataset in "${datasets[@]}"; do
  python3 train_and_test.py --dataset "$dataset" --gpu -1 "$@"
done
