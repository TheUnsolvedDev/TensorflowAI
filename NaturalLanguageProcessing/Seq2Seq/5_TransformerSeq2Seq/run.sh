#!/usr/bin/env bash
set -euo pipefail
datasets=("manythings_english_french" "wikilarge" "cornell_movie_dialogs" "english_german" "english_french" "cnn_dailymail")
for dataset in "${datasets[@]}"; do
  python3 train_and_test.py --dataset "$dataset" --gpu -1 "$@"
done
