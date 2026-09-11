#!/usr/bin/env bash
set -euo pipefail
datasets=("manythings_english_french" "english_french" "english_german" "cornell_movie_dialogs" "cnn_dailymail" "wikilarge")
for dataset in "${datasets[@]}"; do
  python3 train_and_test.py --dataset "$dataset" --gpu -1 "$@"
done
