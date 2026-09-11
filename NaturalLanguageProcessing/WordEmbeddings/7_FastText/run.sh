#!/usr/bin/env bash
set -euo pipefail
corpora=("shakespeare.txt" "wikitext-103/wiki.train.tokens")
for corpus in "${corpora[@]}"; do
  python3 train_and_test.py --corpus "$corpus" --gpu -1 "$@"
done
