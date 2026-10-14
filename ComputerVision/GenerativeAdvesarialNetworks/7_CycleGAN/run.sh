#!/bin/bash

PYTHON_BIN="${PYTHON_BIN:-/home/shuvrajeet/miniconda3/envs/research/bin/python}"

choices=("mnist_fashion" "monet2photo" "cezanne2photo" "ukiyoe2photo" "vangogh2photo")

for choice in "${choices[@]}"; do
    echo "Running dataset: $choice"
    "$PYTHON_BIN" train_and_test.py --type "$choice" --gpu -1 --continue
done
