#!/bin/bash

PYTHON_BIN="${PYTHON_BIN:-/home/shuvrajeet/miniconda3/envs/research/bin/python}"

choices=("bedroom" "celeba" "anime_faces" "chest_x_ray" "skin_cancer")

for choice in "${choices[@]}"; do
    echo "Running dataset: $choice"
    "$PYTHON_BIN" train_and_test.py --type "$choice" --gpu -1 --continue
done
