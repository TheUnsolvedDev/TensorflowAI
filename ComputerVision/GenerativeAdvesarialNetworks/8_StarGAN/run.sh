#!/bin/bash

PYTHON_BIN="${PYTHON_BIN:-/home/shuvrajeet/miniconda3/envs/research/bin/python}"

choices=("celeba" "chest_x_ray" "mnist" "cifar10" "cifar100" "fashion_mnist")

for choice in "${choices[@]}"; do
    echo "Running dataset: $choice"
    "$PYTHON_BIN" train_and_test.py --type "$choice" --gpu -1 --continue
done
