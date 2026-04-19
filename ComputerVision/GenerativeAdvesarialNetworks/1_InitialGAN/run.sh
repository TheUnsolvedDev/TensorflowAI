#!/bin/bash

# Define your choices array
choices=("mnist" "cifar10" "cifar100" "fashion_mnist")

for choice in "${choices[@]}"; do
    echo "Running dataset: $choice"
    python3 train_and_test.py --type "$choice" --gpu -1
done
