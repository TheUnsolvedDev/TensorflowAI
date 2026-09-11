#!/bin/bash

choices=("mnist_fashion" "monet2photo" "cezanne2photo" "ukiyoe2photo" "vangogh2photo")

for choice in "${choices[@]}"; do
    echo "Running dataset: $choice"
    python3 train_and_test.py --type "$choice" --gpu -1
done
