#!/bin/bash

choices=("bedroom" "celeba" "anime_faces" "chest_x_ray" "skin_cancer")

for choice in "${choices[@]}"; do
    echo "Running dataset: $choice"
    python3 train_and_test.py --type "$choice" --gpu -1 --continue
done
