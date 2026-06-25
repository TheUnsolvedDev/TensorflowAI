#!/bin/bash

datasets=(
    "manythings_english_french"
    "cornell_movie_dialogs"
    "wikilarge"
    "cnn_dailymail"
    "english_french"
    "english_german"
)

for dataset in "${datasets[@]}"; do

    echo "Running Dataset: $dataset"

    python3 train_and_test.py \
        --dataset "$dataset" \
        --gpu -1

done