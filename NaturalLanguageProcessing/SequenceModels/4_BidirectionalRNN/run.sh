#!/bin/bash

datasets=("ag_news" "dbpedia" "imdb")

for dataset in "${datasets[@]}"; do
    echo "Running Dataset: $dataset"
    python3 train_and_test.py --dataset "$dataset" --gpu -1
done