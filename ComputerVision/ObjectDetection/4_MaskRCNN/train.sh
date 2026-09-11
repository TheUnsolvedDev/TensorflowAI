#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU="${GPU:--1}"
CLASSIFIERS="${CLASSIFIERS:-}"
DETECTORS="${DETECTORS:-coco voc imagenet}"

if [[ -z "$CLASSIFIERS" ]]; then
    IFS=' ' read -r -a detector_choices <<< "$DETECTORS"
    for detector in "${detector_choices[@]}"; do
        [[ -z "$detector" ]] && continue
        echo "Running detector: $detector"
        "$PYTHON_BIN" "$SCRIPT_DIR/train.py" --detector "$detector" --gpu "$GPU" "$@"
    done
else
    IFS=' ' read -r -a classifier_choices <<< "$CLASSIFIERS"
    IFS=' ' read -r -a detector_choices <<< "$DETECTORS"
    for classifier in "${classifier_choices[@]}"; do
        for detector in "${detector_choices[@]}"; do
            [[ -z "$classifier" || -z "$detector" ]] && continue
            echo "Running classifier: $classifier detector: $detector"
            "$PYTHON_BIN" "$SCRIPT_DIR/train.py" --classifier "$classifier" --detector "$detector" --gpu "$GPU" "$@"
        done
    done
fi
