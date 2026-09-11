#!/usr/bin/env bash
# All eight datasets; overwrite matching results by default. Pass --resume to continue instead.
set -u
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
status=0
for dataset in mnist fashion_mnist cifar10 cifar100 skin_cancer cassava_leaf_disease chest_xray crop_disease; do
    "${PYTHON:-python3}" -u train_and_test.py --type "$dataset" --gpu -1 "$@" || status=1
done
exit "$status"
