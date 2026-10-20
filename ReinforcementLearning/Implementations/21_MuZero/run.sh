#!/bin/sh
set -eu
folder=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python_bin=${PYTHON:-/home/shuvrajeet/miniconda3/envs/research/bin/python}
mode=${1:-train}
if [ "$#" -gt 0 ]; then
    shift
fi
cd "$folder"
case "$mode" in
    train)
        "$python_bin" train_and_test.py --env CartPole-v1 "$@"
        "$python_bin" train_and_test.py --env Acrobot-v1 "$@"
        ;;
    inference)
        exec "$python_bin" inference.py "$@"
        ;;
    *)
        echo "Usage: sh run.sh [train|inference] [arguments]" >&2
        exit 1
        ;;
esac


