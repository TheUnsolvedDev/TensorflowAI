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
        "$python_bin" train_and_test.py --env Pendulum-v1 "$@"
        "$python_bin" train_and_test.py --env MountainCarContinuous-v0 "$@"
        "$python_bin" train_and_test.py --env LunarLanderContinuous-v3 "$@"
        "$python_bin" train_and_test.py --env BipedalWalker-v3 "$@"
        "$python_bin" train_and_test.py --env HalfCheetah-v5 "$@"
        "$python_bin" train_and_test.py --env Hopper-v5 "$@"
        "$python_bin" train_and_test.py --env Walker2d-v5 "$@"
        "$python_bin" train_and_test.py --env Ant-v5 "$@"
        ;;
    inference)
        exec "$python_bin" inference.py "$@"
        ;;
    *)
        echo "Usage: sh run.sh [train|inference] [arguments]" >&2
        exit 1
        ;;
esac


