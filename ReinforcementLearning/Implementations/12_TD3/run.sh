#!/bin/sh
set -eu
folder=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python_bin=${PYTHON:-/home/shuvrajeet/miniconda3/envs/research/bin/python}
mode=${1:-train}
if [ "$#" -gt 0 ]; then shift; fi
if [ ! -x "$python_bin" ]; then
    echo "Python interpreter not found: $python_bin" >&2
    exit 1
fi
cd "$folder"
case "$mode" in
    train)
        mkdir -p checkpoints
        echo "Training Pendulum-v1"
        "$python_bin" train_and_test.py --env Pendulum-v1 --checkpoint "checkpoints/Pendulum-v1.weights.h5" "$@"
        echo "Training MountainCarContinuous-v0"
        "$python_bin" train_and_test.py --env MountainCarContinuous-v0 --checkpoint "checkpoints/MountainCarContinuous-v0.weights.h5" "$@"
        echo "Training LunarLanderContinuous-v3"
        "$python_bin" train_and_test.py --env LunarLanderContinuous-v3 --checkpoint "checkpoints/LunarLanderContinuous-v3.weights.h5" "$@"
        echo "Training BipedalWalker-v3"
        "$python_bin" train_and_test.py --env BipedalWalker-v3 --checkpoint "checkpoints/BipedalWalker-v3.weights.h5" "$@"
        echo "Training HalfCheetah-v5"
        "$python_bin" train_and_test.py --env HalfCheetah-v5 --checkpoint "checkpoints/HalfCheetah-v5.weights.h5" "$@"
        echo "Training Hopper-v5"
        "$python_bin" train_and_test.py --env Hopper-v5 --checkpoint "checkpoints/Hopper-v5.weights.h5" "$@"
        echo "Training Walker2d-v5"
        "$python_bin" train_and_test.py --env Walker2d-v5 --checkpoint "checkpoints/Walker2d-v5.weights.h5" "$@"
        echo "Training Ant-v5"
        "$python_bin" train_and_test.py --env Ant-v5 --checkpoint "checkpoints/Ant-v5.weights.h5" "$@"
        ;;
    inference)
        exec "$python_bin" inference.py "$@"
        ;;
    *)
        echo "Usage: sh run.sh [train|inference] [arguments]" >&2
        exit 1
        ;;
esac

