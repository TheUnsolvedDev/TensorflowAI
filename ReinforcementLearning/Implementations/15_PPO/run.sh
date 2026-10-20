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
        echo "Training CartPole-v1"
        "$python_bin" train_and_test.py --env CartPole-v1 --checkpoint "checkpoints/CartPole-v1.weights.h5" "$@"
        echo "Training Acrobot-v1"
        "$python_bin" train_and_test.py --env Acrobot-v1 --checkpoint "checkpoints/Acrobot-v1.weights.h5" "$@"
        echo "Training MountainCar-v0"
        "$python_bin" train_and_test.py --env MountainCar-v0 --checkpoint "checkpoints/MountainCar-v0.weights.h5" "$@"
        echo "Training LunarLander-v3"
        "$python_bin" train_and_test.py --env LunarLander-v3 --checkpoint "checkpoints/LunarLander-v3.weights.h5" "$@"
        for env_id in Pendulum-v1 LunarLanderContinuous-v3 BipedalWalker-v3 HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5; do
            echo "Training $env_id"
            "$python_bin" train_and_test.py --env "$env_id" --checkpoint "checkpoints/$env_id.weights.h5" "$@"
        done
        ;;
    inference)
        exec "$python_bin" inference.py "$@"
        ;;
    *)
        echo "Usage: sh run.sh [train|inference] [arguments]" >&2
        exit 1
        ;;
esac

