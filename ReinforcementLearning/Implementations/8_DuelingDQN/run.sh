#!/bin/sh

set -eu

folder=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python_bin=${PYTHON:-/home/shuvrajeet/miniconda3/envs/research/bin/python}
mode=${1:-train}
environments="CartPole-v1 Acrobot-v1 MountainCar-v0 LunarLander-v3"

if [ "$#" -gt 0 ]; then
    shift
fi

if [ ! -x "$python_bin" ]; then
    echo "Python interpreter not found: $python_bin" >&2
    echo "Set PYTHON to another interpreter." >&2
    exit 1
fi

case "$mode" in
    train)
        for argument in "$@"; do
            case "$argument" in
                --env|--env=*|--checkpoint|--checkpoint=*)
                    echo "Multi-environment training manages --env and --checkpoint automatically." >&2
                    echo "Use python train_and_test.py directly for a single environment." >&2
                    exit 1
                    ;;
            esac
        done
        cd "$folder"
        for environment in $environments; do
            echo
            echo "Training Dueling DQN on $environment"
            "$python_bin" train_and_test.py --env "$environment" "$@"
        done
        exit 0
        ;;
    inference) entrypoint=inference.py ;;
    *)
        echo "Usage: sh run.sh [train|inference] [arguments]" >&2
        exit 1
        ;;
esac

cd "$folder"
exec "$python_bin" "$entrypoint" "$@"

