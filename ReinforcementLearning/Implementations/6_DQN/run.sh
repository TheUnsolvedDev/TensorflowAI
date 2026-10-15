#!/bin/sh

set -eu

folder=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python_bin=${PYTHON:-/home/shuvrajeet/miniconda3/envs/research/bin/python}
mode=${1:-train}

if [ "$#" -gt 0 ]; then
    shift
fi

case "$mode" in
    train) entrypoint=train_and_test.py ;;
    inference) entrypoint=inference.py ;;
    *)
        echo "Usage: sh run.sh [train|inference] [arguments]" >&2
        exit 1
        ;;
esac

if [ ! -x "$python_bin" ]; then
    echo "Python interpreter not found: $python_bin" >&2
    echo "Set PYTHON to another interpreter." >&2
    exit 1
fi

cd "$folder"
exec "$python_bin" "$entrypoint" "$@"

