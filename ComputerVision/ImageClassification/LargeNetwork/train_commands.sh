#!/usr/bin/env bash
# Enter every standalone model folder, run all eight datasets, then continue.
# Usage from TensorflowAI: bash ComputerVision/ImageClassification/LargeNetwork/train_commands.sh
# Preview without training or writing files: add --dry-run.
# Results stay in each model folder and are overwritten by default; --resume continues them.
# Other arguments (for example --epochs 20) are forwarded to each run.sh.
set -uo pipefail

LARGE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1
export PYTHON=${PYTHON:-python3}
PYTHON=$(command -v -- "$PYTHON") || { printf 'Python interpreter not found.\n' >&2; exit 1; }
if [[ "$PYTHON" != /* ]]; then PYTHON="$PWD/$PYTHON"; fi
export PYTHON

preview=0
extra_args=()
for arg in "$@"; do
    if [[ "$arg" == --dry-run || "$arg" == --list ]]; then
        preview=1
    else
        extra_args+=("$arg")
    fi
done
for arg in "${extra_args[@]}"; do
    case "$arg" in
        --run-id|--run-id=*|--output-root|--output-root=*)
            printf 'Results are saved inside each model folder; %s is no longer used.\n' "$arg" >&2
            exit 1 ;;
        --type|--type=*|--probe|--preflight-data)
            printf '%s is a single-dataset/probe option; this script runs all eight datasets.\n' "$arg" >&2
            exit 1 ;;
    esac
done

# Discover actual launchers, including future nested variants, without visiting
# old launchers copied into logs/checkpoints/dataset caches.
model_dirs=()
while IFS= read -r -d '' launcher; do
    folder=${launcher%/run.sh}
    if [[ -f "$folder/config.py" && -f "$folder/model.py" && -f "$folder/dataset.py" && -f "$folder/train_and_test.py" ]]; then
        model_dirs+=("$folder")
    fi
done < <(find "$LARGE_ROOT" \
    -type d \( -name '.git' -o -name '__pycache__' -o -name 'logs' -o -name 'logs_*' \
    -o -name 'Trash' -o -name 'benchmark_runs' -o -name 'checkpoints' -o -name 'artifacts' \
    -o -name 'wandb' -o -name 'datasets' -o -name 'dataset' \) -prune -o \
    -type f -name 'run.sh' -print0 | sort -z -V)

# Version sorting orders numbered families/depths naturally. Put named ViT
# sizes in their intended small-to-large order within the last family.
ordered_dirs=()
for folder in "${model_dirs[@]}"; do
    case "$folder" in
        "$LARGE_ROOT"/12_VisionTransformer/ViT_Tiny16|"$LARGE_ROOT"/12_VisionTransformer/ViT_Small16|\
        "$LARGE_ROOT"/12_VisionTransformer/ViT_Base16|"$LARGE_ROOT"/12_VisionTransformer/ViT_Large16) ;;
        *) ordered_dirs+=("$folder") ;;
    esac
done
for variant in ViT_Tiny16 ViT_Small16 ViT_Base16 ViT_Large16; do
    for folder in "${model_dirs[@]}"; do
        if [[ "$folder" == "$LARGE_ROOT/12_VisionTransformer/$variant" ]]; then
            ordered_dirs+=("$folder")
        fi
    done
done
model_dirs=("${ordered_dirs[@]}")
if (( ${#model_dirs[@]} == 0 )); then
    printf 'No complete model folders with run.sh found in %s\n' "$LARGE_ROOT" >&2
    exit 1
fi

printf 'Models: %d; datasets per model: 8; results stay inside each model folder.\n' "${#model_dirs[@]}"
if (( preview )); then
    for folder in "${model_dirs[@]}"; do
        printf '(cd %q && bash run.sh' "$folder"
        if (( ${#extra_args[@]} )); then printf ' %q' "${extra_args[@]}"; fi
        printf ')\n'
    done
    exit 0
fi

# Check the same GPU arguments the folder-local trainers will receive. Never
# silently launch the complete benchmark on the CPU if the driver is missing.
"$PYTHON" - "${extra_args[@]}" <<'PY'
import argparse
import tensorflow as tf
p = argparse.ArgumentParser(add_help=False)
p.add_argument('--gpu', type=int, default=-1)
p.add_argument('--require-gpus', type=int, default=0)
p.add_argument('--allow-cpu', action='store_true')
args, _ = p.parse_known_args()
physical = tf.config.list_physical_devices('GPU')
if not args.allow_cpu:
    if not physical or args.gpu < -1 or args.gpu >= len(physical):
        raise SystemExit('No usable selected TensorFlow GPU. Fix the driver before running the training queue.')
    count = len(physical) if args.gpu == -1 else 1
    if count < args.require_gpus:
        raise SystemExit(f'Requires {args.require_gpus} GPUs; only {count} selected.')
    print(f'GPU preflight passed: {count} selected GPU(s).', flush=True)
else:
    print('Explicit CPU mode selected.', flush=True)
PY
if (( $? != 0 )); then exit 1; fi

# Prevent concurrent queues writing the same model-local results.
exec 9>"$LARGE_ROOT/.training_queue.lock"
flock -n 9 || { printf 'This training queue is already running.\n' >&2; exit 1; }
failed=()
trap 'printf "\nTraining queue interrupted. Re-run with --resume to continue; without it, matching results are overwritten.\n" >&2; exit 130' INT TERM

for folder in "${model_dirs[@]}"; do
    relative=${folder#"$LARGE_ROOT"/}
    printf '\n===== %s: running all eight datasets =====\n' "$relative"
    # The subshell restores the parent directory before the next model. Each
    # run.sh invokes train_and_test.py for all datasets and propagates failures.
    (cd -- "$folder" && bash run.sh "${extra_args[@]}")
    code=$?
    if (( code == 0 )); then
        printf 'Finished %s. Its trainer saves dataset metrics and the aggregate accuracy plot.\n' "$relative"
    else
        failed+=("$relative")
        printf 'FAILED: %s (exit %d). Continuing to the next model.\n' "$relative" "$code" >&2
    fi
done

printf '\nQueue finished: %d/%d model launchers succeeded.\n' "$((${#model_dirs[@]} - ${#failed[@]}))" "${#model_dirs[@]}"
printf 'Results: logs_<dataset>_<model>/ and all_datasets_accuracy_*.png inside each model folder.\n'
if (( ${#failed[@]} )); then
    printf 'Failed model: %s\n' "${failed[@]}" >&2
    exit 1
fi
