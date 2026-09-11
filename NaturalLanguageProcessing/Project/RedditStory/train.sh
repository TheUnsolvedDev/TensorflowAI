#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-config.py}"
MODE="all"
FORCE_DATA=0
FORCE_TOKENIZER=0
FORCE_RECORDS=0
RESET_MODEL=0
TRAIN_ARGS=()

# The corrected configuration points at a new model directory.  This explicit
# flag remains available for callers using an older/custom model directory.

usage() {
  cat <<'EOF'
Usage: ./train.sh [options] [train.py options]

Pipeline options:
  --force-data       Rebuild cleaned dataset
  --force-tokenizer  Retrain tokenizer
  --force-records    Rebuild TFRecord shards
  --reset-model      Ignore existing checkpoints
  --prepare-only     Run dataset, tokenizer, and TFRecord stages only
  --train-only       Run model training only
  --evaluate-only    Evaluate the existing model only
  --generation-temperature VALUE
                     Sampling temperature for interval generation
  --generation-top-k VALUE
                     Top-k sampling limit for interval generation
  --generation-top-p VALUE
                     Top-p sampling limit for interval generation
  --generation-repetition-penalty VALUE
                     Repetition penalty for interval generation
  --generation-no-repeat-ngram-size VALUE
                     Block repeated n-grams during interval generation
  -h, --help         Show this help
EOF
}

while (($#)); do
  case "$1" in
    --force-data) FORCE_DATA=1; shift ;;
    --force-tokenizer) FORCE_TOKENIZER=1; shift ;;
    --force-records) FORCE_RECORDS=1; shift ;;
    --reset-model) RESET_MODEL=1; shift ;;
    --prepare-only) MODE="prepare"; shift ;;
    --train-only) MODE="train"; shift ;;
    --evaluate-only) MODE="evaluate"; shift ;;
    --generation-temperature|--generation-top-k|--generation-top-p|--generation-repetition-penalty|--generation-no-repeat-ngram-size)
      TRAIN_ARGS+=("$1" "${2:?Missing value for $1}"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; TRAIN_ARGS+=("$@"); break ;;
    *) TRAIN_ARGS+=("$1"); shift ;;
  esac
done

# Downstream artifacts depend on the cleaned dataset.
if ((FORCE_DATA)); then
  FORCE_TOKENIZER=1
  FORCE_RECORDS=1
fi

has_clean_dataset() {
  [[ -s artifacts/data_v1/clean_train.jsonl && -s artifacts/data_v1/clean_val.jsonl \
     && -s artifacts/data_v1/clean_test.jsonl && -s artifacts/data_v1/stats.json ]]
}

has_tokenizer() {
  [[ -s artifacts/tokenizer_v1/tokenizer.json && -s artifacts/tokenizer_v1/stats.json ]]
}

has_records() {
  [[ -s artifacts/records_v1/stats.json \
     && -n "$(find artifacts/records_v1/train -name '*.tfrecord' -size +0c -print -quit 2>/dev/null)" \
     && -n "$(find artifacts/records_v1/val -name '*.tfrecord' -size +0c -print -quit 2>/dev/null)" \
     && -n "$(find artifacts/records_v1/test -name '*.tfrecord' -size +0c -print -quit 2>/dev/null)" ]]
}

prepare_pipeline() {
  if ((FORCE_DATA)) || ! has_clean_dataset; then
    echo "[1/3] Preparing dataset"
    "$PYTHON_BIN" prepare_dataset.py --config "$CONFIG"
  else
    echo "[1/3] Reusing cleaned dataset"
  fi

  if ((FORCE_TOKENIZER)) || ! has_tokenizer; then
    echo "[2/3] Training tokenizer"
    "$PYTHON_BIN" train_tokenizer.py --config "$CONFIG"
  else
    echo "[2/3] Reusing tokenizer"
  fi

  if ((FORCE_RECORDS)) || ((FORCE_TOKENIZER)) || ! has_records; then
    echo "[3/3] Building TFRecord shards"
    "$PYTHON_BIN" build_records.py --config "$CONFIG"
  else
    echo "[3/3] Reusing TFRecord shards"
  fi
}

if [[ "$MODE" == "all" || "$MODE" == "prepare" ]]; then
  prepare_pipeline
fi

if [[ "$MODE" == "all" || "$MODE" == "train" ]]; then
  echo "[training] Resuming latest checkpoint when available"
  TRAIN_COMMAND=("$PYTHON_BIN" train.py --config "$CONFIG")
  if ((RESET_MODEL)); then
    TRAIN_COMMAND+=(--reset-model)
  fi
  TRAIN_COMMAND+=("${TRAIN_ARGS[@]}")
  "${TRAIN_COMMAND[@]}"
fi

if [[ "$MODE" == "all" || "$MODE" == "evaluate" ]]; then
  echo "[evaluation] Evaluating test split"
  "$PYTHON_BIN" evaluate.py --config "$CONFIG" --split test
fi

echo "Pipeline complete. Artifacts: $ROOT_DIR/artifacts"
