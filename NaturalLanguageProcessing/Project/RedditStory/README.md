# Reddit Story Generator

A local TensorFlow pipeline for collecting Reddit posts, building a story corpus,
training a BPE tokenizer, and training a decoder-only Transformer to continue
prompts. The repository contains implementation only: data, weights, logs,
metrics, and samples are generated locally and ignored by Git.

## Features

- PRAW-based Reddit collection with duplicate-ID tracking and CSV export.
- Cleaning, quality scoring, near-duplicate removal, and deterministic splits.
- A repository-native BPE tokenizer and packed TFRecord data pipeline.
- Float32 causal Transformer with learned token/position embeddings, pre-layer
  normalization, multi-head causal attention, GELU feed-forward blocks, and
  tied input/output embeddings.
- AdamW, warmup + cosine learning rate, gradient clipping, checkpoints,
  finite-value checks, evaluation, and sampled generation.

The default configuration is a 12-layer, 512-hidden-size, 8-head model with a
2,048-unit feed-forward layer and a 1,024-token context window.

## Project layout

| Path | Purpose |
| --- | --- |
| `scraper.py` | Collect posts into the raw JSONL dataset. |
| `prepare_dataset.py` | Clean, score, deduplicate, and split records. |
| `train_tokenizer.py` | Train the BPE tokenizer. |
| `build_records.py` | Convert JSONL splits into TFRecord shards. |
| `train.py` | Train or resume the language model. |
| `evaluate.py` | Evaluate a checkpoint on a dataset split. |
| `generate.py` | Generate from a prompt or prompt file. |
| `storygen/core.py` | Tokenizer, data, model, training, and decoding code. |
| `config.py` | Default paths and training configuration. |
| `config_smoke.py` | Two-step training smoke-test configuration. |
| `train.sh` | Reusable end-to-end launcher. |

## Install

Use Python 3.10+ and a TensorFlow build appropriate for your CPU or CUDA setup:

```bash
python3 -m pip install tensorflow numpy tqdm pandas praw python-dotenv matplotlib pydot
```

To collect data, put Reddit API credentials in your environment or `~/.env`:

```bash
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=reddit-story-generator/1.0
```

## Data and outputs

The default raw dataset path is:

```text
~/Documents/Dataset/reddit_story/output/ghost_stories.jsonl
```

Change `DATA_ROOT`, `RAW_DATA_FILE`, or any `*_DIR` setting in `config.py` if
needed. Generated content is kept locally in:

```text
artifacts/data_v1/             cleaned JSONL splits
artifacts/tokenizer_v1/        tokenizer files
artifacts/records_v1/          TFRecord shards
artifacts/model_v2_float32/    checkpoints and model weights
eval/run_001/                  evaluation metrics
generations/run_001/           generated JSONL
```

These locations are ignored by Git.

## Quick start

Run the complete pipeline from this directory:

```bash
chmod +x train.sh
./train.sh
```

The launcher reuses prepared data, tokenizers, and TFRecords; training resumes
from the latest checkpoint when available.

```bash
./train.sh --prepare-only
./train.sh --train-only
./train.sh --evaluate-only
./train.sh --force-data
./train.sh --force-tokenizer
./train.sh --force-records
./train.sh --reset-model
```

For a fresh corpus, collect data first:

```bash
python3 scraper.py --config config.py
```

## Step-by-step pipeline

```bash
# Clean, filter, deduplicate, and split raw posts.
python3 prepare_dataset.py --config config.py

# Train a 16k-token BPE vocabulary.
python3 train_tokenizer.py --config config.py --vocab-size 16000

# Build fixed-length language-model examples.
python3 build_records.py --config config.py --seq-len 1024

# Train or resume.
python3 train.py --config config.py

# Evaluate the best weights (or last weights if no best weights exist).
python3 evaluate.py --config config.py --split test

# Generate a continuation.
python3 generate.py --config config.py \
  --prompt "When I opened the attic door, the crying stopped."
```

Generate from one prompt per line in `prompts.txt`:

```bash
python3 generate.py --config config.py --prompts prompts.txt \
  --batch-size 8 --max-new-tokens 256 --out-dir generations/run_batch
```

Run an interactive session (use `--typing-wpm 0` for instant output):

```bash
python3 generate.py --config config.py --interactive --typing-wpm 40
```

## Configuration and runtime controls

Edit `config.py` for persistent changes. Important settings include:

- `DATA_SEQ_LEN`, `DATA_SHUFFLE_BUFFER`
- `MODEL_HIDDEN_SIZE`, `MODEL_NUM_LAYERS`, `MODEL_NUM_HEADS`, and
  `MODEL_FFN_HIDDEN_SIZE`
- `TRAIN_BATCH_SIZE`, `TRAIN_LEARNING_RATE`, `TRAIN_WARMUP_STEPS`, and
  `TRAIN_STEPS`
- `TRAIN_EVAL_EVERY`, `TRAIN_SAVE_EVERY`, and `TRAIN_GENERATE_EVERY`

This training path is deliberately float32-only and stops when it detects a
non-finite loss or weight. Training, evaluation, and generation also accept
runtime options such as:

```bash
python3 train.py --config config.py \
  --gpu-memory-growth --enable-xla \
  --allocator cuda_malloc_async \
  --cpu-threads 12 --inter-op-threads 4
```

When multiple GPUs are visible, TensorFlow uses `MirroredStrategy`. CPU-only
execution is supported when no GPU is visible.

## Smoke test

After preparing data, verify the environment with the two-step smoke config:

```bash
CONFIG=config_smoke.py ./train.sh --train-only --reset-model
```

Check TensorFlow device visibility:

```bash
python3 -c 'import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices("GPU"))'
```

## Responsible use

Use Reddit content in accordance with Reddit policies and applicable law. Do
not publish private or sensitive content from source posts. Generated text is
model output, not verified fact.
