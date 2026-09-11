# Reddit Story Generator

Local TensorFlow pipeline for:

- cleaning the raw Reddit dataset
- training a tokenizer from scratch
- building TFRecord shards
- training a decoder-only language model
- evaluating checkpoints
- generating stories from prompts

## Files

- `~/Documents/Dataset/reddit_story/output/ghost_stories.jsonl`: default raw dataset location
- `prepare_dataset.py`: clean, filter, deduplicate, split
- `train_tokenizer.py`: train tokenizer
- `build_records.py`: build TFRecords
- `train.py`: train model
- `evaluate.py`: evaluate model
- `generate.py`: generate text
- `config.py`: training/model/data settings using direct uppercase variables
- `storygen/core.py`: main pipeline logic
- `storygen/utils.py`: runtime and config helpers

The pipeline is resource-adaptive: preprocessing and tokenizer training stream
JSONL data, TFRecords are read with parallel `tf.data` workers, and multiple
visible GPUs use TensorFlow's portable mirrored strategy. Model training is
float32-only for stable long runs; CPU-only execution remains supported.

## Install

```bash
python3 -m pip install tensorflow numpy praw python-dotenv pandas tqdm matplotlib pydot
```

## Config

`config.py` keeps both training settings and project paths. The important path variables are:

```python
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path.home() / "Documents" / "Dataset" / "reddit_story"
RAW_DATA_FILE = DATA_ROOT / "output" / "ghost_stories.jsonl"
DATASET_DIR = PROJECT_ROOT / "artifacts" / "data_v1"
TOKENIZER_DIR = PROJECT_ROOT / "artifacts" / "tokenizer_v1"
```

`PROJECT_ROOT` follows the code checkout. `DATA_ROOT` is only used for the raw source dataset, while the cleaned data, tokenizer, TFRecords, checkpoints, eval outputs, and generations are now stored inside this project folder.

If you want the dataset or artifacts somewhere else, change those path variables in `config.py`.

Relative CLI paths now behave like this:

- `output/...` resolves to `~/Documents/Dataset/reddit_story/output/...`
- `artifacts/...`, `eval/...`, and `generations/...` resolve inside this repo

The training variables still use direct names, for example:

```python
DATA_SEQ_LEN = 1024
MODEL_HIDDEN_SIZE = 512
TRAIN_BATCH_SIZE = 4
TRAIN_STEPS = 5000
```

Edit those variables to change training behavior.

Important ones:

- `DATA_SEQ_LEN`
- `DATA_SHUFFLE_BUFFER`
- `MODEL_HIDDEN_SIZE`
- `MODEL_NUM_LAYERS`
- `MODEL_NUM_HEADS`
- `MODEL_FFN_HIDDEN_SIZE`
- `TRAIN_BATCH_SIZE`
- `TRAIN_EVAL_BATCH_SIZE`
- `TRAIN_LEARNING_RATE`
- `TRAIN_STEPS`
- `TRAIN_CACHE_DATASET` (disabled by default; enable only when the dataset fits RAM)
- `TRAIN_PREFETCH_TO_DEVICE` (disabled by default; enable for GPU input pipelines)
- `TRAIN_GENERATE_EVERY` (default `1000`; set to `0` to disable interval generation)
- `TRAIN_GENERATION_PROMPTS_FILE` (default `prompts.txt`)
- `TRAIN_GENERATION_MAX_NEW_TOKENS`
- `TRAIN_GENERATION_BATCH_SIZE`
- `TRAIN_GENERATION_TEMPERATURE` (default `0.8`)
- `TRAIN_GENERATION_TOP_K` (default `50`)
- `TRAIN_GENERATION_TOP_P` (default `0.9`)
- `TRAIN_GENERATION_REPETITION_PENALTY` (default `1.15`)
- `TRAIN_GENERATION_NO_REPEAT_NGRAM_SIZE` (default `3`)
- `TRAIN_OPTIMIZER_EPSILON`
- `TRAIN_FAIL_ON_NONFINITE`

The default corrected run writes to `artifacts/model_v2_float32/`, so the
older `artifacts/model_v1/` run is preserved. Mixed precision is intentionally
rejected. If a loss or weight becomes NaN/Inf, training stops immediately with
the failing batch reported instead of continuing with corrupted weights.

## Full Pipeline

The complete pipeline can also be run with:

```bash
chmod +x train.sh
./train.sh
```

`train.sh` reuses existing cleaned data, tokenizer files, and TFRecord shards.
It also resumes training from the latest checkpoint and skips training once
`TRAIN_STEPS` has already been reached. Rebuild or reset explicitly when
needed:

```bash
./train.sh --force-data
./train.sh --force-tokenizer
./train.sh --force-records
./train.sh --reset-model
./train.sh --prepare-only
./train.sh --train-only
./train.sh --evaluate-only
```

Interval generation uses the shared sampled decoder. Its defaults can be
overridden through `train.sh`:

```bash
./train.sh --train-only \
  --generation-temperature 0.8 \
  --generation-top-k 50 \
  --generation-top-p 0.9 \
  --generation-repetition-penalty 1.15 \
  --generation-no-repeat-ngram-size 3
```

Additional arguments are passed to `train.py`, for example:

```bash
./train.sh --enable-xla --allocator cuda_malloc_async
```

Run these in order:

```bash
python3 prepare_dataset.py
python3 train_tokenizer.py --vocab-size 16000
python3 build_records.py --seq-len 1024
python3 train.py --config config.py
python3 evaluate.py --config config.py --split test
python3 generate.py --config config.py --prompt "I heard footsteps under the floorboards"
```

For an interactive session that loads the model once and accepts repeated
prompts:

```bash
python3 generate.py \
  --config config.py \
  --model-dir artifacts/model_v2_float32 \
  --interactive \
  --max-new-tokens 256 \
  --typing-wpm 40
```

Interactive stories are displayed at a human-like 40 words per minute rather
than appearing all at once. Change `--typing-wpm` to suit the reader, or use
`--typing-wpm 0` for instant output.

Generation uses temperature/top-k/top-p sampling, repetition penalty, and
3-gram blocking by default. For deterministic decoding, use
`--temperature 0`; for longer samples, adjust `--max-new-tokens`.
When multiple GPUs are visible, generation automatically uses
`MirroredStrategy` and splits each equal-length prompt batch across replicas;
use a batch size divisible by the GPU count for best throughput.

For a throughput-focused GPU run, use the existing runtime controls:

```bash
python3 train.py --config config.py \
  --enable-xla --allocator cuda_malloc_async \
  --cpu-threads 12 --inter-op-threads 4
```

Use `TRAIN_CACHE_DATASET = True` only when the parsed dataset comfortably fits
system memory. `TRAIN_PREFETCH_TO_DEVICE = True` is useful for GPU-bound runs;
it is ignored safely when no GPU is visible.

During training, sample generations are produced every `TRAIN_GENERATE_EVERY`
global steps and saved under the configured model directory. Generation is
batched through the existing model and only runs at the configured interval;
set `TRAIN_GENERATE_EVERY = 0` to remove this callback overhead.

These use the default locations from `config.py`.

If you prefer the old explicit style, this still works:

```bash
python3 prepare_dataset.py --input output/ghost_stories.jsonl --out-dir artifacts/data_v1
python3 train_tokenizer.py --input artifacts/data_v1/clean_train.jsonl --out-dir artifacts/tokenizer_v1 --vocab-size 16000
python3 build_records.py --clean-dir artifacts/data_v1 --tokenizer-dir artifacts/tokenizer_v1 --out-dir artifacts/records_v1 --seq-len 1024
python3 train.py --records-dir artifacts/records_v1 --tokenizer-dir artifacts/tokenizer_v1 --model-dir artifacts/model_v2_float32 --config config.py
python3 evaluate.py --records-dir artifacts/records_v1 --tokenizer-dir artifacts/tokenizer_v1 --model-dir artifacts/model_v2_float32 --config config.py --split test --out-dir eval/run_001
python3 generate.py --tokenizer-dir artifacts/tokenizer_v1 --model-dir artifacts/model_v2_float32 --config config.py --prompt "I heard footsteps under the floorboards" --out-dir generations/run_001
```

## Step By Step

### 1. Clean the dataset

```bash
python3 prepare_dataset.py \
  --min-chars 800 \
  --min-quality-score 25
```

Output:

- `./artifacts/data_v1/clean_train.jsonl`
- `./artifacts/data_v1/clean_val.jsonl`
- `./artifacts/data_v1/clean_test.jsonl`
- `./artifacts/data_v1/stats.json`

### 2. Train tokenizer

```bash
python3 train_tokenizer.py \
  --vocab-size 16000 \
  --min-frequency 2
```

### 3. Build TFRecords

```bash
python3 build_records.py \
  --seq-len 1024 \
  --shard-size 2048
```

### 4. Train

```bash
python3 train.py \
  --config config.py
```

Training outputs:

- `./artifacts/model_v2_float32/checkpoints/`
- `./artifacts/model_v2_float32/best/model.weights.h5`
- `./artifacts/model_v2_float32/best/metrics.json`
- `./artifacts/model_v2_float32/final_metrics.json`
- `./artifacts/model_v2_float32/training_log.jsonl`
- `./artifacts/model_v2_float32/model_summary.txt`
- `./artifacts/model_v2_float32/model_graph.png`
- `./artifacts/model_v2_float32/training_curves.png`

### 5. Evaluate

```bash
python3 evaluate.py \
  --config config.py \
  --split test
```

### 6. Generate

Single prompt:

```bash
python3 generate.py \
  --config config.py \
  --prompt "When I opened the attic door, the crying stopped."
```

Prompt file:

```bash
python3 generate.py \
  --config config.py \
  --prompts prompts.txt \
  --batch-size 8 \
  --max-new-tokens 256 \
  --out-dir generations/run_batch
```

## Runtime Memory Options

Available on `train.py`, `evaluate.py`, and `generate.py`:

- `--cpu-threads`
- `--inter-op-threads`
- `--gpu-memory-growth`
- `--no-gpu-memory-growth`
- `--gpu-memory-limit-mb`
- `--enable-xla`
- `--allocator cuda_malloc_async`

Example:

```bash
python3 train.py \
  --records-dir artifacts/records_v1 \
  --tokenizer-dir artifacts/tokenizer_v1 \
  --model-dir artifacts/model_v2_float32 \
  --config config.py \
  --gpu-memory-growth \
  --gpu-memory-limit-mb 18000 \
  --cpu-threads 12 \
  --inter-op-threads 4 \
  --allocator cuda_malloc_async
```

If TensorFlow sees no GPU, it will fall back to CPU.

## Quick Checks

Show CLI help:

```bash
python3 prepare_dataset.py --help
python3 train_tokenizer.py --help
python3 build_records.py --help
python3 train.py --help
python3 evaluate.py --help
python3 generate.py --help
```

Check GPU visibility:

```bash
python3 - <<'PY'
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
PY
```

If that prints `[]`, your current TensorFlow session is CPU-only.
