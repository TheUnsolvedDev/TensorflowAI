"""Default training configuration for the Reddit story generator."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path.home() / "Documents" / "Dataset" / "reddit_story"
RAW_DATA_DIR = DATA_ROOT / "output"
RAW_DATA_FILE = RAW_DATA_DIR / "ghost_stories.jsonl"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DATASET_DIR = ARTIFACTS_ROOT / "data_v1"
TOKENIZER_DIR = ARTIFACTS_ROOT / "tokenizer_v1"
RECORDS_DIR = ARTIFACTS_ROOT / "records_v1"
MODEL_DIR = ARTIFACTS_ROOT / "model_v2_float32"
EVAL_DIR = PROJECT_ROOT / "eval" / "run_001"
GENERATIONS_DIR = PROJECT_ROOT / "generations" / "run_001"

DATA_SEQ_LEN = 1024
DATA_SHUFFLE_BUFFER = 8192

MODEL_MAX_POSITION_EMBEDDINGS = 1024
MODEL_HIDDEN_SIZE = 512
MODEL_NUM_LAYERS = 12
MODEL_NUM_HEADS = 8
MODEL_FFN_HIDDEN_SIZE = 2048
MODEL_DROPOUT_RATE = 0.1
MODEL_LAYER_NORM_EPSILON = 1e-5

TRAIN_SEED = 42
TRAIN_BATCH_SIZE = 4
TRAIN_EVAL_BATCH_SIZE = 4
TRAIN_LEARNING_RATE = 1e-4
TRAIN_WARMUP_STEPS = 1000
TRAIN_STEPS = 50000
TRAIN_EVAL_EVERY = 250
TRAIN_SAVE_EVERY = 500
TRAIN_LOG_EVERY = 25
TRAIN_WEIGHT_DECAY = 0.01
TRAIN_GRAD_CLIP_NORM = 1.0
# Kept as a compatibility key, but float32 is intentionally mandatory for
# stable long-running training.
TRAIN_MIXED_PRECISION = False
TRAIN_OPTIMIZER_EPSILON = 1e-7
TRAIN_FAIL_ON_NONFINITE = True
TRAIN_CACHE_DATASET = False
TRAIN_PREFETCH_TO_DEVICE = False
TRAIN_AUTOTUNE = True
TRAIN_GENERATE_EVERY = 1000
TRAIN_GENERATION_PROMPTS_FILE = PROJECT_ROOT / "prompts.txt"
TRAIN_GENERATION_MAX_NEW_TOKENS = 128
TRAIN_GENERATION_BATCH_SIZE = 4
TRAIN_GENERATION_TEMPERATURE = 0.8
TRAIN_GENERATION_TOP_K = 50
TRAIN_GENERATION_TOP_P = 0.9
TRAIN_GENERATION_REPETITION_PENALTY = 1.15
TRAIN_GENERATION_NO_REPEAT_NGRAM_SIZE = 3


def get_config():
    return {
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "data_root": str(DATA_ROOT),
            "raw_data_dir": str(RAW_DATA_DIR),
            "raw_data_file": str(RAW_DATA_FILE),
            "artifacts_root": str(ARTIFACTS_ROOT),
            "dataset_dir": str(DATASET_DIR),
            "tokenizer_dir": str(TOKENIZER_DIR),
            "records_dir": str(RECORDS_DIR),
            "model_dir": str(MODEL_DIR),
            "eval_dir": str(EVAL_DIR),
            "generations_dir": str(GENERATIONS_DIR),
        },
        "data": {
            "seq_len": DATA_SEQ_LEN,
            "shuffle_buffer": DATA_SHUFFLE_BUFFER,
        },
        "model": {
            "max_position_embeddings": MODEL_MAX_POSITION_EMBEDDINGS,
            "hidden_size": MODEL_HIDDEN_SIZE,
            "num_layers": MODEL_NUM_LAYERS,
            "num_heads": MODEL_NUM_HEADS,
            "ffn_hidden_size": MODEL_FFN_HIDDEN_SIZE,
            "dropout_rate": MODEL_DROPOUT_RATE,
            "layer_norm_epsilon": MODEL_LAYER_NORM_EPSILON,
        },
        "train": {
            "seed": TRAIN_SEED,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": TRAIN_EVAL_BATCH_SIZE,
            "learning_rate": TRAIN_LEARNING_RATE,
            "warmup_steps": TRAIN_WARMUP_STEPS,
            "train_steps": TRAIN_STEPS,
            "eval_every": TRAIN_EVAL_EVERY,
            "save_every": TRAIN_SAVE_EVERY,
            "log_every": TRAIN_LOG_EVERY,
            "weight_decay": TRAIN_WEIGHT_DECAY,
            "grad_clip_norm": TRAIN_GRAD_CLIP_NORM,
            "mixed_precision": TRAIN_MIXED_PRECISION,
            "optimizer_epsilon": TRAIN_OPTIMIZER_EPSILON,
            "fail_on_nonfinite": TRAIN_FAIL_ON_NONFINITE,
            "cache_dataset": TRAIN_CACHE_DATASET,
            "prefetch_to_device": TRAIN_PREFETCH_TO_DEVICE,
            "autotune": TRAIN_AUTOTUNE,
            "generate_every": TRAIN_GENERATE_EVERY,
            "generation_prompts_file": str(TRAIN_GENERATION_PROMPTS_FILE),
            "generation_max_new_tokens": TRAIN_GENERATION_MAX_NEW_TOKENS,
            "generation_batch_size": TRAIN_GENERATION_BATCH_SIZE,
            "generation_temperature": TRAIN_GENERATION_TEMPERATURE,
            "generation_top_k": TRAIN_GENERATION_TOP_K,
            "generation_top_p": TRAIN_GENERATION_TOP_P,
            "generation_repetition_penalty": TRAIN_GENERATION_REPETITION_PENALTY,
            "generation_no_repeat_ngram_size": TRAIN_GENERATION_NO_REPEAT_NGRAM_SIZE,
        },
    }
