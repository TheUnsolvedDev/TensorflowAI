# config.py

DATASET_ROOT = "/home/shuvrajeet/Documents/Dataset"
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
LOWERCASE = True
SEED = 42
EPOCHS = 35
LEARNING_RATE = 1e-3
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"

EMBEDDING_DIM = 128
ENCODER_UNITS = 128
DECODER_UNITS = 128
SOURCE_MAX_LENGTH = 32
TARGET_MAX_LENGTH = 32
VOCAB_SIZE = 3000

DATASET_CONFIGS = {

    "manythings_english_french": {
        "batch_size": 128,
        "embedding_dim": 128,
        "encoder_units": 128,
        "decoder_units": 128,
        "source_max_length": 32,
        "target_max_length": 32,
        "vocab_size": 8000,
        "num_encoder": 1,
        "num_decoder": 1
    },

    "cornell_movie_dialogs": {
        "batch_size": 128,
        "embedding_dim": 256,
        "encoder_units": 128,
        "decoder_units": 128,
        "source_max_length": 32,
        "target_max_length": 32,
        "vocab_size": 10000,
        "num_encoder": 2,
        "num_decoder": 1
    },

    "wikilarge": {
        "batch_size": 32,
        "embedding_dim": 256,
        "encoder_units": 128,
        "decoder_units": 128,
        "source_max_length": 64,
        "target_max_length": 64,
        "vocab_size": 12000,
        "num_encoder": 2,
        "num_decoder": 1
    },

    "english_french": {
        "batch_size": 32,
        "embedding_dim": 512,
        "encoder_units": 256,
        "decoder_units": 256,
        "source_max_length": 64,
        "target_max_length": 64,
        "vocab_size": 25000,
        "num_encoder": 3,
        "num_decoder": 2
    },

    "english_german": {
        "batch_size": 32,
        "embedding_dim": 512,
        "encoder_units": 256,
        "decoder_units": 256,
        "source_max_length": 96,
        "target_max_length": 96,
        "vocab_size": 20000,
        "num_encoder": 4,
        "num_decoder": 2
    },

    "cnn_dailymail": {
        "batch_size": 32,
        "embedding_dim": 512,
        "encoder_units": 256,
        "decoder_units": 256,
        "source_max_length": 256,
        "target_max_length": 64,
        "vocab_size": 15000,
        "num_encoder": 3,
        "num_decoder": 2
    }
}
