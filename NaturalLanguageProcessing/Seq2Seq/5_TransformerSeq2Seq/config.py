# config.py

DATASET_ROOT = "/home/shuvrajeet/Documents/Dataset"
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
LOWERCASE = True
SEED = 42
EPOCHS = 35
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"

D_MODEL = 128
NUM_HEADS = 4
DFF = 512
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DROPOUT_RATE = 0.1

SOURCE_MAX_LENGTH = 32
TARGET_MAX_LENGTH = 32
VOCAB_SIZE = 3000

DATASET_CONFIGS = {

    "manythings_english_french": {
        "batch_size": 128,
        "d_model": 128,
        "num_heads": 4,
        "dff": 512,
        "source_max_length": 32,
        "target_max_length": 32,
        "vocab_size": 8000,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dropout_rate": 0.1
    },

    "cornell_movie_dialogs": {
        "batch_size": 64,
        "d_model": 256,
        "num_heads": 8,
        "dff": 1024,
        "source_max_length": 32,
        "target_max_length": 32,
        "vocab_size": 10000,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "dropout_rate": 0.1
    },

    "wikilarge": {
        "batch_size": 32,
        "d_model": 256,
        "num_heads": 8,
        "dff": 1024,
        "source_max_length": 64,
        "target_max_length": 64,
        "vocab_size": 12000,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "dropout_rate": 0.1
    },

    "english_french": {
        "batch_size": 32,
        "d_model": 512,
        "num_heads": 8,
        "dff": 2048,
        "source_max_length": 64,
        "target_max_length": 64,
        "vocab_size": 25000,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dropout_rate": 0.1
    },

    "english_german": {
        "batch_size": 16,
        "d_model": 512,
        "num_heads": 8,
        "dff": 2048,
        "source_max_length": 96,
        "target_max_length": 96,
        "vocab_size": 20000,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dropout_rate": 0.1
    },

    "cnn_dailymail": {
        "batch_size": 8,
        "d_model": 512,
        "num_heads": 8,
        "dff": 2048,
        "source_max_length": 256,
        "target_max_length": 64,
        "vocab_size": 15000,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dropout_rate": 0.1
    }
}