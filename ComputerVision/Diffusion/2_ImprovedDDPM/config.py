import os

DATASET_PATHS = [
    "/home/shuvrajeet/Documents/Dataset/",
    "/mnt/storage/da24d402/Documents/Dataset/",
]
DATASET_PATH = next((path for path in DATASET_PATHS if os.path.isdir(path)), None)
print(f"Using dataset path: {DATASET_PATH}")

ARCHITECTURE_VERSION = "improved_ddpm_v1"
SUPPORTED_DATASETS = [
    "mnist", "celeba", "bedroom", "fashion_mnist", "cifar10", "cifar100"
]
DEFAULT_DATASET = "mnist"
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32
SUPPORTED_IMAGE_SIZES = (32, 64, 96, 128)
DEFAULT_IMAGE_SIZES = {
    "mnist": 32,
    "fashion_mnist": 32,
    "cifar10": 32,
    "cifar100": 32,
    "celeba": 96,
    "bedroom": 96,
}
DEFAULT_BATCH_SIZES = {name: 32 for name in SUPPORTED_DATASETS}

EPOCHS = 100
DIFFUSION_STEPS = 200
SAMPLING_STEPS = 50
TIME_EMBED_DIM = 128
BASE_CHANNELS = 64
DENOISER_LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.1
HYBRID_VLB_WEIGHT = 1e-3
SAMPLER_EMA_DECAY = 0.99

SHUFFLE_BUFFER_SIZE = 10000
ENABLE_DATASET_NONDETERMINISM = True
SAVE_EVERY_N_EPOCHS = 5
SAMPLE_EVERY_N_EPOCHS = 5
MAX_DATASET_ITEMS = None
