import os

DATASET_PATHS = ['/home/shuvrajeet/Documents/Dataset/', '/mnt/storage/da24d402/Documents/Dataset/']
DATASET_PATH = next((p for p in DATASET_PATHS if os.path.isdir(p)), None)
print(f"Using dataset path: {DATASET_PATH}")
IMAGE_SIZE = (32,32)
BATCH_SIZE = 32
EPOCHS = 50
LATENT_DIM = 64
GENERATOR_LEARNING_RATE = 1e-4
DISCRIMINATOR_LEARNING_RATE = 1e-4
N_GEN_STEP = 1 # 2 for celeba
N_DISC_STEP = 1 # 1 for celeba