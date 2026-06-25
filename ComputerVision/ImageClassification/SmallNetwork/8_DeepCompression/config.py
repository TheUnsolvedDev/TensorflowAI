import os

DATASET_PATHS = ['/home/shuvrajeet/Documents/Dataset/', '/mnt/storage/da24d402/Documents/Dataset/','/storage/nas/da24d402/Documents/Dataset/']
DATASET_PATH = next((p for p in DATASET_PATHS if os.path.isdir(p)), None)
INPUT_SIZE = [128, 128, 3]
BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 1e-4
