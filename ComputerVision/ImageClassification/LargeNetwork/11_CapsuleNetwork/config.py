"""Defaults for this standalone experiment; CLI flags override training settings."""
import os
DATASET_PATH = os.environ.get('DATASET_ROOT', '/home/shuvrajeet/Documents/Dataset')
INPUT_SIZE = [28, 28, 3]
BATCH_SIZE = 32
EPOCHS = 20  # Default maximum; --epochs overrides this value.
LEARNING_RATE = 0.001
MODEL_FN = 'capsnet_model'
MODEL_ID = '11_CapsuleNetwork'
MODEL_KIND = 'capsule'
FROM_LOGITS = False
