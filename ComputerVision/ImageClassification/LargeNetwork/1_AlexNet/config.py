"""Defaults for this standalone experiment; CLI flags override training settings."""
import os
DATASET_PATH = os.environ.get('DATASET_ROOT', '/home/shuvrajeet/Documents/Dataset')
INPUT_SIZE = [224, 224, 3]
BATCH_SIZE = 32
EPOCHS = 20  # Default maximum; --epochs overrides this value.
LEARNING_RATE = 0.0001
MODEL_FN = 'alexnet_model'
MODEL_ID = '1_AlexNet'
MODEL_KIND = 'cnn'
FROM_LOGITS = False
