"""Defaults for this standalone experiment; CLI flags override training settings."""
import os
DATASET_PATH = os.environ.get('DATASET_ROOT', '/home/shuvrajeet/Documents/Dataset')
INPUT_SIZE = [299, 299, 3]
BATCH_SIZE = 32
EPOCHS = 20  # Default maximum; --epochs overrides this value.
LEARNING_RATE = 0.0001
MODEL_FN = 'inception3_model'
MODEL_ID = '4_InceptionNet__InceptionV3'
MODEL_KIND = 'cnn'
FROM_LOGITS = True
