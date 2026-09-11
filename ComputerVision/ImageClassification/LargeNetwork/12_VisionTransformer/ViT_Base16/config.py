"""Defaults for this standalone experiment; CLI flags override training settings."""
import os
DATASET_PATH = os.environ.get('DATASET_ROOT', '/home/shuvrajeet/Documents/Dataset')
INPUT_SIZE = [224, 224, 3]
BATCH_SIZE = 8
EPOCHS = 20  # Default maximum; --epochs overrides this value.
LEARNING_RATE = 0.0003
MODEL_FN = 'vit_base16_model'
MODEL_ID = '12_VisionTransformer__ViT_Base16'
MODEL_KIND = 'vit'
FROM_LOGITS = False
