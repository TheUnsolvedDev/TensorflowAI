import os


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RCNN_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "RCNN")

DATASET_PATHS = [
    "/home/shuvrajeet/Documents/Dataset",
    "/mnt/storage/da24d402/Documents/Dataset",
    "/storage/nas/da24d402/Documents/Dataset",
]
DATASET_PATH = next((path for path in DATASET_PATHS if os.path.isdir(path)), None)

COCO_ROOT_CANDIDATES = [
    os.path.join(DATASET_PATH, "coco") if DATASET_PATH else None,
    os.path.join(DATASET_PATH, "COCO") if DATASET_PATH else None,
    os.path.join(DATASET_PATH, "coco2017") if DATASET_PATH else None,
    os.path.join(DATASET_PATH, "ObjectDetection", "coco") if DATASET_PATH else None,
]
COCO_ROOT = next((path for path in COCO_ROOT_CANDIDATES if path and os.path.isdir(path)), None)

INPUT_SIZE = (800, 800)
BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

TRAIN_SPLIT = "train2017"
VAL_SPLIT = "val2017"
TEST_SPLIT = "val2017"

USE_SELECTIVE_SEARCH = True
MAX_PROPOSALS = 2000
ROIS_PER_IMAGE = 64
POSITIVE_FRACTION = 0.25
POSITIVE_IOU_THRESHOLD = 0.5
NEGATIVE_IOU_THRESHOLD = 0.1
MIN_BOX_SIZE = 8

ROI_POOL_SIZE = (7, 7)
FC_DIM = 1024

STEPS_PER_EPOCH = 2000
VALIDATION_STEPS = 250
INFERENCE_BATCH_SIZE = 256
INFERENCE_TOPK = 300
SCORE_THRESHOLD = 0.4
NMS_IOU_THRESHOLD = 0.3
MAX_DETECTIONS_PER_CLASS = 25
MAX_DRAW_DETECTIONS = 20

LOG_DIR = os.path.join(PROJECT_DIR, "logs", "coco2017", "FastRCNN")
CHECKPOINT_PATH = os.path.join(LOG_DIR, "fast_rcnn.weights.h5")
HISTORY_PATH = os.path.join(LOG_DIR, "history.json")
STATE_PATH = os.path.join(LOG_DIR, "training_state.json")
INFERENCE_DIR = os.path.join(LOG_DIR, "inference")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache", "coco2017_fast_rcnn")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

