import os


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

CACHE_ENABLED = os.environ.get("DETECTOR_CACHE_ENABLED", "1") != "0"
CACHE_REBUILD = os.environ.get("DETECTOR_CACHE_REBUILD", "0") == "1"
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")
CACHE_ACTIVE_DIR = os.path.join(CACHE_DIR, "active")
CACHE_TARGET_VERSION = 2
# Capacity is storage for a complete split, never a dataset sampling limit.
CACHE_MAX_SAMPLES = 2**22

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

INPUT_SIZE = (256, 256)
GRID_SIZE = 4
BOXES_PER_CELL = 2
BATCH_SIZE = int(os.environ.get("DETECTOR_BATCH_SIZE", "8"))
EPOCHS = 100
CLASSIFIER_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
MIN_BOX_SIZE = 4
MAX_BOXES = 64

TRAIN_SPLIT = "train2017"
VAL_SPLIT = "val2017"
TEST_SPLIT = "val2017"

STEPS_PER_EPOCH = 4000
VALIDATION_STEPS = 250
PROGRESS_UPDATE_INTERVAL = 10
INSPECTION_GRID_SIZE = 16
VALIDATION_METRIC_SAMPLES = 200
SCORE_THRESHOLD = 0.35
NMS_IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 50
MAX_DRAW_DETECTIONS = 20

LOG_DIR = os.path.join(PROJECT_DIR, "logs", "coco2017", "YOLOv1")
CHECKPOINT_PATH = os.path.join(LOG_DIR, "yolov1.weights.h5")
HISTORY_PATH = os.path.join(LOG_DIR, "history.json")
STATE_PATH = os.path.join(LOG_DIR, "training_state.json")
INFERENCE_DIR = os.path.join(LOG_DIR, "inference")

IMAGENET_ROOT_CANDIDATES = [
    os.path.join(DATASET_PATH, "imagenet", "ILSVRC") if DATASET_PATH else None,
    os.path.join(DATASET_PATH, "ImageNet", "ILSVRC") if DATASET_PATH else None,
]
IMAGENET_ROOT = next((path for path in IMAGENET_ROOT_CANDIDATES if path and os.path.isdir(path)), None)
IMAGENET_TRAIN_DIR = os.path.join(IMAGENET_ROOT, "Data", "CLS-LOC", "train") if IMAGENET_ROOT else None
IMAGENET_VAL_DIR = os.path.join(IMAGENET_ROOT, "Data", "CLS-LOC", "val") if IMAGENET_ROOT else None
IMAGENET_TRAIN_ANN_DIR = os.path.join(IMAGENET_ROOT, "Annotations", "CLS-LOC", "train") if IMAGENET_ROOT else None
IMAGENET_VAL_ANN_DIR = os.path.join(IMAGENET_ROOT, "Annotations", "CLS-LOC", "val") if IMAGENET_ROOT else None
IMAGENET_BATCH_SIZE = 128
IMAGENET_NUM_CLASSES = 1000
VOC_ROOT_CANDIDATES = [
    os.path.join(DATASET_PATH, "VOCdevkit", "VOC2012") if DATASET_PATH else None,
    os.path.join(DATASET_PATH, "VOC2012") if DATASET_PATH else None,
]
VOC_ROOT = next((path for path in VOC_ROOT_CANDIDATES if path and os.path.isdir(path)), None)
CLASSIFIER_LOG_DIR = os.path.join(PROJECT_DIR, "logs", "imagenet", "YOLOv1")
CLASSIFIER_CHECKPOINT_PATH = os.path.join(CLASSIFIER_LOG_DIR, "imagenet_classifier.weights.h5")
CLASSIFIER_HISTORY_PATH = os.path.join(CLASSIFIER_LOG_DIR, "history.json")
CLASSIFIER_STATE_PATH = os.path.join(CLASSIFIER_LOG_DIR, "training_state.json")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
