import os


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Cache defaults are consumed explicitly by dataset construction. CLI flags can
# override them for one invocation without mutating process environment state.
CACHE_ENABLED = True
CACHE_REBUILD = False
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")
CACHE_TARGET_VERSION = 2
# Full split cache capacity. 0 opts into the explicit slower no-cache path.
CACHE_MAX_SAMPLES = 2**20

DATASET_PATHS = [
    "/home/shuvrajeet/Documents/Dataset",
    "/mnt/storage/da24d402/Documents/Dataset",
    "/storage/nas/da24d402/Documents/Dataset",
]
DATASET_PATH = next(
    (path for path in DATASET_PATHS if os.path.isdir(path)), None)

COCO_ROOT_CANDIDATES = [
    os.path.join(DATASET_PATH, "coco") if DATASET_PATH else None,
    os.path.join(DATASET_PATH, "COCO") if DATASET_PATH else None,
    os.path.join(DATASET_PATH, "coco2017") if DATASET_PATH else None,
    os.path.join(DATASET_PATH, "ObjectDetection",
                 "coco") if DATASET_PATH else None,
]
COCO_ROOT = next(
    (path for path in COCO_ROOT_CANDIDATES if path and os.path.isdir(path)), None)

INPUT_SIZE = (256, 256)
# ROI tensors amplify activation memory.  Keep the default conservative and
# allow an explicit, measured override instead of forcing global batch 16.
BATCH_SIZE = int(os.environ.get("DETECTOR_BATCH_SIZE", "8"))
# Detector runs are intentionally long-lived; classifier defaults remain below.
EPOCHS = 100
CLASSIFIER_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

TRAIN_SPLIT = "train2017"
VAL_SPLIT = "val2017"
TEST_SPLIT = "val2017"

USE_SELECTIVE_SEARCH = True
MAX_PROPOSALS = 2000
ROIS_PER_IMAGE = 64
POSITIVE_FRACTION = 0.50
POSITIVE_IOU_THRESHOLD = 0.5
NEGATIVE_IOU_THRESHOLD = 0.1
MIN_BOX_SIZE = 8
GT_JITTERS_PER_BOX = 8
GT_JITTER_CENTER_STD = 0.10
GT_JITTER_SCALE_STD = 0.15

ROI_POOL_SIZE = (7, 7)
FC_DIM = 512

STEPS_PER_EPOCH = 2000
VALIDATION_STEPS = 250
VALIDATION_METRIC_SAMPLES = 200
PROGRESS_UPDATE_INTERVAL = 10
INSPECTION_GRID_SIZE = 16
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
IMAGENET_ROOT_CANDIDATES = [
    os.path.join(DATASET_PATH, "imagenet", "ILSVRC") if DATASET_PATH else None,
    os.path.join(DATASET_PATH, "ImageNet", "ILSVRC") if DATASET_PATH else None,
]
IMAGENET_ROOT = next(
    (path for path in IMAGENET_ROOT_CANDIDATES if path and os.path.isdir(path)), None)
IMAGENET_TRAIN_DIR = os.path.join(
    IMAGENET_ROOT, "Data", "CLS-LOC", "train") if IMAGENET_ROOT else None
IMAGENET_VAL_DIR = os.path.join(
    IMAGENET_ROOT, "Data", "CLS-LOC", "val") if IMAGENET_ROOT else None
IMAGENET_TRAIN_ANN_DIR = os.path.join(
    IMAGENET_ROOT, "Annotations", "CLS-LOC", "train") if IMAGENET_ROOT else None
IMAGENET_VAL_ANN_DIR = os.path.join(
    IMAGENET_ROOT, "Annotations", "CLS-LOC", "val") if IMAGENET_ROOT else None
IMAGENET_BATCH_SIZE = 128
IMAGENET_NUM_CLASSES = 1000
VOC_ROOT_CANDIDATES = [
    os.path.join(DATASET_PATH, "VOCdevkit",
                 "VOC2012") if DATASET_PATH else None,
    os.path.join(DATASET_PATH, "VOC2012") if DATASET_PATH else None,
    os.path.join(DATASET_PATH, "archive", "VOC2012_train_val",
                 "VOC2012_train_val") if DATASET_PATH else None,
]
VOC_ROOT = next(
    (path for path in VOC_ROOT_CANDIDATES if path and os.path.isdir(path)), None)
CLASSIFIER_LOG_DIR = os.path.join(PROJECT_DIR, "logs", "imagenet", "FastRCNN")
CLASSIFIER_CHECKPOINT_PATH = os.path.join(
    CLASSIFIER_LOG_DIR, "imagenet_classifier.weights.h5")
CLASSIFIER_HISTORY_PATH = os.path.join(CLASSIFIER_LOG_DIR, "history.json")
CLASSIFIER_STATE_PATH = os.path.join(CLASSIFIER_LOG_DIR, "training_state.json")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
