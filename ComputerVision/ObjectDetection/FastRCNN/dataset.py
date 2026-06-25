import os
import pickle
import random
from collections import defaultdict
import importlib.util

import cv2
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO

from config import (
    BATCH_SIZE,
    CACHE_DIR,
    COCO_ROOT,
    INPUT_SIZE,
    MAX_PROPOSALS,
    MIN_BOX_SIZE,
    NEGATIVE_IOU_THRESHOLD,
    POSITIVE_FRACTION,
    POSITIVE_IOU_THRESHOLD,
    ROIS_PER_IMAGE,
    USE_SELECTIVE_SEARCH,
    VAL_SPLIT,
    TRAIN_SPLIT,
    ensure_dir,
)

_utils_spec = importlib.util.spec_from_file_location(
    "rcnn_fast_utils",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "RCNN", "utils.py"),
)
_utils_module = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(_utils_module)
compute_iou = _utils_module.compute_iou
encode_box = _utils_module.encode_box
generate_region_proposals = _utils_module.generate_region_proposals

COCO_CLASSES = [
    "background",
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
COCO_ID_TO_IDX = {category_id: index + 1 for index, category_id in enumerate(COCO_CATEGORY_IDS)}
NUM_CLASSES = len(COCO_CLASSES)


TARGET_CACHE_DIR = ensure_dir(os.path.join(CACHE_DIR, "targets"))


def collect_coco_samples(coco_root, split):
    annotation_file = f"{coco_root}/annotations/instances_{split}.json"
    image_dir = f"{coco_root}/{split}"
    coco = COCO(annotation_file)
    annotations_by_image = defaultdict(list)
    for annotation in coco.dataset["annotations"]:
        if annotation.get("iscrowd", 0) == 1:
            continue
        category_idx = COCO_ID_TO_IDX.get(annotation["category_id"])
        if category_idx is None:
            continue
        annotations_by_image[annotation["image_id"]].append(annotation)

    samples = []
    for image_id in coco.getImgIds():
        image_info = coco.loadImgs(image_id)[0]
        image_path = f"{image_dir}/{image_info['file_name']}"
        annotations = annotations_by_image.get(image_id, [])
        boxes = []
        labels = []
        areas = []
        for annotation in annotations:
            x, y, width, height = annotation["bbox"]
            if width < MIN_BOX_SIZE or height < MIN_BOX_SIZE:
                continue
            boxes.append([x, y, x + width, y + height])
            labels.append(COCO_ID_TO_IDX[annotation["category_id"]])
            areas.append(annotation.get("area", width * height))
        if not boxes:
            continue
        samples.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "width": int(image_info["width"]),
                "height": int(image_info["height"]),
                "boxes": np.asarray(boxes, dtype=np.float32),
                "labels": np.asarray(labels, dtype=np.int32),
                "areas": np.asarray(areas, dtype=np.float32),
                "file_name": image_info["file_name"],
                "split": split,
            }
        )
    return samples


class FastRCNNDataset:
    def __init__(
        self,
        coco_root=COCO_ROOT,
        split=TRAIN_SPLIT,
        batch_size=BATCH_SIZE,
        input_size=INPUT_SIZE,
        rois_per_image=ROIS_PER_IMAGE,
        max_proposals=MAX_PROPOSALS,
        positive_fraction=POSITIVE_FRACTION,
        positive_iou_threshold=POSITIVE_IOU_THRESHOLD,
        negative_iou_threshold=NEGATIVE_IOU_THRESHOLD,
        augment=False,
        shuffle=True,
        use_selective_search=USE_SELECTIVE_SEARCH,
    ):
        self.coco_root = coco_root
        self.split = split
        self.batch_size = batch_size
        self.input_size = tuple(input_size)
        self.rois_per_image = rois_per_image
        self.max_proposals = max_proposals
        self.positive_fraction = positive_fraction
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold = negative_iou_threshold
        self.augment = augment
        self.shuffle = shuffle
        self.use_selective_search = use_selective_search
        self.samples = collect_coco_samples(coco_root=self.coco_root, split=self.split)
        self.cache_dir = ensure_dir(
            os.path.join(
                TARGET_CACHE_DIR,
                self.split,
                f"rois_{self.rois_per_image}_mp_{self.max_proposals}",
            )
        )
        print(f"[FastRCNNDataset] split={self.split} samples={len(self.samples)}")

    def _cache_path(self, image_id):
        return os.path.join(self.cache_dir, f"{image_id}.pkl")

    def _sample_indices(self, indices, count):
        if len(indices) == 0 or count <= 0:
            return np.array([], dtype=np.int32)
        replace = len(indices) < count
        return np.random.choice(indices, size=count, replace=replace)

    def _build_cached_sample(self, sample):
        cache_path = self._cache_path(sample["image_id"])
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as file:
                return pickle.load(file)

        image_bgr = cv2.imread(sample["image_path"])
        if image_bgr is None:
            return None

        proposals = generate_region_proposals(
            image_bgr=image_bgr,
            max_proposals=self.max_proposals,
            use_selective_search=self.use_selective_search,
        )
        proposals = np.concatenate([proposals, sample["boxes"]], axis=0).astype(np.float32)
        proposals = np.unique(np.round(proposals).astype(np.int32), axis=0).astype(np.float32)

        gt_boxes = sample["boxes"].astype(np.float32)
        gt_labels = sample["labels"].astype(np.int32)
        ious = compute_iou(proposals, gt_boxes)
        max_iou = ious.max(axis=1)
        matched_gt = ious.argmax(axis=1)

        positive_indices = np.where(max_iou >= self.positive_iou_threshold)[0]
        negative_indices = np.where(max_iou < self.negative_iou_threshold)[0]
        fallback_negative_indices = np.where(max_iou < self.positive_iou_threshold)[0]

        num_positive = int(self.rois_per_image * self.positive_fraction)
        num_negative = self.rois_per_image - num_positive
        chosen_positive = self._sample_indices(positive_indices, num_positive)
        chosen_negative = self._sample_indices(negative_indices, num_negative)
        if len(chosen_negative) == 0:
            chosen_negative = self._sample_indices(fallback_negative_indices, num_negative)

        if len(chosen_positive) == 0 and len(chosen_negative) == 0:
            return None

        chosen = np.concatenate([chosen_positive, chosen_negative], axis=0)
        if len(chosen) < self.rois_per_image:
            extra = self._sample_indices(np.arange(len(proposals)), self.rois_per_image - len(chosen))
            chosen = np.concatenate([chosen, extra], axis=0)

        chosen = chosen[: self.rois_per_image]
        positive_lookup = set(chosen_positive.tolist())
        image_h = max(float(sample["height"]), 1.0)
        image_w = max(float(sample["width"]), 1.0)

        rois = []
        labels = []
        bbox_targets = []
        bbox_weights = []

        for index in chosen:
            proposal = proposals[index].astype(np.float32)
            x1, y1, x2, y2 = proposal
            rois.append([y1 / image_h, x1 / image_w, y2 / image_h, x2 / image_w])

            if index in positive_lookup:
                gt_idx = matched_gt[index]
                labels.append(gt_labels[gt_idx])
                bbox_targets.append(encode_box(proposal, gt_boxes[gt_idx]))
                bbox_weights.append(1.0)
            else:
                labels.append(0)
                bbox_targets.append(np.zeros((4,), dtype=np.float32))
                bbox_weights.append(0.0)

        payload = {
            "image_path": sample["image_path"],
            "rois": np.asarray(rois, dtype=np.float32),
            "labels": np.asarray(labels, dtype=np.int32),
            "bbox_targets": np.asarray(bbox_targets, dtype=np.float32),
            "bbox_weights": np.asarray(bbox_weights, dtype=np.float32),
        }
        with open(cache_path, "wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        return payload

    def _generator(self):
        indices = list(range(len(self.samples)))
        if self.shuffle:
            random.shuffle(indices)
        for idx in indices:
            payload = self._build_cached_sample(self.samples[idx])
            if payload is None:
                continue
            yield (
                payload["image_path"].encode(),
                payload["rois"],
                payload["labels"],
                payload["bbox_targets"],
                payload["bbox_weights"],
            )

    def _process(self, image_path, rois, labels, bbox_targets, bbox_weights):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.input_size)

        if self.augment:
            image = tf.image.random_brightness(image, max_delta=0.08)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            image = tf.clip_by_value(image, 0.0, 1.0)

        inputs = {"image": image, "rois": rois}
        targets = {"class_logits": labels, "bbox_regression": bbox_targets}
        weights = {
            "class_logits": tf.ones_like(bbox_weights, dtype=tf.float32),
            "bbox_regression": bbox_weights,
        }
        return inputs, targets, weights

    def build(self):
        output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(self.rois_per_image, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(self.rois_per_image,), dtype=tf.int32),
            tf.TensorSpec(shape=(self.rois_per_image, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(self.rois_per_image,), dtype=tf.float32),
        )
        dataset = tf.data.Dataset.from_generator(self._generator, output_signature=output_signature)
        if self.shuffle:
            dataset = dataset.shuffle(self.batch_size * 32, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(self._process, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


def build_train_dataset(coco_root=COCO_ROOT, split=TRAIN_SPLIT):
    dataset = FastRCNNDataset(coco_root=coco_root, split=split, augment=True, shuffle=True)
    return dataset, dataset.build()


def build_val_dataset(coco_root=COCO_ROOT, split=VAL_SPLIT):
    dataset = FastRCNNDataset(coco_root=coco_root, split=split, augment=False, shuffle=False)
    return dataset, dataset.build()
