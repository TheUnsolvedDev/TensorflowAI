import sys
import os
from collections import defaultdict

import tensorflow as tf
import numpy as np
from pycocotools.coco import COCO

from config import (
    BATCH_SIZE,
    COCO_ROOT,
    INPUT_SIZE,
    MAX_GT_BOXES,
    MIN_BOX_SIZE,
    TRAIN_SPLIT,
    VAL_SPLIT,
)

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
        for annotation in annotations:
            x, y, width, height = annotation["bbox"]
            if width < MIN_BOX_SIZE or height < MIN_BOX_SIZE:
                continue
            boxes.append([x, y, x + width, y + height])
            labels.append(COCO_ID_TO_IDX[annotation["category_id"]])
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
                "file_name": image_info["file_name"],
                "split": split,
            }
        )
    return samples


class FasterRCNNDataset:
    def __init__(
        self,
        coco_root=COCO_ROOT,
        split=TRAIN_SPLIT,
        batch_size=BATCH_SIZE,
        input_size=INPUT_SIZE,
        max_gt_boxes=MAX_GT_BOXES,
        augment=False,
        shuffle=True,
    ):
        self.coco_root = coco_root
        self.split = split
        self.batch_size = batch_size
        self.input_size = tuple(input_size)
        self.max_gt_boxes = max_gt_boxes
        self.augment = augment
        self.shuffle = shuffle
        self.samples = collect_coco_samples(coco_root=self.coco_root, split=self.split)
        print(f"[FasterRCNNDataset] split={self.split} samples={len(self.samples)}")

    def _generator(self):
        indices = tf.range(len(self.samples)).numpy().tolist()
        if self.shuffle:
            import random
            random.shuffle(indices)
        for idx in indices:
            sample = self.samples[idx]
            boxes = sample["boxes"]
            labels = sample["labels"]
            valid = (labels > 0)
            boxes = boxes[valid]
            labels = labels[valid]
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            keep = (widths >= MIN_BOX_SIZE) & (heights >= MIN_BOX_SIZE)
            boxes = boxes[keep]
            labels = labels[keep]
            if len(boxes) == 0:
                continue
            image_h = max(float(sample["height"]), 1.0)
            image_w = max(float(sample["width"]), 1.0)
            norm_boxes = boxes.copy().astype("float32")
            norm_boxes[:, 0] /= image_w
            norm_boxes[:, 2] /= image_w
            norm_boxes[:, 1] /= image_h
            norm_boxes[:, 3] /= image_h
            norm_boxes = norm_boxes[:, [1, 0, 3, 2]]

            padded_boxes = tf.zeros((self.max_gt_boxes, 4), dtype=tf.float32).numpy()
            padded_labels = tf.zeros((self.max_gt_boxes,), dtype=tf.int32).numpy()
            valid_mask = tf.zeros((self.max_gt_boxes,), dtype=tf.bool).numpy()
            count = min(len(norm_boxes), self.max_gt_boxes)
            padded_boxes[:count] = norm_boxes[:count]
            padded_labels[:count] = labels[:count]
            valid_mask[:count] = True
            yield (
                sample["image_path"].encode(),
                padded_boxes,
                padded_labels,
                valid_mask,
            )

    def _process(self, image_path, gt_boxes, gt_labels, valid_mask):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.input_size)
        if self.augment:
            image = tf.image.random_brightness(image, max_delta=0.08)
            image = tf.image.random_contrast(image, 0.9, 1.1)
            image = tf.clip_by_value(image, 0.0, 1.0)
        inputs = {
            "image": image,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "valid_mask": valid_mask,
        }
        return inputs

    def build(self):
        output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(self.max_gt_boxes, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(self.max_gt_boxes,), dtype=tf.int32),
            tf.TensorSpec(shape=(self.max_gt_boxes,), dtype=tf.bool),
        )
        dataset = tf.data.Dataset.from_generator(self._generator, output_signature=output_signature)
        if self.shuffle:
            dataset = dataset.shuffle(self.batch_size * 16, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(self._process, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


def build_train_dataset(coco_root=COCO_ROOT, split=TRAIN_SPLIT):
    dataset = FasterRCNNDataset(coco_root=coco_root, split=split, augment=True, shuffle=True)
    return dataset, dataset.build()


def build_val_dataset(coco_root=COCO_ROOT, split=VAL_SPLIT):
    dataset = FasterRCNNDataset(coco_root=coco_root, split=split, augment=False, shuffle=False)
    return dataset, dataset.build()
