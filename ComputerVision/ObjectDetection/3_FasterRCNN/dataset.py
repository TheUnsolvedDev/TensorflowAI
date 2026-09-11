import sys
import os
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import tensorflow as tf
import numpy as np
from pycocotools.coco import COCO

from config import (
    BATCH_SIZE,
    ANCHOR_RATIOS,
    ANCHOR_SCALES,
    COCO_ROOT,
    IMAGENET_BATCH_SIZE,
    IMAGENET_ROOT,
    IMAGENET_TRAIN_ANN_DIR,
    IMAGENET_TRAIN_DIR,
    IMAGENET_VAL_ANN_DIR,
    IMAGENET_VAL_DIR,
    INPUT_SIZE,
    MAX_GT_BOXES,
    MIN_BOX_SIZE,
    ROI_NEGATIVE_IOU_THRESHOLD,
    ROI_POOL_SIZE,
    ROI_POSITIVE_FRACTION,
    ROI_POSITIVE_IOU_THRESHOLD,
    ROI_SAMPLES_PER_IMAGE,
    RPN_NEGATIVE_IOU_THRESHOLD,
    RPN_NMS_IOU_THRESHOLD,
    RPN_POSITIVE_FRACTION,
    RPN_POSITIVE_IOU_THRESHOLD,
    RPN_POST_NMS_TOPK,
    RPN_PRE_NMS_TOPK,
    RPN_SAMPLES_PER_IMAGE,
    TRAIN_SPLIT,
    VAL_SPLIT,
    VOC_ROOT,
)
from cache_runtime import cached_records
from detection_adapters import VOC_CLASSES, voc_records


def configure_dataset_pipeline(dataset, deterministic):
    options = tf.data.Options()
    options.experimental_deterministic = deterministic
    # Slack is optional and requires a specific final-prefetch graph shape.
    # Keep the terminal prefetch but disable slack to avoid its runtime warning.
    options.experimental_slack = False
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    return dataset.with_options(options)


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
COCO_ID_TO_IDX = {category_id: index + 1 for index,
                  category_id in enumerate(COCO_CATEGORY_IDS)}
NUM_CLASSES = len(COCO_CLASSES)
IMAGENET_DET_NUM_CLASSES = 1001
VOC_DET_NUM_CLASSES = len(VOC_CLASSES) + 1


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


def collect_imagenet_detection_samples(imagenet_root, split):
    if imagenet_root is None:
        raise RuntimeError("IMAGENET_ROOT not found")
    class_names = _imagenet_class_names(imagenet_root)
    class_to_idx = {name: index + 1 for index, name in enumerate(class_names)}
    samples = []
    if split == "train":
        ann_root = Path(IMAGENET_TRAIN_ANN_DIR or "")
        image_root = Path(IMAGENET_TRAIN_DIR or "")
        for class_name in class_names:
            for xml_path in sorted((ann_root / class_name).glob("*.xml")):
                image_path = image_root / class_name / f"{xml_path.stem}.JPEG"
                if not image_path.exists():
                    continue
                root = ET.parse(xml_path).getroot()
                size = root.find("./size")
                width = int(size.findtext("width", "1"))
                height = int(size.findtext("height", "1"))
                boxes = []
                labels = []
                for obj in root.findall("./object"):
                    name = obj.findtext("name")
                    if name not in class_to_idx:
                        continue
                    bbox = obj.find("./bndbox")
                    x1 = float(bbox.findtext("xmin", "0"))
                    y1 = float(bbox.findtext("ymin", "0"))
                    x2 = float(bbox.findtext("xmax", "0"))
                    y2 = float(bbox.findtext("ymax", "0"))
                    if (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE:
                        continue
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_to_idx[name])
                if boxes:
                    samples.append({"image_id": xml_path.stem, "image_path": str(image_path), "width": width, "height": height, "boxes": np.asarray(
                        boxes, dtype=np.float32), "labels": np.asarray(labels, dtype=np.int32), "file_name": image_path.name, "split": split})
    else:
        ann_root = Path(IMAGENET_VAL_ANN_DIR or "")
        image_root = Path(IMAGENET_VAL_DIR or "")
        for xml_path in sorted(ann_root.glob("*.xml")):
            image_path = image_root / f"{xml_path.stem}.JPEG"
            if not image_path.exists():
                continue
            root = ET.parse(xml_path).getroot()
            size = root.find("./size")
            width = int(size.findtext("width", "1"))
            height = int(size.findtext("height", "1"))
            boxes = []
            labels = []
            for obj in root.findall("./object"):
                name = obj.findtext("name")
                if name not in class_to_idx:
                    continue
                bbox = obj.find("./bndbox")
                x1 = float(bbox.findtext("xmin", "0"))
                y1 = float(bbox.findtext("ymin", "0"))
                x2 = float(bbox.findtext("xmax", "0"))
                y2 = float(bbox.findtext("ymax", "0"))
                if (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(class_to_idx[name])
            if boxes:
                samples.append({"image_id": xml_path.stem, "image_path": str(image_path), "width": width, "height": height, "boxes": np.asarray(
                    boxes, dtype=np.float32), "labels": np.asarray(labels, dtype=np.int32), "file_name": image_path.name, "split": split})
    return samples


def collect_voc_detection_samples(voc_root, split):
    if voc_root is None:
        raise RuntimeError("VOC_ROOT not found")
    samples = []
    for record in voc_records(voc_root, split, MIN_BOX_SIZE, background=True):
        image = cv2.imread(record["image_path"])
        if image is None:
            continue
        height, width = image.shape[:2]
        boxes = record["boxes"] * \
            np.asarray([width, height, width, height], dtype=np.float32)
        samples.append({**record, "boxes": boxes.astype(np.float32), "width": width, "height": height,
                        "file_name": os.path.basename(record["image_path"]), "split": split})
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
        dataset_name="coco",
    ):
        self.dataset_name = dataset_name
        self.coco_root = coco_root
        self.split = split
        self.batch_size = batch_size
        self.input_size = tuple(input_size)
        self.max_gt_boxes = max_gt_boxes
        self.augment = augment
        self.shuffle = shuffle
        if dataset_name == "imagenet":
            self.samples = collect_imagenet_detection_samples(
                coco_root, "train" if split == "train" else "val")
        elif dataset_name == "voc":
            self.samples = collect_voc_detection_samples(coco_root, split)
        else:
            self.samples = collect_coco_samples(
                coco_root=self.coco_root, split=self.split)
        self.samples = cached_records("FasterRCNN", dataset_name, split, coco_root, {"input_size": self.input_size, "max_gt_boxes": self.max_gt_boxes, "anchor_scales": ANCHOR_SCALES, "anchor_ratios": ANCHOR_RATIOS, "rpn": [
                                      RPN_PRE_NMS_TOPK, RPN_POST_NMS_TOPK, RPN_NMS_IOU_THRESHOLD, RPN_POSITIVE_IOU_THRESHOLD, RPN_NEGATIVE_IOU_THRESHOLD, RPN_SAMPLES_PER_IMAGE, RPN_POSITIVE_FRACTION], "roi": [ROI_SAMPLES_PER_IMAGE, ROI_POSITIVE_FRACTION, ROI_POSITIVE_IOU_THRESHOLD, ROI_NEGATIVE_IOU_THRESHOLD, ROI_POOL_SIZE]}, self.samples, self._prepare_cached_records)
        print(
            f"[FasterRCNNDataset] dataset={self.dataset_name} split={self.split} samples={len(self.samples)}")

    def _prepare_cached_records(self, records):
        prepared = []
        for sample in records:
            boxes, labels = sample["boxes"], sample["labels"]
            valid = labels > 0
            boxes, labels = boxes[valid], labels[valid]
            keep = ((boxes[:, 2] - boxes[:, 0]) >=
                    MIN_BOX_SIZE) & ((boxes[:, 3] - boxes[:, 1]) >= MIN_BOX_SIZE)
            boxes, labels = boxes[keep], labels[keep]
            if len(boxes) == 0:
                continue
            image_h, image_w = max(float(sample["height"]), 1.0), max(
                float(sample["width"]), 1.0)
            boxes = boxes.astype(np.float32).copy()
            boxes[:, [0, 2]] /= image_w
            boxes[:, [1, 3]] /= image_h
            boxes = boxes[:, [1, 0, 3, 2]]
            padded_boxes = np.zeros((self.max_gt_boxes, 4), dtype=np.float32)
            padded_labels = np.zeros((self.max_gt_boxes,), dtype=np.int32)
            valid_mask = np.zeros((self.max_gt_boxes,), dtype=np.float32)
            count = min(len(boxes), self.max_gt_boxes)
            padded_boxes[:count], padded_labels[:count], valid_mask[:
                                                                    count] = boxes[:count], labels[:count], 1.0
            prepared.append(dict(sample, cache_gt_boxes=padded_boxes,
                            cache_gt_labels=padded_labels, cache_valid_mask=valid_mask))
        return prepared

    def _generator(self):
        indices = tf.range(len(self.samples)).numpy().tolist()
        if self.shuffle:
            import random
            random.shuffle(indices)
        for idx in indices:
            sample = self.samples[idx]
            if "cache_gt_boxes" in sample:
                yield (sample["image_path"].encode(), sample["cache_gt_boxes"], sample["cache_gt_labels"], sample["cache_valid_mask"])
                continue
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

            padded_boxes = tf.zeros(
                (self.max_gt_boxes, 4), dtype=tf.float32).numpy()
            padded_labels = tf.zeros(
                (self.max_gt_boxes,), dtype=tf.int32).numpy()
            # MirroredStrategy reduces each dataset component across replicas;
            # use a numeric mask because collective AddN does not support bool.
            valid_mask = tf.zeros((self.max_gt_boxes,),
                                  dtype=tf.float32).numpy()
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
            tf.TensorSpec(shape=(self.max_gt_boxes,), dtype=tf.float32),
        )
        dataset = tf.data.Dataset.from_generator(
            self._generator, output_signature=output_signature)
        if self.shuffle:
            dataset = dataset.shuffle(
                self.batch_size * 16, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(
            self._process, num_parallel_calls=tf.data.AUTOTUNE)
        # The custom Faster R-CNN train_step iterates over the local batch in
        # Python.  `drop_remainder=True` therefore gives tf.data a fixed global
        # batch signature, which MirroredStrategy can split into fixed local
        # batches instead of tracing it as an unknown dimension.
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return configure_dataset_pipeline(dataset, deterministic=not self.shuffle)


def _imagenet_class_names(imagenet_root=IMAGENET_ROOT):
    train_dir = Path(IMAGENET_TRAIN_DIR or "")
    if imagenet_root is None or not train_dir.is_dir():
        return []
    return sorted(path.name for path in train_dir.iterdir() if path.is_dir())


def _imagenet_val_label(xml_path):
    root = ET.parse(xml_path).getroot()
    node = root.find("./object/name")
    return node.text if node is not None else None


def collect_imagenet_samples(imagenet_root=IMAGENET_ROOT, split="train"):
    if imagenet_root is None:
        raise RuntimeError("IMAGENET_ROOT not found")
    class_names = _imagenet_class_names(imagenet_root)
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    samples = []
    if split == "train":
        for class_name in class_names:
            class_dir = Path(IMAGENET_TRAIN_DIR) / class_name
            for image_path in sorted(class_dir.glob("*.JPEG")):
                samples.append({"image_path": str(image_path),
                               "label": class_to_idx[class_name]})
    else:
        val_dir = Path(IMAGENET_VAL_DIR or "")
        ann_dir = Path(IMAGENET_VAL_ANN_DIR or "")
        for image_path in sorted(val_dir.glob("*.JPEG")):
            class_name = _imagenet_val_label(
                ann_dir / f"{image_path.stem}.xml")
            if class_name in class_to_idx:
                samples.append({"image_path": str(image_path),
                               "label": class_to_idx[class_name]})
    return {"num_classes": len(class_names), "class_names": class_names}, samples


def collect_coco_classification_samples(coco_root=COCO_ROOT, split=TRAIN_SPLIT):
    detection_samples = collect_coco_samples(coco_root=coco_root, split=split)
    samples = []
    for sample in detection_samples:
        if len(sample["labels"]) == 0:
            continue
        samples.append(
            {"image_path": sample["image_path"], "label": int(sample["labels"][0]) - 1})
    return {"num_classes": NUM_CLASSES - 1, "class_names": COCO_CLASSES[1:]}, samples


class COCOClassificationDataset:
    def __init__(self, coco_root=COCO_ROOT, split=TRAIN_SPLIT, batch_size=BATCH_SIZE, augment=False, shuffle=True):
        self.meta, self.samples = collect_coco_classification_samples(
            coco_root=coco_root, split=split)
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle

    def _generator(self):
        indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(indices)
        for idx in indices:
            sample = self.samples[idx]
            yield sample["image_path"].encode(), np.int32(sample["label"])

    def _process(self, image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, INPUT_SIZE)
        if self.augment:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.08)
            image = tf.image.random_contrast(image, 0.9, 1.1)
            image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    def build(self):
        dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )
        if self.shuffle:
            dataset = dataset.shuffle(
                self.batch_size * 8, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(
            self._process, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return configure_dataset_pipeline(dataset, deterministic=not self.shuffle)

    @property
    def steps_per_epoch(self):
        return max(1, int(np.ceil(len(self.samples) / self.batch_size)))


class ImageNetClassificationDataset:
    def __init__(self, imagenet_root=IMAGENET_ROOT, split="train", batch_size=IMAGENET_BATCH_SIZE, augment=False, shuffle=True):
        self.meta, self.samples = collect_imagenet_samples(
            imagenet_root=imagenet_root, split=split)
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle

    def _generator(self):
        indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(indices)
        for idx in indices:
            sample = self.samples[idx]
            yield sample["image_path"].encode(), np.int32(sample["label"])

    def _process(self, image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, INPUT_SIZE)
        if self.augment:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.08)
            image = tf.image.random_contrast(image, 0.9, 1.1)
            image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    def build(self):
        dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )
        if self.shuffle:
            dataset = dataset.shuffle(
                self.batch_size * 8, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(
            self._process, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return configure_dataset_pipeline(dataset, deterministic=not self.shuffle)

    @property
    def steps_per_epoch(self):
        return max(1, int(np.ceil(len(self.samples) / self.batch_size)))


def build_train_dataset(task_name="detector", dataset_name="coco", root=None, split=None, batch_size=None):
    if task_name == "classifier":
        if dataset_name == "imagenet":
            dataset = ImageNetClassificationDataset(
                imagenet_root=root or IMAGENET_ROOT, split=split or "train", augment=True, shuffle=True)
        else:
            dataset = COCOClassificationDataset(
                coco_root=root or COCO_ROOT, split=split or TRAIN_SPLIT, batch_size=BATCH_SIZE, augment=True, shuffle=True)
        return dataset, dataset.build()
    dataset = FasterRCNNDataset(coco_root=root or (IMAGENET_ROOT if dataset_name == "imagenet" else VOC_ROOT if dataset_name == "voc" else COCO_ROOT), split=split or (
        "train" if dataset_name in {"imagenet", "voc"} else TRAIN_SPLIT), batch_size=batch_size or BATCH_SIZE, augment=True, shuffle=True, dataset_name=dataset_name)
    return dataset, dataset.build()


def build_val_dataset(task_name="detector", dataset_name="coco", root=None, split=None, batch_size=None):
    if task_name == "classifier":
        if dataset_name == "imagenet":
            dataset = ImageNetClassificationDataset(
                imagenet_root=root or IMAGENET_ROOT, split=split or "val", augment=False, shuffle=False)
        else:
            dataset = COCOClassificationDataset(
                coco_root=root or COCO_ROOT, split=split or VAL_SPLIT, batch_size=BATCH_SIZE, augment=False, shuffle=False)
        return dataset, dataset.build()
    dataset = FasterRCNNDataset(coco_root=root or (IMAGENET_ROOT if dataset_name == "imagenet" else VOC_ROOT if dataset_name == "voc" else COCO_ROOT), split=split or (
        "val" if dataset_name in {"imagenet", "voc"} else VAL_SPLIT), batch_size=batch_size or BATCH_SIZE, augment=False, shuffle=False, dataset_name=dataset_name)
    return dataset, dataset.build()
