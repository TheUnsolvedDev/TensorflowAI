import os
import random
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO

from config import (
    BATCH_SIZE,
    COCO_ROOT,
    IMAGENET_BATCH_SIZE,
    IMAGENET_ROOT,
    IMAGENET_TRAIN_ANN_DIR,
    IMAGENET_TRAIN_DIR,
    IMAGENET_VAL_ANN_DIR,
    IMAGENET_VAL_DIR,
    INPUT_SIZE,
    MASK_SIZE,
    MAX_PROPOSALS,
    MIN_BOX_SIZE,
    NEGATIVE_IOU_THRESHOLD,
    POSITIVE_FRACTION,
    POSITIVE_IOU_THRESHOLD,
    ROIS_PER_IMAGE,
    TRAIN_SPLIT,
    USE_SELECTIVE_SEARCH,
    VAL_SPLIT,
    VOC_ROOT,
)
from utils import compute_iou, encode_box, generate_region_proposals
from cache_runtime import cached_records
from detection_adapters import VOC_CLASSES, voc_records


def configure_dataset_pipeline(dataset, deterministic):
    options = tf.data.Options()
    options.experimental_deterministic = deterministic
    options.experimental_slack = not deterministic
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
COCO_ID_TO_IDX = {category_id: index + 1 for index, category_id in enumerate(COCO_CATEGORY_IDS)}
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
        masks = []
        for annotation in annotations:
            x, y, width, height = annotation["bbox"]
            if width < MIN_BOX_SIZE or height < MIN_BOX_SIZE:
                continue
            boxes.append([x, y, x + width, y + height])
            labels.append(COCO_ID_TO_IDX[annotation["category_id"]])
            masks.append(coco.annToMask(annotation).astype(np.uint8))
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
                "masks": masks,
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
    ann_root = Path(IMAGENET_TRAIN_ANN_DIR or "") if split == "train" else Path(IMAGENET_VAL_ANN_DIR or "")
    image_root = Path(IMAGENET_TRAIN_DIR or "") if split == "train" else Path(IMAGENET_VAL_DIR or "")
    if split == "train":
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
                masks = []
                for obj in root.findall("./object"):
                    name = obj.findtext("name")
                    if name not in class_to_idx:
                        continue
                    bbox = obj.find("./bndbox")
                    x1 = int(float(bbox.findtext("xmin", "0")))
                    y1 = int(float(bbox.findtext("ymin", "0")))
                    x2 = int(float(bbox.findtext("xmax", "0")))
                    y2 = int(float(bbox.findtext("ymax", "0")))
                    if (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE:
                        continue
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_to_idx[name])
                    mask = np.zeros((height, width), dtype=np.uint8)
                    mask[max(y1,0):min(y2,height), max(x1,0):min(x2,width)] = 1
                    masks.append(mask)
                if boxes:
                    samples.append({"image_id": xml_path.stem, "image_path": str(image_path), "width": width, "height": height, "boxes": np.asarray(boxes, dtype=np.float32), "labels": np.asarray(labels, dtype=np.int32), "masks": masks, "file_name": image_path.name, "split": split})
    else:
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
            masks = []
            for obj in root.findall("./object"):
                name = obj.findtext("name")
                if name not in class_to_idx:
                    continue
                bbox = obj.find("./bndbox")
                x1 = int(float(bbox.findtext("xmin", "0")))
                y1 = int(float(bbox.findtext("ymin", "0")))
                x2 = int(float(bbox.findtext("xmax", "0")))
                y2 = int(float(bbox.findtext("ymax", "0")))
                if (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(class_to_idx[name])
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[max(y1,0):min(y2,height), max(x1,0):min(x2,width)] = 1
                masks.append(mask)
            if boxes:
                samples.append({"image_id": xml_path.stem, "image_path": str(image_path), "width": width, "height": height, "boxes": np.asarray(boxes, dtype=np.float32), "labels": np.asarray(labels, dtype=np.int32), "masks": masks, "file_name": image_path.name, "split": split})
    return samples


def crop_mask(mask, box, output_size):
    x1, y1, x2, y2 = np.round(box).astype(np.int32)
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    crop = mask[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(output_size, dtype=np.float32)
    return cv2.resize(crop.astype(np.float32), (output_size[1], output_size[0]), interpolation=cv2.INTER_NEAREST)


def collect_voc_detection_samples(voc_root, split):
    if voc_root is None:
        raise RuntimeError("VOC_ROOT not found")
    samples = []
    for record in voc_records(voc_root, split, MIN_BOX_SIZE, background=True):
        image = cv2.imread(record["image_path"])
        if image is None:
            continue
        height, width = image.shape[:2]
        boxes = record["boxes"] * np.asarray([width, height, width, height], dtype=np.float32)
        masks = []
        for x1, y1, x2, y2 in boxes:
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[max(int(y1), 0):min(int(np.ceil(y2)), height), max(int(x1), 0):min(int(np.ceil(x2)), width)] = 1
            masks.append(mask)
        samples.append({**record, "boxes": boxes.astype(np.float32), "masks": masks, "width": width, "height": height,
                        "file_name": os.path.basename(record["image_path"]), "split": split})
    return samples


class MaskRCNNDataset:
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
        dataset_name="coco",
    ):
        self.dataset_name = dataset_name
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
        if dataset_name == "imagenet":
            self.samples = collect_imagenet_detection_samples(coco_root, "train" if split == "train" else "val")
        elif dataset_name == "voc":
            self.samples = collect_voc_detection_samples(coco_root, split)
        else:
            self.samples = collect_coco_samples(coco_root=self.coco_root, split=self.split)
        self.samples, self.proposal_cache = cached_records("MaskRCNN", dataset_name, split, coco_root, {"input_size": self.input_size, "max_proposals": max_proposals, "rois_per_image": rois_per_image, "mask_size": MASK_SIZE, "positive_fraction": positive_fraction, "positive_iou": positive_iou_threshold, "negative_iou": negative_iou_threshold}, self.samples)
        print(f"[MaskRCNNDataset] dataset={self.dataset_name} split={self.split} samples={len(self.samples)} (cached metadata; randomized ROI sampling)")

    def _sample_indices(self, indices, count):
        if len(indices) == 0 or count <= 0:
            return np.array([], dtype=np.int32)
        replace = len(indices) < count
        return np.random.choice(indices, size=count, replace=replace)

    def _build_sample(self, sample):
        proposals = self.proposal_cache.load(sample) if self.proposal_cache else None
        if proposals is None:
            image_bgr = cv2.imread(sample["image_path"])
            if image_bgr is None:
                return None
            proposals = generate_region_proposals(image_bgr, self.max_proposals, self.use_selective_search)
            if self.proposal_cache:
                self.proposal_cache.store(sample, proposals)
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
        mask_targets = []
        mask_weights = []

        for index in chosen:
            proposal = proposals[index].astype(np.float32)
            x1, y1, x2, y2 = proposal
            rois.append([y1 / image_h, x1 / image_w, y2 / image_h, x2 / image_w])
            if index in positive_lookup:
                gt_idx = matched_gt[index]
                labels.append(gt_labels[gt_idx])
                bbox_targets.append(encode_box(proposal, gt_boxes[gt_idx]))
                bbox_weights.append(1.0)
                mask_targets.append(crop_mask(sample["masks"][gt_idx], gt_boxes[gt_idx], MASK_SIZE))
                mask_weights.append(1.0)
            else:
                labels.append(0)
                bbox_targets.append(np.zeros((4,), dtype=np.float32))
                bbox_weights.append(0.0)
                mask_targets.append(np.zeros(MASK_SIZE, dtype=np.float32))
                mask_weights.append(0.0)

        return {
            "image_path": sample["image_path"],
            "rois": np.asarray(rois, dtype=np.float32),
            "labels": np.asarray(labels, dtype=np.int32),
            "bbox_targets": np.asarray(bbox_targets, dtype=np.float32),
            "bbox_weights": np.asarray(bbox_weights, dtype=np.float32),
            "mask_targets": np.asarray(mask_targets, dtype=np.float32)[..., None],
            "mask_weights": np.asarray(mask_weights, dtype=np.float32),
        }
    def _generator(self):
        indices = list(range(len(self.samples)))
        if self.shuffle:
            random.shuffle(indices)
        for idx in indices:
            payload = self._build_sample(self.samples[idx])
            if payload is None:
                continue
            yield (
                payload["image_path"].encode(),
                payload["rois"],
                payload["labels"],
                payload["bbox_targets"],
                payload["bbox_weights"],
                payload["mask_targets"],
                payload["mask_weights"],
            )

    def _process(self, image_path, rois, labels, bbox_targets, bbox_weights, mask_targets, mask_weights):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.input_size)
        if self.augment:
            image = tf.image.random_brightness(image, max_delta=0.08)
            image = tf.image.random_contrast(image, 0.9, 1.1)
            image = tf.clip_by_value(image, 0.0, 1.0)
        inputs = {"image": image, "rois": rois}
        targets = {
            "class_logits": labels,
            "bbox_regression": bbox_targets,
            "mask_logits": mask_targets,
        }
        sample_weights = {
            "class_logits": tf.ones_like(tf.cast(labels, tf.float32)),
            "bbox_regression": bbox_weights,
            "mask_logits": mask_weights,
        }
        return inputs, targets, sample_weights

    def build(self):
        output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(self.rois_per_image, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(self.rois_per_image,), dtype=tf.int32),
            tf.TensorSpec(shape=(self.rois_per_image, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(self.rois_per_image,), dtype=tf.float32),
            tf.TensorSpec(shape=(self.rois_per_image, MASK_SIZE[0], MASK_SIZE[1], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(self.rois_per_image,), dtype=tf.float32),
        )
        dataset = tf.data.Dataset.from_generator(self._generator, output_signature=output_signature)
        if self.shuffle:
            dataset = dataset.shuffle(self.batch_size * 8, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(self._process, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = configure_dataset_pipeline(dataset, deterministic=not self.shuffle)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


class COCOInferenceDataset:
    def __init__(self, coco_root=COCO_ROOT, split=VAL_SPLIT):
        self.samples = collect_coco_samples(coco_root=coco_root, split=split)


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
                samples.append({"image_path": str(image_path), "label": class_to_idx[class_name]})
    else:
        val_dir = Path(IMAGENET_VAL_DIR or "")
        ann_dir = Path(IMAGENET_VAL_ANN_DIR or "")
        for image_path in sorted(val_dir.glob("*.JPEG")):
            class_name = _imagenet_val_label(ann_dir / f"{image_path.stem}.xml")
            if class_name in class_to_idx:
                samples.append({"image_path": str(image_path), "label": class_to_idx[class_name]})
    return {"num_classes": len(class_names), "class_names": class_names}, samples


def collect_coco_classification_samples(coco_root=COCO_ROOT, split=TRAIN_SPLIT):
    detection_samples = collect_coco_samples(coco_root=coco_root, split=split)
    samples = []
    for sample in detection_samples:
        if len(sample["labels"]) == 0:
            continue
        samples.append({"image_path": sample["image_path"], "label": int(sample["labels"][0]) - 1})
    return {"num_classes": NUM_CLASSES - 1, "class_names": COCO_CLASSES[1:]}, samples


class COCOClassificationDataset:
    def __init__(self, coco_root=COCO_ROOT, split=TRAIN_SPLIT, batch_size=BATCH_SIZE, augment=False, shuffle=True):
        self.meta, self.samples = collect_coco_classification_samples(coco_root=coco_root, split=split)
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
            dataset = dataset.shuffle(self.batch_size * 8, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(self._process, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = configure_dataset_pipeline(dataset, deterministic=not self.shuffle)
        return dataset.prefetch(tf.data.AUTOTUNE)

    @property
    def steps_per_epoch(self):
        return max(1, int(np.ceil(len(self.samples) / self.batch_size)))


class ImageNetClassificationDataset:
    def __init__(self, imagenet_root=IMAGENET_ROOT, split="train", batch_size=IMAGENET_BATCH_SIZE, augment=False, shuffle=True):
        self.meta, self.samples = collect_imagenet_samples(imagenet_root=imagenet_root, split=split)
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
            dataset = dataset.shuffle(self.batch_size * 8, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(self._process, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = configure_dataset_pipeline(dataset, deterministic=not self.shuffle)
        return dataset.prefetch(tf.data.AUTOTUNE)

    @property
    def steps_per_epoch(self):
        return max(1, int(np.ceil(len(self.samples) / self.batch_size)))


def build_train_dataset(task_name="detector", dataset_name="coco", root=None, split=None):
    if task_name == "classifier":
        if dataset_name == "imagenet":
            dataset = ImageNetClassificationDataset(imagenet_root=root or IMAGENET_ROOT, split=split or "train", augment=True, shuffle=True)
        else:
            dataset = COCOClassificationDataset(coco_root=root or COCO_ROOT, split=split or TRAIN_SPLIT, batch_size=BATCH_SIZE, augment=True, shuffle=True)
        return dataset, dataset.build()
    dataset = MaskRCNNDataset(coco_root=root or (IMAGENET_ROOT if dataset_name == "imagenet" else VOC_ROOT if dataset_name == "voc" else COCO_ROOT), split=split or ("train" if dataset_name in {"imagenet", "voc"} else TRAIN_SPLIT), augment=True, shuffle=True, dataset_name=dataset_name)
    return dataset, dataset.build()


def build_val_dataset(task_name="detector", dataset_name="coco", root=None, split=None):
    if task_name == "classifier":
        if dataset_name == "imagenet":
            dataset = ImageNetClassificationDataset(imagenet_root=root or IMAGENET_ROOT, split=split or "val", augment=False, shuffle=False)
        else:
            dataset = COCOClassificationDataset(coco_root=root or COCO_ROOT, split=split or VAL_SPLIT, batch_size=BATCH_SIZE, augment=False, shuffle=False)
        return dataset, dataset.build()
    dataset = MaskRCNNDataset(coco_root=root or (IMAGENET_ROOT if dataset_name == "imagenet" else VOC_ROOT if dataset_name == "voc" else COCO_ROOT), split=split or ("val" if dataset_name in {"imagenet", "voc"} else VAL_SPLIT), augment=False, shuffle=False, dataset_name=dataset_name)
    return dataset, dataset.build()
