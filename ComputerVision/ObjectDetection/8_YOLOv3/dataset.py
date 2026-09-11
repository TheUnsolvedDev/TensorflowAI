from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO

from config import ANCHORS, BATCH_SIZE, COCO_ROOT, IMAGENET_BATCH_SIZE, IMAGENET_ROOT, IMAGENET_TRAIN_ANN_DIR, IMAGENET_TRAIN_DIR, IMAGENET_VAL_ANN_DIR, IMAGENET_VAL_DIR, INPUT_SIZE, MIN_BOX_SIZE, SCALES, TRAIN_SPLIT, VAL_SPLIT
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
COCO_ID_TO_IDX = {category_id: index for index, category_id in enumerate(COCO_CATEGORY_IDS)}
NUM_CLASSES = len(COCO_CATEGORY_IDS)
IMAGENET_DET_NUM_CLASSES = 1000


def collect_coco_samples(coco_root, split):
    annotation_file = f"{coco_root}/annotations/instances_{split}.json"
    image_dir = f"{coco_root}/{split}"
    coco = COCO(annotation_file)
    annotations_by_image = defaultdict(list)
    for annotation in coco.dataset["annotations"]:
        if annotation.get("iscrowd", 0) == 1:
            continue
        label = COCO_ID_TO_IDX.get(annotation["category_id"])
        if label is None:
            continue
        annotations_by_image[annotation["image_id"]].append(annotation)
    samples = []
    for image_id in coco.getImgIds():
        image_info = coco.loadImgs(image_id)[0]
        image_path = f"{image_dir}/{image_info['file_name']}"
        width = float(image_info["width"])
        height = float(image_info["height"])
        boxes = []
        labels = []
        for annotation in annotations_by_image.get(image_id, []):
            x, y, w, h = annotation["bbox"]
            if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                continue
            boxes.append([x / width, y / height, (x + w) / width, (y + h) / height])
            labels.append(COCO_ID_TO_IDX[annotation["category_id"]])
        if boxes:
            samples.append({"image_id": image_id, "image_path": image_path, "boxes": np.asarray(boxes, dtype=np.float32), "labels": np.asarray(labels, dtype=np.int32)})
    return samples


def wh_iou(box_wh, anchors_wh):
    inter = np.minimum(box_wh[0], anchors_wh[:, 0]) * np.minimum(box_wh[1], anchors_wh[:, 1])
    union = box_wh[0] * box_wh[1] + anchors_wh[:, 0] * anchors_wh[:, 1] - inter
    return inter / np.maximum(union, 1e-6)


def detection_num_classes(dataset_name):
    return IMAGENET_DET_NUM_CLASSES if dataset_name == "imagenet" else len(VOC_CLASSES) if dataset_name == "voc" else NUM_CLASSES


def encode_targets(boxes, labels, num_classes):
    targets = [np.zeros((scale, scale, 3, num_classes + 5), dtype=np.float32) for scale in SCALES]
    flat_anchors = np.asarray([anchor for group in ANCHORS for anchor in group], dtype=np.float32) / INPUT_SIZE[0]
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        cx = np.clip((x1 + x2) * 0.5, 0.0, 0.9999)
        cy = np.clip((y1 + y2) * 0.5, 0.0, 0.9999)
        bw = np.clip(x2 - x1, 1e-6, 1.0)
        bh = np.clip(y2 - y1, 1e-6, 1.0)
        best_anchor = int(np.argmax(wh_iou((bw, bh), flat_anchors)))
        scale_idx = best_anchor // 3
        anchor_idx = best_anchor % 3
        scale = SCALES[scale_idx]
        gx = min(int(cx * scale), scale - 1)
        gy = min(int(cy * scale), scale - 1)
        targets[scale_idx][gy, gx, anchor_idx, :num_classes] = 0.0
        targets[scale_idx][gy, gx, anchor_idx, label] = 1.0
        targets[scale_idx][gy, gx, anchor_idx, num_classes: num_classes + 5] = [1.0, cx * scale - gx, cy * scale - gy, bw, bh]
    return tuple(targets)


def _parse_imagenet_detection_annotation(xml_path, class_to_idx):
    root = ET.parse(xml_path).getroot()
    size = root.find("./size")
    width = max(float(size.findtext("width", "1")), 1.0)
    height = max(float(size.findtext("height", "1")), 1.0)
    boxes = []
    labels = []
    for obj in root.findall("./object"):
        class_name = obj.findtext("name")
        if class_name not in class_to_idx:
            continue
        bbox = obj.find("./bndbox")
        if bbox is None:
            continue
        x1 = float(bbox.findtext("xmin", "0")) / width
        y1 = float(bbox.findtext("ymin", "0")) / height
        x2 = float(bbox.findtext("xmax", "0")) / width
        y2 = float(bbox.findtext("ymax", "0")) / height
        if (x2 - x1) * width < MIN_BOX_SIZE or (y2 - y1) * height < MIN_BOX_SIZE:
            continue
        boxes.append([np.clip(x1, 0.0, 1.0), np.clip(y1, 0.0, 1.0), np.clip(x2, 0.0, 1.0), np.clip(y2, 0.0, 1.0)])
        labels.append(class_to_idx[class_name])
    return boxes, labels


def collect_imagenet_detection_samples(imagenet_root=IMAGENET_ROOT, split="train"):
    if imagenet_root is None:
        raise RuntimeError("IMAGENET_ROOT not found")
    class_names = _imagenet_class_names(imagenet_root)
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    samples = []
    if split == "train":
        ann_root = Path(IMAGENET_TRAIN_ANN_DIR or "")
        image_root = Path(IMAGENET_TRAIN_DIR or "")
        for class_name in class_names:
            for xml_path in sorted((ann_root / class_name).glob("*.xml")):
                image_path = image_root / class_name / f"{xml_path.stem}.JPEG"
                if not image_path.exists():
                    continue
                boxes, labels = _parse_imagenet_detection_annotation(xml_path, class_to_idx)
                if boxes:
                    samples.append({"image_path": str(image_path), "boxes": np.asarray(boxes, dtype=np.float32), "labels": np.asarray(labels, dtype=np.int32)})
    else:
        ann_root = Path(IMAGENET_VAL_ANN_DIR or "")
        image_root = Path(IMAGENET_VAL_DIR or "")
        for xml_path in sorted(ann_root.glob("*.xml")):
            image_path = image_root / f"{xml_path.stem}.JPEG"
            if not image_path.exists():
                continue
            boxes, labels = _parse_imagenet_detection_annotation(xml_path, class_to_idx)
            if boxes:
                samples.append({"image_path": str(image_path), "boxes": np.asarray(boxes, dtype=np.float32), "labels": np.asarray(labels, dtype=np.int32)})
    return samples


class YOLOv3Dataset:
    def __init__(self, coco_root=COCO_ROOT, split=TRAIN_SPLIT, batch_size=BATCH_SIZE, augment=False, shuffle=True, dataset_name="coco"):
        self.dataset_name = dataset_name
        if dataset_name == "imagenet":
            self.samples = collect_imagenet_detection_samples(coco_root, "train" if split == "train" else "val")
        elif dataset_name == "voc":
            self.samples = voc_records(coco_root, split, MIN_BOX_SIZE)
        else:
            self.samples = collect_coco_samples(coco_root, split)
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.num_classes = detection_num_classes(dataset_name)
        self.samples = cached_records("YOLOv3", dataset_name, split, coco_root, {"input_size": INPUT_SIZE, "scales": SCALES, "anchors": ANCHORS, "num_classes": self.num_classes}, self.samples, lambda records: [dict(sample, cache_targets=encode_targets(sample["boxes"], sample["labels"], self.num_classes)) for sample in records])
        print(f"[YOLOv3Dataset] dataset={dataset_name} split={split} samples={len(self.samples)}")

    def _generator(self):
        indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(indices)
        for idx in indices:
            sample = self.samples[idx]
            t_small, t_medium, t_large = sample.get("cache_targets", encode_targets(sample["boxes"], sample["labels"], self.num_classes))
            yield sample["image_path"].encode(), t_small, t_medium, t_large

    def _process(self, image_path, t_small, t_medium, t_large):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, INPUT_SIZE)
        if self.augment:
            image = tf.image.random_brightness(image, max_delta=0.08)
            image = tf.image.random_contrast(image, 0.9, 1.1)
            image = tf.clip_by_value(image, 0.0, 1.0)
        return image, {"yolo_s": t_small, "yolo_m": t_medium, "yolo_l": t_large}

    def build(self):
        output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(SCALES[0], SCALES[0], 3, self.num_classes + 5), dtype=tf.float32),
            tf.TensorSpec(shape=(SCALES[1], SCALES[1], 3, self.num_classes + 5), dtype=tf.float32),
            tf.TensorSpec(shape=(SCALES[2], SCALES[2], 3, self.num_classes + 5), dtype=tf.float32),
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

    @property
    def steps_per_epoch(self):
        return max(1, int(np.ceil(len(self.samples) / self.batch_size)))


class COCOInferenceDataset:
    def __init__(self, coco_root=COCO_ROOT, split=VAL_SPLIT):
        self.samples = collect_coco_samples(coco_root, split)


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


def collect_coco_classification_samples(coco_root=COCO_ROOT, split=TRAIN_SPLIT, refresh_cache=False):
    detection_samples = collect_coco_samples(coco_root=coco_root, split=split)
    samples = []
    for sample in detection_samples:
        if len(sample["labels"]) == 0:
            continue
        samples.append({"image_path": sample["image_path"], "label": int(sample["labels"][0])})
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
    dataset = YOLOv3Dataset(coco_root=root or (IMAGENET_ROOT if dataset_name == "imagenet" else COCO_ROOT), split=split or ("train" if dataset_name == "imagenet" else TRAIN_SPLIT), augment=True, shuffle=True, dataset_name=dataset_name)
    return dataset, dataset.build()


def build_val_dataset(task_name="detector", dataset_name="coco", root=None, split=None):
    if task_name == "classifier":
        if dataset_name == "imagenet":
            dataset = ImageNetClassificationDataset(imagenet_root=root or IMAGENET_ROOT, split=split or "val", augment=False, shuffle=False)
        else:
            dataset = COCOClassificationDataset(coco_root=root or COCO_ROOT, split=split or VAL_SPLIT, batch_size=BATCH_SIZE, augment=False, shuffle=False)
        return dataset, dataset.build()
    dataset = YOLOv3Dataset(coco_root=root or (IMAGENET_ROOT if dataset_name == "imagenet" else COCO_ROOT), split=split or ("val" if dataset_name == "imagenet" else VAL_SPLIT), augment=False, shuffle=False, dataset_name=dataset_name)
    return dataset, dataset.build()
