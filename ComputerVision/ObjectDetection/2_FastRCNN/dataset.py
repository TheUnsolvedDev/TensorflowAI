import os
import zlib
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO

from config import (
    BATCH_SIZE,
    CACHE_ENABLED,
    CACHE_MAX_SAMPLES,
    CACHE_REBUILD,
    COCO_ROOT,
    IMAGENET_BATCH_SIZE,
    IMAGENET_ROOT,
    IMAGENET_TRAIN_ANN_DIR,
    IMAGENET_TRAIN_DIR,
    IMAGENET_VAL_ANN_DIR,
    IMAGENET_VAL_DIR,
    INPUT_SIZE,
    GT_JITTER_CENTER_STD,
    GT_JITTER_SCALE_STD,
    GT_JITTERS_PER_BOX,
    MAX_PROPOSALS,
    MIN_BOX_SIZE,
    NEGATIVE_IOU_THRESHOLD,
    POSITIVE_FRACTION,
    POSITIVE_IOU_THRESHOLD,
    ROIS_PER_IMAGE,
    USE_SELECTIVE_SEARCH,
    VAL_SPLIT,
    TRAIN_SPLIT,
    VOC_ROOT,
)
from utils import compute_iou, generate_region_proposals
from cache_runtime import PersistentProposalCache


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
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


class BaseDetectionDataset:
    """Folder-local adapter contract for a labelled detection dataset."""

    name = None
    root = None
    class_names = ()

    def __init__(self, root=None):
        self.root = root or self.root
        if not self.root or not os.path.isdir(self.root):
            raise RuntimeError(
                f"{self.name} dataset root not found: {self.root}")

    @property
    def num_classes(self):
        return len(self.class_names)

    def split_name(self, phase):
        raise NotImplementedError

    def records(self, phase):
        raise NotImplementedError

    def annotation_sources(self, phase):
        """Files/directories which identify the labelled split for cache reuse."""
        return []


class COCODetectionDataset(BaseDetectionDataset):
    name = "coco"
    root = COCO_ROOT
    class_names = COCO_CLASSES

    def split_name(self, phase):
        return TRAIN_SPLIT if phase == "train" else VAL_SPLIT

    def records(self, phase):
        return collect_coco_samples(self.root, self.split_name(phase))

    def annotation_sources(self, phase):
        return [os.path.join(self.root, "annotations", f"instances_{self.split_name(phase)}.json")]


class VOCDataset(BaseDetectionDataset):
    name = "voc"
    root = VOC_ROOT
    class_names = VOC_CLASSES

    def split_name(self, phase):
        # VOC2012 test annotations are not public, so it is intentionally absent.
        return "train" if phase == "train" else "val"

    def records(self, phase):
        split = self.split_name(phase)
        split_file = Path(self.root) / "ImageSets" / "Main" / f"{split}.txt"
        annotation_dir = Path(self.root) / "Annotations"
        image_dir = Path(self.root) / "JPEGImages"
        if not split_file.is_file():
            raise RuntimeError(f"VOC split file not found: {split_file}")
        class_to_idx = {name: index for index,
                        name in enumerate(self.class_names)}
        samples = []
        for image_id in split_file.read_text(encoding="utf-8").splitlines():
            image_id = image_id.strip()
            if not image_id:
                continue
            xml_path, image_path = annotation_dir / \
                f"{image_id}.xml", image_dir / f"{image_id}.jpg"
            if not xml_path.is_file() or not image_path.is_file():
                continue
            root = ET.parse(xml_path).getroot()
            width = int(root.findtext("./size/width", "1"))
            height = int(root.findtext("./size/height", "1"))
            boxes, labels, areas = [], [], []
            for obj in root.findall("./object"):
                if obj.findtext("difficult", "0") == "1":
                    continue
                label = class_to_idx.get(obj.findtext("name", ""))
                bbox = obj.find("./bndbox")
                if label is None or bbox is None:
                    continue
                x1, y1 = float(bbox.findtext("xmin", "0")), float(
                    bbox.findtext("ymin", "0"))
                x2, y2 = float(bbox.findtext("xmax", "0")), float(
                    bbox.findtext("ymax", "0"))
                if x2 - x1 < MIN_BOX_SIZE or y2 - y1 < MIN_BOX_SIZE:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(label)
                areas.append((x2 - x1) * (y2 - y1))
            if boxes:
                samples.append({"image_id": image_id, "image_path": str(image_path), "width": width, "height": height,
                                "boxes": np.asarray(boxes, np.float32), "labels": np.asarray(labels, np.int32),
                                "areas": np.asarray(areas, np.float32), "file_name": image_path.name, "split": split})
        return samples

    def annotation_sources(self, phase):
        split = self.split_name(phase)
        return [str(Path(self.root) / "ImageSets" / "Main" / f"{split}.txt"), str(Path(self.root) / "Annotations")]


class ImageNetLocalizationDataset(BaseDetectionDataset):
    name = "imagenet"
    root = IMAGENET_ROOT

    def __init__(self, root=None):
        super().__init__(root)
        self.class_names = ["background", *_imagenet_class_names(self.root)]

    def split_name(self, phase):
        return "train" if phase == "train" else "val"

    def records(self, phase):
        return collect_imagenet_detection_samples(self.root, self.split_name(phase))

    def annotation_sources(self, phase):
        return [str(Path(self.root) / "Annotations" / "CLS-LOC" / self.split_name(phase))]


DATASET_ADAPTERS = {"coco": COCODetectionDataset,
                    "voc": VOCDataset, "imagenet": ImageNetLocalizationDataset}


def resolve_detection_dataset(name, root=None):
    try:
        return DATASET_ADAPTERS[name](root=root)
    except KeyError as error:
        raise ValueError(
            f"Unsupported detection dataset {name!r}; choose from {sorted(DATASET_ADAPTERS)}") from error


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


def collect_imagenet_detection_samples(imagenet_root, split):
    if imagenet_root is None:
        raise RuntimeError("IMAGENET_ROOT not found")
    class_names = _imagenet_class_names(imagenet_root)
    class_to_idx = {name: index + 1 for index, name in enumerate(class_names)}
    samples = []
    root_path = Path(imagenet_root)
    if split == "train":
        ann_root = root_path / "Annotations" / "CLS-LOC" / "train"
        image_root = root_path / "Data" / "CLS-LOC" / "train"
        iterator = ((ann_root / class_name).glob("*.xml")
                    for class_name in class_names)
        for class_name, xml_paths in zip(class_names, iterator):
            for xml_path in sorted(xml_paths):
                image_path = image_root / class_name / f"{xml_path.stem}.JPEG"
                if not image_path.exists():
                    continue
                root = ET.parse(xml_path).getroot()
                size = root.find("./size")
                width = int(size.findtext("width", "1"))
                height = int(size.findtext("height", "1"))
                boxes = []
                labels = []
                areas = []
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
                    areas.append((x2 - x1) * (y2 - y1))
                if boxes:
                    samples.append({"image_id": xml_path.stem, "image_path": str(image_path), "width": width, "height": height, "boxes": np.asarray(
                        boxes, dtype=np.float32), "labels": np.asarray(labels, dtype=np.int32), "areas": np.asarray(areas, dtype=np.float32), "file_name": image_path.name, "split": split})
    else:
        ann_root = root_path / "Annotations" / "CLS-LOC" / "val"
        image_root = root_path / "Data" / "CLS-LOC" / "val"
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
            areas = []
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
                areas.append((x2 - x1) * (y2 - y1))
            if boxes:
                samples.append({"image_id": xml_path.stem, "image_path": str(image_path), "width": width, "height": height, "boxes": np.asarray(
                    boxes, dtype=np.float32), "labels": np.asarray(labels, dtype=np.int32), "areas": np.asarray(areas, dtype=np.float32), "file_name": image_path.name, "split": split})
    return samples


class FastRCNNDataset:
    def __init__(
        self,
        dataset_root=None,
        split=None,
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
        cache_enabled=CACHE_ENABLED,
        cache_rebuild=CACHE_REBUILD,
        cache_max_samples=CACHE_MAX_SAMPLES,
        cache_workers=None,
    ):
        self.adapter = resolve_detection_dataset(
            dataset_name, root=dataset_root)
        self.dataset_name = self.adapter.name
        self.dataset_root = self.adapter.root
        self.split = split or self.adapter.split_name("train")
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
        if self.split == self.adapter.split_name("train"):
            phase = "train"
        elif self.split == self.adapter.split_name("val"):
            phase = "val"
        else:
            raise ValueError(
                f"{self.dataset_name} has no labelled detection split {self.split!r}; "
                f"use {self.adapter.split_name('train')!r} or {self.adapter.split_name('val')!r}."
            )
        self.samples = self.adapter.records(phase)
        self.class_names = self.adapter.class_names
        self.num_classes = self.adapter.num_classes
        if not self.samples:
            raise RuntimeError(
                f"No labelled {self.dataset_name} records found for split {self.split}")
        max_jittered = max(
            (len(sample["boxes"]) for sample in self.samples), default=0) * GT_JITTERS_PER_BOX
        self.cache_manager = PersistentProposalCache("FastRCNN", self.dataset_name, self.split, self.dataset_root, {"input_size": self.input_size, "max_proposals": max_proposals, "max_jittered_proposals": max_jittered, "rois_per_image": rois_per_image, "positive_fraction": positive_fraction, "positive_iou": positive_iou_threshold, "negative_iou": negative_iou_threshold, "selective_search": use_selective_search, "gt_jitter": [
                                                     GT_JITTERS_PER_BOX, GT_JITTER_CENTER_STD, GT_JITTER_SCALE_STD], "annotation_sources": self.adapter.annotation_sources(phase)}, self.samples, enabled=cache_enabled, rebuild=cache_rebuild, max_samples=cache_max_samples, workers=cache_workers)
        self.proposal_cache = None
        print(
            f"[FastRCNNDataset] dataset={self.dataset_name} split={self.split} samples={len(self.samples)}")

    def _jitter_gt_boxes(self, sample):
        """Create deterministic, near-GT positive candidates without exact GT boxes."""
        seed = zlib.crc32(str(sample["image_id"]).encode("utf-8"))
        rng = np.random.default_rng(seed)
        boxes = sample["boxes"].astype(np.float32)
        image_h = float(sample["height"])
        image_w = float(sample["width"])
        proposals = []
        for x1, y1, x2, y2 in boxes:
            width = max(x2 - x1, 1.0)
            height = max(y2 - y1, 1.0)
            center_x = x1 + 0.5 * width
            center_y = y1 + 0.5 * height
            for _ in range(GT_JITTERS_PER_BOX):
                scale_x = np.exp(rng.normal(0.0, GT_JITTER_SCALE_STD))
                scale_y = np.exp(rng.normal(0.0, GT_JITTER_SCALE_STD))
                jitter_x = rng.normal(0.0, GT_JITTER_CENTER_STD * width)
                jitter_y = rng.normal(0.0, GT_JITTER_CENTER_STD * height)
                proposal_w = np.clip(width * scale_x, MIN_BOX_SIZE, image_w)
                proposal_h = np.clip(height * scale_y, MIN_BOX_SIZE, image_h)
                proposal_x1 = np.clip(
                    center_x + jitter_x - 0.5 * proposal_w, 0.0, image_w - 1.0)
                proposal_y1 = np.clip(
                    center_y + jitter_y - 0.5 * proposal_h, 0.0, image_h - 1.0)
                proposal_x2 = np.clip(
                    proposal_x1 + proposal_w, proposal_x1 + 1.0, image_w)
                proposal_y2 = np.clip(
                    proposal_y1 + proposal_h, proposal_y1 + 1.0, image_h)
                proposals.append(
                    [proposal_x1, proposal_y1, proposal_x2, proposal_y2])
        if not proposals:
            return np.empty((0, 4), dtype=np.float32)
        proposals = np.asarray(proposals, dtype=np.float32)
        positive = compute_iou(proposals, boxes).max(
            axis=1) >= self.positive_iou_threshold
        return proposals[positive]

    def _build_proposals(self, sample, image_bgr):
        """Cold-cache-only proposal generation; warm training never calls it."""
        proposals = generate_region_proposals(
            image_bgr, self.max_proposals, self.use_selective_search)
        jittered_gt_proposals = self._jitter_gt_boxes(sample)
        proposals = np.concatenate(
            [proposals, jittered_gt_proposals], axis=0).astype(np.float32)
        proposals = np.unique(np.round(proposals).astype(
            np.int32), axis=0).astype(np.float32)

        return proposals

    def prepare_cache(self):
        self.proposal_cache = self.cache_manager.prepare(self._build_proposals)
        if self.proposal_cache is None:
            # Do not precompute an entire volatile split: --no-cache should let
            # model.fit begin immediately and generate proposals only on demand.
            print(
                "[cache] no-cache lazy mode; proposals will be generated per input batch")
        return self.proposal_cache

    def _lazy_proposals(self, image_path, sample_index):
        """Python callback for --no-cache; output keeps the fixed cache shape."""
        if isinstance(image_path, np.ndarray):
            image_path = image_path.item()
        path = image_path.decode(
            "utf-8") if isinstance(image_path, (bytes, np.bytes_)) else str(image_path)
        sample = self.samples[int(sample_index)]
        image = cv2.imread(path)
        if image is None:
            raise RuntimeError(f"Cannot read image: {path}")
        value = self._build_proposals(sample, image)[
            :self.cache_manager.capacity]
        if not len(value):
            raise RuntimeError(f"No proposals built for {path}")
        proposals = np.zeros((self.cache_manager.capacity, 4), np.float32)
        proposals[:len(value)] = value
        return proposals, np.int32(len(value))

    def _process_lazy(self, image_path, sample_index, boxes, labels, box_count, height, width):
        proposals, proposal_count = tf.numpy_function(
            self._lazy_proposals, [image_path, sample_index], [
                tf.float32, tf.int32]
        )
        proposals.set_shape((self.cache_manager.capacity, 4))
        proposal_count.set_shape(())
        return self._process(image_path, proposals, proposal_count, boxes, labels, box_count, height, width)

    def _sample_targets(self, proposals, proposal_count, boxes, labels, box_count, height, width):
        valid_p = tf.range(tf.shape(proposals)[0]) < proposal_count
        valid_g = tf.range(tf.shape(boxes)[0]) < box_count
        low = tf.maximum(proposals[:, None, :2], boxes[None, :, :2])
        high = tf.minimum(proposals[:, None, 2:], boxes[None, :, 2:])
        inter_hw = tf.maximum(high - low, 0.0)
        inter = inter_hw[..., 0] * inter_hw[..., 1]
        pa = tf.maximum(proposals[:, 2] - proposals[:, 0], 0.0) * \
            tf.maximum(proposals[:, 3] - proposals[:, 1], 0.0)
        ga = tf.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * \
            tf.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
        denom = pa[:, None] + ga[None, :] - inter
        iou = tf.where(denom > 0, inter / denom, tf.zeros_like(denom))
        iou = tf.where(valid_g[None, :], iou, -tf.ones_like(iou))
        max_iou = tf.reduce_max(iou, axis=1)
        matched = tf.argmax(iou, axis=1, output_type=tf.int32)
        positive = valid_p & (max_iou >= self.positive_iou_threshold)
        negative = valid_p & (max_iou < self.negative_iou_threshold)
        fallback = valid_p & (max_iou < self.positive_iou_threshold)
        npos = int(self.rois_per_image * self.positive_fraction)
        scores = tf.random.uniform(tf.shape(max_iou))
        pos_idx = tf.math.top_k(
            tf.where(positive, scores, -tf.ones_like(scores)), k=npos).indices
        neg_mask = tf.where(tf.reduce_any(negative), negative, fallback)
        neg_idx = tf.math.top_k(tf.where(neg_mask, tf.random.uniform(tf.shape(
            scores)), -tf.ones_like(scores)), k=self.rois_per_image - npos).indices
        chosen = tf.concat([pos_idx, neg_idx], axis=0)
        chosen_p = tf.gather(proposals, chosen)
        chosen_positive = tf.gather(positive, chosen)
        matched = tf.gather(matched, chosen)
        gt = tf.gather(boxes, matched)
        chosen_labels = tf.where(chosen_positive, tf.gather(
            labels, matched), tf.zeros((self.rois_per_image,), tf.int32))
        pw = tf.maximum(chosen_p[:, 2] - chosen_p[:, 0], 1.0)
        ph = tf.maximum(chosen_p[:, 3] - chosen_p[:, 1], 1.0)
        gw = tf.maximum(gt[:, 2] - gt[:, 0], 1.0)
        gh = tf.maximum(gt[:, 3] - gt[:, 1], 1.0)
        targets = tf.stack([(gt[:, 0] + .5 * gw - chosen_p[:, 0] - .5 * pw) / pw, (gt[:, 1] + .5 *
                           gh - chosen_p[:, 1] - .5 * ph) / ph, tf.math.log(gw / pw), tf.math.log(gh / ph)], axis=1)
        targets = tf.where(
            chosen_positive[:, None], targets, tf.zeros_like(targets))
        rois = tf.stack([chosen_p[:, 1] / height, chosen_p[:, 0] / width,
                        chosen_p[:, 3] / height, chosen_p[:, 2] / width], axis=1)
        return rois, chosen_labels, targets, tf.cast(chosen_positive, tf.float32)

    def _process(self, image_path, proposals, proposal_count, boxes, labels, box_count, height, width):
        rois, labels, bbox_targets, bbox_weights = self._sample_targets(
            proposals, proposal_count, boxes, labels, box_count, height, width)
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
        if self.proposal_cache is None:
            self.prepare_cache()
        max_boxes = max(len(sample["boxes"]) for sample in self.samples)
        boxes = np.zeros((len(self.samples), max_boxes, 4), np.float32)
        labels = np.zeros((len(self.samples), max_boxes), np.int32)
        box_counts = np.zeros(len(self.samples), np.int32)
        for index, sample in enumerate(self.samples):
            count = len(sample["boxes"])
            boxes[index, :count] = sample["boxes"]
            labels[index, :count] = sample["labels"]
            box_counts[index] = count
        paths = np.asarray([sample["image_path"] for sample in self.samples])
        heights = np.asarray([sample["height"]
                             for sample in self.samples], np.float32)
        widths = np.asarray([sample["width"]
                            for sample in self.samples], np.float32)
        # Keep the full proposal arrays and annotation tensors in host memory.
        # Without this explicit placement TensorFlow may materialize the large
        # cached proposal tensor on GPU:0 before MirroredStrategy can shard
        # batches, producing a large and unnecessary GPU-memory imbalance.
        with tf.device("/CPU:0"):
            if self.proposal_cache is None:
                dataset = tf.data.Dataset.from_tensor_slices((paths, np.arange(
                    len(self.samples), dtype=np.int32), boxes, labels, box_counts, heights, widths))
                process = self._process_lazy
            else:
                dataset = tf.data.Dataset.from_tensor_slices(
                    (paths, self.proposal_cache.proposals, self.proposal_cache.counts, boxes, labels, box_counts, heights, widths))
                process = self._process
        if self.shuffle:
            dataset = dataset.shuffle(
                self.batch_size * 32, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(process, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return configure_dataset_pipeline(dataset, deterministic=not self.shuffle)


def _imagenet_class_names(imagenet_root=IMAGENET_ROOT):
    train_dir = Path(imagenet_root) / "Data" / "CLS-LOC" / \
        "train" if imagenet_root else Path()
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
            class_dir = Path(imagenet_root) / "Data" / \
                "CLS-LOC" / "train" / class_name
            for image_path in sorted(class_dir.glob("*.JPEG")):
                samples.append({"image_path": str(image_path),
                               "label": class_to_idx[class_name]})
    else:
        val_dir = Path(imagenet_root) / "Data" / "CLS-LOC" / "val"
        ann_dir = Path(imagenet_root) / "Annotations" / "CLS-LOC" / "val"
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
        label_index = int(np.argmax(sample["areas"]))
        samples.append({"image_path": sample["image_path"], "label": int(
            sample["labels"][label_index]) - 1})
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


def build_train_dataset(task_name="detector", dataset_name="coco", root=None, split=None, **cache_options):
    if task_name == "classifier":
        if dataset_name == "imagenet":
            dataset = ImageNetClassificationDataset(
                imagenet_root=root or IMAGENET_ROOT, split=split or "train", augment=True, shuffle=True)
        else:
            dataset = COCOClassificationDataset(
                coco_root=root or COCO_ROOT, split=split or TRAIN_SPLIT, batch_size=BATCH_SIZE, augment=True, shuffle=True)
        return dataset, dataset.build()
    adapter = resolve_detection_dataset(dataset_name, root=root)
    dataset = FastRCNNDataset(dataset_root=adapter.root, split=split or adapter.split_name(
        "train"), augment=True, shuffle=True, dataset_name=dataset_name, **cache_options)
    return dataset, dataset.build()


def build_val_dataset(task_name="detector", dataset_name="coco", root=None, split=None, **cache_options):
    if task_name == "classifier":
        if dataset_name == "imagenet":
            dataset = ImageNetClassificationDataset(
                imagenet_root=root or IMAGENET_ROOT, split=split or "val", augment=False, shuffle=False)
        else:
            dataset = COCOClassificationDataset(
                coco_root=root or COCO_ROOT, split=split or VAL_SPLIT, batch_size=BATCH_SIZE, augment=False, shuffle=False)
        return dataset, dataset.build()
    adapter = resolve_detection_dataset(dataset_name, root=root)
    dataset = FastRCNNDataset(dataset_root=adapter.root, split=split or adapter.split_name(
        "val"), augment=False, shuffle=False, dataset_name=dataset_name, **cache_options)
    return dataset, dataset.build()
