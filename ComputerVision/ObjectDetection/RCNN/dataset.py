import os
import pickle
import random
from collections import defaultdict

import cv2
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO

from config import (
    BATCH_SIZE,
    COCO_ROOT,
    INPUT_SIZE,
    MAX_PROPOSALS,
    MIN_BOX_SIZE,
    NEGATIVE_IOU_THRESHOLD,
    POSITIVE_FRACTION,
    POSITIVE_IOU_THRESHOLD,
    PROJECT_DIR,
    TRAIN_PROPOSALS_PER_IMAGE,
    USE_SELECTIVE_SEARCH,
    VAL_PROPOSALS_PER_IMAGE,
    ensure_dir,
)
from utils import compute_iou, encode_box, generate_region_proposals


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
IDX_TO_COCO_ID = {index + 1: category_id for index, category_id in enumerate(COCO_CATEGORY_IDS)}
NUM_CLASSES = len(COCO_CLASSES)

CACHE_ROOT = ensure_dir(os.path.join(PROJECT_DIR, "cache", "coco2017_rcnn"))
METADATA_CACHE_DIR = ensure_dir(os.path.join(CACHE_ROOT, "metadata"))
TARGET_CACHE_DIR = ensure_dir(os.path.join(CACHE_ROOT, "targets"))


def coco_annotation_path(coco_root, split):
    return f"{coco_root}/annotations/instances_{split}.json"


def coco_image_dir(coco_root, split):
    return f"{coco_root}/{split}"


def _metadata_cache_path(split):
    return os.path.join(METADATA_CACHE_DIR, f"{split}_samples.pkl")


def collect_coco_samples(coco_root, split, refresh_cache=False):
    cache_path = _metadata_cache_path(split)
    if os.path.exists(cache_path) and not refresh_cache:
        with open(cache_path, "rb") as file:
            return pickle.load(file)

    annotation_file = coco_annotation_path(coco_root, split)
    image_dir = coco_image_dir(coco_root, split)
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

    with open(cache_path, "wb") as file:
        pickle.dump(samples, file, protocol=pickle.HIGHEST_PROTOCOL)

    return samples


class RCNNProposalDataset:
    def __init__(
        self,
        coco_root=COCO_ROOT,
        split="train2017",
        batch_size=BATCH_SIZE,
        input_size=INPUT_SIZE,
        proposals_per_image=TRAIN_PROPOSALS_PER_IMAGE,
        max_proposals=MAX_PROPOSALS,
        positive_fraction=POSITIVE_FRACTION,
        positive_iou_threshold=POSITIVE_IOU_THRESHOLD,
        negative_iou_threshold=NEGATIVE_IOU_THRESHOLD,
        augment=False,
        shuffle=True,
        use_selective_search=USE_SELECTIVE_SEARCH,
        refresh_cache=False,
    ):
        self.coco_root = coco_root
        self.split = split
        self.batch_size = batch_size
        self.input_size = tuple(input_size)
        self.proposals_per_image = proposals_per_image
        self.max_proposals = max_proposals
        self.positive_fraction = positive_fraction
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold = negative_iou_threshold
        self.augment = augment
        self.shuffle = shuffle
        self.use_selective_search = use_selective_search
        self.refresh_cache = refresh_cache
        self.target_cache_dir = ensure_dir(
            os.path.join(
                TARGET_CACHE_DIR,
                self.split,
                f"ppi_{self.proposals_per_image}_mp_{self.max_proposals}"
                f"_piou_{str(self.positive_iou_threshold).replace('.', '_')}"
                f"_niou_{str(self.negative_iou_threshold).replace('.', '_')}",
            )
        )

        self.samples = collect_coco_samples(
            coco_root=self.coco_root,
            split=self.split,
            refresh_cache=self.refresh_cache,
        )
        print(
            f"[RCNNDataset2] split={self.split} samples={len(self.samples)} "
            f"proposals_per_image={self.proposals_per_image} cache_dir={self.target_cache_dir}"
        )

    def _sample_indices(self, indices, count):
        if len(indices) == 0 or count <= 0:
            return np.array([], dtype=np.int32)
        replace = len(indices) < count
        return np.random.choice(indices, size=count, replace=replace)

    def _target_cache_path(self, image_id):
        return os.path.join(self.target_cache_dir, f"{image_id}.npz")

    def _build_image_target_cache(self, sample):
        cache_path = self._target_cache_path(sample["image_id"])
        if os.path.exists(cache_path) and not self.refresh_cache:
            return cache_path

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

        num_positive = int(self.proposals_per_image * self.positive_fraction)
        num_negative = self.proposals_per_image - num_positive

        chosen_positive = self._sample_indices(positive_indices, num_positive)
        chosen_negative = self._sample_indices(negative_indices, num_negative)
        if len(chosen_negative) == 0:
            chosen_negative = self._sample_indices(fallback_negative_indices, num_negative)

        if len(chosen_positive) == 0 and len(chosen_negative) == 0:
            return None

        chosen = np.concatenate([chosen_positive, chosen_negative], axis=0)
        positive_lookup = set(chosen_positive.tolist())

        proposal_boxes = []
        class_labels = []
        bbox_targets = []
        bbox_weights = []

        for index in chosen:
            proposal = proposals[index].astype(np.float32)
            is_positive = index in positive_lookup

            if is_positive:
                gt_idx = matched_gt[index]
                class_label = gt_labels[gt_idx]
                bbox_target = encode_box(proposal, gt_boxes[gt_idx])
                bbox_weight = 1.0
            else:
                class_label = 0
                bbox_target = np.zeros((4,), dtype=np.float32)
                bbox_weight = 0.0

            proposal_boxes.append(proposal)
            class_labels.append(np.int32(class_label))
            bbox_targets.append(bbox_target.astype(np.float32))
            bbox_weights.append(np.float32(bbox_weight))

        np.savez_compressed(
            cache_path,
            boxes=np.asarray(proposal_boxes, dtype=np.float32),
            labels=np.asarray(class_labels, dtype=np.int32),
            bbox_targets=np.asarray(bbox_targets, dtype=np.float32),
            bbox_weights=np.asarray(bbox_weights, dtype=np.float32),
        )
        return cache_path

    def _load_cached_records(self, sample):
        cache_path = self._build_image_target_cache(sample)
        if cache_path is None:
            return None

        payload = np.load(cache_path)
        boxes = payload["boxes"]
        labels = payload["labels"]
        bbox_targets = payload["bbox_targets"]
        bbox_weights = payload["bbox_weights"]

        if len(boxes) == 0:
            return None

        indices = np.arange(len(boxes))
        if self.shuffle:
            np.random.shuffle(indices)

        return {
            "image_path": sample["image_path"],
            "boxes": boxes[indices],
            "labels": labels[indices],
            "bbox_targets": bbox_targets[indices],
            "bbox_weights": bbox_weights[indices],
        }

    def _generator(self):
        indices = list(range(len(self.samples)))
        if self.shuffle:
            random.shuffle(indices)

        for idx in indices:
            sample = self.samples[idx]
            cached = self._load_cached_records(sample)
            if cached is None:
                continue

            image_path = cached["image_path"]
            for box, label, bbox_target, bbox_weight in zip(
                cached["boxes"],
                cached["labels"],
                cached["bbox_targets"],
                cached["bbox_weights"],
            ):
                yield (
                    image_path.encode(),
                    box.astype(np.float32),
                    np.int32(label),
                    bbox_target.astype(np.float32),
                    np.float32(bbox_weight),
                )

    def _process_record(self, image_path, box, class_label, bbox_target, bbox_weight):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        image_shape = tf.shape(image)
        image_height = tf.cast(image_shape[0], tf.float32)
        image_width = tf.cast(image_shape[1], tf.float32)

        x1, y1, x2, y2 = tf.unstack(box)
        normalized_box = tf.stack(
            [
                y1 / image_height,
                x1 / image_width,
                y2 / image_height,
                x2 / image_width,
            ]
        )
        normalized_box = tf.expand_dims(normalized_box, axis=0)
        crop = tf.image.crop_and_resize(
            image=tf.expand_dims(image, axis=0),
            boxes=normalized_box,
            box_indices=tf.constant([0], dtype=tf.int32),
            crop_size=self.input_size,
        )
        crop = tf.squeeze(crop, axis=0)

        if self.augment:
            crop = tf.image.random_flip_left_right(crop)
            crop = tf.image.random_brightness(crop, max_delta=0.08)
            crop = tf.image.random_contrast(crop, lower=0.9, upper=1.1)
            crop = tf.clip_by_value(crop, 0.0, 1.0)

        targets = {
            "class_logits": class_label,
            "bbox_regression": bbox_target,
        }
        sample_weights = {
            "class_logits": tf.constant(1.0, dtype=tf.float32),
            "bbox_regression": bbox_weight,
        }
        return crop, targets, sample_weights

    def build(self):
        output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(4,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(4,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )

        dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=output_signature,
        )

        if self.shuffle:
            dataset = dataset.shuffle(self.batch_size * 128, reshuffle_each_iteration=True)

        dataset = dataset.repeat()
        dataset = dataset.map(
            self._process_record,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not self.shuffle,
        )
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    @property
    def steps_per_epoch(self):
        total = len(self.samples) * self.proposals_per_image
        return max(1, int(np.ceil(total / self.batch_size)))


class COCOInferenceDataset:
    def __init__(self, coco_root=COCO_ROOT, split="val2017", refresh_cache=False):
        self.samples = collect_coco_samples(
            coco_root=coco_root,
            split=split,
            refresh_cache=refresh_cache,
        )

    def __len__(self):
        return len(self.samples)


def build_train_dataset(coco_root=COCO_ROOT, split="train2017"):
    dataset = RCNNProposalDataset(
        coco_root=coco_root,
        split=split,
        batch_size=BATCH_SIZE,
        input_size=INPUT_SIZE,
        proposals_per_image=TRAIN_PROPOSALS_PER_IMAGE,
        augment=True,
        shuffle=True,
    )
    return dataset, dataset.build()


def build_val_dataset(coco_root=COCO_ROOT, split="val2017"):
    dataset = RCNNProposalDataset(
        coco_root=coco_root,
        split=split,
        batch_size=BATCH_SIZE,
        input_size=INPUT_SIZE,
        proposals_per_image=VAL_PROPOSALS_PER_IMAGE,
        augment=False,
        shuffle=False,
    )
    return dataset, dataset.build()


if __name__ == "__main__":
    dataset, tf_dataset = build_train_dataset()
    print("steps_per_epoch:", dataset.steps_per_epoch)
    counter = 0
    for images, targets, weights in tf_dataset:# .take(100):
        counter += 1
        print(f"Batch {counter}")
        # print("images:", images.shape, images.dtype)
        # print("class_logits:", targets["class_logits"].shape)
        # print("bbox_regression:", targets["bbox_regression"].shape)
        # print("bbox_weights:", weights["bbox_regression"].shape)




# import random
# from collections import defaultdict

# import cv2
# import numpy as np
# import tensorflow as tf
# from pycocotools.coco import COCO

# from config import (
#     BATCH_SIZE,
#     COCO_ROOT,
#     INPUT_SIZE,
#     MAX_PROPOSALS,
#     MIN_BOX_SIZE,
#     NEGATIVE_IOU_THRESHOLD,
#     POSITIVE_FRACTION,
#     POSITIVE_IOU_THRESHOLD,
#     TRAIN_PROPOSALS_PER_IMAGE,
#     USE_SELECTIVE_SEARCH,
#     VAL_PROPOSALS_PER_IMAGE,
# )
# from utils import compute_iou, crop_and_resize, encode_box, generate_region_proposals


# COCO_CLASSES = [
#     "background",
#     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
#     "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#     "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
#     "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
#     "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
#     "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
#     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
#     "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
#     "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
#     "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
# ]

# COCO_CATEGORY_IDS = [
#     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
#     22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
#     43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
#     62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
#     85, 86, 87, 88, 89, 90,
# ]

# COCO_ID_TO_IDX = {category_id: index + 1 for index, category_id in enumerate(COCO_CATEGORY_IDS)}
# IDX_TO_COCO_ID = {index + 1: category_id for index, category_id in enumerate(COCO_CATEGORY_IDS)}
# NUM_CLASSES = len(COCO_CLASSES)


# def coco_annotation_path(coco_root, split):
#     return f"{coco_root}/annotations/instances_{split}.json"


# def coco_image_dir(coco_root, split):
#     return f"{coco_root}/{split}"


# def collect_coco_samples(coco_root, split):
#     annotation_file = coco_annotation_path(coco_root, split)
#     image_dir = coco_image_dir(coco_root, split)
#     coco = COCO(annotation_file)

#     annotations_by_image = defaultdict(list)
#     for annotation in coco.dataset["annotations"]:
#         if annotation.get("iscrowd", 0) == 1:
#             continue
#         category_idx = COCO_ID_TO_IDX.get(annotation["category_id"])
#         if category_idx is None:
#             continue
#         annotations_by_image[annotation["image_id"]].append(annotation)

#     samples = []
#     for image_id in coco.getImgIds():
#         image_info = coco.loadImgs(image_id)[0]
#         image_path = f"{image_dir}/{image_info['file_name']}"
#         annotations = annotations_by_image.get(image_id, [])

#         boxes = []
#         labels = []
#         areas = []

#         for annotation in annotations:
#             x, y, width, height = annotation["bbox"]
#             if width < MIN_BOX_SIZE or height < MIN_BOX_SIZE:
#                 continue
#             boxes.append([x, y, x + width, y + height])
#             labels.append(COCO_ID_TO_IDX[annotation["category_id"]])
#             areas.append(annotation.get("area", width * height))

#         if not boxes:
#             continue

#         samples.append(
#             {
#                 "image_id": image_id,
#                 "image_path": image_path,
#                 "width": int(image_info["width"]),
#                 "height": int(image_info["height"]),
#                 "boxes": np.asarray(boxes, dtype=np.float32),
#                 "labels": np.asarray(labels, dtype=np.int32),
#                 "areas": np.asarray(areas, dtype=np.float32),
#                 "file_name": image_info["file_name"],
#                 "split": split,
#             }
#         )

#     return samples


# class RCNNProposalDataset:
#     def __init__(
#         self,
#         coco_root=COCO_ROOT,
#         split="train2017",
#         batch_size=BATCH_SIZE,
#         input_size=INPUT_SIZE,
#         proposals_per_image=TRAIN_PROPOSALS_PER_IMAGE,
#         max_proposals=MAX_PROPOSALS,
#         positive_fraction=POSITIVE_FRACTION,
#         positive_iou_threshold=POSITIVE_IOU_THRESHOLD,
#         negative_iou_threshold=NEGATIVE_IOU_THRESHOLD,
#         augment=False,
#         shuffle=True,
#         use_selective_search=USE_SELECTIVE_SEARCH,
#     ):
#         self.coco_root = coco_root
#         self.split = split
#         self.batch_size = batch_size
#         self.input_size = tuple(input_size)
#         self.proposals_per_image = proposals_per_image
#         self.max_proposals = max_proposals
#         self.positive_fraction = positive_fraction
#         self.positive_iou_threshold = positive_iou_threshold
#         self.negative_iou_threshold = negative_iou_threshold
#         self.augment = augment
#         self.shuffle = shuffle
#         self.use_selective_search = use_selective_search

#         self.samples = collect_coco_samples(coco_root=self.coco_root, split=self.split)
#         print(
#             f"[RCNNDataset] split={self.split} samples={len(self.samples)} "
#             f"proposals_per_image={self.proposals_per_image}"
#         )

#     def _sample_indices(self, indices, count):
#         if len(indices) == 0 or count <= 0:
#             return np.array([], dtype=np.int32)
#         replace = len(indices) < count
#         return np.random.choice(indices, size=count, replace=replace)

#     def _augment_crop(self, crop):
#         crop = crop.astype(np.float32)
#         if np.random.rand() < 0.5:
#             crop = np.clip(crop * np.random.uniform(0.9, 1.1), 0.0, 1.0)
#         if np.random.rand() < 0.5:
#             crop = np.clip(crop + np.random.uniform(-0.05, 0.05), 0.0, 1.0)
#         return crop

#     def _proposal_targets(self, sample):
#         image_bgr = cv2.imread(sample["image_path"])
#         if image_bgr is None:
#             return None

#         image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#         proposals = generate_region_proposals(
#             image_bgr=image_bgr,
#             max_proposals=self.max_proposals,
#             use_selective_search=self.use_selective_search,
#         )
#         proposals = np.concatenate([proposals, sample["boxes"]], axis=0).astype(np.float32)
#         proposals = np.unique(np.round(proposals).astype(np.int32), axis=0).astype(np.float32)

#         gt_boxes = sample["boxes"].astype(np.float32)
#         gt_labels = sample["labels"].astype(np.int32)
#         ious = compute_iou(proposals, gt_boxes)
#         max_iou = ious.max(axis=1)
#         matched_gt = ious.argmax(axis=1)

#         positive_indices = np.where(max_iou >= self.positive_iou_threshold)[0]
#         negative_indices = np.where(max_iou < self.negative_iou_threshold)[0]
#         fallback_negative_indices = np.where(max_iou < self.positive_iou_threshold)[0]

#         num_positive = int(self.proposals_per_image * self.positive_fraction)
#         num_negative = self.proposals_per_image - num_positive

#         chosen_positive = self._sample_indices(positive_indices, num_positive)
#         chosen_negative = self._sample_indices(negative_indices, num_negative)
#         if len(chosen_negative) == 0:
#             chosen_negative = self._sample_indices(fallback_negative_indices, num_negative)

#         if len(chosen_positive) == 0 and len(chosen_negative) == 0:
#             return None

#         chosen = np.concatenate([chosen_positive, chosen_negative], axis=0)
#         if self.shuffle:
#             np.random.shuffle(chosen)

#         positive_lookup = set(chosen_positive.tolist())
#         crops = []
#         class_labels = []
#         bbox_targets = []
#         bbox_weights = []

#         for index in chosen:
#             proposal = proposals[index]
#             crop = crop_and_resize(image_rgb, proposal, self.input_size)
#             is_positive = index in positive_lookup

#             if self.augment:
#                 crop = self._augment_crop(crop)

#             if is_positive:
#                 gt_idx = matched_gt[index]
#                 class_label = gt_labels[gt_idx]
#                 bbox_target = encode_box(proposal, gt_boxes[gt_idx])
#                 bbox_weight = 1.0
#             else:
#                 class_label = 0
#                 bbox_target = np.zeros((4,), dtype=np.float32)
#                 bbox_weight = 0.0

#             crops.append(crop.astype(np.float32))
#             class_labels.append(np.int32(class_label))
#             bbox_targets.append(bbox_target.astype(np.float32))
#             bbox_weights.append(np.float32(bbox_weight))

#         return (
#             np.asarray(crops, dtype=np.float32),
#             np.asarray(class_labels, dtype=np.int32),
#             np.asarray(bbox_targets, dtype=np.float32),
#             np.asarray(bbox_weights, dtype=np.float32),
#         )

#     def _generator(self):
#         indices = list(range(len(self.samples)))
#         if self.shuffle:
#             random.shuffle(indices)

#         for idx in indices:
#             targets = self._proposal_targets(self.samples[idx])
#             if targets is None:
#                 continue

#             crops, class_labels, bbox_targets, bbox_weights = targets
#             for crop, class_label, bbox_target, bbox_weight in zip(
#                 crops, class_labels, bbox_targets, bbox_weights
#             ):
#                 yield {
#                     "image": crop,
#                     "class_label": class_label,
#                     "bbox_target": bbox_target,
#                     "bbox_weight": bbox_weight,
#                 }

#     def build(self):
#         output_signature = {
#             "image": tf.TensorSpec(shape=(self.input_size[0], self.input_size[1], 3), dtype=tf.float32),
#             "class_label": tf.TensorSpec(shape=(), dtype=tf.int32),
#             "bbox_target": tf.TensorSpec(shape=(4,), dtype=tf.float32),
#             "bbox_weight": tf.TensorSpec(shape=(), dtype=tf.float32),
#         }

#         dataset = tf.data.Dataset.from_generator(self._generator, output_signature=output_signature)
#         if self.shuffle:
#             dataset = dataset.shuffle(self.batch_size * 16)

#         dataset = dataset.batch(self.batch_size)
#         dataset = dataset.repeat()

#         def to_supervised(batch):
#             images = batch["image"]
#             targets = {
#                 "class_logits": batch["class_label"],
#                 "bbox_regression": batch["bbox_target"],
#             }
#             sample_weights = {
#                 "class_logits": tf.ones_like(batch["bbox_weight"], dtype=tf.float32),
#                 "bbox_regression": batch["bbox_weight"],
#             }
#             return images, targets, sample_weights

#         dataset = dataset.map(to_supervised, num_parallel_calls=tf.data.AUTOTUNE)
#         dataset = dataset.prefetch(tf.data.AUTOTUNE)
#         return dataset

#     @property
#     def steps_per_epoch(self):
#         total = len(self.samples) * self.proposals_per_image
#         return max(1, int(np.ceil(total / self.batch_size)))


# class COCOInferenceDataset:
#     def __init__(self, coco_root=COCO_ROOT, split="val2017"):
#         self.samples = collect_coco_samples(coco_root=coco_root, split=split)

#     def __len__(self):
#         return len(self.samples)


# def build_train_dataset(coco_root=COCO_ROOT, split="train2017"):
#     dataset = RCNNProposalDataset(
#         coco_root=coco_root,
#         split=split,
#         batch_size=BATCH_SIZE,
#         input_size=INPUT_SIZE,
#         proposals_per_image=TRAIN_PROPOSALS_PER_IMAGE,
#         augment=True,
#         shuffle=True,
#     )
#     return dataset, dataset.build()


# def build_val_dataset(coco_root=COCO_ROOT, split="val2017"):
#     dataset = RCNNProposalDataset(
#         coco_root=coco_root,
#         split=split,
#         batch_size=BATCH_SIZE,
#         input_size=INPUT_SIZE,
#         proposals_per_image=VAL_PROPOSALS_PER_IMAGE,
#         augment=False,
#         shuffle=False,
#     )
#     return dataset, dataset.build()


# if __name__ == "__main__":
#     dataset, tf_dataset = build_train_dataset()
#     print("steps_per_epoch:", dataset.steps_per_epoch)
#     for images, targets, weights in tf_dataset.take(1):
#         print("images:", images.shape, images.dtype)
#         print("class_logits:", targets["class_logits"].shape)
#         print("bbox_regression:", targets["bbox_regression"].shape)
#         print("bbox_weights:", weights["bbox_regression"].shape)
