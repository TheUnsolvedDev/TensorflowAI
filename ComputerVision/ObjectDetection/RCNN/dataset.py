"""
dataset.py — Pascal VOC 2007/2012 TF2 Data Pipeline for Faster R-CNN
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import tensorflow as tf

# ──────────────────────────────────────────────
# Pascal VOC class labels (20 classes + background)
# ──────────────────────────────────────────────
VOC_CLASSES = [
    "background",   # 0  (reserved for R-CNN)
    "aeroplane",    # 1
    "bicycle",      # 2
    "bird",         # 3
    "boat",         # 4
    "bottle",       # 5
    "bus",          # 6
    "car",          # 7
    "cat",          # 8
    "chair",        # 9
    "cow",          # 10
    "diningtable",  # 11
    "dog",          # 12
    "horse",        # 13
    "motorbike",    # 14
    "person",       # 15
    "pottedplant",  # 16
    "sheep",        # 17
    "sofa",         # 18
    "train",        # 19
    "tvmonitor",    # 20
]

CLASS_TO_IDX: Dict[str, int] = {
    cls: idx for idx, cls in enumerate(VOC_CLASSES)}
NUM_CLASSES: int = len(VOC_CLASSES)  # 21


# ──────────────────────────────────────────────
# Pascal VOC annotation parser
# ──────────────────────────────────────────────

def parse_voc_annotation(xml_path: str) -> Dict:
    """
    Parse a single Pascal VOC XML annotation file.

    Returns
    -------
    dict with keys:
        image_path  : str
        width       : int
        height      : int
        boxes       : np.ndarray  shape (N, 4) float32  [y1, x1, y2, x2] normalised [0,1]
        labels      : np.ndarray  shape (N,)   int32
        difficult   : np.ndarray  shape (N,)   bool
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    folder = root.findtext("folder", default="VOC2007")
    filename = root.findtext("filename")

    # Derive image path relative to VOC root (caller resolves absolute path)
    image_rel = os.path.join("JPEGImages", filename)

    size_node = root.find("size")
    width = int(size_node.findtext("width"))
    height = int(size_node.findtext("height"))

    boxes, labels, difficult = [], [], []

    for obj in root.iter("object"):
        name = obj.findtext("name")
        diff_flag = int(obj.findtext("difficult", default="0"))
        bndbox = obj.find("bndbox")

        xmin = float(bndbox.findtext("xmin"))
        ymin = float(bndbox.findtext("ymin"))
        xmax = float(bndbox.findtext("xmax"))
        ymax = float(bndbox.findtext("ymax"))

        # Normalise to [0, 1] and store as [y1, x1, y2, x2] (TF convention)
        boxes.append([
            ymin / height,
            xmin / width,
            ymax / height,
            xmax / width,
        ])
        labels.append(CLASS_TO_IDX.get(name, 0))
        difficult.append(bool(diff_flag))

    return {
        "image_path": image_rel,
        "width":      width,
        "height":     height,
        "boxes":      np.array(boxes,     dtype=np.float32),
        "labels":     np.array(labels,    dtype=np.int32),
        "difficult":  np.array(difficult, dtype=bool),
    }


# ──────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────

class Dataset:
    """
    Pascal VOC 2007 / 2012 TF2 data-pipeline for Faster R-CNN.

    Directory layout expected
    ─────────────────────────
    <voc_root>/
        VOC2007/
            Annotations/   *.xml
            JPEGImages/    *.jpg
            ImageSets/
                Main/
                    train.txt  val.txt  test.txt
        VOC2012/           (optional)
            ...

    Parameters
    ----------
    voc_root     : root directory that contains VOCyear/ sub-folders
    years        : list of dataset years, e.g. ["2007"] or ["2007", "2012"]
    split        : "train" | "val" | "trainval" | "test"
    image_size   : (H, W) to resize images to
    batch_size   : samples per batch
    augment      : whether to apply training augmentations
    skip_difficult: skip objects marked as difficult
    max_boxes    : pad/truncate ground-truth boxes to this fixed length
                   (needed for tf.data batching; set to None to disable)
    """

    def __init__(
        self,
        voc_root:        str,
        years:           List[str] = ("2007",),
        split:           str = "train",
        image_size:      Tuple[int, int] = (600, 600),
        batch_size:      int = 2,
        augment:         bool = True,
        skip_difficult:  bool = True,
        max_boxes:       Optional[int] = 100,
    ):
        self.voc_root = Path(voc_root)
        self.years = list(years)
        self.split = split
        self.image_size = image_size          # (H, W)
        self.batch_size = batch_size
        self.augment = augment
        self.skip_difficult = skip_difficult
        self.max_boxes = max_boxes

        # Collect all (image_abs_path, annotation_dict) pairs
        self.samples: List[Dict] = self._collect_samples()
        print(f"[Dataset] split={split!r}  samples={len(self.samples)}  "
              f"years={self.years}  image_size={image_size}")

    # ── internal helpers ───────────────────────

    def _collect_samples(self) -> List[Dict]:
        samples = []
        for year in self.years:
            year_dir = self.voc_root / f"VOC{year}"
            sets_dir = year_dir / "ImageSets" / "Main"
            ann_dir = year_dir / "Annotations"
            img_dir = year_dir / "JPEGImages"

            split_file = sets_dir / f"{self.split}.txt"
            if not split_file.exists():
                raise FileNotFoundError(f"Split file not found: {split_file}")

            with open(split_file) as f:
                ids = [line.strip() for line in f if line.strip()]

            for img_id in ids:
                xml_path = ann_dir / f"{img_id}.xml"
                jpg_path = img_dir / f"{img_id}.jpg"
                if not xml_path.exists() or not jpg_path.exists():
                    continue  # silently skip missing files

                ann = parse_voc_annotation(str(xml_path))

                if self.skip_difficult and ann["difficult"].any():
                    # Keep the sample but drop difficult objects
                    keep = ~ann["difficult"]
                    ann["boxes"] = ann["boxes"][keep]
                    ann["labels"] = ann["labels"][keep]
                    ann["difficult"] = ann["difficult"][keep]

                if len(ann["labels"]) == 0:
                    continue  # skip images with no valid objects

                ann["image_path"] = str(jpg_path)
                samples.append(ann)

        return samples

    # ── tf.data generator ─────────────────────

    def _generator(self):
        """Python generator yielding one sample at a time."""
        indices = np.arange(len(self.samples))
        if self.augment:
            np.random.shuffle(indices)

        for idx in indices:
            s = self.samples[idx]
            yield (
                s["image_path"].encode(),          # bytes
                s["boxes"].astype(np.float32),     # (N, 4)
                s["labels"].astype(np.int32),      # (N,)
            )

    def _load_image(self, path: tf.Tensor) -> tf.Tensor:
        """Read & decode JPEG → float32 [0, 1]."""
        raw = tf.io.read_file(path)
        image = tf.image.decode_jpeg(raw, channels=3)
        # image = tf.cast(image, tf.float32) / 255.0
        return image

    def _resize(
        self,
        image: tf.Tensor,
        boxes: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Resize image to self.image_size.
        Boxes are already normalised [0,1] so they need no adjustment.
        """
        h, w = self.image_size
        image = tf.image.resize(image, [h, w])
        return image, boxes

    def _augment(
        self,
        image: tf.Tensor,
        boxes: tf.Tensor,
        labels: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Training-time augmentations (all box-safe).
        """
        # 1. Random horizontal flip
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            # x-coords: x1_new = 1 - x2_old,  x2_new = 1 - x1_old
            y1, x1, y2, x2 = tf.unstack(boxes, axis=-1)
            boxes = tf.stack([y1, 1.0 - x2, y2, 1.0 - x1], axis=-1)

        # 2. Random brightness / contrast / saturation
        image = tf.image.random_brightness(image, max_delta=0.12)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, boxes, labels

    def _pad_boxes(
        self,
        boxes:  tf.Tensor,
        labels: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Pad (or truncate) boxes/labels to self.max_boxes.
        Returns also a boolean valid_mask of shape (max_boxes,).
        """
        n = tf.shape(boxes)[0]
        m = self.max_boxes

        # Truncate if needed
        boxes = boxes[:m]
        labels = labels[:m]
        actual = tf.minimum(n, m)

        # Pad with zeros
        pad_b = m - actual
        boxes = tf.pad(boxes,  [[0, pad_b], [0, 0]])
        labels = tf.pad(labels, [[0, pad_b]])

        # valid_mask: True for real boxes, False for padding
        valid_mask = tf.sequence_mask(actual, maxlen=m)  # (m,)

        # Restore shape information (needed for static shape in model)
        boxes = tf.ensure_shape(boxes,      [m, 4])
        labels = tf.ensure_shape(labels,     [m])
        valid_mask = tf.ensure_shape(valid_mask, [m])

        return boxes, labels, valid_mask

    def _process(
        self,
        image_path: tf.Tensor,
        boxes:      tf.Tensor,
        labels:     tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """
        Full preprocessing pipeline for one sample.
        Returns a dict suitable for model input.
        """
        image = self._load_image(image_path)
        image, boxes = self._resize(image, boxes)

        if self.augment:
            image, boxes, labels = self._augment(image, boxes, labels)

        valid_mask = None
        if self.max_boxes is not None:
            boxes, labels, valid_mask = self._pad_boxes(boxes, labels)

        output = {
            "image":  image,          # (H, W, 3)  float32
            # (N or max_boxes, 4) float32  [y1,x1,y2,x2]
            "boxes":  boxes,
            "labels": labels,         # (N or max_boxes,)  int32
        }
        if valid_mask is not None:
            output["valid_mask"] = valid_mask  # (max_boxes,) bool

        return output

    # ── public API ────────────────────────────

    def build(self) -> tf.data.Dataset:
        """
        Build and return a ready-to-iterate tf.data.Dataset.

        Each batch is a dict:
            image      : (B, H, W, 3)          float32
            boxes      : (B, max_boxes, 4)      float32
            labels     : (B, max_boxes)         int32
            valid_mask : (B, max_boxes)         bool
        """
        output_sig = (
            tf.TensorSpec(shape=(),      dtype=tf.string),   # image_path
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),  # boxes
            tf.TensorSpec(shape=(None,),   dtype=tf.int32),  # labels
        )

        ds = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=output_sig,
        )

        # Map preprocessing (parallelised)
        ds = ds.map(
            self._process,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not self.augment,
        )

        # Shuffle only for training
        if self.augment:
            ds = ds.shuffle(buffer_size=min(500, len(self.samples)))

        ds = ds.batch(self.batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def steps_per_epoch(self) -> int:
        return int(np.ceil(len(self.samples) / self.batch_size))

    @staticmethod
    def class_name(idx: int) -> str:
        return VOC_CLASSES[idx] if 0 <= idx < NUM_CLASSES else "unknown"


# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # voc_root = sys.argv[1] if len(sys.argv) > 1 else "./VOCdevkit"
    voc_root = ""
    train_ds = Dataset(
        voc_root=voc_root,
        years=["2007"],
        split="train",
        image_size=(600, 600),
        batch_size=2,
        augment=True,
        max_boxes=100,
    ).build()

    for batch in train_ds.take(1):
        print("image     :", batch["image"].shape,      batch["image"].dtype)
        print("boxes     :", batch["boxes"].shape,      batch["boxes"].dtype)
        print("labels    :", batch["labels"].shape,     batch["labels"].dtype)
        print("valid_mask:", batch["valid_mask"].shape,
              batch["valid_mask"].dtype)
        print("label sample:", [Dataset.class_name(i.numpy())
                                for i in batch["labels"][0]
                                if i.numpy() > 0][:5])
