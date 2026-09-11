"""Folder-local labelled-detection adapters shared by this detector's datasets.

The adapters deliberately return normalized xyxy boxes; each detector keeps its
own target encoder and loss in ``dataset.py``/``model.py``.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


def voc_records(root, split, min_box_size, background=False):
    """Load the labelled VOC 2012 train/val split (never the unlabelled test)."""
    root = Path(root)
    split_file = root / "ImageSets" / "Main" / f"{split}.txt"
    if not split_file.is_file():
        raise RuntimeError(f"VOC split file not found: {split_file}")
    class_to_index = {name: index + int(background) for index, name in enumerate(VOC_CLASSES)}
    records = []
    for image_id in split_file.read_text(encoding="utf-8").splitlines():
        image_id = image_id.strip()
        xml_path = root / "Annotations" / f"{image_id}.xml"
        image_path = root / "JPEGImages" / f"{image_id}.jpg"
        if not image_id or not xml_path.is_file() or not image_path.is_file():
            continue
        annotation = ET.parse(xml_path).getroot()
        width = max(float(annotation.findtext("./size/width", "1")), 1.0)
        height = max(float(annotation.findtext("./size/height", "1")), 1.0)
        boxes, labels = [], []
        for obj in annotation.findall("./object"):
            if obj.findtext("difficult", "0") == "1":
                continue
            label = class_to_index.get(obj.findtext("name", ""))
            bbox = obj.find("./bndbox")
            if label is None or bbox is None:
                continue
            x1, y1 = float(bbox.findtext("xmin", "0")), float(bbox.findtext("ymin", "0"))
            x2, y2 = float(bbox.findtext("xmax", "0")), float(bbox.findtext("ymax", "0"))
            if x2 - x1 < min_box_size or y2 - y1 < min_box_size:
                continue
            boxes.append([max(0., x1 / width), max(0., y1 / height), min(1., x2 / width), min(1., y2 / height)])
            labels.append(label)
        if boxes:
            records.append({"image_id": image_id, "image_path": str(image_path),
                            "boxes": np.asarray(boxes, np.float32), "labels": np.asarray(labels, np.int32)})
    return records


def dataset_root(name, coco_root, voc_root, imagenet_root):
    roots = {"coco": coco_root, "voc": voc_root, "imagenet": imagenet_root}
    try:
        return roots[name]
    except KeyError as error:
        raise ValueError(f"unknown detector dataset: {name}") from error


def split_name(name, phase, coco_train, coco_val):
    if name == "voc":
        return "train" if phase == "train" else "val"
    if name == "imagenet":
        return "train" if phase == "train" else "val"
    return coco_train if phase == "train" else coco_val
