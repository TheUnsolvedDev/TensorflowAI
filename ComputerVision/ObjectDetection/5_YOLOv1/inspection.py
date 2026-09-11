"""Folder-local labelled validation inspection grid."""
import os

import cv2
import numpy as np
import tensorflow as tf


def _name(class_names, class_id):
    return class_names[class_id] if 0 <= int(class_id) < len(class_names) else f"class_{class_id}"


def draw_labelled_boxes(image_bgr, boxes, labels, class_names, *, normalized, prediction=False, max_draw=20):
    canvas = image_bgr.copy()
    height, width = canvas.shape[:2]
    color = (0, 220, 0) if prediction else (0, 0, 255)
    for box, label in list(zip(boxes, labels))[:max_draw]:
        if prediction:
            class_id, score = label
            text = f"{_name(class_names, class_id)}: {float(score):.2f}"
        else:
            class_id = label
            text = f"GT {_name(class_names, class_id)}"
        y1, x1, y2, x2 = box
        if normalized:
            x1, x2, y1, y2 = x1 * width, x2 * width, y1 * height, y2 * height
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, text, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas


class DetectionInspectionCallback(tf.keras.callbacks.Callback):
    """Write a deterministic 4x4 labelled validation grid after every epoch."""

    def __init__(self, samples, output_dir, class_names, infer_image, *, gt_normalized, pred_normalized=True):
        super().__init__()
        self.samples, self.class_names, self.infer_image = list(samples[:16]), class_names, infer_image
        self.gt_normalized, self.pred_normalized = gt_normalized, pred_normalized
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _render(self, sample):
        image = cv2.imread(sample["image_path"])
        if image is None:
            return None
        canvas = draw_labelled_boxes(image, sample["boxes"], sample["labels"], self.class_names, normalized=self.gt_normalized)
        detections, _ = self.infer_image(self.model, sample["image_path"])
        boxes = [item["box"] for item in detections]
        labels = [(item["class_id"], item["score"]) for item in detections]
        canvas = draw_labelled_boxes(canvas, boxes, labels, self.class_names, normalized=self.pred_normalized, prediction=True)
        canvas = cv2.resize(canvas, (320, 240), interpolation=cv2.INTER_AREA)
        return canvas

    def on_epoch_end(self, epoch, logs=None):
        panels = [panel for sample in self.samples if (panel := self._render(sample)) is not None]
        if not panels:
            print("[inspection] No validation images could be rendered.")
            return
        panels.extend(np.zeros_like(panels[0]) for _ in range(16 - len(panels)))
        grid = np.concatenate([np.concatenate(panels[row:row + 4], axis=1) for row in range(0, 16, 4)], axis=0)
        path = os.path.join(self.output_dir, f"epoch_{epoch + 1:03d}.png")
        if not cv2.imwrite(path, grid):
            raise RuntimeError(f"Could not write inspection image: {path}")
        print(f"[inspection] Saved 4x4 ground-truth/prediction grid: {path}")
