import argparse
import gc
import os

import cv2
import numpy as np
import tensorflow as tf

from config import (
    CHECKPOINT_PATH,
    COCO_ROOT,
    INFERENCE_DIR,
    INPUT_SIZE,
    MAX_DRAW_DETECTIONS,
    TEST_SPLIT,
    ensure_dir,
)
from dataset import COCO_CLASSES, FasterRCNNDataset
from model import create_compiled_model


def setup_runtime(gpu_id):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpu_id != -1 and gpus and 0 <= gpu_id < len(gpus):
        tf.config.set_visible_devices(gpus[gpu_id], "GPU")
    visible = len(tf.config.get_visible_devices("GPU"))
    if visible > 1:
        return tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())
    if visible == 1:
        return tf.distribute.OneDeviceStrategy("/gpu:0")
    return tf.distribute.OneDeviceStrategy("/cpu:0")


def load_model(strategy):
    tf.keras.backend.clear_session()
    with strategy.scope():
        model = create_compiled_model()
    model.load_weights(CHECKPOINT_PATH)
    return model


def draw_detections(image_bgr, detections):
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]
    for det in detections[:MAX_DRAW_DETECTIONS]:
        y1, x1, y2, x2 = det["box"]
        x1 = int(x1 * w)
        x2 = int(x2 * w)
        y1 = int(y1 * h)
        y2 = int(y2 * h)
        label = COCO_CLASSES[det["class_id"]]
        score = det["score"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(canvas, f"{label}: {score:.2f}", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    return canvas


def infer_image(model, image_path):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image_rgb, (INPUT_SIZE[1], INPUT_SIZE[0])).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    detections = model.detect(image)
    return detections, image_bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=5)
    args = parser.parse_args()

    ensure_dir(INFERENCE_DIR)
    strategy = setup_runtime(args.gpu)
    model = load_model(strategy)

    if args.image:
        detections, image_bgr = infer_image(model, args.image)
        cv2.imwrite(os.path.join(INFERENCE_DIR, "single_pred.jpg"), draw_detections(image_bgr, detections))
    else:
        dataset = FasterRCNNDataset(COCO_ROOT, TEST_SPLIT, augment=False, shuffle=False)
        for sample in dataset.samples[: args.num_images]:
            detections, image_bgr = infer_image(model, sample["image_path"])
            cv2.imwrite(os.path.join(INFERENCE_DIR, f"{sample['image_id']}_pred.jpg"), draw_detections(image_bgr, detections))

    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
