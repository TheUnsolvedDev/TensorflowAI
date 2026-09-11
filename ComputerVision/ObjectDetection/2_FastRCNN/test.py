import argparse
import gc
import os

import cv2
import numpy as np
import tensorflow as tf

from config import MAX_DRAW_DETECTIONS, ensure_dir
from dataset import resolve_detection_dataset
from model import create_compiled_model
from train import infer_image, select_artifacts
from utils import draw_detections


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


def load_model(strategy, dataset_name):
    adapter = resolve_detection_dataset(dataset_name)
    _, _, checkpoint_path, _, _ = select_artifacts("detector", dataset_name)
    tf.keras.backend.clear_session()
    with strategy.scope():
        model = create_compiled_model(dataset_name=dataset_name, num_classes=adapter.num_classes)
    model.load_weights(checkpoint_path)
    return model, adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--detector", choices=["coco", "voc", "imagenet"], default="coco")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=5)
    args = parser.parse_args()

    _, log_dir, _, _, _ = select_artifacts("detector", args.detector)
    inference_dir = ensure_dir(os.path.join(log_dir, "inference"))
    strategy = setup_runtime(args.gpu)
    model, adapter = load_model(strategy, args.detector)
    if args.image:
        detections, image_bgr = infer_image(model, args.image, adapter.class_names)
        rendered = draw_detections(image_bgr, detections, adapter.class_names, max_draw=MAX_DRAW_DETECTIONS)
        cv2.imwrite(os.path.join(inference_dir, "single_pred.jpg"), rendered)
    else:
        for sample in adapter.records("val")[: args.num_images]:
            detections, image_bgr = infer_image(model, sample["image_path"], adapter.class_names)
            rendered = draw_detections(image_bgr, detections, adapter.class_names, max_draw=MAX_DRAW_DETECTIONS)
            cv2.imwrite(os.path.join(inference_dir, f"{sample['image_id']}_pred.jpg"), rendered)
    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
