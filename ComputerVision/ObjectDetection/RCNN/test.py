import argparse
import gc
import os

import cv2
import numpy as np
import tensorflow as tf

from config import (
    CHECKPOINT_PATH,
    COCO_ROOT,
    INFERENCE_BATCH_SIZE,
    INFERENCE_DIR,
    INFERENCE_TOPK,
    INPUT_SIZE,
    MAX_DETECTIONS_PER_CLASS,
    MAX_DRAW_DETECTIONS,
    MAX_PROPOSALS,
    NMS_IOU_THRESHOLD,
    SCORE_THRESHOLD,
    TEST_SPLIT,
    USE_SELECTIVE_SEARCH,
    ensure_dir,
)
from dataset import COCO_CLASSES, COCOInferenceDataset
from model import create_compiled_model
from utils import (
    batched,
    clip_box,
    crop_and_resize,
    decode_box,
    draw_detections,
    generate_region_proposals,
    make_output_path,
    nms,
)


def setup_runtime(gpu_id):
    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    if gpu_id == -1 or not physical_devices:
        visible_gpus = physical_devices
    elif 0 <= gpu_id < len(physical_devices):
        tf.config.set_visible_devices(physical_devices[gpu_id], "GPU")
        visible_gpus = [physical_devices[gpu_id]]
    else:
        visible_gpus = []

    visible_gpu_count = len(tf.config.get_visible_devices("GPU"))
    if visible_gpu_count > 1:
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.NcclAllReduce()
        )
        print(f"Using {visible_gpu_count} GPUs with MirroredStrategy + NcclAllReduce")
    elif visible_gpu_count == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print("Using single GPU")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        print("Using CPU")

    return strategy, visible_gpus


def load_model(strategy):
    tf.keras.backend.clear_session()
    with strategy.scope():
        model = create_compiled_model()
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    model.load_weights(CHECKPOINT_PATH)
    return model


def infer_image(model, image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    proposals = generate_region_proposals(
        image_bgr=image_bgr,
        max_proposals=MAX_PROPOSALS,
        use_selective_search=USE_SELECTIVE_SEARCH,
    )

    crops = np.asarray(
        [crop_and_resize(image_rgb, proposal, INPUT_SIZE) for proposal in proposals],
        dtype=np.float32,
    )

    class_logits = []
    bbox_deltas = []

    for crop_batch in batched(crops, INFERENCE_BATCH_SIZE):
        predictions = model(crop_batch, training=False)
        class_logits.append(predictions["class_logits"].numpy())
        bbox_deltas.append(predictions["bbox_regression"].numpy())

    class_logits = np.concatenate(class_logits, axis=0)
    bbox_deltas = np.concatenate(bbox_deltas, axis=0)
    probabilities = tf.nn.softmax(class_logits, axis=-1).numpy()

    class_ids = probabilities.argmax(axis=1)
    class_scores = probabilities.max(axis=1)

    decoded_boxes = []
    for proposal, delta in zip(proposals, bbox_deltas):
        decoded_boxes.append(decode_box(proposal, delta))
    decoded_boxes = np.asarray(decoded_boxes, dtype=np.float32)

    height, width = image_bgr.shape[:2]
    decoded_boxes = np.asarray([clip_box(box, height, width) for box in decoded_boxes], dtype=np.float32)

    ranked = np.argsort(class_scores)[::-1][:INFERENCE_TOPK]
    detections = []

    for class_id in range(1, len(COCO_CLASSES)):
        class_indices = ranked[(class_ids[ranked] == class_id) & (class_scores[ranked] >= SCORE_THRESHOLD)]
        if len(class_indices) == 0:
            continue

        keep = nms(
            boxes=decoded_boxes[class_indices],
            scores=class_scores[class_indices],
            iou_threshold=NMS_IOU_THRESHOLD,
            max_keep=MAX_DETECTIONS_PER_CLASS,
        )
        for keep_idx in keep:
            proposal_idx = class_indices[keep_idx]
            detections.append(
                {
                    "class_id": int(class_id),
                    "score": float(class_scores[proposal_idx]),
                    "box": decoded_boxes[proposal_idx],
                }
            )

    detections.sort(key=lambda item: item["score"], reverse=True)
    del crops, class_logits, bbox_deltas, probabilities, decoded_boxes
    return detections, image_bgr


def run_single_image(model, image_path):
    ensure_dir(INFERENCE_DIR)
    detections, image_bgr = infer_image(model, image_path)
    rendered = draw_detections(
        image_bgr=image_bgr,
        detections=detections,
        class_names=COCO_CLASSES,
        max_draw=MAX_DRAW_DETECTIONS,
    )
    output_path = make_output_path(INFERENCE_DIR, image_path)
    cv2.imwrite(output_path, rendered)
    print(f"Saved detections to: {output_path}")
    for detection in detections[:10]:
        label = COCO_CLASSES[detection["class_id"]]
        box = detection["box"].round(1).tolist()
        print(f"{label:15s} score={detection['score']:.3f} box={box}")


def run_split(model, num_images):
    if COCO_ROOT is None:
        raise RuntimeError("COCO_ROOT was not found. Update config.py with a valid COCO 2017 root.")

    dataset = COCOInferenceDataset(COCO_ROOT, TEST_SPLIT)
    ensure_dir(INFERENCE_DIR)

    for sample in dataset.samples[:num_images]:
        print(f"Inferencing {sample['image_id']} ...")
        run_single_image(model, sample["image_path"])


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained scratch R-CNN model.")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, use -1 for all GPUs/CPU.")
    parser.add_argument("--image", type=str, default=None, help="Path to an image for single-image inference.")
    parser.add_argument("--num_images", type=int, default=5, help="How many images to run when using dataset split inference.")
    args = parser.parse_args()

    strategy, _ = setup_runtime(args.gpu)
    model = load_model(strategy)

    if args.image:
        run_single_image(model, args.image)
    else:
        run_split(model, args.num_images)
    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
