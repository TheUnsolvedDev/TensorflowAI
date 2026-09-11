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
    MASK_SIZE,
    MASK_THRESHOLD,
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
from utils import batched, clip_box, decode_box, draw_detections, generate_region_proposals, nms


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
        model = create_compiled_model(dataset_name="coco")
    model.load_weights(CHECKPOINT_PATH)
    return model


def paste_mask(mask, box, image_shape):
    output = np.zeros(image_shape[:2], dtype=np.float32)
    x1, y1, x2, y2 = np.round(box).astype(np.int32)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_shape[1], max(x2, x1 + 1))
    y2 = min(image_shape[0], max(y2, y1 + 1))
    resized = cv2.resize(mask.astype(np.float32), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    output[y1:y2, x1:x2] = resized
    return output


def infer_image(model, image_path, class_names=COCO_CLASSES):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    proposals = generate_region_proposals(image_bgr, max_proposals=MAX_PROPOSALS, use_selective_search=USE_SELECTIVE_SEARCH)
    height, width = image_bgr.shape[:2]
    rois = np.asarray([[p[1] / height, p[0] / width, p[3] / height, p[2] / width] for p in proposals], dtype=np.float32)
    image = cv2.resize(image_rgb, (INPUT_SIZE[1], INPUT_SIZE[0])).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    class_logits = []
    bbox_deltas = []
    mask_logits = []
    for roi_batch in batched(rois, INFERENCE_BATCH_SIZE):
        preds = model({"image": image, "rois": np.expand_dims(roi_batch, axis=0)}, training=False)
        class_logits.append(preds["class_logits"].numpy()[0])
        bbox_deltas.append(preds["bbox_regression"].numpy()[0])
        mask_logits.append(preds["mask_logits"].numpy()[0])

    class_logits = np.concatenate(class_logits, axis=0)
    bbox_deltas = np.concatenate(bbox_deltas, axis=0)
    mask_logits = np.concatenate(mask_logits, axis=0)
    probs = tf.nn.softmax(class_logits, axis=-1).numpy()
    class_ids = probs.argmax(axis=1)
    class_scores = probs.max(axis=1)
    decoded_boxes = np.asarray([decode_box(p.astype(np.float32), d.astype(np.float32)) for p, d in zip(proposals, bbox_deltas)])
    decoded_boxes = np.asarray([clip_box(box, height, width) for box in decoded_boxes], dtype=np.float32)
    masks = tf.nn.sigmoid(mask_logits).numpy()[..., 0]

    ranked = np.argsort(class_scores)[::-1][:INFERENCE_TOPK]
    detections = []
    for class_id in range(1, len(class_names)):
        class_indices = ranked[(class_ids[ranked] == class_id) & (class_scores[ranked] >= SCORE_THRESHOLD)]
        if len(class_indices) == 0:
            continue
        keep = nms(decoded_boxes[class_indices], class_scores[class_indices], iou_threshold=NMS_IOU_THRESHOLD, max_keep=MAX_DETECTIONS_PER_CLASS)
        for keep_idx in keep:
            proposal_idx = class_indices[keep_idx]
            detections.append(
                {
                    "class_id": int(class_id),
                    "score": float(class_scores[proposal_idx]),
                    "box": decoded_boxes[proposal_idx],
                    "mask": paste_mask(masks[proposal_idx], decoded_boxes[proposal_idx], image_bgr.shape),
                }
            )
    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections, image_bgr


def render_masks(image_bgr, detections, class_names=COCO_CLASSES):
    canvas = image_bgr.copy()
    for index, det in enumerate(detections[:MAX_DRAW_DETECTIONS]):
        mask = det["mask"] >= MASK_THRESHOLD
        color = np.asarray([40 + (index * 53) % 180, 100 + (index * 31) % 120, 200 - (index * 29) % 140], dtype=np.uint8)
        canvas[mask] = np.clip(0.6 * canvas[mask] + 0.4 * color, 0, 255).astype(np.uint8)
    return draw_detections(canvas, detections, class_names, max_draw=MAX_DRAW_DETECTIONS)


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
        cv2.imwrite(os.path.join(INFERENCE_DIR, "single_pred.jpg"), render_masks(image_bgr, detections))
    else:
        dataset = COCOInferenceDataset(COCO_ROOT, TEST_SPLIT)
        for sample in dataset.samples[: args.num_images]:
            detections, image_bgr = infer_image(model, sample["image_path"])
            cv2.imwrite(os.path.join(INFERENCE_DIR, f"{sample['image_id']}_pred.jpg"), render_masks(image_bgr, detections))

    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
