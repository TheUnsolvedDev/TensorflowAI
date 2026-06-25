import argparse
import gc
import json
import os

import cv2
import numpy as np
import tensorflow as tf

from config import (
    CHECKPOINT_PATH,
    COCO_ROOT,
    EPOCHS,
    HISTORY_PATH,
    INFERENCE_BATCH_SIZE,
    INPUT_SIZE,
    LOG_DIR,
    MAX_DETECTIONS_PER_CLASS,
    MAX_DRAW_DETECTIONS,
    MAX_PROPOSALS,
    NMS_IOU_THRESHOLD,
    SCORE_THRESHOLD,
    STEPS_PER_EPOCH,
    STATE_PATH,
    TRAIN_SPLIT,
    USE_SELECTIVE_SEARCH,
    VAL_SPLIT,
    VALIDATION_STEPS,
    ensure_dir,
)
from dataset import COCO_CLASSES, build_train_dataset, build_val_dataset
from model import create_compiled_model
from utils import (
    batched,
    clip_box,
    crop_and_resize,
    decode_box,
    draw_detections,
    generate_region_proposals,
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
        print(
            f"Using {visible_gpu_count} GPUs with MirroredStrategy + NcclAllReduce")
    elif visible_gpu_count == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print("Using single GPU")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        print("Using CPU")

    return strategy, visible_gpus


def load_state():
    if not os.path.exists(STATE_PATH):
        return 0
    with open(STATE_PATH, "r", encoding="utf-8") as file:
        state = json.load(file)
    return int(state.get("epoch", 0))


def save_state(epoch):
    with open(STATE_PATH, "w", encoding="utf-8") as file:
        json.dump({"epoch": int(epoch)}, file)


class HistorySaver(tf.keras.callbacks.Callback):
    def __init__(self, history_path):
        super().__init__()
        self.history_path = history_path
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            self.history.setdefault(key, []).append(float(value))
        with open(self.history_path, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=2)


class StateSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        save_state(epoch + 1)


class InspectionCallback(tf.keras.callbacks.Callback):
    def __init__(self, samples, log_dir, grid_size=16):
        super().__init__()
        self.samples = samples[:grid_size]
        self.grid_size = min(grid_size, len(self.samples))
        self.samples_dir = ensure_dir(os.path.join(log_dir, "samples"))
        self.writer = tf.summary.create_file_writer(
            ensure_dir(os.path.join(log_dir, "inspection"))
        )

    def _draw_ground_truth(self, image_bgr, sample, max_draw=10):
        canvas = image_bgr.copy()
        boxes = sample.get("boxes", [])
        labels = sample.get("labels", [])
        for box, label in zip(boxes[:max_draw], labels[:max_draw]):
            x1, y1, x2, y2 = box.astype(np.int32)
            class_name = COCO_CLASSES[int(label)] if 0 <= int(label) < len(COCO_CLASSES) else str(label)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (220, 40, 40), 2)
            cv2.putText(
                canvas,
                f"gt:{class_name}",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 40, 40),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _infer_image(self, sample):
        image_path = sample["image_path"]
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return None

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
            predictions = self.model(crop_batch, training=False)
            class_logits.append(predictions["class_logits"].numpy())
            bbox_deltas.append(predictions["bbox_regression"].numpy())

        class_logits = np.concatenate(class_logits, axis=0)
        bbox_deltas = np.concatenate(bbox_deltas, axis=0)
        probabilities = tf.nn.softmax(class_logits, axis=-1).numpy()
        class_ids = probabilities.argmax(axis=1)
        class_scores = probabilities.max(axis=1)

        decoded_boxes = np.asarray(
            [decode_box(proposal, delta) for proposal, delta in zip(proposals, bbox_deltas)],
            dtype=np.float32,
        )
        height, width = image_bgr.shape[:2]
        decoded_boxes = np.asarray(
            [clip_box(box, height, width) for box in decoded_boxes],
            dtype=np.float32,
        )

        ranked = np.argsort(class_scores)[::-1][:300]
        detections = []
        for class_id in range(1, len(COCO_CLASSES)):
            class_indices = ranked[
                (class_ids[ranked] == class_id) & (class_scores[ranked] >= SCORE_THRESHOLD)
            ]
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

        if not detections:
            # Early epochs often have no predictions over the normal threshold.
            fallback_indices = ranked[class_ids[ranked] > 0][:5]
            for proposal_idx in fallback_indices:
                detections.append(
                    {
                        "class_id": int(class_ids[proposal_idx]),
                        "score": float(class_scores[proposal_idx]),
                        "box": decoded_boxes[proposal_idx],
                    }
                )

        detections.sort(key=lambda item: item["score"], reverse=True)
        rendered = self._draw_ground_truth(
            image_bgr=image_bgr,
            sample=sample,
            max_draw=10,
        )
        rendered = draw_detections(
            image_bgr=rendered,
            detections=detections,
            class_names=COCO_CLASSES,
            max_draw=MAX_DRAW_DETECTIONS,
        )
        return rendered

    def _make_grid(self, images, tile_size=(320, 320)):
        rows, cols = 4, 4
        tile_h, tile_w = tile_size
        grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

        for index, image in enumerate(images[: rows * cols]):
            row = index // cols
            col = index % cols
            resized = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
            y1 = row * tile_h
            y2 = y1 + tile_h
            x1 = col * tile_w
            x2 = x1 + tile_w
            grid[y1:y2, x1:x2] = resized

        return grid

    def on_epoch_end(self, epoch, logs=None):
        rendered_images = []
        for sample in self.samples:
            rendered = self._infer_image(sample)
            if rendered is not None:
                rendered_images.append(rendered)

        if not rendered_images:
            return

        grid = self._make_grid(rendered_images)
        output_path = os.path.join(self.samples_dir, f"epoch_{epoch + 1}.png")
        cv2.imwrite(output_path, grid)

        grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
        grid_rgb = np.expand_dims(grid_rgb, axis=0)
        with self.writer.as_default():
            tf.summary.image("inspection/predictions", grid_rgb, step=epoch + 1)
            self.writer.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Train a scratch R-CNN detector on COCO 2017.")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU id, use -1 for all available GPUs/CPU.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint.")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of epochs to train.")
    args = parser.parse_args()

    if COCO_ROOT is None:
        raise RuntimeError(
            "COCO_ROOT was not found. Update config.py with a valid COCO 2017 root.")

    ensure_dir(LOG_DIR)
    strategy, _ = setup_runtime(args.gpu)

    train_meta, train_dataset = build_train_dataset(COCO_ROOT, TRAIN_SPLIT)
    val_meta, val_dataset = build_val_dataset(COCO_ROOT, VAL_SPLIT)

    tf.keras.backend.clear_session()
    with strategy.scope():
        model = create_compiled_model()
    model.summary()

    initial_epoch = 0
    if args.resume and os.path.exists(CHECKPOINT_PATH):
        model.load_weights(CHECKPOINT_PATH)
        initial_epoch = load_state()
        print(f"Resuming from epoch {initial_epoch}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            save_weights_only=True,
            monitor="val_class_logits_accuracy",
            mode="max",
            save_best_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_class_logits_accuracy",
            mode="max",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_class_logits_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
        InspectionCallback(val_meta.samples, LOG_DIR, grid_size=16),
        HistorySaver(HISTORY_PATH),
        StateSaver(),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=train_meta.steps_per_epoch,
        validation_steps=VALIDATION_STEPS,
        callbacks=callbacks,
    )

    model.save_weights(CHECKPOINT_PATH)
    completed_epochs = initial_epoch + len(history.epoch)
    save_state(completed_epochs)
    print(f"Training finished. Best artifacts saved under: {LOG_DIR}")
    print("History keys:", list(history.history.keys()))
    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
