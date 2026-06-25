import argparse
import gc
import json
import os

import tensorflow as tf

from config import (
    CHECKPOINT_PATH,
    COCO_ROOT,
    EPOCHS,
    HISTORY_PATH,
    LOG_DIR,
    STATE_PATH,
    STEPS_PER_EPOCH,
    TRAIN_SPLIT,
    VALIDATION_STEPS,
    VAL_SPLIT,
    ensure_dir,
)
from dataset import build_train_dataset, build_val_dataset
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


def load_state():
    if not os.path.exists(STATE_PATH):
        return 0
    with open(STATE_PATH, "r", encoding="utf-8") as file:
        return int(json.load(file).get("epoch", 0))


def save_state(epoch):
    with open(STATE_PATH, "w", encoding="utf-8") as file:
        json.dump({"epoch": int(epoch)}, file)


class HistorySaver(tf.keras.callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            self.history.setdefault(key, []).append(float(value))
        with open(self.path, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=2)


class StateSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        save_state(epoch + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    if COCO_ROOT is None:
        raise RuntimeError("COCO_ROOT not found")

    ensure_dir(LOG_DIR)
    strategy = setup_runtime(args.gpu)
    train_meta, train_dataset = build_train_dataset(COCO_ROOT, TRAIN_SPLIT)
    _, val_dataset = build_val_dataset(COCO_ROOT, VAL_SPLIT)

    tf.keras.backend.clear_session()
    with strategy.scope():
        model = create_compiled_model()

    initial_epoch = 0
    if args.resume and os.path.exists(CHECKPOINT_PATH):
        model.load_weights(CHECKPOINT_PATH)
        initial_epoch = load_state()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_class_logits_accuracy", mode="max", factor=0.5, patience=2),
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
        HistorySaver(HISTORY_PATH),
        StateSaver(),
    ]
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        callbacks=callbacks,
    )
    model.save_weights(CHECKPOINT_PATH)
    save_state(initial_epoch + len(history.epoch))
    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()

