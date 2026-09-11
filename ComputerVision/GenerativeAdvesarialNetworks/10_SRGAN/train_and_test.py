import argparse
import json
import os
import signal
import sys

import matplotlib.pyplot as plt
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf

from config import *
from dataset import *
from model import *

MODEL_NAME = "SRGAN"


def setup_gpu(gpu_id):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpu_id == -1:
        print("Using all GPUs")
    elif 0 <= gpu_id < len(gpus):
        tf.config.set_visible_devices(gpus[gpu_id], "GPU")
        print(f"Using GPU {gpu_id}")
    else:
        print("Invalid GPU ID, using CPU")


def build_strategy():
    return tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce()
    )


def get_weight_paths(log_dir):
    return (
        os.path.join(log_dir, "generator.weights.h5"),
        os.path.join(log_dir, "discriminator.weights.h5"),
    )


def get_state_path(log_dir):
    return os.path.join(log_dir, "training_state.json")


def save_training_state(log_dir, epoch):
    with open(get_state_path(log_dir), "w") as handle:
        json.dump({"epoch": epoch}, handle)


def load_training_state(log_dir):
    path = get_state_path(log_dir)
    if os.path.exists(path):
        with open(path, "r") as handle:
            return json.load(handle).get("epoch", 0)
    return 0


def load_weights_if_needed(model, log_dir, resume):
    gen_path, disc_path = get_weight_paths(log_dir)
    if resume and os.path.exists(gen_path) and os.path.exists(disc_path):
        model.generator.load_weights(gen_path)
        model.discriminator.load_weights(disc_path)
        print("Loaded existing weights")
    elif resume:
        print("No saved weights found, starting fresh")


def setup_interrupt_handler(model, log_dir):
    def handler(sig, frame):
        print("\nInterrupt received. Saving state...")

        gen_path, disc_path = get_weight_paths(log_dir)
        model.generator.save_weights(gen_path)
        model.discriminator.save_weights(disc_path)

        current_epoch = getattr(model, "_current_epoch", 0)
        save_training_state(log_dir, current_epoch)

        print(f"Saved at epoch {current_epoch}. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)


def should_run_periodic_callback(callback, epoch, frequency):
    current_epoch = epoch + 1
    params = getattr(callback, "params", None) or {}
    total_epochs = params.get("epochs", current_epoch)
    return current_epoch % frequency == 0 or current_epoch == total_epochs


class EpochTracker(tf.keras.callbacks.Callback):
    def __init__(self, model):
        self.model_ref = model

    def on_epoch_begin(self, epoch, logs=None):
        self.model_ref._current_epoch = epoch


class GANLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.history = {"gen_loss": [], "disc_loss": []}

    def on_epoch_end(self, epoch, logs=None):
        self.history["gen_loss"].append(float(logs["gen_loss"]))
        self.history["disc_loss"].append(float(logs["disc_loss"]))

        with open(os.path.join(self.log_dir, "history.json"), "w") as handle:
            json.dump(self.history, handle)


class WeightSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, log_dir):
        super().__init__()
        self.model_ref = model
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        if not should_run_periodic_callback(self, epoch, SAVE_EVERY_N_EPOCHS):
            return

        gen_path, disc_path = get_weight_paths(self.log_dir)
        self.model_ref.generator.save_weights(gen_path)
        self.model_ref.discriminator.save_weights(disc_path)
        save_training_state(self.log_dir, epoch + 1)
        print(f"Saved weights and state at epoch {epoch + 1}")


class SampleImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, log_dir, fixed_lr, fixed_hr):
        super().__init__()
        self.model_ref = model
        self.img_dir = os.path.join(log_dir, "samples")
        os.makedirs(self.img_dir, exist_ok=True)
        self.fixed_lr = fixed_lr
        self.fixed_hr = fixed_hr

    def on_epoch_end(self, epoch, logs=None):
        if not should_run_periodic_callback(self, epoch, SAMPLE_EVERY_N_EPOCHS):
            return

        sr_images = self.model_ref.generator(self.fixed_lr, training=False)

        lr_images = tf.image.resize((self.fixed_lr + 1.0) / 2.0, HR_IMAGE_SIZE).numpy()
        sr_images = ((sr_images + 1.0) / 2.0).numpy()
        hr_images = ((self.fixed_hr + 1.0) / 2.0).numpy()

        fig, axes = plt.subplots(3, 4, figsize=(8, 6))
        for index in range(4):
            axes[0, index].imshow(np.clip(lr_images[index], 0.0, 1.0))
            axes[0, index].axis("off")
            axes[0, index].set_title("LR", fontsize=8)

            axes[1, index].imshow(np.clip(sr_images[index], 0.0, 1.0))
            axes[1, index].axis("off")
            axes[1, index].set_title("SR", fontsize=8)

            axes[2, index].imshow(np.clip(hr_images[index], 0.0, 1.0))
            axes[2, index].axis("off")
            axes[2, index].set_title("HR", fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.img_dir, f"epoch_{epoch + 1}.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--type", type=str, default="celeba")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue", dest="resume", action="store_true")
    args = parser.parse_args()

    setup_gpu(args.gpu)

    dataset = Dataset()
    train_ds, channels = dataset.load_data(args.type)
    fixed_lr, fixed_hr = dataset.get_preview_batch(args.type, count=4)

    strategy = build_strategy()
    train_ds = strategy.experimental_distribute_dataset(train_ds)

    model = SRGAN(
        strategy=strategy,
        lr_input_shape=(LR_IMAGE_SIZE[0], LR_IMAGE_SIZE[1], channels),
        hr_input_shape=(HR_IMAGE_SIZE[0], HR_IMAGE_SIZE[1], channels),
        batch_size=BATCH_SIZE,
    )

    log_dir = f"logs/{args.type}/{MODEL_NAME}"
    os.makedirs(log_dir, exist_ok=True)

    start_epoch = 0
    if args.resume:
        load_weights_if_needed(model, log_dir, True)
        start_epoch = load_training_state(log_dir)
        print(f"Resuming from epoch {start_epoch}")

    setup_interrupt_handler(model, log_dir)
    callbacks = [
        EpochTracker(model),
        GANLogger(log_dir),
        WeightSaveCallback(model, log_dir),
        SampleImageCallback(model, log_dir, fixed_lr, fixed_hr),
    ]

    try:
        model.fit(
            train_ds,
            epochs=EPOCHS,
            initial_epoch=start_epoch,
            callbacks=callbacks,
        )
    finally:
        dataset.cleanup_cache()


if __name__ == "__main__":
    main()
