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

MODEL_NAME = "StarGAN"


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
    total_epochs = callback.params.get("epochs", current_epoch)
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
    def __init__(self, model, log_dir, condition_spec, fixed_images):
        super().__init__()
        self.model_ref = model
        self.condition_spec = condition_spec
        self.img_dir = os.path.join(log_dir, "samples")
        os.makedirs(self.img_dir, exist_ok=True)
        self.fixed_images = fixed_images

    def on_epoch_end(self, epoch, logs=None):
        if not should_run_periodic_callback(self, epoch, SAMPLE_EVERY_N_EPOCHS):
            return

        target_labels = tf.convert_to_tensor(
            self.condition_spec.make_sample_labels(8),
            dtype=tf.float32
        )
        translated = self.model_ref.generator((self.fixed_images, target_labels), training=False).numpy()
        real_images = ((self.fixed_images + 1.0) / 2.0).numpy()
        translated = (translated + 1.0) / 2.0

        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for index in range(8):
            row, col = divmod(index, 4)
            axes[row * 2, col].imshow(np.squeeze(real_images[index]), cmap="gray" if real_images[index].shape[-1] == 1 else None)
            axes[row * 2, col].axis("off")
            axes[row * 2, col].set_title("source", fontsize=8)

            axes[row * 2 + 1, col].imshow(np.squeeze(translated[index]), cmap="gray" if translated[index].shape[-1] == 1 else None)
            axes[row * 2 + 1, col].axis("off")
            axes[row * 2 + 1, col].set_title(self.condition_spec.format_label(target_labels[index].numpy()), fontsize=6)

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
    train_ds, channels, condition_spec = dataset.load_data(args.type)
    fixed_images = dataset.get_preview_images(args.type, count=8)

    strategy = build_strategy()
    train_ds = strategy.experimental_distribute_dataset(train_ds)

    input_shape = (
        IMAGE_SIZE[0] * 2,
        IMAGE_SIZE[1] * 2,
        channels,
    ) if args.type in {"celeba", "chest_x_ray"} else (
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        channels,
    )

    model = StarGAN(
        strategy=strategy,
        input_shape=input_shape,
        batch_size=BATCH_SIZE,
        label_dim=condition_spec.label_dim,
        label_mode=condition_spec.mode,
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
        SampleImageCallback(model, log_dir, condition_spec, fixed_images),
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
