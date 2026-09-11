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

MODEL_CLASS = GAN
LARGE_IMAGE_DATASETS = {"celeba", "anime_faces"}


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
    def __init__(self, model, log_dir, latent_dim):
        super().__init__()
        self.model_ref = model
        self.latent_dim = latent_dim
        self.fixed_noise = np.random.normal(0, 1, (8, latent_dim))

        self.img_dir = os.path.join(log_dir, "samples")
        os.makedirs(self.img_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if not should_run_periodic_callback(self, epoch, SAMPLE_EVERY_N_EPOCHS):
            return

        random_noise = np.random.normal(0, 1, (8, self.latent_dim))
        noise = np.concatenate([self.fixed_noise, random_noise], axis=0)

        generated = self.model_ref.generator(noise, training=False)
        generated = ((generated + 1.0) / 2.0).numpy()

        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for index in range(16):
            row, col = divmod(index, 4)
            axes[row, col].imshow(generated[index])
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(self.img_dir, f"epoch_{epoch + 1}.png"))
        plt.close()


class GANLRScheduler(tf.keras.callbacks.Callback):
    def __init__(self, gen_opt, disc_opt, factor=0.5, patience=15):
        super().__init__()
        self.gen_opt = gen_opt
        self.disc_opt = disc_opt
        self.factor = factor
        self.patience = patience
        self.wait = 0
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        gen_loss = logs.get("gen_loss", 0.0)

        if gen_loss < self.best:
            self.best = gen_loss
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            new_lr_g = self.gen_opt.learning_rate * self.factor
            new_lr_d = self.disc_opt.learning_rate * self.factor

            self.gen_opt.learning_rate.assign(new_lr_g)
            self.disc_opt.learning_rate.assign(new_lr_d)

            print(f"LR reduced -> G: {new_lr_g.numpy()}, D: {new_lr_d.numpy()}")
            self.wait = 0


class ModeCollapseCallback(tf.keras.callbacks.Callback):
    def __init__(self, latent_dim, num_samples=32, threshold=0.05):
        super().__init__()
        self.threshold = threshold
        self.fixed_noise = tf.random.normal([num_samples, latent_dim])

    @tf.function
    def compute_diversity_graph(self, generator, noise):
        samples = generator(noise, training=False)
        flattened = tf.reshape(samples, [tf.shape(samples)[0], -1])
        diffs = tf.expand_dims(flattened, 1) - tf.expand_dims(flattened, 0)
        distances = tf.linalg.norm(diffs, axis=-1)
        mask = 1.0 - tf.eye(tf.shape(flattened)[0])
        return tf.reduce_sum(distances * mask) / tf.reduce_sum(mask)

    def on_epoch_end(self, epoch, logs=None):
        if not should_run_periodic_callback(self, epoch, SAMPLE_EVERY_N_EPOCHS):
            return

        diversity = self.compute_diversity_graph(
            self.model.generator,
            self.fixed_noise,
        )
        diversity_val = float(diversity.numpy())

        print(f"\n[ModeCollapse] Diversity: {diversity_val:.6f}")
        if diversity_val < self.threshold:
            print("[WARNING] Mode collapse likely detected")

        if logs is not None:
            logs["diversity"] = diversity_val


def save_final_generated_grid(model, log_dir, latent_dim):
    noise = tf.random.normal([16, latent_dim])
    generated = ((model.generator(noise, training=False) + 1.0) / 2.0).numpy()

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for index in range(16):
        row, col = divmod(index, 4)
        axes[row, col].imshow(generated[index])
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "final_grid.png"))
    plt.close()


def build_model(strategy, dataset_type, channels):
    is_large = dataset_type in LARGE_IMAGE_DATASETS
    latent_dim = LATENT_DIM * 4 if is_large else LATENT_DIM
    input_shape = (
        IMAGE_SIZE[0] * 2,
        IMAGE_SIZE[1] * 2,
        channels,
    ) if is_large else (
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        channels,
    )

    model = MODEL_CLASS(
        strategy=strategy,
        input_shape=input_shape,
        latent_dim=(latent_dim,),
        batch_size=BATCH_SIZE,
    )
    return model, latent_dim


def build_callbacks(model, log_dir, latent_dim):
    return [
        EpochTracker(model),
        GANLogger(log_dir),
        WeightSaveCallback(model, log_dir),
        SampleImageCallback(model, log_dir, latent_dim),
        ModeCollapseCallback(latent_dim),
        GANLRScheduler(
            model.generator_optimizer,
            model.discriminator_optimizer,
        ),
    ]


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

    strategy = build_strategy()
    train_ds = strategy.experimental_distribute_dataset(train_ds)

    log_dir = f"logs/{args.type}/GAN"
    os.makedirs(log_dir, exist_ok=True)

    model, latent_dim = build_model(strategy, args.type, channels)

    start_epoch = 0
    if args.resume:
        load_weights_if_needed(model, log_dir, True)
        start_epoch = load_training_state(log_dir)
        print(f"Resuming from epoch {start_epoch}")

    setup_interrupt_handler(model, log_dir)
    callbacks = build_callbacks(model, log_dir, latent_dim)

    try:
        model.fit(
            train_ds,
            epochs=EPOCHS,
            initial_epoch=start_epoch,
            path=f"{args.type}/GAN",
            callbacks=callbacks,
        )
        save_final_generated_grid(model, log_dir, latent_dim)
    finally:
        dataset.cleanup_cache()


if __name__ == "__main__":
    main()
