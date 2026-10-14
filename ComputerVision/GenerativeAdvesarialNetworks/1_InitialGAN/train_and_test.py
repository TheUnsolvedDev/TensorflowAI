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

MODEL_CLASS = NN_GAN
LARGE_IMAGE_DATASETS = {"celeba", "anime_faces"}


def setup_gpu(gpu_id):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpu_id == -1:
        print(f"Using all {len(gpus)} visible GPU(s)" if gpus else "Using CPU")
    elif 0 <= gpu_id < len(gpus):
        tf.config.set_visible_devices(gpus[gpu_id], "GPU")
        print(f"Using GPU {gpu_id}")
    else:
        tf.config.set_visible_devices([], "GPU")
        print("Using CPU")


def build_strategy():
    if len(tf.config.get_visible_devices("GPU")) > 1:
        return tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.NcclAllReduce()
        )
    return tf.distribute.MirroredStrategy()


def get_weight_paths(log_dir):
    return (
        os.path.join(log_dir, "generator.weights.h5"),
        os.path.join(log_dir, "discriminator.weights.h5"),
    )


def get_state_path(log_dir):
    return os.path.join(log_dir, "training_state.json")


def save_training_state(log_dir, epoch):
    with open(get_state_path(log_dir), "w") as handle:
        json.dump({"epoch": epoch, "model_version": MODEL_VERSION}, handle)


def load_training_state(log_dir):
    path = get_state_path(log_dir)
    if os.path.exists(path):
        with open(path, "r") as handle:
            state = json.load(handle)
        if state.get("model_version") == MODEL_VERSION:
            return state.get("epoch", 0)
    return 0


def build_checkpoint(model, log_dir):
    checkpoint = tf.train.Checkpoint(
        generator=model.generator,
        discriminator=model.discriminator,
        generator_optimizer=model.generator_optimizer,
        discriminator_optimizer=model.discriminator_optimizer,
        generator_ema=model.ema,
    )
    return checkpoint, tf.train.CheckpointManager(
        checkpoint,
        os.path.join(log_dir, "checkpoints"),
        max_to_keep=2,
    )


def load_weights_if_needed(model, log_dir, resume, checkpoint, manager):
    gen_path, disc_path = get_weight_paths(log_dir)
    if resume and manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"Restored checkpoint: {manager.latest_checkpoint}")
        return True
    if (
        load_training_state(log_dir) > 0
        and os.path.exists(gen_path)
        and os.path.exists(disc_path)
    ):
        model.generator.load_weights(gen_path)
        model.discriminator.load_weights(disc_path)
        print("Loaded legacy model weights (optimizer starts fresh)")
        return True
    print("No compatible weights found, starting fresh")
    return False


def setup_interrupt_handler(model, log_dir, manager):
    def handler(sig, frame):
        print("\nInterrupt received. Saving state...")

        gen_path, disc_path = get_weight_paths(log_dir)
        model.generator.save_weights(gen_path)
        model.discriminator.save_weights(disc_path)
        manager.save()

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
    def __init__(self, log_dir, resume=False):
        super().__init__()
        self.log_dir = log_dir
        self.history = {"gen_loss": [], "disc_loss": []}
        history_path = os.path.join(log_dir, "history.json")
        if resume and os.path.exists(history_path):
            with open(history_path, "r") as handle:
                self.history.update(json.load(handle))

    def on_epoch_end(self, epoch, logs=None):
        for name, value in logs.items():
            self.history.setdefault(name, []).append(float(value))

        with open(os.path.join(self.log_dir, "history.json"), "w") as handle:
            json.dump(self.history, handle)


class WeightSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, log_dir, manager):
        super().__init__()
        self.model_ref = model
        self.log_dir = log_dir
        self.manager = manager

    def on_epoch_end(self, epoch, logs=None):
        if not should_run_periodic_callback(self, epoch, SAVE_EVERY_N_EPOCHS):
            return

        gen_path, disc_path = get_weight_paths(self.log_dir)

        self.model_ref.generator.save_weights(gen_path)
        self.model_ref.discriminator.save_weights(disc_path)
        self.manager.save()
        save_training_state(self.log_dir, epoch + 1)

        print(f"Saved weights and state at epoch {epoch + 1}")


class SampleImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, log_dir, latent_dim):
        super().__init__()
        self.model_ref = model
        self.fixed_noise = tf.random.normal([16, latent_dim])

        self.img_dir = os.path.join(log_dir, "samples")
        os.makedirs(self.img_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if not should_run_periodic_callback(self, epoch, SAMPLE_EVERY_N_EPOCHS):
            return

        with self.model_ref.ema.average_weights():
            generated = self.model_ref.generator(self.fixed_noise, training=False)
        generated = ((generated + 1.0) / 2.0).numpy()

        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for index in range(16):
            row, col = divmod(index, 4)
            axes[row, col].imshow(generated[index])
            axes[row, col].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(self.img_dir, f"epoch_{epoch + 1}.png"))
        plt.close()


class CollapseMonitor(tf.keras.callbacks.Callback):
    def __init__(self, model, latent_dim):
        super().__init__()
        self.model_ref = model
        self.fixed_noise = tf.random.normal([32, latent_dim], seed=42)

    def on_epoch_end(self, epoch, logs=None):
        generated = self.model_ref.generator(self.fixed_noise, training=False)
        sample_stddev = float(
            tf.reduce_mean(tf.math.reduce_std(generated, axis=0)).numpy()
        )
        logs["sample_stddev"] = sample_stddev
        if sample_stddev < COLLAPSE_STDDEV_THRESHOLD:
            print(
                f"[WARNING] Low sample diversity ({sample_stddev:.4f}); "
                "mode collapse may be occurring"
            )


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


def save_final_generated_grid(model, log_dir, latent_dim):
    noise = tf.random.normal([16, latent_dim])
    with model.ema.average_weights():
        generated = ((model.generator(noise, training=False) + 1.0) / 2.0).numpy()

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for index in range(16):
        row, col = divmod(index, 4)
        axes[row, col].imshow(generated[index])
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "final_grid.png"))
    plt.close()


def build_model(strategy, dataset_type, channels, batch_size):
    is_large = dataset_type in LARGE_IMAGE_DATASETS
    latent_dim = LATENT_DIM * 4 if is_large else LATENT_DIM
    input_shape = (
        IMAGE_SIZE[0] * 4,
        IMAGE_SIZE[1] * 4,
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
        batch_size=batch_size,
    )
    return model, latent_dim


def build_callbacks(model, log_dir, latent_dim, manager, resume):
    return [
        EpochTracker(model),
        CollapseMonitor(model, latent_dim),
        GANLogger(log_dir, resume=resume),
        WeightSaveCallback(model, log_dir, manager),
        SampleImageCallback(model, log_dir, latent_dim),
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
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    setup_gpu(args.gpu)

    strategy = build_strategy()
    replicas = strategy.num_replicas_in_sync
    if args.batch_size < replicas or args.batch_size % replicas:
        parser.error(
            f"--batch-size must be at least {replicas} and divisible by {replicas}"
        )

    dataset = Dataset(batch_size=args.batch_size)
    train_ds, _, channels = dataset.load_data(args.type)
    train_ds = strategy.experimental_distribute_dataset(train_ds)

    log_dir = f"logs/{args.type}/GAN"
    os.makedirs(log_dir, exist_ok=True)

    model, latent_dim = build_model(strategy, args.type, channels, args.batch_size)
    checkpoint, manager = build_checkpoint(model, log_dir)

    start_epoch = 0
    weights_loaded = load_weights_if_needed(
        model, log_dir, args.resume, checkpoint, manager
    )
    if args.resume and weights_loaded:
        start_epoch = load_training_state(log_dir)
        print(f"Resuming from epoch {start_epoch}")
    elif weights_loaded:
        print("Starting from epoch 0 with loaded weights")

    setup_interrupt_handler(model, log_dir, manager)
    callbacks = build_callbacks(
        model,
        log_dir,
        latent_dim,
        manager,
        args.resume and start_epoch > 0,
    )

    model.fit(
        train_ds,
        epochs=args.epochs,
        initial_epoch=start_epoch,
        path=f"{args.type}/GAN",
        callbacks=callbacks,
    )
    save_final_generated_grid(model, log_dir, latent_dim)


if __name__ == "__main__":
    main()
