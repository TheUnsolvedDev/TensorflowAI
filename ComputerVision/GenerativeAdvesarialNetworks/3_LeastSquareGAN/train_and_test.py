import silence_tensorflow.auto
import tensorflow as tf
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import signal
import sys

from model import *
from dataset import *
from config import *


# =========================
# GPU SETUP
# =========================
def setup_gpu(gpu_id):
    gpus = tf.config.list_physical_devices('GPU')
    [tf.config.experimental.set_memory_growth(g, True) for g in gpus]

    if gpu_id == -1:
        print("Using all GPUs")
    elif 0 <= gpu_id < len(gpus):
        tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
        print(f"Using GPU {gpu_id}")
    else:
        print("Invalid GPU ID, using CPU")

def get_weight_paths(log_dir):
    gen_path = os.path.join(log_dir, "generator.weights.h5")
    disc_path = os.path.join(log_dir, "discriminator.weights.h5")
    return gen_path, disc_path


def get_state_path(log_dir):
    return os.path.join(log_dir, "training_state.json")


def save_training_state(log_dir, epoch):
    state = {"epoch": epoch}
    with open(get_state_path(log_dir), "w") as f:
        json.dump(state, f)


def load_training_state(log_dir):
    path = get_state_path(log_dir)
    if os.path.exists(path):
        with open(path, "r") as f:
            state = json.load(f)
        return state.get("epoch", 0)
    return 0

def load_weights_if_needed(model, log_dir, resume):
    gen_path, disc_path = get_weight_paths(log_dir)

    if resume:
        if os.path.exists(gen_path) and os.path.exists(disc_path):
            model.generator.load_weights(gen_path)
            model.discriminator.load_weights(disc_path)
            print("Loaded existing weights")
        else:
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

        with open(os.path.join(self.log_dir, "history.json"), "w") as f:
            json.dump(self.history, f)


class WeightSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, log_dir):
        super().__init__()
        self.model_ref = model
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        gen_path, disc_path = get_weight_paths(self.log_dir)

        self.model_ref.generator.save_weights(gen_path)
        self.model_ref.discriminator.save_weights(disc_path)

        save_training_state(self.log_dir, epoch + 1)

        print(f"Saved weights and state at epoch {epoch+1}")


class SampleImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, log_dir, latent_dim):
        super().__init__()
        self.model_ref = model
        self.fixed_noise = tf.random.normal([16, latent_dim])

        self.img_dir = os.path.join(log_dir, "samples")
        os.makedirs(self.img_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        gen = self.model_ref.generator(self.fixed_noise, training=False)
        gen = (gen + 1.0) / 2.0
        gen = gen.numpy()

        fig, ax = plt.subplots(4, 4, figsize=(6, 6))
        k = 0

        for i in range(4):
            for j in range(4):
                ax[i, j].imshow(gen[k])
                ax[i, j].axis("off")
                k += 1

        plt.tight_layout()
        plt.savefig(os.path.join(self.img_dir, f"epoch_{epoch+1}.png"))
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
        g_loss = logs.get("gen_loss", 0.0)

        if g_loss < self.best:
            self.best = g_loss
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            new_lr_g = self.gen_opt.learning_rate * self.factor
            new_lr_d = self.disc_opt.learning_rate * self.factor

            self.gen_opt.learning_rate.assign(new_lr_g)
            self.disc_opt.learning_rate.assign(new_lr_d)

            print(
                f"LR reduced -> G: {new_lr_g.numpy()}, D: {new_lr_d.numpy()}")
            self.wait = 0
            
import tensorflow as tf

class ModeCollapseCallback(tf.keras.callbacks.Callback):
    def __init__(self, latent_dim, num_samples=32, threshold=0.05):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.threshold = threshold

        self.fixed_noise = tf.random.normal([num_samples, latent_dim])

    @tf.function
    def compute_diversity_graph(self, generator, noise):
        samples = generator(noise, training=False)
        x = tf.reshape(samples, [tf.shape(samples)[0], -1])
        diffs = tf.expand_dims(x, 1) - tf.expand_dims(x, 0)
        dists = tf.linalg.norm(diffs, axis=-1)
        mask = 1.0 - tf.eye(tf.shape(x)[0])
        mean_dist = tf.reduce_sum(dists * mask) / tf.reduce_sum(mask)
        return mean_dist

    def on_epoch_end(self, epoch, logs=None):
        diversity = self.compute_diversity_graph(
            self.model.generator,
            self.fixed_noise
        )

        diversity_val = float(diversity.numpy())

        print(f"\n[ModeCollapse] Diversity: {diversity_val:.6f}")

        if diversity_val < self.threshold:
            print("[WARNING] Mode collapse likely detected")

        if logs is not None:
            logs["diversity"] = diversity_val

def save_final_generated_grid(model, log_dir, latent_dim):
    noise = tf.random.normal([16, latent_dim])

    gen = (model.generator(noise, training=False) + 1.0) / 2.0
    gen = gen.numpy()

    fig, ax = plt.subplots(4, 4, figsize=(6, 6))
    k = 0

    for i in range(4):
        for j in range(4):
            ax[i, j].imshow(gen[k])
            ax[i, j].axis("off")
            k += 1

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "final_grid.png"))
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=-1)
    p.add_argument('--type', type=str, default='celeba')
    p.add_argument('--continue', dest='resume', action='store_true')
    a = p.parse_args()

    setup_gpu(a.gpu)

    d = Dataset()
    train_ds, _, ch = d.load_data(a.type)

    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce()
    )

    train_ds = strategy.experimental_distribute_dataset(train_ds)

    log_dir = f"logs/{a.type}/GAN"
    os.makedirs(log_dir, exist_ok=True)

    if a.type in ['celeba', 'anime_faces']:
        model = LS_GAN(
            strategy=strategy,
            input_shape=(IMAGE_SIZE[0]*2, IMAGE_SIZE[1]*2, ch),
            latent_dim=(LATENT_DIM*2,),
            batch_size=BATCH_SIZE//2
        )
        latent_dim = LATENT_DIM * 2
    else:
        model = LS_GAN(
            strategy=strategy,
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], ch),
            latent_dim=(LATENT_DIM,),
            batch_size=BATCH_SIZE
        )
        latent_dim = LATENT_DIM

    start_epoch = 0

    if a.resume:
        load_weights_if_needed(model, log_dir, True)
        start_epoch = load_training_state(log_dir)
        print(f"Resuming from epoch {start_epoch}")

    setup_interrupt_handler(model, log_dir)
    callbacks = [
        EpochTracker(model),
        GANLogger(log_dir),
        WeightSaveCallback(model, log_dir),
        SampleImageCallback(model, log_dir, latent_dim),
        ModeCollapseCallback(latent_dim),
        GANLRScheduler(
            model.generator_optimizer,
            model.discriminator_optimizer
        ),
    ]
    model.fit(
        train_ds,
        epochs=EPOCHS,
        initial_epoch=start_epoch,
        path=f"{a.type}/GAN",
        callbacks=callbacks
    )

    save_final_generated_grid(model, log_dir, latent_dim)


if __name__ == "__main__":
    main()