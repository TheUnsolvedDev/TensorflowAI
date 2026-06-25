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
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

    if gpu_id == -1:
        print("Using all GPUs")
    elif 0 <= gpu_id < len(gpus):
        tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
        print(f"Using GPU {gpu_id}")
    else:
        print("Invalid GPU ID, using CPU")


# =========================
# CHECKPOINT UTILS
# =========================
def get_weight_paths(log_dir):
    return (
        os.path.join(log_dir, "generator.weights.h5"),
        os.path.join(log_dir, "discriminator.weights.h5"),
    )


def get_state_path(log_dir):
    return os.path.join(log_dir, "training_state.json")


def save_training_state(log_dir, epoch):
    with open(get_state_path(log_dir), "w") as f:
        json.dump({"epoch": epoch}, f)


def load_training_state(log_dir):
    path = get_state_path(log_dir)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f).get("epoch", 0)
    return 0


def load_weights_if_needed(model, log_dir, resume):
    gen_path, disc_path = get_weight_paths(log_dir)

    if resume and os.path.exists(gen_path):
        model.generator.load_weights(gen_path)
        model.discriminator.load_weights(disc_path)
        print("Loaded saved weights")


def setup_interrupt_handler(model, log_dir):
    def handler(sig, frame):
        print("\nInterrupt detected. Saving...")

        gen_path, disc_path = get_weight_paths(log_dir)
        model.generator.save_weights(gen_path)
        model.discriminator.save_weights(disc_path)

        epoch = getattr(model, "_current_epoch", 0)
        save_training_state(log_dir, epoch)

        print(f"Saved at epoch {epoch}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)


# =========================
# CALLBACKS
# =========================
class EpochTracker(tf.keras.callbacks.Callback):
    def __init__(self, model):
        self.model_ref = model

    def on_epoch_begin(self, epoch, logs=None):
        self.model_ref._current_epoch = epoch


class GANLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.history = {"gen_loss": [], "disc_loss": []}

    def on_epoch_end(self, epoch, logs=None):
        self.history["gen_loss"].append(float(logs["gen_loss"]))
        self.history["disc_loss"].append(float(logs["disc_loss"]))

        with open(os.path.join(self.log_dir, "history.json"), "w") as f:
            json.dump(self.history, f)


class WeightSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, log_dir):
        self.model_ref = model
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        gen_path, disc_path = get_weight_paths(self.log_dir)

        self.model_ref.generator.save_weights(gen_path)
        self.model_ref.discriminator.save_weights(disc_path)

        save_training_state(self.log_dir, epoch + 1)
        print(f"Saved checkpoint at epoch {epoch+1}")


class SampleImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, log_dir, latent_dim, num_classes, label_mode="onehot", multilabel_dim=None):
        self.model_ref = model
        self.fixed_noise = np.random.normal(0, 1, (8, latent_dim))
        self.label_mode = label_mode
        self.latent_dim = latent_dim
        base = np.arange(16) % num_classes

        if label_mode == "multilabel":
            if multilabel_dim is None: raise ValueError("multilabel_dim required")
            labels = (np.random.rand(16, multilabel_dim) > 0.5).astype(np.float32)
        else:
            labels = np.eye(num_classes)[base].astype(np.float32)

        self.labels = tf.convert_to_tensor(labels)

        self.img_dir = os.path.join(log_dir, "samples")
        os.makedirs(self.img_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        random_noise = np.random.normal(0, 1, (8, self.latent_dim))
        self.noise = tf.convert_to_tensor(
            np.concatenate([self.fixed_noise, random_noise], axis=0),
            dtype=tf.float32
        )
        gen = self.model_ref.generator((self.noise, self.labels), training=False)
        gen = ((gen + 1.0) / 2.0).numpy()

        fig, ax = plt.subplots(4, 4, figsize=(6, 6))
        for i in range(16):
            r, c = i // 4, i % 4
            ax[r, c].imshow(gen[i])
            ax[r, c].axis("off")

            if self.label_mode != "multilabel":
                label = int(np.argmax(self.labels[i].numpy()))
                ax[r, c].set_title(str(label), fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.img_dir, f"epoch_{epoch+1}.png"))
        plt.close()

class GANLRScheduler(tf.keras.callbacks.Callback):
    def __init__(self, gen_opt, disc_opt, factor=0.5, patience=15):
        self.gen_opt = gen_opt
        self.disc_opt = disc_opt
        self.factor = factor
        self.patience = patience
        self.wait = 0
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        g_loss = logs["gen_loss"]

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

            print(f"LR reduced: G={new_lr_g.numpy()} D={new_lr_d.numpy()}")
            self.wait = 0

class ModeCollapseCallback(tf.keras.callbacks.Callback):
    def __init__(self, latent_dim, num_classes, num_samples=32, threshold=0.05, label_mode="onehot", multilabel_dim=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.threshold = threshold
        self.label_mode = label_mode
        self.num_classes = num_classes
        self.multilabel_dim = multilabel_dim

        self.fixed_noise = tf.random.normal([num_samples, latent_dim])

        base = np.arange(num_samples) % num_classes

        if label_mode == "multilabel":
            if multilabel_dim is None: raise ValueError("multilabel_dim required")
            labels = (np.random.rand(num_samples, multilabel_dim) > 0.5).astype(np.float32)
        else:
            labels = np.eye(num_classes)[base].astype(np.float32)

        self.fixed_labels = tf.convert_to_tensor(labels)

    @tf.function
    def compute_diversity_graph(self, generator, noise, labels):
        samples = generator((noise, labels), training=False)
        x = tf.reshape(samples, [tf.shape(samples)[0], -1])
        diffs = tf.expand_dims(x, 1) - tf.expand_dims(x, 0)
        dists = tf.linalg.norm(diffs, axis=-1)
        mask = 1.0 - tf.eye(tf.shape(x)[0])
        return tf.reduce_sum(dists * mask) / tf.reduce_sum(mask)

    def on_epoch_end(self, epoch, logs=None):
        diversity = self.compute_diversity_graph(
            self.model.generator,
            self.fixed_noise,
            self.fixed_labels
        )

        diversity_val = float(diversity.numpy())

        print(f"\n[ModeCollapse] Diversity: {diversity_val:.6f}")

        if diversity_val < self.threshold:
            print("[WARNING] Mode collapse likely detected")

        if logs is not None:
            logs["diversity"] = diversity_val

def save_final_grid(model, log_dir, latent_dim, num_classes, label_mode="onehot", multilabel_dim=None):
    noise = tf.random.normal([16, latent_dim])
    base = np.arange(16) % num_classes

    if label_mode == "multilabel":
        if multilabel_dim is None: raise ValueError("multilabel_dim required")
        labels = (np.random.rand(16, multilabel_dim) > 0.5).astype(np.float32)
    else:
        labels = np.eye(num_classes)[base].astype(np.float32)

    labels = tf.convert_to_tensor(labels)

    gen = model.generator((noise, labels), training=False)
    gen = ((gen + 1.0) / 2.0).numpy()

    fig, ax = plt.subplots(4, 4, figsize=(6, 6))
    for i in range(16):
        r, c = i // 4, i % 4
        ax[r, c].imshow(gen[i])
        ax[r, c].axis("off")

        if label_mode != "multilabel":
            label = int(np.argmax(labels[i].numpy()))
            ax[r, c].set_title(f"class {label}", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "final_grid.png"))
    plt.close()


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--type', type=str, default='cifar10')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    setup_gpu(args.gpu)

    dataset = Dataset()
    train_ds, ch = dataset.load_data(args.type)

    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce())
    # train_ds = train_ds
    train_ds = strategy.experimental_distribute_dataset(train_ds)

    log_dir = f"logs/{args.type}/CondGAN"
    os.makedirs(log_dir, exist_ok=True)

    num_classes = {
        "mnist": 10,
        "fashion_mnist": 10,
        "cifar10": 10,
        "cifar100": 100,
        "celeba": 40
    }.get(args.type, 10)

    if args.type in ['celeba', 'anime_faces']:
        latent_dim = LATENT_DIM * 4
        model = Cond_GAN(
            strategy=strategy,
            input_shape=(IMAGE_SIZE[0]*2, IMAGE_SIZE[1]*2, ch),
            latent_dim=(latent_dim,),
            batch_size=BATCH_SIZE,
            num_classes=num_classes
        )
    else:
        model = Cond_GAN(
            strategy=strategy,
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], ch),
            latent_dim=(LATENT_DIM,),
            batch_size=BATCH_SIZE,
            num_classes=num_classes
        )
        latent_dim = LATENT_DIM

    start_epoch = 0
    if args.resume:
        load_weights_if_needed(model, log_dir, True)
        start_epoch = load_training_state(log_dir)
        print(f"Resuming from epoch {start_epoch}")

    setup_interrupt_handler(model, log_dir)
    label_mode = "multilabel" if args.type == 'celeba' else "onehot"
    multilabel_dim = 40 if args.type == 'celeba' else None

    callbacks = [
        EpochTracker(model),
        GANLogger(log_dir),
        WeightSaveCallback(model, log_dir),
        SampleImageCallback(model, log_dir, latent_dim,
                            num_classes, label_mode, multilabel_dim),
        GANLRScheduler(
            model.generator_optimizer,
            model.discriminator_optimizer
        ),
        ModeCollapseCallback(
            latent_dim,
            num_classes,
            label_mode=label_mode,
            multilabel_dim=multilabel_dim
        )
    ]

    model.fit(
        train_ds,
        epochs=EPOCHS,
        initial_epoch=start_epoch,
        path=f"{args.type}/CondGAN",
        callbacks=callbacks
    )

    save_final_grid(model, log_dir, latent_dim, num_classes, label_mode, multilabel_dim)


if __name__ == "__main__":
    main()
