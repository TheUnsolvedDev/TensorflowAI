import silence_tensorflow.auto
import tensorflow as tf
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from model import *
from dataset import *
from config import *


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

def get_weight_paths(log_dir):
    gen_path = os.path.join(log_dir, "generator.weights.h5")
    disc_path = os.path.join(log_dir, "discriminator.weights.h5")
    return gen_path, disc_path


def load_weights_if_needed(model, log_dir, resume):
    gen_path, disc_path = get_weight_paths(log_dir)

    if resume:
        if os.path.exists(gen_path) and os.path.exists(disc_path):
            model.generator.load_weights(gen_path)
            model.discriminator.load_weights(disc_path)
            print("Loaded existing weights")
        else:
            print("No saved weights found, starting fresh")


class WeightSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, log_dir):
        super().__init__()
        self.model_ref = model
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        gen_path, disc_path = get_weight_paths(self.log_dir)

        self.model_ref.generator.save_weights(gen_path)
        self.model_ref.discriminator.save_weights(disc_path)

        print(f"Saved weights at epoch {epoch+1}")

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

class GeneratorEMA(tf.keras.callbacks.Callback):
    def __init__(self, model, decay=0.999):
        super().__init__()
        self.model_ref = model
        self.decay = decay
        self.ema_weights = [tf.Variable(w, trainable=False)
                            for w in model.generator.weights]

    def on_train_batch_end(self, batch, logs=None):
        for ema_w, w in zip(self.ema_weights, self.model_ref.generator.weights):
            ema_w.assign(self.decay * ema_w + (1.0 - self.decay) * w)

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

class GradientMonitor(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super().__init__()
        self.model_ref = model

    def on_train_batch_end(self, batch, logs=None):
        g_norm = tf.linalg.global_norm(self.model_ref.gen_gradients)
        d_norm = tf.linalg.global_norm(self.model_ref.disc_gradients)

        if tf.math.is_nan(g_norm) or tf.math.is_nan(d_norm):
            print("NaN gradients detected. Stopping.")
            self.model.stop_training = True

class CollapseDetector(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def on_epoch_end(self, epoch, logs=None):
        g_loss = logs["gen_loss"]
        d_loss = logs["disc_loss"]

        if g_loss > 5.0 and d_loss < 0.1:
            self.counter += 1
        else:
            self.counter = 0

        if self.counter >= 3:
            print("Mode collapse detected. Stopping.")
            self.model.stop_training = True


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
        model = WGAN(
            strategy=strategy,
            input_shape=(IMAGE_SIZE[0]*4, IMAGE_SIZE[1]*4, ch),
            latent_dim=(LATENT_DIM*4,),
            batch_size=BATCH_SIZE//4
        )
        latent_dim = LATENT_DIM * 4
    else:
        model = WGAN(
            strategy=strategy,
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], ch),
            latent_dim=(LATENT_DIM,),
            batch_size=BATCH_SIZE
        )
        latent_dim = LATENT_DIM

    load_weights_if_needed(model, log_dir, a.resume)

    callbacks = [
        GANLogger(log_dir),
        WeightSaveCallback(model, log_dir),
        SampleImageCallback(model, log_dir, latent_dim),
        GeneratorEMA(model),
        GANLRScheduler(model.generator_optimizer,
                       model.discriminator_optimizer),
        # GradientMonitor(model),
        # CollapseDetector()
    ]

    model.fit(
        train_ds,
        epochs=EPOCHS,
        path=f"{a.type}/GAN",
        callbacks=callbacks
    )

    save_final_generated_grid(model, log_dir, latent_dim)


if __name__ == "__main__":
    main()
