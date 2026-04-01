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
    def __init__(self, log_dir): super().__init__(
    ); self.log_dir = log_dir; self.history = {"gen_loss": [], "disc_loss": []}

    def on_epoch_end(self, epoch, logs=None):
        self.history["gen_loss"].append(float(logs["gen_loss"]))
        self.history["disc_loss"].append(float(logs["disc_loss"]))
        with open(os.path.join(self.log_dir, "history.json"), "w") as f:
            json.dump(self.history, f)


def save_final_generated_grid(model, log_dir, fixed_noise):
    os.makedirs(log_dir, exist_ok=True)
    gen = (model.generator(fixed_noise, training=False)+1.0)/2.0
    gen = gen.numpy()
    fig, ax = plt.subplots(4, 4, figsize=(6, 6))
    k = 0
    for i in range(4):
        for j in range(4):
            img = gen[k]
            ax[i, j].imshow(
                img[:, :, 0], cmap="gray") if img.shape[-1] == 1 else ax[i, j].imshow(img)
            ax[i, j].axis("off")
            k += 1
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "final_grid.png"))
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=-1)
    p.add_argument('--type', type=str, default='cifar10', choices=[
                   'cifar10', 'fashion_mnist', 'mnist', 'cifar100', 'celeba', 'anime_faces'])
    a = p.parse_args()

    setup_gpu(a.gpu)

    d = Dataset()
    train_ds, test_ds, ch = d.load_data(a.type)

    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce())
    train_ds = strategy.experimental_distribute_dataset(train_ds)

    print(f"Dataset: {a.type}")
    print(f"Devices: {strategy.num_replicas_in_sync}")

    log_dir = f"logs/{a.type}/GAN"
    os.makedirs(log_dir, exist_ok=True)


    if a.type in ['celeba', 'anime_faces']:
        model = GAN(strategy=strategy, input_shape=(
            IMAGE_SIZE[0]*4, IMAGE_SIZE[1]*4, ch), latent_dim=(LATENT_DIM*4,), batch_size=BATCH_SIZE)
        print("Using larger latent dimension for high-res dataset")
    else:
        model = GAN(strategy=strategy, input_shape=(
            IMAGE_SIZE[0], IMAGE_SIZE[1], ch), latent_dim=(LATENT_DIM,), batch_size=BATCH_SIZE)
        print("Using standard latent dimension for low-res dataset")

    model.fit(train_ds, epochs=EPOCHS,
              path=f"{a.type}/GAN", callbacks=[GANLogger(log_dir)])

    if a.type in ['celeba', 'anime_faces']:
        print("Saving final grid with larger latent dimension")
        fixed_noise = tf.random.normal([16, LATENT_DIM*4])
        save_final_generated_grid(model, log_dir, fixed_noise)
    else:
        print("Saving final grid with standard latent dimension")
        fixed_noise = tf.random.normal([16, LATENT_DIM])
        save_final_generated_grid(model, log_dir, fixed_noise)


if __name__ == "__main__":
    main()
