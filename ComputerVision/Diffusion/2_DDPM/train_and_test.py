import argparse
import json
import os
import signal

import numpy as np
import tensorflow as tf

from config import (BATCH_SIZE, DEFAULT_DATASET, DIFFUSION_STEPS, EPOCHS,
                    IMAGE_SIZE, SAMPLE_EVERY_N_EPOCHS, SAVE_EVERY_N_EPOCHS,
                    SUPPORTED_DATASETS)
from dataset import Dataset
from model import BaseDiffusionModel

MODEL_NAME = "DDPM"


def setup_gpu(gpu_id):
    gpus = tf.config.list_physical_devices("GPU")
    selected = gpus if gpu_id == -1 else ([gpus[gpu_id]] if 0 <= gpu_id < len(gpus) else [])
    for gpu in selected:
        tf.config.experimental.set_memory_growth(gpu, True)
    if selected:
        tf.config.set_visible_devices(selected, "GPU")
    print(f"Using {len(selected)} GPU(s)" if selected else "Using CPU")


def build_strategy():
    replicas = len(tf.config.list_logical_devices("GPU"))
    strategy = tf.distribute.MirroredStrategy() if replicas > 1 else tf.distribute.get_strategy()
    print(f"Distributed replicas: {strategy.num_replicas_in_sync}")
    return strategy


def paths(dataset_type):
    directory = os.path.join("logs", dataset_type, f"{MODEL_NAME}_v2", "diffusion")
    return directory, os.path.join(directory, "denoiser.weights.h5"), os.path.join(directory, "training_state.json")


def save_checkpoint(model, directory, weight_file, state_file, epoch):
    model.denoiser.save_weights(weight_file)
    with open(state_file, "w") as handle:
        json.dump({"epoch": epoch, "diffusion_steps": DIFFUSION_STEPS}, handle)


def load_epoch(model, weight_file, state_file):
    if not os.path.exists(weight_file):
        return 0
    model.denoiser.load_weights(weight_file)
    try:
        with open(state_file) as handle:
            state = json.load(handle)
        return state.get("epoch", 0) if state.get("diffusion_steps") == DIFFUSION_STEPS else 0
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


def save_grid(images, filename):
    images = np.clip((images + 1.0) / 2.0, 0.0, 1.0)
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    columns = 4
    rows = [images[start:start + columns] for start in range(0, len(images), columns)]
    width = max(len(row) for row in rows)
    if any(len(row) < width for row in rows):
        blank = np.zeros_like(images[:1])
        rows[-1] = np.concatenate([rows[-1], np.repeat(blank, width - len(rows[-1]), axis=0)])
    grid = np.concatenate([np.concatenate(list(row), axis=1) for row in rows], axis=0)
    tf.io.write_file(filename, tf.io.encode_png(tf.cast(grid * 255.0, tf.uint8)))


def due(callback, epoch, frequency):
    total = (getattr(callback, "params", None) or {}).get("epochs", epoch + 1)
    return (epoch + 1) % frequency == 0 or epoch + 1 == total


class Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, model, directory, weight_file, state_file):
        self.model_ref, self.directory, self.weight_file, self.state_file = model, directory, weight_file, state_file

    def on_epoch_end(self, epoch, logs=None):
        if due(self, epoch, SAVE_EVERY_N_EPOCHS):
            save_checkpoint(self.model_ref, self.directory, self.weight_file, self.state_file, epoch + 1)


class History(tf.keras.callbacks.Callback):
    def __init__(self, directory):
        self.path, self.records = os.path.join(directory, "history.json"), []

    def on_epoch_end(self, epoch, logs=None):
        self.records.append({"epoch": epoch + 1, "loss": float((logs or {}).get("loss", 0.0))})
        with open(self.path, "w") as handle:
            json.dump(self.records, handle)


class Samples(tf.keras.callbacks.Callback):
    def __init__(self, model, preview, directory, count):
        self.model_ref, self.preview, self.directory, self.count = model, preview, directory, count
        os.makedirs(os.path.join(directory, "samples"), exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if due(self, epoch, SAMPLE_EVERY_N_EPOCHS):
            generated = self.model_ref.generate_samples(self.count).numpy()
            save_grid(np.concatenate([self.preview.numpy(), generated]), os.path.join(self.directory, "samples", f"epoch_{epoch + 1}.png"))


def verify_sampler(model):
    assert list(model.reverse_timesteps()) == list(range(DIFFUSION_STEPS - 1, -1, -1))
    sample = tf.zeros((1, *model.input_shape_value))
    timestep = tf.zeros((1,), dtype=tf.int32)
    first = model.p_sample(sample, timestep)
    second = model.p_sample(sample, timestep)
    assert bool(tf.reduce_all(tf.equal(first, second)))


def main():
    parser = argparse.ArgumentParser(description="Train an unconditional DDPM.")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--type", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--resume", "--continue", dest="resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--verify-sampler", action="store_true")
    parser.add_argument("--sample-count", type=int, default=4)
    args = parser.parse_args()
    setup_gpu(args.gpu)
    strategy = build_strategy()
    dataset = Dataset()
    channels = 1 if args.type in {"mnist", "fashion_mnist"} else 3
    model = BaseDiffusionModel(strategy, (IMAGE_SIZE[0], IMAGE_SIZE[1], channels), BATCH_SIZE, model_label=MODEL_NAME)
    if args.verify_sampler:
        verify_sampler(model)
        print("DDPM sampler verification passed.")
        return
    train_ds, channels = dataset.load_data(args.type)
    directory, weight_file, state_file = paths(args.type)
    os.makedirs(directory, exist_ok=True)
    if args.reset:
        for filename in (weight_file, state_file):
            if os.path.exists(filename):
                os.remove(filename)
    start_epoch = load_epoch(model, weight_file, state_file) if args.resume else 0
    preview = dataset.get_preview_batch(args.type, count=min(args.sample_count, 4))[0]
    distributed_ds = strategy.experimental_distribute_dataset(train_ds)

    def interrupt(sig, frame):
        del sig, frame
        save_checkpoint(model, directory, weight_file, state_file, getattr(model, "_current_epoch", start_epoch))
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupt)
    model.fit(distributed_ds, epochs=args.epochs, initial_epoch=start_epoch,
              callbacks=[History(directory), Checkpoint(model, directory, weight_file, state_file),
                         Samples(model, preview, directory, min(args.sample_count, 4))],
              steps_per_epoch=len(train_ds))
    save_checkpoint(model, directory, weight_file, state_file, args.epochs)
    dataset.cleanup_cache()


if __name__ == "__main__":
    main()
