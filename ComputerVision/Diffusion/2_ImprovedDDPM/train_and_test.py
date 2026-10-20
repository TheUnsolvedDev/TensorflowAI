import argparse
import json
import os
import shutil
import signal

import numpy as np
try:
    import silence_tensorflow.auto  # noqa: F401
except ImportError:
    pass
import tensorflow as tf

from config import (
    ARCHITECTURE_VERSION,
    DEFAULT_BATCH_SIZES,
    DEFAULT_DATASET,
    DEFAULT_IMAGE_SIZES,
    DIFFUSION_STEPS,
    EPOCHS,
    SAMPLE_EVERY_N_EPOCHS,
    SAMPLING_STEPS,
    SAVE_EVERY_N_EPOCHS,
    SUPPORTED_DATASETS,
    SUPPORTED_IMAGE_SIZES,
)
MODEL_NAME = "ImprovedDDPM"


def setup_gpu(gpu_id):
    gpus = tf.config.list_physical_devices("GPU")
    if gpu_id == -1 and len(gpus) < 2:
        raise RuntimeError(f"--gpu -1 requires at least 2 visible GPUs, found {len(gpus)}.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpu_id == -1:
        tf.config.set_visible_devices(gpus[:2], "GPU")
        print(f"Using both GPUs with NCCL: {gpus[0].name}, {gpus[1].name}")
    elif 0 <= gpu_id < len(gpus):
        tf.config.set_visible_devices(gpus[gpu_id], "GPU")
        print(f"Using GPU {gpu_id}")
    else:
        print("Invalid GPU ID, using CPU")


def build_strategy():
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce()
    )
    print(f"Distributed replicas: {strategy.num_replicas_in_sync}")
    return strategy


def export_model_diagram(model):
    try:
        tf.keras.utils.plot_model(
            model.denoiser,
            to_file="denoiser.png",
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
        )
        print("Saved Functional model diagram: denoiser.png")
    except (ImportError, OSError) as exc:
        print(f"Skipping model diagram export: {exc}")


def run_paths(dataset_type):
    directory = os.path.join("logs", dataset_type, MODEL_NAME, "diffusion")
    return (
        directory,
        os.path.join(directory, "denoiser.weights.h5"),
        os.path.join(directory, "training_state.json"),
    )


def save_checkpoint(model, weight_file, state_file, epoch):
    model.denoiser.save_weights(weight_file)
    with open(state_file, "w") as handle:
        json.dump({
            "epoch": epoch,
            "architecture": ARCHITECTURE_VERSION,
            "diffusion_steps": DIFFUSION_STEPS,
            "sampling_steps": SAMPLING_STEPS,
            "image_size": model.input_shape_value[0],
            "schedule": "cosine",
            "loss_second_moment": model.loss_second_moment.numpy().tolist(),
        }, handle)


def load_checkpoint(model, weight_file, state_file):
    if not os.path.exists(weight_file) or not os.path.exists(state_file):
        return 0
    try:
        with open(state_file) as handle:
            state = json.load(handle)
        compatible = (
            state.get("architecture") == ARCHITECTURE_VERSION
            and state.get("diffusion_steps") == DIFFUSION_STEPS
            and state.get("sampling_steps") == SAMPLING_STEPS
            and state.get("image_size") == model.input_shape_value[0]
            and state.get("schedule") == "cosine"
        )
        moments = state.get("loss_second_moment", [])
        if not compatible or len(moments) != DIFFUSION_STEPS:
            print("Saved weights use incompatible Improved DDPM settings. Starting fresh.")
            return 0
        model.denoiser.load_weights(weight_file)
        model.loss_second_moment.assign(moments)
        print("Loaded Improved DDPM weights and timestep sampler state")
        return state.get("epoch", 0)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not resume saved weights. Starting fresh: {exc}")
        return 0


def save_grid(images, filename):
    images = np.clip((images + 1.0) / 2.0, 0.0, 1.0)
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    rows = [images[start:start + 4] for start in range(0, len(images), 4)]
    width = max(len(row) for row in rows)
    if len(rows[-1]) < width:
        blank = np.zeros_like(images[:1])
        rows[-1] = np.concatenate([
            rows[-1], np.repeat(blank, width - len(rows[-1]), axis=0)
        ])
    grid = np.concatenate([
        np.concatenate(list(row), axis=1) for row in rows
    ], axis=0)
    encoded = tf.io.encode_png(tf.cast(grid * 255.0, tf.uint8))
    tf.io.write_file(filename, encoded)


def callback_due(callback, epoch, frequency):
    total = (getattr(callback, "params", None) or {}).get("epochs", epoch + 1)
    return (epoch + 1) % frequency == 0 or epoch + 1 == total


class History(tf.keras.callbacks.Callback):
    def __init__(self, directory):
        self.path = os.path.join(directory, "history.json")
        if os.path.exists(self.path):
            with open(self.path) as handle:
                self.records = json.load(handle)
        else:
            self.records = []

    def on_epoch_end(self, epoch, logs=None):
        self.records.append({
            "epoch": epoch + 1,
            "loss": float((logs or {}).get("loss", 0.0)),
        })
        with open(self.path, "w") as handle:
            json.dump(self.records, handle)


class Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, model, weight_file, state_file):
        self.model_ref = model
        self.weight_file = weight_file
        self.state_file = state_file

    def on_epoch_end(self, epoch, logs=None):
        if callback_due(self, epoch, SAVE_EVERY_N_EPOCHS):
            save_checkpoint(
                self.model_ref, self.weight_file, self.state_file, epoch + 1
            )
            print(f"Saved weights and state at epoch {epoch + 1}")


class Samples(tf.keras.callbacks.Callback):
    def __init__(self, model, directory, count=16):
        self.model_ref = model
        self.directory = os.path.join(directory, "samples")
        self.count = count
        os.makedirs(self.directory, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if callback_due(self, epoch, SAMPLE_EVERY_N_EPOCHS):
            print(
                f"Generating {self.count} samples "
                f"({SAMPLING_STEPS} respaced denoising steps)..."
            )
            generated = self.model_ref.generate_samples(self.count).numpy()
            save_grid(generated, os.path.join(self.directory, f"epoch_{epoch + 1}.png"))


def verify_implementation(model):
    pairs = model.reverse_timestep_pairs()
    assert len(pairs) == SAMPLING_STEPS
    assert pairs[0][0] == DIFFUSION_STEPS - 1
    assert pairs[-1] == (0, -1)
    assert all(current > previous for current, previous in pairs)
    assert bool(model.alpha_bars[-1] < 0.01)
    assert model.denoiser.output_shape[-1] == model.input_shape_value[-1] * 2
    probabilities = model.timestep_probabilities()
    assert bool(tf.reduce_all(probabilities > 0.0))
    assert bool(tf.abs(tf.reduce_sum(probabilities) - 1.0) < 1e-6)

    def verify_on_replica():
        sample = tf.zeros((1, *model.input_shape_value))
        first = model.p_sample(sample, 0, -1)
        second = model.p_sample(sample, 0, -1)
        return tf.reduce_all(tf.equal(first, second))

    results = model.strategy.run(verify_on_replica)
    assert all(bool(result) for result in model.strategy.experimental_local_results(results))
    print("Improved DDPM verification passed.")


def main():
    parser = argparse.ArgumentParser(description="Train Improved DDPM.")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--type", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--resume", "--continue", dest="resume", action="store_true")
    run_mode.add_argument("--reset", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--image-size", type=int, choices=SUPPORTED_IMAGE_SIZES)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.sample_count < 1:
        parser.error("--sample-count must be at least 1")

    image_size = args.image_size or DEFAULT_IMAGE_SIZES[args.type]
    batch_size = args.batch_size or DEFAULT_BATCH_SIZES[args.type]
    setup_gpu(args.gpu)
    # Import model code only after visible devices and memory growth are fixed.
    from dataset import Dataset
    from model import ImprovedDDPMTrainer

    strategy = build_strategy()
    if batch_size % strategy.num_replicas_in_sync:
        parser.error(
            f"--batch-size must be divisible by "
            f"the {strategy.num_replicas_in_sync} active replicas"
        )
    channels = 1 if args.type in {"mnist", "fashion_mnist"} else 3
    model = ImprovedDDPMTrainer(strategy, (image_size, image_size, channels))
    export_model_diagram(model)
    if args.verify:
        verify_implementation(model)
        return

    dataset = Dataset((image_size, image_size), batch_size)
    train_dataset, _ = dataset.load_data(args.type)
    distributed_dataset = strategy.experimental_distribute_dataset(train_dataset)
    directory, weight_file, state_file = run_paths(args.type)
    if (args.reset or not args.resume) and os.path.isdir(directory):
        shutil.rmtree(directory)
        print(f"Reset run directory: {directory}")
    os.makedirs(directory, exist_ok=True)
    start_epoch = load_checkpoint(model, weight_file, state_file) if args.resume else 0
    if args.resume and start_epoch == 0:
        shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

    def interrupt(sig, frame):
        del sig, frame
        print("\nInterrupt received. Stopping after current step...")
        model._interrupt_requested = True

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    callbacks = [
        History(directory),
        Checkpoint(model, weight_file, state_file),
        Samples(model, directory, args.sample_count),
    ]
    try:
        model.fit(
            distributed_dataset,
            epochs=args.epochs,
            initial_epoch=start_epoch,
            callbacks=callbacks,
            steps_per_epoch=len(train_dataset),
        )
        save_checkpoint(model, weight_file, state_file, args.epochs)
    except KeyboardInterrupt:
        epoch = getattr(model, "_current_epoch", start_epoch)
        save_checkpoint(model, weight_file, state_file, epoch)
        print(f"Interrupt checkpoint saved at epoch {epoch}")
        raise


if __name__ == "__main__":
    main()
