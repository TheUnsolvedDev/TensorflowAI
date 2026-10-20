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
    SAVE_EVERY_N_EPOCHS,
    SUPPORTED_DATASETS,
    SUPPORTED_IMAGE_SIZES,
)
from dataset import Dataset
from model import DDPMTrainer

MODEL_NAME = "DiffusionProbabilisticModels"


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
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())
    print(f"Distributed replicas: {strategy.num_replicas_in_sync}")
    return strategy


def get_weight_path(log_dir):
    return os.path.join(log_dir, "denoiser.weights.h5")


def get_state_path(log_dir):
    return os.path.join(log_dir, "training_state.json")


def get_log_dir(dataset_type):
    return os.path.join("logs", dataset_type, MODEL_NAME, "diffusion")


def reset_run(log_dir):
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
        print(f"Reset run directory: {log_dir}")


def verify_ddpm_sampler(model):
    schedule = list(model.reverse_timesteps())
    if schedule != list(range(DIFFUSION_STEPS - 1, -1, -1)) or len(schedule) != DIFFUSION_STEPS:
        raise AssertionError("DDPM reverse schedule must contain each timestep exactly once.")

    alpha_bars = model.alpha_bars
    if not bool(alpha_bars[-1] < 0.01):
        raise AssertionError("The final forward-diffusion state must be approximately pure noise.")
    previous_alpha_bars = tf.concat([tf.ones((1,), dtype=tf.float32), alpha_bars[:-1]], axis=0)
    posterior_variance = model.betas * (1.0 - previous_alpha_bars) / (1.0 - alpha_bars)
    if not bool(tf.reduce_all(tf.math.is_finite(posterior_variance))):
        raise AssertionError("DDPM posterior variance must be finite.")
    if not bool(tf.reduce_all(posterior_variance >= 0.0)) or not bool(tf.abs(posterior_variance[0]) < 1e-7):
        raise AssertionError("DDPM posterior variance must be non-negative and zero at timestep 0.")

    sample = tf.zeros((1, *model.input_shape_value), dtype=tf.float32)
    timestep_zero = tf.zeros((1,), dtype=tf.int32)
    first = model.p_sample(sample, timestep_zero)
    second = model.p_sample(sample, timestep_zero)
    if not bool(tf.reduce_all(tf.equal(first, second))):
        raise AssertionError("DDPM timestep 0 must not add random noise.")
    print("DDPM sampler verification passed.")


def save_training_state(log_dir, epoch, image_size):
    with open(get_state_path(log_dir), "w") as handle:
        json.dump({"epoch": epoch, "architecture": ARCHITECTURE_VERSION,
                   "schedule": "cosine", "diffusion_steps": DIFFUSION_STEPS,
                   "image_size": image_size}, handle)


def load_training_state(log_dir):
    path = get_state_path(log_dir)
    if os.path.exists(path):
        with open(path, "r") as handle:
            payload = json.load(handle)
        return payload.get("epoch", 0)
    return 0


def save_weights(model, log_dir):
    model.denoiser.save_weights(get_weight_path(log_dir))


def load_weights_if_needed(model, log_dir, resume):
    if not resume:
        return False
    state_path = get_state_path(log_dir)
    if os.path.exists(state_path):
        with open(state_path, "r") as handle:
            state = json.load(handle)
        if (state.get("architecture") != ARCHITECTURE_VERSION or
                state.get("schedule") != "cosine" or
                state.get("diffusion_steps") != DIFFUSION_STEPS or
                state.get("image_size") != model.input_shape_value[0]):
            print("Saved weights use an incompatible diffusion schedule. Starting fresh.")
            return False
    weight_path = get_weight_path(log_dir)
    if not os.path.exists(weight_path):
        print("No saved weights found. Starting fresh.")
        return False
    try:
        model.denoiser.load_weights(weight_path)
    except (OSError, ValueError) as exc:
        print(f"Incompatible saved weights detected. Starting fresh: {exc}")
        return False
    print("Loaded existing denoiser weights")
    return True


def setup_interrupt_handler(model):
    def handler(sig, frame):
        del sig, frame
        print("\nInterrupt received. Stopping after current step...")
        model._interrupt_requested = True
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def save_grid(images, filename):
    rendered_images = np.clip((images + 1.0) / 2.0, 0.0, 1.0)
    if rendered_images.shape[-1] == 1:
        rendered_images = np.repeat(rendered_images, 3, axis=-1)
    rendered_images = list(rendered_images)
    columns = 4
    grid_rows = [rendered_images[start:start + columns] for start in range(0, len(rendered_images), columns)]
    width = max(len(row) for row in grid_rows)
    if len(grid_rows[-1]) < width:
        blank = np.zeros_like(rendered_images[0])
        grid_rows[-1].extend([blank] * (width - len(grid_rows[-1])))
    grid = np.concatenate([np.concatenate(row, axis=1) for row in grid_rows], axis=0)
    image_tensor = tf.cast(tf.clip_by_value(grid * 255.0, 0.0, 255.0), tf.uint8)
    tf.io.write_file(filename, tf.io.encode_png(image_tensor))


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


class LossLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        history_path = os.path.join(log_dir, "history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as handle:
                self.history = json.load(handle)
        else:
            self.history = []

    def on_epoch_end(self, epoch, logs=None):
        record = {"epoch": epoch + 1}
        if logs:
            for key, value in logs.items():
                record[key] = value if isinstance(value, str) else float(value)
        self.history.append(record)
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
        save_weights(self.model_ref, self.log_dir)
        save_training_state(self.log_dir, epoch + 1, self.model_ref.input_shape_value[0])
        print(f"Saved weights and state at epoch {epoch + 1}")


class SampleImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, log_dir, sample_count=16):
        super().__init__()
        self.model_ref = model
        self.sample_count = sample_count
        self.img_dir = os.path.join(log_dir, "samples")
        os.makedirs(self.img_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if not should_run_periodic_callback(self, epoch, SAMPLE_EVERY_N_EPOCHS):
            return
        print(f"Generating {self.sample_count} samples ({DIFFUSION_STEPS} denoising steps)...")
        generated = self.model_ref.generate_samples(sample_count=self.sample_count).numpy()
        save_grid(generated, os.path.join(self.img_dir, f"epoch_{epoch + 1}.png"))


def build_model(strategy, channels, image_size):
    return DDPMTrainer(strategy, (image_size, image_size, channels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index, or -1 for two-GPU NCCL training.")
    parser.add_argument("--type", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--resume", "--continue", dest="resume", action="store_true")
    run_mode.add_argument("--reset", action="store_true", help="Delete this dataset run before training.")
    parser.add_argument("--verify-sampler", action="store_true", help="Run DDPM reverse-process regression checks and exit.")
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--image-size", type=int, choices=SUPPORTED_IMAGE_SIZES)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    image_size = args.image_size or DEFAULT_IMAGE_SIZES[args.type]
    batch_size = args.batch_size or DEFAULT_BATCH_SIZES[args.type]
    if args.sample_count < 1:
        parser.error("--sample-count must be at least 1")

    setup_gpu(args.gpu)
    dataset = Dataset((image_size, image_size), batch_size)
    strategy = build_strategy()
    channels = 1 if args.type in {"mnist", "fashion_mnist"} else 3
    if batch_size % strategy.num_replicas_in_sync:
        parser.error(f"--batch-size must be divisible by the {strategy.num_replicas_in_sync} active replicas")
    model = build_model(strategy, channels, image_size)
    if args.verify_sampler:
        verify_ddpm_sampler(model)
        return

    train_ds, channels = dataset.load_data(args.type)
    distributed_ds = strategy.experimental_distribute_dataset(train_ds)
    log_dir = get_log_dir(args.type)
    if args.reset or not args.resume:
        reset_run(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    resumed = load_weights_if_needed(model, log_dir, args.resume)
    if args.resume and not resumed:
        reset_run(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    start_epoch = load_training_state(log_dir) if resumed else 0
    setup_interrupt_handler(model)
    callbacks = [
        EpochTracker(model),
        LossLogger(log_dir),
        WeightSaveCallback(model, log_dir),
        SampleImageCallback(model, log_dir, args.sample_count),
    ]

    try:
        model.fit(
            distributed_ds,
            epochs=EPOCHS,
            initial_epoch=start_epoch,
            callbacks=callbacks,
            steps_per_epoch=len(train_ds),
        )
        save_weights(model, log_dir)
        save_training_state(log_dir, EPOCHS, image_size)
    except KeyboardInterrupt:
        current_epoch = getattr(model, "_current_epoch", start_epoch)
        print(f"\nSaving interrupt checkpoint at epoch {current_epoch}...")
        save_weights(model, log_dir)
        save_training_state(log_dir, current_epoch, image_size)
        print("Interrupt checkpoint saved.")
        raise

if __name__ == "__main__":
    main()
