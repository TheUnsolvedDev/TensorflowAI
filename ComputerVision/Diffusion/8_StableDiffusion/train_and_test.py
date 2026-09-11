import argparse
import json
import os
import signal
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
try:
    import silence_tensorflow.auto  # noqa: F401
except ImportError:
    pass
import tensorflow as tf

from config import *
from dataset import *
from model import *

MODEL_KIND = "latent"
MODEL_NAME = "StableDiffusion"


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
    return tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())


def get_weight_paths(log_dir):
    return {
        "denoiser": os.path.join(log_dir, "denoiser.weights.h5"),
        "text_encoder": os.path.join(log_dir, "text_encoder.weights.h5"),
        "autoencoder": os.path.join(log_dir, "autoencoder.weights.h5"),
        "clip_encoder": os.path.join(log_dir, "clip_encoder.weights.h5"),
        "prior": os.path.join(log_dir, "prior.weights.h5"),
        "consistency": os.path.join(log_dir, "consistency.weights.h5"),
    }


def get_state_path(log_dir):
    return os.path.join(log_dir, "training_state.json")


def save_training_state(log_dir, epoch, stage):
    with open(get_state_path(log_dir), "w") as handle:
        json.dump({"epoch": epoch, "stage": stage}, handle)


def load_training_state(log_dir):
    path = get_state_path(log_dir)
    if os.path.exists(path):
        with open(path, "r") as handle:
            payload = json.load(handle)
        return payload.get("epoch", 0), payload.get("stage")
    return 0, None


def save_weights(model, log_dir):
    paths = get_weight_paths(log_dir)
    model.denoiser.save_weights(paths["denoiser"])
    if model.text_encoder is not None:
        model.text_encoder.save_weights(paths["text_encoder"])
    if model.autoencoder is not None:
        model.autoencoder.save_weights(paths["autoencoder"])
    if model.clip_encoder is not None:
        model.clip_encoder.save_weights(paths["clip_encoder"])
    if model.prior_projection is not None:
        model.prior_projection.save_weights(paths["prior"])
    if model.consistency_model is not None:
        model.consistency_model.save_weights(paths["consistency"])


def warmup_model(model, stage):
    image_batch = tf.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32)
    latent_batch = tf.zeros((1, LATENT_IMAGE_SIZE[0], LATENT_IMAGE_SIZE[1], model.input_shape_value[-1]), dtype=tf.float32)
    timesteps = tf.zeros((1,), dtype=tf.int32)
    if model.text_encoder is not None:
        model.text_encoder(tf.zeros((1, TEXT_SEQUENCE_LENGTH), dtype=tf.int32), training=False)
    if model.autoencoder is not None:
        model.autoencoder(image_batch, training=False)
    if model.clip_encoder is not None:
        model.clip_encoder.encode_image(image_batch, training=False)
        model.clip_encoder.encode_text(tf.zeros((1, TEXT_SEQUENCE_LENGTH), dtype=tf.int32), training=False)
        model.prior_projection(tf.zeros((1, EMBED_DIM), dtype=tf.float32), training=False)
    model.denoiser(latent_batch if model.autoencoder is not None and stage in {"diffusion", "control"} else tf.zeros((1, *model.input_shape_value), dtype=tf.float32), timesteps, training=False)
    if model.consistency_model is not None:
        model.consistency_model(tf.zeros((1, *model.input_shape_value), dtype=tf.float32), timesteps, training=False)


def load_weights_if_needed(model, log_dir, resume, stage):
    if not resume:
        return
    warmup_model(model, stage)
    paths = get_weight_paths(log_dir)
    if os.path.exists(paths["denoiser"]):
        model.denoiser.load_weights(paths["denoiser"])
    if model.text_encoder is not None and os.path.exists(paths["text_encoder"]):
        model.text_encoder.load_weights(paths["text_encoder"])
    if model.autoencoder is not None and os.path.exists(paths["autoencoder"]):
        model.autoencoder.load_weights(paths["autoencoder"])
    if model.clip_encoder is not None and os.path.exists(paths["clip_encoder"]):
        model.clip_encoder.load_weights(paths["clip_encoder"])
    if model.prior_projection is not None and os.path.exists(paths["prior"]):
        model.prior_projection.load_weights(paths["prior"])
    if model.consistency_model is not None and os.path.exists(paths["consistency"]):
        model.consistency_model.load_weights(paths["consistency"])
    print("Loaded existing weights where available")


def setup_interrupt_handler(model, log_dir, stage):
    def handler(sig, frame):
        print("\nInterrupt received. Saving state...")
        save_weights(model, log_dir)
        current_epoch = getattr(model, "_current_epoch", 0)
        save_training_state(log_dir, current_epoch, stage)
        print(f"Saved at epoch {current_epoch}. Exiting.")
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)


def save_rows_grid(rows, filename):
    rendered_rows = []
    for row in rows:
        images = np.clip((row + 1.0) / 2.0, 0.0, 1.0)
        if images.shape[-1] == 1:
            images = np.repeat(images, 3, axis=-1)
        rendered_rows.append(np.concatenate(list(images), axis=1))
    grid = np.concatenate(rendered_rows, axis=0)
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
    def __init__(self, model, log_dir, stage):
        super().__init__()
        self.model_ref = model
        self.log_dir = log_dir
        self.stage = stage

    def on_epoch_end(self, epoch, logs=None):
        if not should_run_periodic_callback(self, epoch, SAVE_EVERY_N_EPOCHS):
            return
        save_weights(self.model_ref, self.log_dir)
        save_training_state(self.log_dir, epoch + 1, self.stage)
        print(f"Saved weights and state at epoch {epoch + 1}")


class SampleImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, dataset_ref, log_dir, stage, preview_images=None, preview_prompt=None, preview_control=None, sample_count=4):
        super().__init__()
        self.model_ref = model
        self.dataset_ref = dataset_ref
        self.stage = stage
        self.preview_images = preview_images
        self.preview_prompt = preview_prompt
        self.preview_control = preview_control
        self.sample_count = sample_count
        self.cond_tokens = None
        self.img_dir = os.path.join(log_dir, "samples")
        os.makedirs(self.img_dir, exist_ok=True)

    def on_train_begin(self, logs=None):
        if self.preview_prompt and (self.model_ref.text_encoder is not None or self.model_ref.clip_encoder is not None):
            token_row = self.dataset_ref._tokenize_text(self.preview_prompt)
            token_rows = np.stack([token_row for _ in range(self.sample_count)])
            self.cond_tokens = tf.convert_to_tensor(token_rows, dtype=tf.int32)

    def _to_display(self, batch):
        batch = np.clip((batch + 1.0) / 2.0, 0.0, 1.0)
        if batch.shape[-1] == 1:
            batch = np.repeat(batch, 3, axis=-1)
        return batch

    def _save_grid(self, rows, titles, filename):
        del titles
        rows = [self._to_display(row) for row in rows]
        save_rows_grid(rows, filename)

    def on_epoch_end(self, epoch, logs=None):
        if not should_run_periodic_callback(self, epoch, SAMPLE_EVERY_N_EPOCHS):
            return
        if self.stage == "vae" and self.preview_images is not None and self.model_ref.autoencoder is not None:
            _, recon = self.model_ref.autoencoder(self.preview_images, training=False)
            self._save_grid([self.preview_images.numpy(), recon.numpy()], ["target", "recon"], os.path.join(self.img_dir, f"epoch_{epoch + 1}.png"))
            return
        sample_stage = "sample_consistency" if self.stage == "consistency" else ("control" if MODEL_KIND == "controlnet" and self.preview_control is not None else "diffusion")
        generated = self.model_ref.generate_samples(sample_count=self.sample_count, cond_tokens=self.cond_tokens, control=self.preview_control, stage=sample_stage).numpy()
        rows = [generated]
        titles = ["generated"]
        if self.preview_images is not None:
            rows.insert(0, self.preview_images.numpy())
            titles.insert(0, "target")
        if self.preview_control is not None:
            rows.insert(0, np.repeat(self.preview_control.numpy(), 3, axis=-1))
            titles.insert(0, "control")
        self._save_grid(rows, titles, os.path.join(self.img_dir, f"epoch_{epoch + 1}.png"))


def build_model(strategy, channels, mode):
    if MODEL_KIND == "latent":
        return BaseDiffusionModel(strategy, (LATENT_IMAGE_SIZE[0], LATENT_IMAGE_SIZE[1], 4), BATCH_SIZE, use_text=True, use_latent=True, model_label=MODEL_NAME)
    if MODEL_KIND == "controlnet":
        return BaseDiffusionModel(strategy, (LATENT_IMAGE_SIZE[0], LATENT_IMAGE_SIZE[1], 4), BATCH_SIZE, use_text=True, use_latent=True, use_control=True, model_label=MODEL_NAME)
    if MODEL_KIND == "text":
        return BaseDiffusionModel(strategy, (IMAGE_SIZE[0], IMAGE_SIZE[1], channels), BATCH_SIZE, use_text=True, model_label=MODEL_NAME)
    if MODEL_KIND == "text_sr":
        shape = (SUPER_RES_IMAGE_SIZE[0], SUPER_RES_IMAGE_SIZE[1], channels) if mode == "sr" else (IMAGE_SIZE[0], IMAGE_SIZE[1], channels)
        return BaseDiffusionModel(strategy, shape, BATCH_SIZE, use_text=True, model_label=MODEL_NAME)
    if MODEL_KIND == "dalle2":
        return BaseDiffusionModel(strategy, (IMAGE_SIZE[0], IMAGE_SIZE[1], channels), BATCH_SIZE, use_text=True, use_clip=True, model_label=MODEL_NAME)
    if MODEL_KIND == "consistency":
        return BaseDiffusionModel(strategy, (IMAGE_SIZE[0], IMAGE_SIZE[1], channels), BATCH_SIZE, use_consistency=True, model_label=MODEL_NAME)
    return BaseDiffusionModel(strategy, (IMAGE_SIZE[0], IMAGE_SIZE[1], channels), BATCH_SIZE, model_label=MODEL_NAME)


def prepare_dataset(dataset, dataset_type, mode):
    if MODEL_KIND == "controlnet":
        return dataset.load_control_data(dataset_type)
    if MODEL_KIND == "text_sr" and mode == "sr":
        return dataset.load_super_resolution_data(dataset_type)
    return dataset.load_data(dataset_type)


def parse_modes():
    if MODEL_KIND == "text_sr":
        return ["base", "sr"]
    if MODEL_KIND == "dalle2":
        return ["align", "prior", "decoder"]
    if MODEL_KIND == "latent":
        return ["vae", "diffusion"]
    if MODEL_KIND == "controlnet":
        return ["vae", "diffusion", "control"]
    if MODEL_KIND == "consistency":
        return ["teacher", "consistency", "sample"]
    return ["diffusion"]


def resolve_stages(mode):
    valid = parse_modes()
    if mode == "all":
        return valid
    if mode not in valid:
        raise ValueError(f"Invalid mode '{mode}'. Expected one of {valid + ['all']}")
    return [mode]


def translate_stage(stage):
    mapping = {
        "base": "diffusion",
        "decoder": "diffusion",
        "teacher": "diffusion",
        "control": "diffusion",
        "sample": "consistency",
    }
    return mapping.get(stage, stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--type", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue", dest="resume", action="store_true")
    parser.add_argument("--mode", type=str, default="all" if len(parse_modes()) > 1 else parse_modes()[0])
    parser.add_argument("--sample-prompt", type=str, default="a bright street scene with people and buildings")
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--control-type", type=str, default="canny")
    args = parser.parse_args()

    setup_gpu(args.gpu)
    dataset = Dataset()
    strategy = build_strategy()

    for stage in resolve_stages(args.mode):
        if MODEL_KIND == "consistency" and stage == "sample":
            train_ds, channels = dataset.load_data(args.type)
            model = build_model(strategy, channels, stage)
            load_weights_if_needed(model, os.path.join("logs", args.type, MODEL_NAME, "consistency"), args.resume, "consistency")
            sample_log_dir = os.path.join("logs", args.type, MODEL_NAME, "sample")
            os.makedirs(os.path.join(sample_log_dir, "samples"), exist_ok=True)
            prompt_tokens = None
            if dataset.data_types and args.type in {"coco", "flickr30k"}:
                pairs = dataset._load_caption_pairs(args.type)
                dataset._build_vocab([caption for _, caption in pairs[:128]])
                token_row = dataset._tokenize_text(args.sample_prompt)
                prompt_tokens = tf.convert_to_tensor(np.stack([token_row for _ in range(args.sample_count)]), dtype=tf.int32)
            generated = model.generate_samples(sample_count=args.sample_count, cond_tokens=prompt_tokens, stage="sample_consistency").numpy()
            save_rows_grid([generated], os.path.join(sample_log_dir, "samples", "sample.png"))
            continue

        train_stage = translate_stage(stage)
        train_ds, channels = prepare_dataset(dataset, args.type, stage)
        distributed_ds = strategy.experimental_distribute_dataset(train_ds)
        model = build_model(strategy, channels, stage)
        log_dir = os.path.join("logs", args.type, MODEL_NAME, stage)
        os.makedirs(log_dir, exist_ok=True)
        load_weights_if_needed(model, log_dir, args.resume, train_stage)
        start_epoch, _ = load_training_state(log_dir) if args.resume else (0, None)
        setup_interrupt_handler(model, log_dir, stage)

        preview_images = None
        preview_control = None
        if args.type in {"coco", "flickr30k"}:
            pairs = dataset._load_caption_pairs(args.type)
            dataset._build_vocab([caption for _, caption in pairs[:2048]])
            if MODEL_KIND == "controlnet":
                preview_images, preview_control, _ = dataset.get_preview_batch(args.type, count=min(args.sample_count, 4), include_control=True)
            else:
                preview_images, _ = dataset.get_preview_batch(args.type, count=min(args.sample_count, 4))
        else:
            preview_images = dataset.get_preview_batch(args.type, count=min(args.sample_count, 4))[0]

        callbacks = [
            EpochTracker(model),
            LossLogger(log_dir),
            WeightSaveCallback(model, log_dir, stage),
            SampleImageCallback(model, dataset, log_dir, train_stage if stage != "teacher" else "diffusion", preview_images=preview_images, preview_prompt=args.sample_prompt, preview_control=preview_control, sample_count=min(args.sample_count, 4)),
        ]

        model.fit(distributed_ds, epochs=EPOCHS, initial_epoch=start_epoch, callbacks=callbacks, stage=train_stage)
        save_weights(model, log_dir)
        save_training_state(log_dir, EPOCHS, stage)

    dataset.cleanup_cache()


if __name__ == "__main__":
    main()
