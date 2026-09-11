import argparse
import gc
import json
import os
import shutil
import time

import tensorflow as tf
from tensorflow.keras import mixed_precision

from config import BATCH_SIZE, CHECKPOINT_PATH, CLASSIFIER_CHECKPOINT_PATH, CLASSIFIER_EPOCHS, CLASSIFIER_HISTORY_PATH, CLASSIFIER_LOG_DIR, CLASSIFIER_STATE_PATH, COCO_ROOT, EPOCHS, HISTORY_PATH, IMAGENET_ROOT, LOG_DIR, STATE_PATH, STEPS_PER_EPOCH, TRAIN_SPLIT, VALIDATION_STEPS, VAL_SPLIT, VOC_ROOT, ensure_dir
from dataset import COCO_CLASSES, _imagenet_class_names, build_train_dataset, build_val_dataset
from detection_adapters import VOC_CLASSES
from inspection import DetectionInspectionCallback
from model import create_compiled_model
from test import infer_image
from cache_runtime import cleanup_cache, default_cache_workers


def setup_runtime(gpu_id):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpu_id != -1 and gpus and 0 <= gpu_id < len(gpus):
        tf.config.set_visible_devices(gpus[gpu_id], "GPU")
    visible = len(tf.config.get_visible_devices("GPU"))
    tf.config.optimizer.set_experimental_options(
        {
            "layout_optimizer": True,
            "constant_folding": True,
            "shape_optimization": True,
            "remapping": True,
            "arithmetic_optimization": True,
        }
    )
    mixed_precision.set_global_policy("mixed_float16" if gpus else "float32")
    if visible > 1:
        if BATCH_SIZE % visible:
            raise RuntimeError(f"BATCH_SIZE={BATCH_SIZE} must be divisible by {visible} NCCL replicas.")
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())
    elif visible == 1:
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
    print(f"[runtime] visible_gpus={visible} strategy={type(strategy).__name__} replicas={strategy.num_replicas_in_sync} global_batch={BATCH_SIZE} per_replica_batch={BATCH_SIZE // strategy.num_replicas_in_sync}")
    return strategy


def load_state(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Saved training state is required for --continue: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return int(json.load(file).get("epoch", 0))


def save_state(path, epoch):
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"epoch": int(epoch)}, file)


def scope_artifact_paths(base_log_dir, base_checkpoint_path, base_history_path, base_state_path, dataset_name):
    model_name = os.path.basename(base_log_dir)
    logs_root = os.path.dirname(os.path.dirname(base_log_dir))
    log_dir = os.path.join(logs_root, dataset_name, model_name)
    return (
        log_dir,
        os.path.join(log_dir, os.path.basename(base_checkpoint_path)),
        os.path.join(log_dir, os.path.basename(base_history_path)),
        os.path.join(log_dir, os.path.basename(base_state_path)),
    )


def classifier_checkpoint_name(dataset_name):
    return f"{dataset_name}_classifier.weights.h5"


def inspection_class_names(dataset_name):
    if dataset_name == "voc": return VOC_CLASSES
    return _imagenet_class_names() if dataset_name == "imagenet" else COCO_CLASSES


class HistorySaver(tf.keras.callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            self.history.setdefault(key, []).append(float(value))
        with open(self.path, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=2)


class StateSaver(tf.keras.callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        save_state(self.path, epoch + 1)


class CompactProgress(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None): self.epoch, self.started, self.steps = epoch + 1, time.monotonic(), self.params.get("steps")
    def on_train_batch_end(self, batch, logs=None):
        step = batch + 1
        if step % 10 and step != self.steps: return
        elapsed = time.monotonic() - self.started; eta = elapsed * (self.steps - step) / step if self.steps else 0.0
        print(f"\rE{self.epoch} {step}/{self.steps} loss={float((logs or {}).get('loss', float('nan'))):.3f} eta={int(eta // 60)}:{int(eta % 60):02d}", end="", flush=True)
    def on_epoch_end(self, epoch, logs=None): print()


def save_model_report(model, task_name):
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, f"{task_name}_model_summary.txt"), "w", encoding="utf-8") as file: model.summary(expand_nested=True, show_trainable=True, print_fn=lambda line: file.write(f"{line}\n"))
    try: tf.keras.utils.plot_model(model, to_file=os.path.join(root, f"{task_name}_model.png"), show_shapes=True, show_trainable=True, expand_nested=True)
    except (ImportError, OSError) as error: print(f"[model report] diagram unavailable: {error}")


def select_artifacts(task_name, dataset_name):
    if task_name == "classifier":
        dataset_root = IMAGENET_ROOT if dataset_name == "imagenet" else VOC_ROOT if dataset_name == "voc" else COCO_ROOT
        log_dir, checkpoint_path, history_path, state_path = scope_artifact_paths(
            CLASSIFIER_LOG_DIR,
            CLASSIFIER_CHECKPOINT_PATH,
            CLASSIFIER_HISTORY_PATH,
            CLASSIFIER_STATE_PATH,
            dataset_name,
        )
        checkpoint_path = os.path.join(log_dir, classifier_checkpoint_name(dataset_name))
        return dataset_root, log_dir, checkpoint_path, history_path, state_path
    dataset_root = IMAGENET_ROOT if dataset_name == "imagenet" else VOC_ROOT if dataset_name == "voc" else COCO_ROOT
    log_dir, checkpoint_path, history_path, state_path = scope_artifact_paths(
        LOG_DIR,
        CHECKPOINT_PATH,
        HISTORY_PATH,
        STATE_PATH,
        dataset_name,
    )
    return dataset_root, log_dir, checkpoint_path, history_path, state_path


def fit_task(strategy, task_name, dataset_name, continue_training, epochs, initialize_from=None, full_reports=False, reset=False):
    dataset_root, log_dir, checkpoint_path, history_path, state_path = select_artifacts(task_name, dataset_name)
    if dataset_root is None:
        raise RuntimeError(f"{task_name} dataset root not found for {dataset_name}")
    if reset and os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    ensure_dir(log_dir)
    print(f"[run] mode={'reset' if reset else 'continue' if continue_training else 'warm-start'} checkpoint={checkpoint_path} state={state_path}")
    train_split = "train" if dataset_name in ("imagenet", "voc") else TRAIN_SPLIT
    val_split = "val" if dataset_name in ("imagenet", "voc") else VAL_SPLIT
    train_builder, train_dataset = build_train_dataset(
        task_name=task_name,
        dataset_name=dataset_name,
        root=dataset_root,
        split=train_split,
    )
    val_builder, val_dataset = build_val_dataset(
        task_name=task_name,
        dataset_name=dataset_name,
        root=dataset_root,
        split=val_split,
    )
    steps_per_epoch = getattr(train_builder, "steps_per_epoch", STEPS_PER_EPOCH)
    validation_steps = getattr(val_builder, "steps_per_epoch", VALIDATION_STEPS)

    tf.keras.backend.clear_session()
    with strategy.scope():
        model = create_compiled_model(task_name=task_name, dataset_name=dataset_name)
    if full_reports:
        save_model_report(model, task_name)

    if initialize_from and os.path.exists(initialize_from):
        try:
            model.load_weights(initialize_from, skip_mismatch=True)
        except (TypeError, ValueError):
            try:
                model.load_weights(initialize_from)
            except ValueError:
                pass

    initial_epoch = 0
    if continue_training and not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"--continue requires a compatible checkpoint: {checkpoint_path}")
    if not reset and os.path.exists(checkpoint_path):
        try:
            model.load_weights(checkpoint_path)
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Selected checkpoint is incompatible and was preserved: {checkpoint_path} ({error})") from error
        else:
            if continue_training:
                initial_epoch = load_state(state_path)
                print(f"[checkpoint] Continuing {checkpoint_path} from epoch {initial_epoch + 1}.")
            else:
                print(f"[checkpoint] Warm-started matching weights from {checkpoint_path}; epoch numbering starts at 1.")
    elif reset:
        print("[checkpoint] --reset replaced the selected task/dataset artifacts; using new weights at epoch 0.")

    callbacks = [
        CompactProgress(), tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=0),
        HistorySaver(history_path),
        StateSaver(state_path),
    ]
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
    if task_name == "detector":
        callbacks.append(DetectionInspectionCallback(val_builder.samples, os.path.join(log_dir, "inspection"), inspection_class_names(dataset_name), infer_image, gt_normalized=True))
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=0,
    )
    model.save_weights(checkpoint_path)
    save_state(state_path, initial_epoch + len(history.epoch))
    tf.keras.backend.clear_session()
    gc.collect()
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", choices=["coco", "imagenet"], default=None)
    parser.add_argument("--detector", choices=["coco", "voc", "imagenet"], default=None)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--continue", dest="continue_training", action="store_true", help="Require checkpoint and saved state, then restore its epoch.")
    parser.add_argument("--resume", dest="continue_training", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reset", action="store_true", help="Delete and recreate only the selected task/dataset artifact directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override task defaults (detector: 100; classifier: 20).")
    parser.add_argument("--full-reports", action="store_true", help="Enable expensive reports; the labelled 4x4 inspection grid is written every detector epoch.")
    parser.add_argument("--no-cache", action="store_true", help="Disable the transient metadata/target cache.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Replace the active transient cache generation.")
    parser.add_argument("--cache-max-samples", type=int, default=2048, help="Bound cache payload storage without limiting dataset coverage.")
    parser.add_argument("--cache-workers", type=int, default=default_cache_workers(), help="CPU cache preparation workers (default: 4).")
    args = parser.parse_args()
    if args.reset and args.continue_training:
        parser.error("--reset and --continue cannot be used together")

    if args.no_cache:
        os.environ["DETECTOR_CACHE_ENABLED"] = "0"
    if args.rebuild_cache:
        os.environ["DETECTOR_CACHE_REBUILD"] = "1"
    if args.cache_max_samples < 0:
        parser.error("--cache-max-samples must be non-negative")
    if args.cache_workers < 1:
        parser.error("--cache-workers must be at least 1")
    os.environ["DETECTOR_CACHE_MAX_SAMPLES"] = str(args.cache_max_samples)
    os.environ["DETECTOR_CACHE_WORKERS"] = str(args.cache_workers)
    print(f"[cache] enabled={not args.no_cache} rebuild={args.rebuild_cache} max_samples={args.cache_max_samples} workers={args.cache_workers}")

    if args.classifier is None and args.detector is None:
        args.detector = "coco"

    strategy = setup_runtime(args.gpu)
    classifier_checkpoint = None
    if args.classifier is not None:
        classifier_checkpoint = fit_task(strategy, "classifier", args.classifier, args.continue_training, args.epochs or CLASSIFIER_EPOCHS, full_reports=args.full_reports, reset=args.reset)
    if args.detector is not None:
        fit_task(strategy, "detector", args.detector, args.continue_training, args.epochs or EPOCHS, initialize_from=classifier_checkpoint, full_reports=args.full_reports, reset=args.reset)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_cache()
