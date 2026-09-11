import argparse
import gc
import json
import os
import shutil
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

from config import BATCH_SIZE, CACHE_MAX_SAMPLES, CHECKPOINT_PATH, CLASSIFIER_CHECKPOINT_PATH, CLASSIFIER_EPOCHS, CLASSIFIER_HISTORY_PATH, CLASSIFIER_LOG_DIR, CLASSIFIER_STATE_PATH, COCO_ROOT, EPOCHS, HISTORY_PATH, IMAGENET_ROOT, LOG_DIR, STATE_PATH, STEPS_PER_EPOCH, TRAIN_SPLIT, VALIDATION_STEPS, VAL_SPLIT, VOC_ROOT, ensure_dir
from dataset import COCO_CLASSES, VOC_CLASSES, _imagenet_class_names, build_train_dataset, build_val_dataset
from model import build_faster_rcnn_report_model, create_compiled_model
from cache_runtime import cleanup_cache, default_cache_workers


def setup_runtime(gpu_id, global_batch):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpu_id != -1 and gpus and 0 <= gpu_id < len(gpus):
        tf.config.set_visible_devices(gpus[gpu_id], "GPU")
    visible = len(tf.config.get_visible_devices("GPU"))
    if visible > 1 and global_batch % visible:
        raise RuntimeError(
            f"global batch={global_batch} must be divisible by {visible} NCCL replicas.")
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
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.NcclAllReduce())
    elif visible == 1:
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
    print(
        f"[runtime] visible_gpus={visible} strategy={type(strategy).__name__} "
        f"replicas={strategy.num_replicas_in_sync} global_batch={global_batch} "
        f"per_replica_batch={global_batch // strategy.num_replicas_in_sync}"
    )
    return strategy


def load_state(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Saved training state is required for --continue: {path}")
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
    if dataset_name == "imagenet":
        return ["background", *_imagenet_class_names()]
    return ["background", *VOC_CLASSES] if dataset_name == "voc" else COCO_CLASSES


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
    """Compact terminal metrics; TensorBoard and history retain full names."""

    def __init__(self, update_every=10):
        super().__init__()
        self.update_every = update_every

    @staticmethod
    def _metric(logs, name):
        value = logs.get(name)
        return float(value) if value is not None else float("nan")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
        self.steps = self.params.get("steps")
        self.started_at = time.monotonic()

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        step = batch + 1
        if step % self.update_every and step != self.steps:
            return
        elapsed = time.monotonic() - self.started_at
        eta = elapsed * (self.steps - step) / step if self.steps else 0.0
        print(
            f"\rE{self.epoch} {step}/{self.steps} "
            f"loss={self._metric(logs, 'loss'):.3f} "
            f"rpnC={self._metric(logs, 'rpn_cls_loss'):.3f} "
            f"rpnB={self._metric(logs, 'rpn_box_loss'):.3f} "
            f"roiC={self._metric(logs, 'roi_cls_loss'):.3f} "
            f"roiB={self._metric(logs, 'roi_box_loss'):.3f} "
            f"acc={self._metric(logs, 'roi_accuracy'):.1%} "
            f"eta={int(eta // 60)}:{int(eta % 60):02d}",
            end="",
            flush=True,
        )

    def on_epoch_end(self, epoch, logs=None):
        print()


class FasterRCNNValidationCallback(tf.keras.callbacks.Callback):
    """Run the native RPN/ROI `detect` path on a fixed validation subset."""

    def __init__(self, samples, output_dir, class_names, write_inspection=True, limit=200):
        super().__init__()
        self.samples = list(samples[:limit])
        self.output_dir = ensure_dir(output_dir)
        self.class_names = class_names
        self.inspection_dir = ensure_dir(os.path.join(
            output_dir, "inspection")) if write_inspection else None
        self.writer = tf.summary.create_file_writer(self.output_dir)
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        rendered, count, top_score = [], 0, 0.0
        for sample in self.samples:
            image = cv2.imread(sample["image_path"])
            if image is None:
                continue
            height, width = image.shape[:2]
            resized = cv2.resize(cv2.cvtColor(
                image, cv2.COLOR_BGR2RGB), (256, 256)).astype(np.float32) / 255.0
            detections = self.model.detect(
                tf.convert_to_tensor(resized[None, ...]))
            canvas = image.copy()
            for (x1, y1, x2, y2), class_id in zip(sample["boxes"][:20], sample["labels"][:20]):
                cv2.rectangle(canvas, (int(x1), int(y1)),
                              (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(canvas, f"GT {self.class_names[int(class_id)]}", (int(x1), max(
                    16, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            for detection in detections:
                y1, x1, y2, x2 = detection["box"]
                cv2.rectangle(canvas, (int(x1 * width), int(y1 * height)),
                              (int(x2 * width), int(y2 * height)), (0, 255, 0), 2)
                class_id = int(detection["class_id"])
                cv2.putText(canvas, f"{self.class_names[class_id]}: {float(detection['score']):.2f}", (int(
                    x1 * width), max(16, int(y1 * height) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                count += 1
                top_score = max(top_score, float(detection["score"]))
            if len(rendered) < 16:
                rendered.append(cv2.resize(canvas, (320, 240)))
        # `detect` has already applied the detector's own RPN, ROI head, score
        # threshold and NMS. COCO AP remains zero when the dataset cannot supply
        # COCO annotations or there are no decoded detections.
        metrics = {"bbox_map": 0.0, "bbox_ap50": 0.0, "bbox_ap75": 0.0,
                   "proposal_recall_iou50": 0.0, "proposal_mean_best_iou": 0.0,
                   "foreground_detection_count": count, "highest_foreground_score": top_score}
        self.history.append({"epoch": epoch + 1, **metrics})
        with open(os.path.join(self.output_dir, "validation_metrics.json"), "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=2)
        if rendered and self.inspection_dir:
            rendered.extend(np.zeros_like(rendered[0])
                            for _ in range(16 - len(rendered)))
            grid = np.concatenate([np.concatenate(
                rendered[row:row + 4], axis=1) for row in range(0, 16, 4)], axis=0)
            path = os.path.join(self.inspection_dir,
                                f"epoch_{epoch + 1:03d}.png")
            cv2.imwrite(path, grid)
            print(
                f"[inspection] Saved 4x4 ground-truth/prediction grid: {path}")
        with self.writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=epoch + 1)
        self.writer.flush()
        print(
            f"[validation] mAP={metrics['bbox_map']:.3f} AP50={metrics['bbox_ap50']:.3f} AP75={metrics['bbox_ap75']:.3f} fg={count} top_fg={top_score:.3f}")


def save_model_report(model, task_name):
    """Print and save the full functional report in this Faster R-CNN folder."""
    model.summary(expand_nested=True, show_trainable=True)
    report_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(report_dir, f"{task_name}_model_summary.txt")
    diagram_path = os.path.join(report_dir, f"{task_name}_model.png")
    with open(summary_path, "w", encoding="utf-8") as file:
        model.summary(expand_nested=True, show_trainable=True,
                      print_fn=lambda line: file.write(f"{line}\n"))
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=diagram_path,
            show_shapes=True,
            show_layer_names=True,
            show_trainable=True,
            expand_nested=True,
        )
    except (ImportError, OSError) as error:
        print(f"[model report] Could not create {diagram_path}: {error}")
    else:
        print(f"[model report] Summary: {summary_path}")
        print(f"[model report] Architecture diagram: {diagram_path}")


def select_artifacts(task_name, dataset_name):
    if task_name == "classifier":
        dataset_root = IMAGENET_ROOT if dataset_name == "imagenet" else COCO_ROOT
        log_dir, checkpoint_path, history_path, state_path = scope_artifact_paths(
            CLASSIFIER_LOG_DIR,
            CLASSIFIER_CHECKPOINT_PATH,
            CLASSIFIER_HISTORY_PATH,
            CLASSIFIER_STATE_PATH,
            dataset_name,
        )
        checkpoint_path = os.path.join(
            log_dir, classifier_checkpoint_name(dataset_name))
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


def fit_task(strategy, task_name, dataset_name, continue_training, epochs, initialize_from=None, reset=False, full_reports=False, detector_batch_size=None):
    dataset_root, log_dir, checkpoint_path, history_path, state_path = select_artifacts(
        task_name, dataset_name)
    if dataset_root is None:
        raise RuntimeError(
            f"{task_name} dataset root not found for {dataset_name}")
    if reset and os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    ensure_dir(log_dir)
    print(f"[run] mode={'reset' if reset else 'continue' if continue_training else 'warm-start'} checkpoint={checkpoint_path} state={state_path}")
    train_split = "train" if dataset_name in {
        "imagenet", "voc"} else TRAIN_SPLIT
    val_split = "val" if dataset_name in {"imagenet", "voc"} else VAL_SPLIT
    train_builder, train_dataset = build_train_dataset(
        task_name=task_name,
        dataset_name=dataset_name,
        root=dataset_root,
        split=train_split,
        batch_size=detector_batch_size if task_name == "detector" else None,
    )
    val_builder, val_dataset = build_val_dataset(
        task_name=task_name,
        dataset_name=dataset_name,
        root=dataset_root,
        split=val_split,
        batch_size=detector_batch_size if task_name == "detector" else None,
    )
    steps_per_epoch = getattr(
        train_builder, "steps_per_epoch", STEPS_PER_EPOCH)
    validation_steps = getattr(
        val_builder, "steps_per_epoch", VALIDATION_STEPS)
    tf.keras.backend.clear_session()
    with strategy.scope():
        model = create_compiled_model(
            task_name=task_name, dataset_name=dataset_name)
        report_model = model if task_name == "classifier" else build_faster_rcnn_report_model(
            model.detector)
    if full_reports:
        save_model_report(report_model, task_name)

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
        raise FileNotFoundError(
            f"--continue requires a compatible checkpoint: {checkpoint_path}")
    if not reset and os.path.exists(checkpoint_path):
        try:
            model.load_weights(checkpoint_path)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"Selected checkpoint is incompatible and was preserved: {checkpoint_path} ({error})") from error
        else:
            if continue_training:
                initial_epoch = load_state(state_path)
                print(
                    f"[checkpoint] Continuing {checkpoint_path} from epoch {initial_epoch + 1}.")
            else:
                print(
                    f"[checkpoint] Warm-started matching weights from {checkpoint_path}; epoch numbering starts at 1.")
    elif reset:
        print("[checkpoint] --reset replaced the selected task/dataset artifacts; using new weights at epoch 0.")

    callbacks = [
        CompactProgress(),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_weights_only=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        HistorySaver(history_path),
        StateSaver(state_path),
    ]
    if task_name == "detector":
        callbacks.append(FasterRCNNValidationCallback(
            val_builder.samples, log_dir, inspection_class_names(dataset_name)))
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
    parser.add_argument(
        "--classifier", choices=["coco", "imagenet"], default=None)
    parser.add_argument(
        "--detector", choices=["coco", "voc", "imagenet"], default=None)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--continue", dest="continue_training", action="store_true",
                        help="Require checkpoint and saved state, then restore its epoch.")
    parser.add_argument("--resume", dest="continue_training",
                        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reset", action="store_true",
                        help="Delete and recreate only the selected task/dataset artifact directory.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override task defaults (detector: 100; classifier: 20).")
    parser.add_argument("--full-reports", action="store_true",
                        help="Write expanded architecture diagram and summary. Detector inspection grids are always enabled.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the transient metadata/target cache.")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Replace the active transient cache generation.")
    parser.add_argument("--cache-max-samples", type=int, default=CACHE_MAX_SAMPLES,
                        help="Bound lazy cache payloads; never limits dataset enumeration.")
    parser.add_argument("--cache-workers", type=int, default=default_cache_workers(),
                        help="CPU cache preparation workers (default: 4).")
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
    print(
        f"[cache] enabled={not args.no_cache} rebuild={args.rebuild_cache} max_samples={args.cache_max_samples} workers={args.cache_workers}")

    if args.classifier is None and args.detector is None:
        args.detector = "coco"

    # Detector training is graph-compiled and requires NCCL all-reduce across
    # all visible GPUs; no single-device fallback is permitted.
    if args.detector is not None:
        print("[runtime] Detector training requires forced NCCL all-reduce.")
    visible_gpus = len(tf.config.list_physical_devices("GPU"))
    if args.gpu != -1 and (args.gpu < 0 or args.gpu >= visible_gpus):
        parser.error(
            f"--gpu must select one of {visible_gpus} visible GPUs, or use -1 for all GPUs")
    selected_gpu_count = 1 if args.gpu != -1 and visible_gpus else visible_gpus
    replicas_for_batch = max(selected_gpu_count, 1)
    detector_batch_size = BATCH_SIZE
    if detector_batch_size % replicas_for_batch:
        parser.error(
            "The global detector batch must divide evenly across visible replicas.")
    strategy = setup_runtime(
        args.gpu, detector_batch_size if args.detector is not None else BATCH_SIZE)
    classifier_checkpoint = None
    if args.classifier is not None:
        classifier_checkpoint = fit_task(strategy, "classifier", args.classifier, args.continue_training,
                                         args.epochs or CLASSIFIER_EPOCHS, reset=args.reset, full_reports=args.full_reports)
    if args.detector is not None:
        fit_task(strategy, "detector", args.detector, args.continue_training, args.epochs or EPOCHS, initialize_from=classifier_checkpoint,
                 reset=args.reset, full_reports=args.full_reports, detector_batch_size=detector_batch_size)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_cache()
