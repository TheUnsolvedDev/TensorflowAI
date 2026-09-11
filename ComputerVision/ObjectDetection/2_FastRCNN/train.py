import argparse
import gc
import json
import os
import shutil
import time

import cv2
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tensorflow.keras import mixed_precision

from config import BATCH_SIZE, CHECKPOINT_PATH, CLASSIFIER_CHECKPOINT_PATH, CLASSIFIER_EPOCHS, CLASSIFIER_HISTORY_PATH, CLASSIFIER_LOG_DIR, CLASSIFIER_STATE_PATH, EPOCHS, HISTORY_PATH, INFERENCE_BATCH_SIZE, INFERENCE_TOPK, INPUT_SIZE, LOG_DIR, MAX_DETECTIONS_PER_CLASS, MAX_DRAW_DETECTIONS, CACHE_MAX_SAMPLES, MAX_PROPOSALS, NMS_IOU_THRESHOLD, SCORE_THRESHOLD, STATE_PATH, STEPS_PER_EPOCH, USE_SELECTIVE_SEARCH, VALIDATION_METRIC_SAMPLES, VALIDATION_STEPS, ensure_dir
from dataset import COCO_CATEGORY_IDS, COCO_CLASSES, build_train_dataset, build_val_dataset, resolve_detection_dataset
from model import create_compiled_model
from cache_runtime import cleanup_cache, default_cache_workers
from utils import batched, clip_box, compute_iou, decode_box, draw_detections, draw_ground_truth, generate_region_proposals, nms


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
            raise RuntimeError(
                f"BATCH_SIZE={BATCH_SIZE} must be divisible by {visible} NCCL replicas.")
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.NcclAllReduce())
    elif visible == 1:
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
    print(f"[runtime] visible_gpus={visible} strategy={type(strategy).__name__} replicas={strategy.num_replicas_in_sync} global_batch={BATCH_SIZE} per_replica_batch={BATCH_SIZE // strategy.num_replicas_in_sync}")
    return strategy


def load_state(path, required=False):
    """Load the completed epoch count, optionally requiring a valid state file."""
    if not os.path.exists(path):
        if required:
            raise RuntimeError(
                f"--continue requires saved epoch state: {path}")
        return 0
    try:
        with open(path, "r", encoding="utf-8") as file:
            epoch = json.load(file)["epoch"]
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        return epoch
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        if required:
            raise RuntimeError(
                f"--continue requires valid saved epoch state: {path}") from error
        raise RuntimeError(f"Invalid saved epoch state: {path}") from error


def save_state(path, epoch):
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"epoch": int(epoch)}, file)


def reset_artifacts(log_dir):
    """Discard only the selected task/dataset run artifacts for ``--reset``."""
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    ensure_dir(log_dir)


def load_task_checkpoint(model, checkpoint_path, *, mode):
    """Strictly restore a task checkpoint; never skip an incompatible head."""
    try:
        model.load_weights(checkpoint_path)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Cannot {mode} from incompatible checkpoint {checkpoint_path}; it was preserved. "
            "Use --reset to start with newly initialized weights."
        ) from error


def configure_run(model, checkpoint_path, state_path, *, continue_run=False, reset=False, initialize_from=None):
    """Apply explicit run controls and return the initial epoch for ``model.fit``."""
    if continue_run and reset:
        raise ValueError("--continue and --reset cannot be used together")

    if reset:
        print(
            f"[run] mode=reset checkpoint={checkpoint_path} state={state_path}")
        return 0, "reset"

    if continue_run:
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(
                f"--continue requires checkpoint: {checkpoint_path}")
        initial_epoch = load_state(state_path, required=True)
        load_task_checkpoint(model, checkpoint_path, mode="continue")
        print(
            f"[run] mode=continue initial_epoch={initial_epoch} checkpoint={checkpoint_path} state={state_path}")
        return initial_epoch, "continue"

    if os.path.exists(checkpoint_path):
        load_task_checkpoint(model, checkpoint_path, mode="load weights")
        print(
            f"[run] mode=default-load initial_epoch=0 checkpoint={checkpoint_path} state={state_path}")
        return 0, "default-load"

    if initialize_from and os.path.exists(initialize_from):
        try:
            model.load_weights(initialize_from, skip_mismatch=True)
        except (OSError, TypeError, ValueError) as error:
            print(
                f"[run] classifier initialization skipped: {initialize_from} ({error})")
        else:
            print(
                f"[run] mode=default-initialize initial_epoch=0 checkpoint={checkpoint_path} state={state_path}")
            return 0, "default-initialize"

    print(
        f"[run] mode=default-fresh initial_epoch=0 checkpoint={checkpoint_path} state={state_path}")
    return 0, "default-fresh"


def requested_run_mode(continue_run, reset):
    if reset:
        return "reset"
    if continue_run:
        return "continue"
    return "default"


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


def infer_image(model, image_path, class_names, proposals=None, return_score_diagnostics=False):
    """Folder-local detector inference used by the epoch-end inspection callback."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(
            f"Could not read inspection image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if proposals is None:
        proposals = generate_region_proposals(
            image_bgr,
            max_proposals=MAX_PROPOSALS,
            use_selective_search=USE_SELECTIVE_SEARCH,
        )
    height, width = image_bgr.shape[:2]
    rois = np.asarray([[box[1] / height, box[0] / width, box[3] /
                      height, box[2] / width] for box in proposals], dtype=np.float32)
    image = cv2.resize(image_rgb, (INPUT_SIZE[1], INPUT_SIZE[0])).astype(
        np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    class_logits, bbox_deltas = [], []
    for roi_batch in batched(rois, INFERENCE_BATCH_SIZE):
        predictions = model(
            {"image": image, "rois": np.expand_dims(roi_batch, axis=0)}, training=False)
        class_logits.append(predictions["class_logits"].numpy()[0])
        bbox_deltas.append(predictions["bbox_regression"].numpy()[0])
    class_logits = np.concatenate(class_logits, axis=0)
    bbox_deltas = np.concatenate(bbox_deltas, axis=0)
    probabilities = tf.nn.softmax(class_logits, axis=-1).numpy()
    class_ids = probabilities.argmax(axis=1)
    class_scores = probabilities.max(axis=1)
    foreground_mask = class_ids > 0
    score_diagnostics = {
        "foreground_predictions_pre_threshold": int(np.count_nonzero(foreground_mask)),
        "foreground_predictions_post_threshold": int(np.count_nonzero(foreground_mask & (class_scores >= SCORE_THRESHOLD))),
        "highest_foreground_score": float(class_scores[foreground_mask].max()) if np.any(foreground_mask) else 0.0,
    }
    decoded_boxes = np.asarray([decode_box(proposal, delta) for proposal, delta in zip(
        proposals, bbox_deltas)], dtype=np.float32)
    decoded_boxes = np.asarray([clip_box(box, height, width)
                               for box in decoded_boxes], dtype=np.float32)
    ranked = np.argsort(class_scores)[::-1][:INFERENCE_TOPK]
    detections = []
    for class_id in range(1, len(class_names)):
        indices = ranked[(class_ids[ranked] == class_id) & (
            class_scores[ranked] >= SCORE_THRESHOLD)]
        if len(indices) == 0:
            continue
        keep = nms(decoded_boxes[indices], class_scores[indices],
                   NMS_IOU_THRESHOLD, MAX_DETECTIONS_PER_CLASS)
        detections.extend(
            {"class_id": int(class_id), "score": float(
                class_scores[indices[keep_idx]]), "box": decoded_boxes[indices[keep_idx]]}
            for keep_idx in keep
        )
    detections.sort(key=lambda item: item["score"], reverse=True)
    if return_score_diagnostics:
        return detections, image_bgr, score_diagnostics
    return detections, image_bgr


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
    """Short terminal status while full metric names remain in logs/history."""

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
        steps_per_second = step / max(elapsed, 1e-6)
        print(
            f"\rE{self.epoch} {step}/{self.steps} "
            f"loss={self._metric(logs, 'loss'):.3f} "
            f"cls={self._metric(logs, 'class_logits_loss'):.3f} "
            f"acc={self._metric(logs, 'class_logits_accuracy'):.1%} "
            f"box={self._metric(logs, 'bbox_regression_loss'):.3f} "
            f"mae={self._metric(logs, 'bbox_regression_mae'):.3f} "
            f"in={steps_per_second * BATCH_SIZE:.1f} ex/s step/s={steps_per_second:.2f} "
            f"eta={int(eta // 60)}:{int(eta % 60):02d}",
            end="",
            flush=True,
        )

    def on_epoch_end(self, epoch, logs=None):
        print()


class DetectionInspectionCallback(tf.keras.callbacks.Callback):
    """Save a fixed 4x4 validation grid after each completed epoch."""

    def __init__(self, samples, output_dir, class_names, tile_size=(320, 240)):
        super().__init__()
        self.samples = list(samples[:16])
        self.output_dir = ensure_dir(output_dir)
        self.tile_size = tile_size
        self.class_names = class_names

    def _render_sample(self, sample):
        image_bgr = cv2.imread(sample["image_path"])
        if image_bgr is None:
            return None
        canvas = draw_ground_truth(
            image_bgr,
            sample["boxes"],
            sample["labels"],
            self.class_names,
            max_draw=MAX_DRAW_DETECTIONS,
        )
        detections, _ = infer_image(
            self.model, sample["image_path"], self.class_names)
        canvas = draw_detections(
            canvas, detections, self.class_names, max_draw=MAX_DRAW_DETECTIONS)
        canvas = cv2.resize(canvas, self.tile_size,
                            interpolation=cv2.INTER_AREA)
        cv2.putText(canvas, "GT: red | prediction: green", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "GT: red | prediction: green", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        return canvas

    def on_epoch_end(self, epoch, logs=None):
        panels = [panel for sample in self.samples if (
            panel := self._render_sample(sample)) is not None]
        if not panels:
            print("[inspection] No validation images could be rendered.")
            return
        blank = np.zeros_like(panels[0])
        panels.extend(blank.copy() for _ in range(16 - len(panels)))
        grid = np.concatenate([np.concatenate(
            panels[row: row + 4], axis=1) for row in range(0, 16, 4)], axis=0)
        path = os.path.join(self.output_dir, f"epoch_{epoch + 1:03d}.png")
        if not cv2.imwrite(path, grid):
            raise RuntimeError(f"Could not write inspection image: {path}")
        print(f"[inspection] Saved 4x4 ground-truth/prediction grid: {path}")


class COCOValidationMetricsCallback(tf.keras.callbacks.Callback):
    """Measure actual proposal quality and COCO-style detection AP on a fixed subset."""

    def __init__(self, samples, annotation_path, output_path, tensorboard_dir):
        super().__init__()
        self.samples = list(samples[:VALIDATION_METRIC_SAMPLES])
        self.annotation_path = annotation_path
        self.output_path = output_path
        self.writer = tf.summary.create_file_writer(tensorboard_dir)
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        results, recalls, best_ious, image_ids = [], [], [], []
        foreground_pre_threshold = 0
        foreground_post_threshold = 0
        highest_foreground_score = 0.0
        for sample in self.samples:
            image_bgr = cv2.imread(sample["image_path"])
            if image_bgr is None:
                continue
            proposals = generate_region_proposals(
                image_bgr,
                max_proposals=MAX_PROPOSALS,
                use_selective_search=USE_SELECTIVE_SEARCH,
            )
            proposal_ious = compute_iou(proposals, sample["boxes"])
            max_ious = proposal_ious.max(axis=0) if len(proposals) else np.zeros(
                (len(sample["boxes"]),), dtype=np.float32)
            recalls.extend((max_ious >= 0.5).astype(np.float32))
            best_ious.extend(max_ious.astype(np.float32))
            detections, _, score_diagnostics = infer_image(
                self.model,
                sample["image_path"], COCO_CLASSES,
                proposals=proposals,
                return_score_diagnostics=True,
            )
            foreground_pre_threshold += score_diagnostics["foreground_predictions_pre_threshold"]
            foreground_post_threshold += score_diagnostics["foreground_predictions_post_threshold"]
            highest_foreground_score = max(
                highest_foreground_score, score_diagnostics["highest_foreground_score"])
            image_ids.append(int(sample["image_id"]))
            for detection in detections:
                x1, y1, x2, y2 = detection["box"].tolist()
                results.append(
                    {
                        "image_id": int(sample["image_id"]),
                        "category_id": COCO_CATEGORY_IDS[int(detection["class_id"]) - 1],
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(detection["score"]),
                    }
                )

        if results:
            coco_gt = COCO(self.annotation_path)
            coco_dt = coco_gt.loadRes(results)
            evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
            evaluator.params.imgIds = image_ids
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            subset_map, subset_ap50, subset_ap75 = (
                float(evaluator.stats[index]) for index in range(3))
        else:
            subset_map = subset_ap50 = subset_ap75 = 0.0
            print(
                "[validation] No detections above SCORE_THRESHOLD; subset AP values are 0.0.")
        metrics = {
            "proposal_recall_iou50": float(np.mean(recalls)) if recalls else 0.0,
            "proposal_mean_best_iou": float(np.mean(best_ious)) if best_ious else 0.0,
            "subset_map": subset_map,
            "subset_ap50": subset_ap50,
            "subset_ap75": subset_ap75,
            "foreground_predictions_pre_threshold": foreground_pre_threshold,
            "foreground_predictions_post_threshold": foreground_post_threshold,
            "highest_foreground_score": highest_foreground_score,
        }
        logs.update(metrics)
        self.history.append({"epoch": epoch + 1, **metrics})
        with open(self.output_path, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=2)
        with self.writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=epoch + 1)
        self.writer.flush()
        print(
            "[validation] "
            f"recall@0.5={metrics['proposal_recall_iou50']:.3f} "
            f"best_iou={metrics['proposal_mean_best_iou']:.3f} "
            f"mAP={metrics['subset_map']:.3f} "
            f"AP50={metrics['subset_ap50']:.3f} AP75={metrics['subset_ap75']:.3f} "
            f"fg={metrics['foreground_predictions_pre_threshold']}->"
            f"{metrics['foreground_predictions_post_threshold']} "
            f"top_fg={metrics['highest_foreground_score']:.3f}"
        )


def save_model_report(model, task_name):
    """Print and save a full, folder-local Keras architecture report."""
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
        dataset_root = resolve_detection_dataset(dataset_name).root
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
    dataset_root = resolve_detection_dataset(dataset_name).root
    log_dir, checkpoint_path, history_path, state_path = scope_artifact_paths(
        LOG_DIR,
        CHECKPOINT_PATH,
        HISTORY_PATH,
        STATE_PATH,
        dataset_name,
    )
    return dataset_root, log_dir, checkpoint_path, history_path, state_path


def fit_task(strategy, task_name, dataset_name, continue_run, reset, epochs, initialize_from=None, full_reports=False, **cache_options):
    dataset_root, log_dir, checkpoint_path, history_path, state_path = select_artifacts(
        task_name, dataset_name)
    if dataset_root is None:
        raise RuntimeError(
            f"{task_name} dataset root not found for {dataset_name}")
    print(
        f"[run] requested_mode={requested_run_mode(continue_run, reset)} "
        f"checkpoint={checkpoint_path} state={state_path}"
    )
    if continue_run:
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(
                f"--continue requires checkpoint: {checkpoint_path}")
        load_state(state_path, required=True)
    if reset:
        reset_artifacts(log_dir)
    else:
        ensure_dir(log_dir)
    adapter = resolve_detection_dataset(dataset_name, root=dataset_root)
    train_split = adapter.split_name("train")
    val_split = adapter.split_name("val")
    train_builder, train_dataset = build_train_dataset(
        task_name=task_name,
        dataset_name=dataset_name,
        root=dataset_root,
        split=train_split,
        **cache_options,
    )
    val_builder, val_dataset = build_val_dataset(
        task_name=task_name,
        dataset_name=dataset_name,
        root=dataset_root,
        split=val_split,
        **cache_options,
    )
    # build_train_dataset/build_val_dataset already prepare each detector cache
    # and construct its tf.data pipeline.  Reopening them here only repeats warm
    # cache startup without changing the datasets.
    steps_per_epoch = getattr(
        train_builder, "steps_per_epoch", STEPS_PER_EPOCH)
    validation_steps = getattr(
        val_builder, "steps_per_epoch", VALIDATION_STEPS)

    tf.keras.backend.clear_session()
    with strategy.scope():
        model = create_compiled_model(
            task_name=task_name,
            dataset_name=dataset_name,
            num_classes=getattr(train_builder, "num_classes", None),
        )
    initial_epoch, _ = configure_run(
        model,
        checkpoint_path,
        state_path,
        continue_run=continue_run,
        reset=reset,
        initialize_from=initialize_from,
    )
    if full_reports:
        save_model_report(model, task_name)

    callbacks = [
        CompactProgress(),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_weights_only=True, verbose=1),
        HistorySaver(history_path),
        StateSaver(state_path),
    ]
    if full_reports and task_name == "detector" and dataset_name == "coco":
        callbacks.append(
            COCOValidationMetricsCallback(
                val_builder.samples,
                os.path.join(dataset_root, "annotations",
                             f"instances_{val_split}.json"),
                os.path.join(log_dir, "validation_metrics.json"),
                os.path.join(log_dir, "validation_tensorboard"),
            )
        )
    if full_reports:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
    if task_name == "detector":
        callbacks.append(
            DetectionInspectionCallback(
                val_builder.samples,
                os.path.join(log_dir, "inspection"),
                val_builder.class_names,
            )
        )
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
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--continue", dest="continue_run", action="store_true",
                          help="Require checkpoint and saved epoch state, then continue toward --epochs.")
    run_mode.add_argument("--reset", action="store_true",
                          help="Discard selected task/dataset run artifacts and train from newly initialized weights.")
    run_mode.add_argument("--resume", dest="continue_run",
                          action="store_true", help="Compatibility alias for --continue.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the task default (detector: 100, classifier: 20).")
    parser.add_argument("--full-reports", action="store_true",
                        help="Enable TensorBoard, model diagrams, and COCO-only 200-image metrics. Detector inspection grids are always enabled.")
    parser.add_argument("--use-cache", action="store_true",
                        help="Compatibility alias; persistent proposal caching is enabled by default.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the default persistent cache and generate proposals lazily per input batch.")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Replace only the selected split-scoped persistent cache generation.")
    parser.add_argument("--cache-max-samples", type=int, default=CACHE_MAX_SAMPLES,
                        help="Persistent cache capacity; it must cover every sample in a split (0 selects no-cache).")
    parser.add_argument("--cache-workers", type=int, default=default_cache_workers(),
                        help="Processes for proposal preparation (default: 4; use 1 for serial).")
    parser.add_argument("--prepare-cache", action="store_true",
                        help="Build and verify selected detector train/validation caches, then exit.")
    args = parser.parse_args()

    if args.no_cache and args.use_cache:
        parser.error("--no-cache and --use-cache cannot be used together")
    if args.no_cache and (args.prepare_cache or args.rebuild_cache):
        parser.error(
            "--no-cache cannot be used with cache preparation or rebuild commands")
    if args.prepare_cache and (args.continue_run or args.reset):
        parser.error(
            "--prepare-cache cannot be used with --continue or --reset")
    if args.cache_max_samples < 0:
        parser.error("--cache-max-samples must be non-negative")
    if args.cache_workers < 1:
        parser.error("--cache-workers must be at least 1")
    cache_options = {
        "cache_enabled": not args.no_cache,
        "cache_rebuild": args.rebuild_cache,
        "cache_max_samples": args.cache_max_samples,
        "cache_workers": args.cache_workers,
    }
    print(
        "[cache config] "
        f"enabled={cache_options['cache_enabled']} "
        f"rebuild={cache_options['cache_rebuild']} "
        f"max_samples={cache_options['cache_max_samples']} "
        f"workers={cache_options['cache_workers']}"
    )

    if args.classifier is None and args.detector is None:
        args.detector = "coco"

    if args.epochs is None:
        args.epochs = CLASSIFIER_EPOCHS if args.classifier is not None and args.detector is None else EPOCHS

    strategy = setup_runtime(args.gpu)
    classifier_checkpoint = None
    if args.classifier is not None:
        classifier_checkpoint = fit_task(
            strategy, "classifier", args.classifier, args.continue_run, args.reset, args.epochs,
            full_reports=args.full_reports,
        )
    if args.detector is not None:
        if args.prepare_cache:
            dataset_root, _, _, _, _ = select_artifacts(
                "detector", args.detector)
            adapter = resolve_detection_dataset(
                args.detector, root=dataset_root)
            split_train = adapter.split_name("train")
            split_val = adapter.split_name("val")
            # Dataset construction prepares and verifies each selected cache once.
            build_train_dataset("detector", args.detector,
                                dataset_root, split_train, **cache_options)
            build_val_dataset("detector", args.detector,
                              dataset_root, split_val, **cache_options)
            return
        fit_task(
            strategy, "detector", args.detector, args.continue_run, args.reset, args.epochs,
            initialize_from=classifier_checkpoint, full_reports=args.full_reports, **cache_options,
        )


if __name__ == "__main__":
    main()
