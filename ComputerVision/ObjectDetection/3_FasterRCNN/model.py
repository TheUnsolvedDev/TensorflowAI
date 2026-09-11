"""Functional Faster R-CNN network plus its eager target/proposal trainer."""

import math
import re

import numpy as np
import tensorflow as tf

from config import (ANCHOR_RATIOS, ANCHOR_SCALES, FC_DIM, IMAGENET_NUM_CLASSES,
                    INPUT_SIZE, LEARNING_RATE, MAX_DETECTIONS_PER_CLASS,
                    NMS_IOU_THRESHOLD, ROI_NEGATIVE_IOU_THRESHOLD, ROI_POOL_SIZE,
                    ROI_POSITIVE_FRACTION, ROI_POSITIVE_IOU_THRESHOLD,
                    ROI_SAMPLES_PER_IMAGE, RPN_NEGATIVE_IOU_THRESHOLD,
                    RPN_NMS_IOU_THRESHOLD, RPN_POST_NMS_TOPK,
                    RPN_POSITIVE_FRACTION, RPN_POSITIVE_IOU_THRESHOLD,
                    RPN_PRE_NMS_TOPK, RPN_SAMPLES_PER_IMAGE, SCORE_THRESHOLD,
                    WEIGHT_DECAY)
from dataset import NUM_CLASSES, VOC_DET_NUM_CLASSES


def iou_yxyx(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    top = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    left = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    bottom = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    right = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.clip(bottom - top, 0., None) * np.clip(right - left, 0., None)
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0., None) * \
        np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0., None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0., None) * \
        np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0., None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0., inter / union, 0.).astype(np.float32)


def encode_boxes_yxyx(anchors, gt_boxes):
    ah, aw = np.maximum(anchors[:, 2] - anchors[:, 0],
                        1e-6), np.maximum(anchors[:, 3] - anchors[:, 1], 1e-6)
    gh, gw = np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0],
                        1e-6), np.maximum(gt_boxes[:, 3] - gt_boxes[:, 1], 1e-6)
    return np.stack([(gt_boxes[:, 0] + .5 * gh - anchors[:, 0] - .5 * ah) / ah,
                     (gt_boxes[:, 1] + .5 * gw - anchors[:, 1] - .5 * aw) / aw,
                     np.log(gh / ah), np.log(gw / aw)], axis=-1).astype(np.float32)


def decode_boxes_yxyx(anchors, deltas):
    ah, aw = tf.maximum(anchors[:, 2] - anchors[:, 0],
                        1e-6), tf.maximum(anchors[:, 3] - anchors[:, 1], 1e-6)
    ay, ax = anchors[:, 0] + .5 * ah, anchors[:, 1] + .5 * aw
    ty, tx, th, tw = tf.unstack(deltas, axis=-1)
    gh, gw = ah * tf.exp(tf.clip_by_value(th, -4., 4.)), aw * \
        tf.exp(tf.clip_by_value(tw, -4., 4.))
    gy, gx = ay + ty * ah, ax + tx * aw
    return tf.clip_by_value(tf.stack([gy - .5 * gh, gx - .5 * gw, gy + .5 * gh, gx + .5 * gw], axis=-1), 0., 1.)


class BatchedROIPooling(tf.keras.layers.Layer):
    """Crop feature maps for each image while retaining the dynamic ROI axis."""

    def __init__(self, pool_size=ROI_POOL_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = tuple(pool_size)

    def call(self, inputs):
        feature_map, rois = inputs

        def crop_one(values):
            fmap, boxes = values
            return tf.image.crop_and_resize(tf.expand_dims(fmap, 0), boxes,
                                            tf.zeros(tf.shape(boxes)[0], tf.int32), self.pool_size)
        return tf.map_fn(crop_one, (feature_map, rois),
                         fn_output_signature=tf.TensorSpec((None, *self.pool_size, feature_map.shape[-1]), feature_map.dtype))


def residual_block(inputs, filters, stride, name):
    regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY)
    shortcut = inputs
    x = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same",
                               use_bias=False, kernel_regularizer=regularizer, name=f"{name}_conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = tf.keras.layers.ReLU(name=f"{name}_relu1")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False,
                               kernel_regularizer=regularizer, name=f"{name}_conv2")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding="same",
                                          use_bias=False, kernel_regularizer=regularizer, name=f"{name}_proj_conv")(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(
            name=f"{name}_proj_bn")(shortcut)
    return tf.keras.layers.ReLU(name=f"{name}_out")(tf.keras.layers.Add(name=f"{name}_add")([x, shortcut]))


def build_faster_rcnn_backbone(input_shape=(*INPUT_SIZE, 3)):
    inputs = tf.keras.Input(shape=input_shape, name="backbone_image")
    regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY)
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False,
                               kernel_regularizer=regularizer, name="stem_conv")(inputs)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)
    x = tf.keras.layers.MaxPooling2D(3, 2, padding="same", name="stem_pool")(x)
    for stage, filters, stride in ((2, 64, 1), (3, 128, 2), (4, 256, 2), (5, 512, 1)):
        x = residual_block(x, filters, stride, f"stage{stage}_block1")
        x = residual_block(x, filters, 1, f"stage{stage}_block2")
    return tf.keras.Model(inputs, x, name="FasterRCNNBackbone")


def build_imagenet_classifier(input_shape=(*INPUT_SIZE, 3), num_classes=IMAGENET_NUM_CLASSES, name="FasterRCNNClassifier"):
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = build_faster_rcnn_backbone(input_shape)(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="classifier_gap")(x)
    x = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(
        WEIGHT_DECAY), name="classifier_fc")(x)
    x = tf.keras.layers.Dropout(.4, name="classifier_dropout")(x)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(num_classes, name="logits", dtype="float32")(x), name=name)


def build_faster_rcnn(num_classes=NUM_CLASSES, input_shape=(*INPUT_SIZE, 3)):
    """Return the executable Functional detector: image/ROIs -> all head outputs."""
    image = tf.keras.Input(shape=input_shape, name="image")
    rois = tf.keras.Input(shape=(None, 4), name="rois")
    backbone = build_faster_rcnn_backbone(input_shape)
    feature_map = backbone(image)
    rpn_feature = tf.keras.layers.Conv2D(
        256, 3, padding="same", activation="relu", name="rpn_conv")(feature_map)
    rpn_objectness = tf.keras.layers.Conv2D(len(
        ANCHOR_SCALES) * len(ANCHOR_RATIOS), 1, padding="same", name="rpn_obj")(rpn_feature)
    rpn_bbox = tf.keras.layers.Conv2D(len(
        ANCHOR_SCALES) * len(ANCHOR_RATIOS) * 4, 1, padding="same", name="rpn_reg")(rpn_feature)

    # Keep the ROI path Functional, but make it a reusable head.  The trainer
    # first computes the backbone/RPN for the complete batch, then feeds each
    # image's sampled ROIs through this head.  Calling the full detector in
    # that loop used to rerun and retain the backbone once per image, which is
    # what exhausted a 16 GiB GPU at the old global batch of 16.
    roi_feature_map = tf.keras.Input(
        shape=(None, None, feature_map.shape[-1]), name="roi_feature_map")
    roi_boxes = tf.keras.Input(shape=(None, 4), name="roi_boxes")
    pooled = BatchedROIPooling(name="roi_pool")([roi_feature_map, roi_boxes])
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Flatten(), name="flatten_rois")(pooled)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
        FC_DIM, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)), name="fc1")(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(
        FC_DIM, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)), name="fc2")(x)
    roi_head = tf.keras.Model(
        [roi_feature_map, roi_boxes],
        {"roi_class_logits": tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, dtype="float32"), name="cls_head")(x),
         "roi_bbox_regression": tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4, dtype="float32"), name="box_head")(x)},
        name="FasterRCNNROIHead",
    )
    roi_outputs = roi_head([feature_map, rois])
    return tf.keras.Model({"image": image, "rois": rois}, {"feature_map": feature_map, "rpn_objectness": rpn_objectness,
                                                           "rpn_bbox": rpn_bbox, **roi_outputs}, name="FasterRCNN")


class FasterRCNNTrainer(tf.keras.Model):
    """Owns sampling, loss/metrics, optimization and native decoded inference."""

    def __init__(self, detector, num_classes=NUM_CLASSES):
        super().__init__(name="FasterRCNNTrainer")
        self.detector, self.num_classes = detector, num_classes
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.rpn_cls_tracker = tf.keras.metrics.Mean(name="rpn_cls_loss")
        self.rpn_box_tracker = tf.keras.metrics.Mean(name="rpn_box_loss")
        self.roi_cls_tracker = tf.keras.metrics.Mean(name="roi_cls_loss")
        self.roi_box_tracker = tf.keras.metrics.Mean(name="roi_box_loss")
        self.acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="roi_accuracy")
        # The Functional detector owns all weight-bearing layers and is already
        # built. Mark this thin orchestration wrapper built so legacy H5 weights
        # can be restored before the first fit/detect call.
        self.build(None)

    @property
    def metrics(self): return [self.loss_tracker, self.rpn_cls_tracker,
                               self.rpn_box_tracker, self.roi_cls_tracker, self.roi_box_tracker, self.acc_tracker]

    def build(self, input_shape=None):
        super().build(input_shape)

    @staticmethod
    def _natural_keys(names):
        return sorted(names, key=lambda name: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)])

    @staticmethod
    def _assign_layer(layer, values, checkpoint):
        variables = layer.variables
        if len(variables) != len(values) or any(tuple(var.shape) != tuple(value.shape) for var, value in zip(variables, values)):
            raise ValueError(
                f"{checkpoint} does not match layer {layer.name}.")
        for variable, value in zip(variables, values):
            variable.assign(value)

    def _load_legacy_h5(self, path, original_error):
        """Migrate the pre-Functional subclass H5 layout when its shapes match."""
        try:
            import h5py
            with h5py.File(path, "r") as file:
                if not {"backbone", "fc1", "fc2", "cls_head", "box_head", "layers"}.issubset(file):
                    raise ValueError(
                        "not a recognized legacy Faster R-CNN checkpoint")
                old_backbone = file["backbone/layers"]
                new_backbone = self.detector.get_layer("FasterRCNNBackbone")
                for old_prefix, layer_type in (("conv2d", tf.keras.layers.Conv2D), ("batch_normalization", tf.keras.layers.BatchNormalization)):
                    old_names = self._natural_keys(
                        [name for name in old_backbone if name == old_prefix or name.startswith(old_prefix + "_")])
                    new_layers = [
                        layer for layer in new_backbone.layers if isinstance(layer, layer_type)]
                    if len(old_names) != len(new_layers):
                        raise ValueError("backbone layer count differs")
                    for old_name, new_layer in zip(old_names, new_layers):
                        self._assign_layer(new_layer, [old_backbone[f"{old_name}/vars/{key}"][(
                        )] for key in self._natural_keys(old_backbone[f"{old_name}/vars"].keys())], path)
                for old_name, new_name in zip(("conv2d", "conv2d_1", "conv2d_2"), ("rpn_conv", "rpn_obj", "rpn_reg")):
                    group = file[f"layers/{old_name}/vars"]
                    self._assign_layer(self.detector.get_layer(new_name), [
                                       group[key][()] for key in self._natural_keys(group.keys())], path)
                roi_head = self.detector.get_layer("FasterRCNNROIHead")
                for old_name, new_name in (("fc1", "fc1"), ("fc2", "fc2"), ("cls_head", "cls_head"), ("box_head", "box_head")):
                    group = file[f"{old_name}/vars"]
                    self._assign_layer(roi_head.get_layer(new_name).layer, [
                                       group[key][()] for key in self._natural_keys(group.keys())], path)
        except Exception as error:
            raise RuntimeError(
                f"Checkpoint is incompatible and was preserved: {path} ({error})") from original_error

    def load_weights(self, filepath, *args, **kwargs):
        try:
            return super().load_weights(filepath, *args, **kwargs)
        except ValueError as original_error:
            self._load_legacy_h5(filepath, original_error)
            return self

    def call(self, inputs, training=False): return self.detector(
        inputs, training=training)

    @staticmethod
    def _sample(indices, count): return np.random.choice(indices, count, replace=len(
        indices) < count).astype(np.int32) if len(indices) and count else np.empty(0, np.int32)

    @staticmethod
    def _anchors(h, w):
        result = []
        for y in range(h):
            for x in range(w):
                for scale in ANCHOR_SCALES:
                    for ratio in ANCHOR_RATIOS:
                        hh, ww = scale / \
                            INPUT_SIZE[0] * math.sqrt(ratio), scale / \
                            INPUT_SIZE[1] / math.sqrt(ratio)
                        result.append([(y + .5) / h - hh / 2, (x + .5) / w -
                                      ww / 2, (y + .5) / h + hh / 2, (x + .5) / w + ww / 2])
        return np.clip(np.asarray(result, np.float32), 0., 1.)

    def _rpn_targets(self, anchors, boxes):
        overlaps = iou_yxyx(anchors, boxes)
        best, match = overlaps.max(1), overlaps.argmax(1)
        pos = self._sample(np.where(best >= RPN_POSITIVE_IOU_THRESHOLD)[
                           0], int(RPN_SAMPLES_PER_IMAGE * RPN_POSITIVE_FRACTION))
        neg = self._sample(np.where(best < RPN_NEGATIVE_IOU_THRESHOLD)[
                           0], RPN_SAMPLES_PER_IMAGE - len(pos))
        selected = np.concatenate((pos, neg))
        cls = np.concatenate(
            (np.ones(len(pos), np.float32), np.zeros(len(neg), np.float32)))
        targets = np.zeros((len(selected), 4), np.float32)
        mask = np.zeros(len(selected), np.float32)
        if len(pos):
            targets[:len(pos)], mask[:len(pos)] = encode_boxes_yxyx(
                anchors[pos], boxes[match[pos]]), 1.
        return selected, cls, targets, mask

    def _roi_targets(self, proposals, boxes, labels):
        proposals = np.concatenate((proposals, boxes)).astype(np.float32)
        overlaps = iou_yxyx(proposals, boxes)
        best, match = overlaps.max(1), overlaps.argmax(1)
        pos = self._sample(np.where(best >= ROI_POSITIVE_IOU_THRESHOLD)[
                           0], int(ROI_SAMPLES_PER_IMAGE * ROI_POSITIVE_FRACTION))
        neg_pool = np.where(best < ROI_NEGATIVE_IOU_THRESHOLD)[0]
        neg = self._sample(neg_pool if len(neg_pool) else np.where(
            best < ROI_POSITIVE_IOU_THRESHOLD)[0], ROI_SAMPLES_PER_IMAGE - len(pos))
        chosen = np.concatenate((pos, neg))
        if len(chosen) < ROI_SAMPLES_PER_IMAGE:
            chosen = np.concatenate((chosen, self._sample(
                np.arange(len(proposals)), ROI_SAMPLES_PER_IMAGE - len(chosen))))
        chosen = chosen[:ROI_SAMPLES_PER_IMAGE]
        result_labels, target, weights = np.zeros(len(chosen), np.int32), np.zeros(
            (len(chosen), 4), np.float32), np.zeros(len(chosen), np.float32)
        positive = set(pos.tolist())
        for index, proposal_index in enumerate(chosen):
            if proposal_index in positive:
                result_labels[index], target[index], weights[index] = labels[match[proposal_index]], encode_boxes_yxyx(
                    proposals[proposal_index:proposal_index+1], boxes[match[proposal_index:proposal_index+1]])[0], 1.
        return proposals[chosen], result_labels, target, weights

    @staticmethod
    def _numpy(fn, args, shapes, types):
        values = tf.numpy_function(fn, args, types)
        for value, shape in zip(values, shapes):
            value.set_shape(shape)
        return values

    def _proposals(self, anchors, logits, deltas):
        scores, boxes = tf.nn.sigmoid(
            logits), decode_boxes_yxyx(anchors, deltas)
        count = min(int(scores.shape[0]), RPN_PRE_NMS_TOPK)
        indices = tf.math.top_k(scores, count).indices
        boxes, scores = tf.gather(boxes, indices), tf.gather(scores, indices)
        return tf.gather(boxes, tf.image.non_max_suppression(boxes, scores, RPN_POST_NMS_TOPK, RPN_NMS_IOU_THRESHOLD))

    def _forward_losses(self, inputs, training):
        images, boxes, labels, valid = inputs["image"], inputs["gt_boxes"], inputs["gt_labels"], tf.cast(
            inputs["valid_mask"], tf.bool)
        batch = images.shape[0]
        if batch is None:
            raise ValueError(
                "FasterRCNN requires a statically sized per-replica batch.")
        # A disposable ROI keeps TimeDistributed valid for this RPN pass; actual
        # sampled ROIs are fed below for the ROI losses.
        outputs = self.detector({"image": images, "rois": tf.zeros(
            (batch, 1, 4), tf.float32)}, training=training)
        anchors = tf.convert_to_tensor(self._anchors(
            int(outputs["feature_map"].shape[1]), int(outputs["feature_map"].shape[2])))
        losses, all_logits, all_labels = [0.] * 4, [], []
        for index in range(batch):
            gt_boxes, gt_labels = tf.boolean_mask(
                boxes[index], valid[index]), tf.boolean_mask(labels[index], valid[index])
            tf.debugging.assert_positive(tf.shape(gt_boxes)[0])
            obj, delta = tf.reshape(
                outputs["rpn_objectness"][index], (-1,)), tf.reshape(outputs["rpn_bbox"][index], (-1, 4))
            selected, cls, target, mask = self._numpy(self._rpn_targets, [anchors, gt_boxes], [(
                None,), (None,), (None, 4), (None,)], [tf.int32, tf.float32, tf.float32, tf.float32])
            rpn_c = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=cls, logits=tf.gather(obj, selected)))
            rpn_b = tf.reduce_mean(tf.keras.losses.huber(
                target, tf.gather(delta, selected)) * mask)
            proposal = self._proposals(anchors, obj, delta)
            roi, roi_label, roi_target, roi_weight = self._numpy(self._roi_targets, [proposal, gt_boxes, gt_labels], [(ROI_SAMPLES_PER_IMAGE, 4), (
                ROI_SAMPLES_PER_IMAGE,), (ROI_SAMPLES_PER_IMAGE, 4), (ROI_SAMPLES_PER_IMAGE,)], [tf.float32, tf.int32, tf.float32, tf.float32])
            # Reuse this batch's feature map: only the small ROI head is run
            # per image, so GradientTape does not retain B additional backbone
            # graphs for every training step.
            roi_outputs = self.detector.get_layer("FasterRCNNROIHead")(
                [outputs["feature_map"][index:index+1], roi[None, ...]], training=training)
            roi_logits, roi_delta = roi_outputs["roi_class_logits"][0], roi_outputs["roi_bbox_regression"][0]
            roi_c = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                roi_label, roi_logits, from_logits=True))
            roi_b = tf.reduce_mean(tf.keras.losses.huber(
                roi_target, roi_delta) * roi_weight)
            losses = [losses[0]+rpn_c, losses[1] +
                      rpn_b, losses[2]+roi_c, losses[3]+roi_b]
            all_logits.append(roi_logits)
            all_labels.append(roi_label)
        return [value / batch for value in losses], all_logits, all_labels

    def _step(self, data, training):
        inputs = data[0] if isinstance(data, (tuple, list)) else data
        if training:
            with tf.GradientTape() as tape:
                losses, logits, labels = self._forward_losses(inputs, True)
                total = tf.add_n(
                    losses) + (tf.add_n(self.detector.losses) if self.detector.losses else 0.)
            self.optimizer.apply_gradients(zip(tape.gradient(
                total, self.detector.trainable_variables), self.detector.trainable_variables))
        else:
            losses, logits, labels = self._forward_losses(inputs, False)
            total = tf.add_n(losses)
        self.loss_tracker.update_state(total)
        self.rpn_cls_tracker.update_state(losses[0])
        self.rpn_box_tracker.update_state(losses[1])
        self.roi_cls_tracker.update_state(losses[2])
        self.roi_box_tracker.update_state(losses[3])
        self.acc_tracker.update_state(
            tf.concat(labels, 0), tf.concat(logits, 0))
        return {metric.name: metric.result() for metric in self.metrics}

    def train_step(self, data): return self._step(data, True)
    def test_step(self, data): return self._step(data, False)

    def detect(self, image):
        outputs = self.detector(
            {"image": image, "rois": tf.zeros((1, 1, 4), tf.float32)}, training=False)
        fmap = outputs["feature_map"][0]
        anchors = tf.convert_to_tensor(self._anchors(
            int(fmap.shape[0]), int(fmap.shape[1])))
        obj, delta = tf.reshape(
            outputs["rpn_objectness"][0], (-1,)), tf.reshape(outputs["rpn_bbox"][0], (-1, 4))
        proposals = self._proposals(anchors, obj, delta)
        roi = self.detector(
            {"image": image, "rois": proposals[None, ...]}, training=False)
        probs = tf.nn.softmax(roi["roi_class_logits"][0]).numpy()
        refined = decode_boxes_yxyx(
            proposals, roi["roi_bbox_regression"][0]).numpy()
        ids, scores = probs.argmax(1), probs.max(1)
        found = []
        for class_id in range(1, self.num_classes):
            selected = np.where((ids == class_id) & (
                scores >= SCORE_THRESHOLD))[0]
            if len(selected):
                for keep in tf.image.non_max_suppression(refined[selected], scores[selected], MAX_DETECTIONS_PER_CLASS, NMS_IOU_THRESHOLD).numpy():
                    found.append({"class_id": int(class_id), "score": float(
                        scores[selected[keep]]), "box": refined[selected[keep]]})
        return sorted(found, key=lambda item: item["score"], reverse=True)


def build_faster_rcnn_report_model(detector_model):
    return detector_model


def create_compiled_model(task_name="detector", dataset_name="coco"):
    if task_name == "classifier":
        classes = IMAGENET_NUM_CLASSES if dataset_name == "imagenet" else NUM_CLASSES - 1
        model = build_imagenet_classifier(
            num_classes=classes, name="FasterRCNNClassifier")
        model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")], steps_per_execution=1, run_eagerly=False)
        return model
    classes = IMAGENET_NUM_CLASSES + \
        1 if dataset_name == "imagenet" else VOC_DET_NUM_CLASSES if dataset_name == "voc" else NUM_CLASSES
    trainer = FasterRCNNTrainer(build_faster_rcnn(classes), classes)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(
        LEARNING_RATE), steps_per_execution=1, run_eagerly=False)
    return trainer
