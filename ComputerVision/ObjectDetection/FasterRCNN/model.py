import math

import numpy as np
import tensorflow as tf

from config import (
    ANCHOR_RATIOS,
    ANCHOR_SCALES,
    FC_DIM,
    INPUT_SIZE,
    LEARNING_RATE,
    MAX_DETECTIONS_PER_CLASS,
    NMS_IOU_THRESHOLD,
    ROI_NEGATIVE_IOU_THRESHOLD,
    ROI_POOL_SIZE,
    ROI_POSITIVE_FRACTION,
    ROI_POSITIVE_IOU_THRESHOLD,
    ROI_SAMPLES_PER_IMAGE,
    RPN_NEGATIVE_IOU_THRESHOLD,
    RPN_NMS_IOU_THRESHOLD,
    RPN_POST_NMS_TOPK,
    RPN_POSITIVE_FRACTION,
    RPN_POSITIVE_IOU_THRESHOLD,
    RPN_PRE_NMS_TOPK,
    RPN_SAMPLES_PER_IMAGE,
    SCORE_THRESHOLD,
    WEIGHT_DECAY,
)
from dataset import NUM_CLASSES


def iou_yxyx(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    top = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    left = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    bottom = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    right = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    h = np.clip(bottom - top, 0.0, None)
    w = np.clip(right - left, 0.0, None)
    inter = h * w
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0.0, None) * np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0.0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0.0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0).astype(np.float32)


def encode_boxes_yxyx(anchors, gt_boxes):
    ah = np.maximum(anchors[:, 2] - anchors[:, 0], 1e-6)
    aw = np.maximum(anchors[:, 3] - anchors[:, 1], 1e-6)
    ay = anchors[:, 0] + 0.5 * ah
    ax = anchors[:, 1] + 0.5 * aw
    gh = np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0], 1e-6)
    gw = np.maximum(gt_boxes[:, 3] - gt_boxes[:, 1], 1e-6)
    gy = gt_boxes[:, 0] + 0.5 * gh
    gx = gt_boxes[:, 1] + 0.5 * gw
    ty = (gy - ay) / ah
    tx = (gx - ax) / aw
    th = np.log(gh / ah)
    tw = np.log(gw / aw)
    return np.stack([ty, tx, th, tw], axis=-1).astype(np.float32)


def decode_boxes_yxyx(anchors, deltas):
    ah = tf.maximum(anchors[:, 2] - anchors[:, 0], 1e-6)
    aw = tf.maximum(anchors[:, 3] - anchors[:, 1], 1e-6)
    ay = anchors[:, 0] + 0.5 * ah
    ax = anchors[:, 1] + 0.5 * aw
    ty, tx, th, tw = tf.unstack(deltas, axis=-1)
    gy = ay + ty * ah
    gx = ax + tx * aw
    gh = ah * tf.exp(tf.clip_by_value(th, -4.0, 4.0))
    gw = aw * tf.exp(tf.clip_by_value(tw, -4.0, 4.0))
    y1 = gy - 0.5 * gh
    x1 = gx - 0.5 * gw
    y2 = gy + 0.5 * gh
    x2 = gx + 0.5 * gw
    boxes = tf.stack([y1, x1, y2, x2], axis=-1)
    return tf.clip_by_value(boxes, 0.0, 1.0)


class ROIPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=ROI_POOL_SIZE):
        super().__init__()
        self.pool_size = tuple(pool_size)

    def call(self, feature_map, rois):
        box_indices = tf.zeros((tf.shape(rois)[0],), dtype=tf.int32)
        return tf.image.crop_and_resize(
            image=tf.expand_dims(feature_map, axis=0),
            boxes=rois,
            box_indices=box_indices,
            crop_size=self.pool_size,
        )


class FasterRCNN(tf.keras.Model):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__(name="FasterRCNN")
        self.num_classes = num_classes
        self.num_anchors = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
        self.backbone = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", activation="relu"),
                tf.keras.layers.MaxPooling2D(3, 2, padding="same"),
                tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
                tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
                tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
            ]
        )
        self.rpn_conv = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")
        self.rpn_obj = tf.keras.layers.Conv2D(self.num_anchors, 1, padding="same")
        self.rpn_reg = tf.keras.layers.Conv2D(self.num_anchors * 4, 1, padding="same")
        self.roi_pool = ROIPooling()
        self.fc1 = tf.keras.layers.Dense(FC_DIM, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))
        self.fc2 = tf.keras.layers.Dense(FC_DIM, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))
        self.cls_head = tf.keras.layers.Dense(num_classes, dtype="float32")
        self.box_head = tf.keras.layers.Dense(4, dtype="float32")
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.rpn_cls_tracker = tf.keras.metrics.Mean(name="rpn_cls_loss")
        self.rpn_box_tracker = tf.keras.metrics.Mean(name="rpn_box_loss")
        self.roi_cls_tracker = tf.keras.metrics.Mean(name="roi_cls_loss")
        self.roi_box_tracker = tf.keras.metrics.Mean(name="roi_box_loss")
        self.acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="roi_accuracy")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.rpn_cls_tracker,
            self.rpn_box_tracker,
            self.roi_cls_tracker,
            self.roi_box_tracker,
            self.acc_tracker,
        ]

    def call(self, inputs, training=False):
        images = inputs["image"] if isinstance(inputs, dict) else inputs
        feature_map = self.backbone(images, training=training)
        rpn_feature = self.rpn_conv(feature_map, training=training)
        rpn_obj = self.rpn_obj(rpn_feature, training=training)
        rpn_reg = self.rpn_reg(rpn_feature, training=training)
        return {"feature_map": feature_map, "rpn_objectness": rpn_obj, "rpn_bbox": rpn_reg}

    def _generate_anchors(self, feature_h, feature_w):
        anchors = []
        for y in range(feature_h):
            cy = (y + 0.5) / feature_h
            for x in range(feature_w):
                cx = (x + 0.5) / feature_w
                for scale in ANCHOR_SCALES:
                    for ratio in ANCHOR_RATIOS:
                        h = (scale / INPUT_SIZE[0]) * math.sqrt(ratio)
                        w = (scale / INPUT_SIZE[1]) / math.sqrt(ratio)
                        anchors.append([cy - 0.5 * h, cx - 0.5 * w, cy + 0.5 * h, cx + 0.5 * w])
        anchors = np.asarray(anchors, dtype=np.float32)
        return np.clip(anchors, 0.0, 1.0)

    def _sample_indices(self, indices, count):
        if len(indices) == 0 or count <= 0:
            return np.array([], dtype=np.int32)
        replace = len(indices) < count
        return np.random.choice(indices, size=count, replace=replace)

    def _sample_rpn_targets(self, anchors, gt_boxes):
        ious = iou_yxyx(anchors, gt_boxes)
        max_iou = ious.max(axis=1)
        matched_gt = ious.argmax(axis=1)
        positive = np.where(max_iou >= RPN_POSITIVE_IOU_THRESHOLD)[0]
        negative = np.where(max_iou < RPN_NEGATIVE_IOU_THRESHOLD)[0]
        num_positive = int(RPN_SAMPLES_PER_IMAGE * RPN_POSITIVE_FRACTION)
        num_negative = RPN_SAMPLES_PER_IMAGE - num_positive
        pos_idx = self._sample_indices(positive, num_positive)
        neg_idx = self._sample_indices(negative, num_negative)
        selected = np.concatenate([pos_idx, neg_idx], axis=0)
        cls_targets = np.concatenate([np.ones((len(pos_idx),), dtype=np.float32), np.zeros((len(neg_idx),), dtype=np.float32)], axis=0)
        bbox_targets = np.zeros((len(selected), 4), dtype=np.float32)
        bbox_mask = np.zeros((len(selected),), dtype=np.float32)
        if len(pos_idx) > 0:
            bbox_targets[: len(pos_idx)] = encode_boxes_yxyx(anchors[pos_idx], gt_boxes[matched_gt[pos_idx]])
            bbox_mask[: len(pos_idx)] = 1.0
        return selected, cls_targets, bbox_targets, bbox_mask

    def _generate_proposals(self, anchors, obj_logits, bbox_deltas):
        scores = tf.nn.sigmoid(obj_logits)
        proposals = decode_boxes_yxyx(anchors, bbox_deltas)
        topk = min(int(scores.shape[0]), RPN_PRE_NMS_TOPK)
        indices = tf.math.top_k(scores, k=topk).indices
        proposals = tf.gather(proposals, indices)
        scores = tf.gather(scores, indices)
        keep = tf.image.non_max_suppression(
            boxes=proposals,
            scores=scores,
            max_output_size=RPN_POST_NMS_TOPK,
            iou_threshold=RPN_NMS_IOU_THRESHOLD,
        )
        return tf.gather(proposals, keep)

    def _sample_rois(self, proposals, gt_boxes, gt_labels):
        proposals_np = proposals.numpy()
        proposals_np = np.concatenate([proposals_np, gt_boxes], axis=0).astype(np.float32)
        ious = iou_yxyx(proposals_np, gt_boxes)
        max_iou = ious.max(axis=1)
        matched_gt = ious.argmax(axis=1)
        positive = np.where(max_iou >= ROI_POSITIVE_IOU_THRESHOLD)[0]
        negative = np.where(max_iou < ROI_NEGATIVE_IOU_THRESHOLD)[0]
        fallback_negative = np.where(max_iou < ROI_POSITIVE_IOU_THRESHOLD)[0]
        num_positive = int(ROI_SAMPLES_PER_IMAGE * ROI_POSITIVE_FRACTION)
        num_negative = ROI_SAMPLES_PER_IMAGE - num_positive
        pos_idx = self._sample_indices(positive, num_positive)
        neg_idx = self._sample_indices(negative, num_negative)
        if len(neg_idx) == 0:
            neg_idx = self._sample_indices(fallback_negative, num_negative)
        chosen = np.concatenate([pos_idx, neg_idx], axis=0)
        if len(chosen) < ROI_SAMPLES_PER_IMAGE:
            extra = self._sample_indices(np.arange(len(proposals_np)), ROI_SAMPLES_PER_IMAGE - len(chosen))
            chosen = np.concatenate([chosen, extra], axis=0)
        chosen = chosen[:ROI_SAMPLES_PER_IMAGE]
        chosen_boxes = proposals_np[chosen]
        labels = np.zeros((len(chosen_boxes),), dtype=np.int32)
        bbox_targets = np.zeros((len(chosen_boxes), 4), dtype=np.float32)
        bbox_weights = np.zeros((len(chosen_boxes),), dtype=np.float32)
        positive_lookup = set(pos_idx.tolist())
        for i, idx in enumerate(chosen):
            if idx in positive_lookup:
                gt_idx = matched_gt[idx]
                labels[i] = gt_labels[gt_idx]
                bbox_targets[i] = encode_boxes_yxyx(chosen_boxes[i: i + 1], gt_boxes[gt_idx: gt_idx + 1])[0]
                bbox_weights[i] = 1.0
        return (
            tf.convert_to_tensor(chosen_boxes, dtype=tf.float32),
            tf.convert_to_tensor(labels, dtype=tf.int32),
            tf.convert_to_tensor(bbox_targets, dtype=tf.float32),
            tf.convert_to_tensor(bbox_weights, dtype=tf.float32),
        )

    def _roi_forward(self, feature_map, rois, training=False):
        pooled = self.roi_pool(feature_map, rois)
        pooled = tf.reshape(pooled, (tf.shape(pooled)[0], -1))
        x = self.fc1(pooled, training=training)
        x = self.fc2(x, training=training)
        return self.cls_head(x, training=training), self.box_head(x, training=training)

    def train_step(self, data):
        inputs = data[0] if isinstance(data, (tuple, list)) else data
        images = inputs["image"]
        gt_boxes = inputs["gt_boxes"]
        gt_labels = inputs["gt_labels"]
        valid_mask = inputs["valid_mask"]

        with tf.GradientTape() as tape:
            outputs = self({"image": images}, training=True)
            feature_map = outputs["feature_map"]
            rpn_obj = outputs["rpn_objectness"]
            rpn_reg = outputs["rpn_bbox"]
            feature_h = int(feature_map.shape[1])
            feature_w = int(feature_map.shape[2])
            anchors_np = self._generate_anchors(feature_h, feature_w)
            anchors_tf = tf.convert_to_tensor(anchors_np, dtype=tf.float32)

            total_rpn_cls = 0.0
            total_rpn_box = 0.0
            total_roi_cls = 0.0
            total_roi_box = 0.0
            total_acc_logits = []
            total_acc_labels = []
            valid_images = 0

            batch_size = images.shape[0]
            for batch_idx in range(batch_size):
                gtb = gt_boxes[batch_idx][valid_mask[batch_idx]].numpy()
                gtl = gt_labels[batch_idx][valid_mask[batch_idx]].numpy()
                if len(gtb) == 0:
                    continue

                valid_images += 1
                obj_logits = tf.reshape(rpn_obj[batch_idx], (-1,))
                bbox_deltas = tf.reshape(rpn_reg[batch_idx], (-1, 4))
                selected, cls_targets, bbox_targets, bbox_mask = self._sample_rpn_targets(anchors_np, gtb)

                sel_logits = tf.gather(obj_logits, selected)
                sel_bbox = tf.gather(bbox_deltas, selected)
                rpn_cls_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.convert_to_tensor(cls_targets, dtype=tf.float32),
                        logits=sel_logits,
                    )
                )
                rpn_box_loss = tf.reduce_mean(
                    tf.keras.losses.Huber(reduction="none")(
                        tf.convert_to_tensor(bbox_targets, dtype=tf.float32),
                        sel_bbox,
                    ) * tf.expand_dims(tf.convert_to_tensor(bbox_mask, dtype=tf.float32), axis=-1)
                )

                proposals = self._generate_proposals(anchors_tf, obj_logits, bbox_deltas)
                roi_boxes, roi_labels, roi_bbox_targets, roi_bbox_weights = self._sample_rois(proposals, gtb, gtl)
                roi_logits, roi_deltas = self._roi_forward(feature_map[batch_idx], roi_boxes, training=True)
                roi_cls_loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(roi_labels, roi_logits, from_logits=True)
                )
                roi_box_loss = tf.reduce_mean(
                    tf.keras.losses.Huber(reduction="none")(roi_bbox_targets, roi_deltas)
                    * tf.expand_dims(roi_bbox_weights, axis=-1)
                )

                total_rpn_cls += rpn_cls_loss
                total_rpn_box += rpn_box_loss
                total_roi_cls += roi_cls_loss
                total_roi_box += roi_box_loss
                total_acc_logits.append(roi_logits)
                total_acc_labels.append(roi_labels)

            denom = max(valid_images, 1)
            total_loss = (total_rpn_cls + total_rpn_box + total_roi_cls + total_roi_box) / denom
            if self.losses:
                total_loss += tf.add_n(self.losses)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(total_loss)
        self.rpn_cls_tracker.update_state(total_rpn_cls / max(valid_images, 1))
        self.rpn_box_tracker.update_state(total_rpn_box / max(valid_images, 1))
        self.roi_cls_tracker.update_state(total_roi_cls / max(valid_images, 1))
        self.roi_box_tracker.update_state(total_roi_box / max(valid_images, 1))
        if total_acc_logits:
            self.acc_tracker.update_state(tf.concat(total_acc_labels, axis=0), tf.concat(total_acc_logits, axis=0))
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        inputs = data[0] if isinstance(data, (tuple, list)) else data
        images = inputs["image"]
        gt_boxes = inputs["gt_boxes"]
        gt_labels = inputs["gt_labels"]
        valid_mask = inputs["valid_mask"]
        outputs = self({"image": images}, training=False)
        feature_map = outputs["feature_map"]
        rpn_obj = outputs["rpn_objectness"]
        rpn_reg = outputs["rpn_bbox"]
        feature_h = int(feature_map.shape[1])
        feature_w = int(feature_map.shape[2])
        anchors_np = self._generate_anchors(feature_h, feature_w)
        anchors_tf = tf.convert_to_tensor(anchors_np, dtype=tf.float32)

        total_rpn_cls = 0.0
        total_rpn_box = 0.0
        total_roi_cls = 0.0
        total_roi_box = 0.0
        total_acc_logits = []
        total_acc_labels = []
        valid_images = 0

        batch_size = images.shape[0]
        for batch_idx in range(batch_size):
            gtb = gt_boxes[batch_idx][valid_mask[batch_idx]].numpy()
            gtl = gt_labels[batch_idx][valid_mask[batch_idx]].numpy()
            if len(gtb) == 0:
                continue
            valid_images += 1
            obj_logits = tf.reshape(rpn_obj[batch_idx], (-1,))
            bbox_deltas = tf.reshape(rpn_reg[batch_idx], (-1, 4))
            selected, cls_targets, bbox_targets, bbox_mask = self._sample_rpn_targets(anchors_np, gtb)
            sel_logits = tf.gather(obj_logits, selected)
            sel_bbox = tf.gather(bbox_deltas, selected)
            total_rpn_cls += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.convert_to_tensor(cls_targets, dtype=tf.float32), logits=sel_logits))
            total_rpn_box += tf.reduce_mean(tf.keras.losses.Huber(reduction="none")(tf.convert_to_tensor(bbox_targets, dtype=tf.float32), sel_bbox) * tf.expand_dims(tf.convert_to_tensor(bbox_mask, dtype=tf.float32), axis=-1))
            proposals = self._generate_proposals(anchors_tf, obj_logits, bbox_deltas)
            roi_boxes, roi_labels, roi_bbox_targets, roi_bbox_weights = self._sample_rois(proposals, gtb, gtl)
            roi_logits, roi_deltas = self._roi_forward(feature_map[batch_idx], roi_boxes, training=False)
            total_roi_cls += tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(roi_labels, roi_logits, from_logits=True))
            total_roi_box += tf.reduce_mean(tf.keras.losses.Huber(reduction="none")(roi_bbox_targets, roi_deltas) * tf.expand_dims(roi_bbox_weights, axis=-1))
            total_acc_logits.append(roi_logits)
            total_acc_labels.append(roi_labels)

        denom = max(valid_images, 1)
        total_loss = (total_rpn_cls + total_rpn_box + total_roi_cls + total_roi_box) / denom
        self.loss_tracker.update_state(total_loss)
        self.rpn_cls_tracker.update_state(total_rpn_cls / denom)
        self.rpn_box_tracker.update_state(total_rpn_box / denom)
        self.roi_cls_tracker.update_state(total_roi_cls / denom)
        self.roi_box_tracker.update_state(total_roi_box / denom)
        if total_acc_logits:
            self.acc_tracker.update_state(tf.concat(total_acc_labels, axis=0), tf.concat(total_acc_logits, axis=0))
        return {metric.name: metric.result() for metric in self.metrics}

    def detect(self, image):
        outputs = self({"image": image}, training=False)
        feature_map = outputs["feature_map"][0]
        obj_logits = tf.reshape(outputs["rpn_objectness"][0], (-1,))
        bbox_deltas = tf.reshape(outputs["rpn_bbox"][0], (-1, 4))
        anchors = tf.convert_to_tensor(self._generate_anchors(int(feature_map.shape[0]), int(feature_map.shape[1])), dtype=tf.float32)
        proposals = self._generate_proposals(anchors, obj_logits, bbox_deltas)
        roi_logits, roi_deltas = self._roi_forward(feature_map, proposals, training=False)
        probs = tf.nn.softmax(roi_logits, axis=-1).numpy()
        class_ids = probs.argmax(axis=1)
        class_scores = probs.max(axis=1)
        refined_boxes = decode_boxes_yxyx(proposals, roi_deltas).numpy()
        detections = []
        for class_id in range(1, self.num_classes):
            indices = np.where((class_ids == class_id) & (class_scores >= SCORE_THRESHOLD))[0]
            if len(indices) == 0:
                continue
            keep = tf.image.non_max_suppression(
                boxes=refined_boxes[indices],
                scores=class_scores[indices],
                max_output_size=MAX_DETECTIONS_PER_CLASS,
                iou_threshold=NMS_IOU_THRESHOLD,
            ).numpy()
            for keep_idx in keep:
                proposal_idx = indices[keep_idx]
                detections.append(
                    {
                        "class_id": int(class_id),
                        "score": float(class_scores[proposal_idx]),
                        "box": refined_boxes[proposal_idx],
                    }
                )
        detections.sort(key=lambda item: item["score"], reverse=True)
        return detections


def create_compiled_model():
    model = FasterRCNN()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    return model

