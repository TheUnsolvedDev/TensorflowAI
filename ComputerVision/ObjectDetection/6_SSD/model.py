import numpy as np
import tensorflow as tf

from config import IMAGENET_NUM_CLASSES, INPUT_SIZE, LEARNING_RATE, MAX_DETECTIONS, NMS_IOU_THRESHOLD, SCORE_THRESHOLD, WEIGHT_DECAY
from dataset import NUM_CLASSES, NUM_PRIORS, PRIORS


def compute_iou(boxes_a, boxes_b):
    top = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(bottom - top, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0.0, None) * np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0.0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0.0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / union, 0.0)


def nms(boxes, scores, iou_threshold=NMS_IOU_THRESHOLD, max_keep=MAX_DETECTIONS):
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_keep:
        idx = order[0]
        keep.append(idx)
        if order.size == 1:
            break
        ious = compute_iou(boxes[idx: idx + 1], boxes[order[1:]])[0]
        order = order[1:][ious < iou_threshold]
    return np.asarray(keep, dtype=np.int32)


def decode_boxes(box_deltas):
    pri_xy = PRIORS[:, :2]
    pri_wh = PRIORS[:, 2:]
    centers = pri_xy + box_deltas[:, :2] * pri_wh
    sizes = pri_wh * np.exp(np.clip(box_deltas[:, 2:], -4.0, 4.0))
    return np.concatenate([centers - 0.5 * sizes, centers + 0.5 * sizes], axis=-1)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, stride=1):
        super().__init__()
        self.filters = filters
        self.stride = stride
        regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY)
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, kernel_regularizer=regularizer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_regularizer=regularizer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.proj = None

    def build(self, input_shape):
        if self.stride != 1 or input_shape[-1] != self.filters:
            regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY)
            self.proj = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(self.filters, 1, strides=self.stride, padding="same", use_bias=False, kernel_regularizer=regularizer),
                    tf.keras.layers.BatchNormalization(),
                ]
            )
        super().build(input_shape)

    def call(self, inputs, training=False):
        shortcut = inputs if self.proj is None else self.proj(inputs, training=training)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        return tf.nn.relu(x + shortcut)


class ResidualStage(tf.keras.layers.Layer):
    def __init__(self, filters, blocks, stride):
        super().__init__()
        self.blocks = [ResidualBlock(filters, stride=stride)]
        self.blocks.extend(ResidualBlock(filters, stride=1) for _ in range(blocks - 1))

    def call(self, inputs, training=False):
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)
        return x


class SSDBackbone(tf.keras.Model):
    def __init__(self):
        super().__init__(name="SSDBackbone")
        regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY)
        self.stem = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False, kernel_regularizer=regularizer),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPooling2D(3, 2, padding="same"),
            ]
        )
        self.stage2 = ResidualStage(64, 2, stride=1)
        self.stage3 = ResidualStage(128, 2, stride=2)
        self.stage4 = ResidualStage(256, 2, stride=2)
        self.stage5 = ResidualStage(512, 2, stride=2)
        self.stage6 = ResidualStage(512, 2, stride=2)

    def call(self, inputs, training=False):
        x = self.stem(inputs, training=training)
        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        c4 = self.stage4(x, training=training)
        c5 = self.stage5(c4, training=training)
        c6 = self.stage6(c5, training=training)
        return c4, c5, c6


def build_imagenet_classifier(num_classes=IMAGENET_NUM_CLASSES, name="SSDClassifier"):
    inputs = tf.keras.layers.Input(shape=(*INPUT_SIZE, 3), name="image")
    backbone = SSDBackbone()
    _, _, c6 = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(c6)
    x = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    logits = tf.keras.layers.Dense(num_classes, name="logits", dtype="float32")(x)
    return tf.keras.Model(inputs=inputs, outputs=logits, name=name)


class SSD(tf.keras.Model):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__(name="SSD")
        self.num_classes = num_classes
        regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY)
        self.backbone = SSDBackbone()
        self.cls_convs = [
            tf.keras.layers.Conv2D(3 * self.num_classes, 3, padding="same", kernel_regularizer=regularizer),
            tf.keras.layers.Conv2D(3 * self.num_classes, 3, padding="same", kernel_regularizer=regularizer),
            tf.keras.layers.Conv2D(3 * self.num_classes, 3, padding="same", kernel_regularizer=regularizer),
        ]
        self.box_convs = [
            tf.keras.layers.Conv2D(3 * 4, 3, padding="same", kernel_regularizer=regularizer),
            tf.keras.layers.Conv2D(3 * 4, 3, padding="same", kernel_regularizer=regularizer),
            tf.keras.layers.Conv2D(3 * 4, 3, padding="same", kernel_regularizer=regularizer),
        ]
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs, training=False):
        feature_maps = self.backbone(inputs, training=training)
        cls_outputs = []
        box_outputs = []
        for fmap, cls_conv, box_conv in zip(feature_maps, self.cls_convs, self.box_convs):
            cls_map = cls_conv(fmap, training=training)
            box_map = box_conv(fmap, training=training)
            cls_outputs.append(tf.reshape(cls_map, (tf.shape(inputs)[0], -1, self.num_classes)))
            box_outputs.append(tf.reshape(box_map, (tf.shape(inputs)[0], -1, 4)))
        cls_logits = tf.concat(cls_outputs, axis=1)
        box_deltas = tf.concat(box_outputs, axis=1)
        return {"cls_logits": cls_logits, "box_deltas": box_deltas}

    def _total_loss(self, targets, preds):
        cls_targets = targets["cls_targets"]
        box_targets = tf.cast(targets["box_targets"], tf.float32)
        cls_logits = tf.cast(preds["cls_logits"], tf.float32)
        box_deltas = tf.cast(preds["box_deltas"], tf.float32)
        positive = tf.cast(cls_targets > 0, tf.float32)
        cls_loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(cls_targets, cls_logits, from_logits=True))
        box_loss = tf.reduce_sum(positive * tf.keras.losses.Huber(reduction="none")(box_targets, box_deltas))
        reg = tf.add_n(self.losses) if self.losses else 0.0
        return (cls_loss + 5.0 * box_loss) / tf.cast(tf.shape(cls_targets)[0], tf.float32) + reg

    def train_step(self, data):
        images, targets = data
        with tf.GradientTape() as tape:
            preds = self(images, training=True)
            loss = self._total_loss(targets, preds)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        images, targets = data
        preds = self(images, training=False)
        loss = self._total_loss(targets, preds)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def detect(self, images):
        preds = self(images, training=False)
        batch_cls = tf.nn.softmax(preds["cls_logits"], axis=-1).numpy()
        batch_boxes = preds["box_deltas"].numpy()
        detections = []
        for cls_probs, box_deltas in zip(batch_cls, batch_boxes):
            boxes = decode_boxes(box_deltas)
            sample_detections = []
            for prior_idx in range(NUM_PRIORS):
                class_id = int(np.argmax(cls_probs[prior_idx, 1:])) + 1
                score = float(cls_probs[prior_idx, class_id])
                if score < SCORE_THRESHOLD:
                    continue
                sample_detections.append({"class_id": class_id, "score": score, "box": np.asarray([boxes[prior_idx, 1], boxes[prior_idx, 0], boxes[prior_idx, 3], boxes[prior_idx, 2]], dtype=np.float32)})
            if sample_detections:
                sample_boxes = np.asarray([det["box"] for det in sample_detections], dtype=np.float32)
                sample_scores = np.asarray([det["score"] for det in sample_detections], dtype=np.float32)
                keep = nms(sample_boxes, sample_scores)
                sample_detections = [sample_detections[idx] for idx in keep]
            detections.append(sample_detections[:MAX_DETECTIONS])
        return detections


def create_compiled_model(task_name="detector", dataset_name="coco"):
    if task_name == "classifier":
        num_classes = IMAGENET_NUM_CLASSES if dataset_name == "imagenet" else NUM_CLASSES - 1
        model = build_imagenet_classifier(num_classes=num_classes, name="SSDClassifier")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            steps_per_execution=1,
            run_eagerly=False,
        )
        return model
    num_classes = IMAGENET_NUM_CLASSES + 1 if dataset_name == "imagenet" else 21 if dataset_name == "voc" else NUM_CLASSES
    model = SSD(num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        steps_per_execution=1,
        run_eagerly=False,
    )
    # Materialize variables with a real forward call before checkpoint loading.
    model(tf.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 3)), training=False)
    return model
