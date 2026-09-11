import numpy as np
import tensorflow as tf

from config import ANCHORS, GRID_SIZE, IMAGENET_NUM_CLASSES, INPUT_SIZE, LEARNING_RATE, MAX_DETECTIONS, NMS_IOU_THRESHOLD, SCORE_THRESHOLD, WEIGHT_DECAY
from dataset import NUM_ANCHORS, NUM_CLASSES


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


class ResidualBackbone(tf.keras.Model):
    def __init__(self):
        super().__init__(name="YOLOv2Backbone")
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

    def call(self, inputs, training=False):
        x = self.stem(inputs, training=training)
        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)
        x = self.stage5(x, training=training)
        return x


def build_imagenet_classifier(num_classes=IMAGENET_NUM_CLASSES, name="YOLOv2Classifier"):
    inputs = tf.keras.layers.Input(shape=(*INPUT_SIZE, 3), name="image")
    backbone = ResidualBackbone()
    features = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(features)
    x = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    logits = tf.keras.layers.Dense(num_classes, name="logits", dtype="float32")(x)
    return tf.keras.Model(inputs=inputs, outputs=logits, name=name)


class YOLOv2(tf.keras.Model):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__(name="YOLOv2")
        self.num_classes = num_classes
        regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY)
        self.backbone = ResidualBackbone()
        self.detector = tf.keras.layers.Conv2D(NUM_ANCHORS * (self.num_classes + 5), 1, padding="same", kernel_regularizer=regularizer)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs, training=False):
        outputs = self.detector(self.backbone(inputs, training=training), training=training)
        outputs = tf.reshape(outputs, (-1, GRID_SIZE, GRID_SIZE, NUM_ANCHORS, self.num_classes + 5))
        class_logits = outputs[..., :self.num_classes]
        objectness = tf.sigmoid(outputs[..., self.num_classes: self.num_classes + 1])
        xy = tf.sigmoid(outputs[..., self.num_classes + 1: self.num_classes + 3])
        wh = tf.sigmoid(outputs[..., self.num_classes + 3:])
        return tf.concat([class_logits, objectness, xy, wh], axis=-1)

    def compute_total_loss(self, target, pred):
        target = tf.cast(target, tf.float32)
        pred = tf.cast(pred, tf.float32)
        target_cls = target[..., :self.num_classes]
        target_obj = target[..., self.num_classes: self.num_classes + 1]
        target_box = target[..., self.num_classes + 1:]
        pred_cls = pred[..., :self.num_classes]
        pred_obj = pred[..., self.num_classes: self.num_classes + 1]
        pred_box = pred[..., self.num_classes + 1:]
        cls_loss = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(target_cls, (-1, self.num_classes)), logits=tf.reshape(pred_cls, (-1, self.num_classes)))
            * tf.reshape(target_obj, (-1,))
        )
        obj_loss = tf.reduce_sum(tf.square(target_obj - pred_obj))
        box_loss = tf.reduce_sum(target_obj * tf.reduce_sum(tf.square(target_box - pred_box), axis=-1, keepdims=True))
        noobj_loss = tf.reduce_sum((1.0 - target_obj) * tf.square(pred_obj))
        reg_loss = tf.add_n(self.losses) if self.losses else 0.0
        return (cls_loss + 5.0 * box_loss + obj_loss + 0.5 * noobj_loss) / tf.cast(tf.shape(target)[0], tf.float32) + reg_loss

    def train_step(self, data):
        images, target_bundle = data
        targets = target_bundle["yolo_output"]
        with tf.GradientTape() as tape:
            preds = self(images, training=True)
            loss = self.compute_total_loss(targets, preds)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        images, target_bundle = data
        targets = target_bundle["yolo_output"]
        preds = self(images, training=False)
        loss = self.compute_total_loss(targets, preds)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def detect(self, images):
        preds = self(images, training=False).numpy()
        anchors = np.asarray(ANCHORS, dtype=np.float32) / GRID_SIZE
        detections = []
        for sample_pred in preds:
            sample_detections = []
            class_probs = tf.nn.softmax(sample_pred[..., :self.num_classes], axis=-1).numpy()
            objectness = sample_pred[..., self.num_classes]
            xy = sample_pred[..., self.num_classes + 1: self.num_classes + 3]
            wh = sample_pred[..., self.num_classes + 3:]
            for gy in range(GRID_SIZE):
                for gx in range(GRID_SIZE):
                    for anchor_idx in range(NUM_ANCHORS):
                        score = float(objectness[gy, gx, anchor_idx])
                        if score < SCORE_THRESHOLD:
                            continue
                        cls = int(np.argmax(class_probs[gy, gx, anchor_idx])) + 1
                        cls_score = float(class_probs[gy, gx, anchor_idx, cls - 1] * score)
                        if cls_score < SCORE_THRESHOLD:
                            continue
                        tx, ty = xy[gy, gx, anchor_idx]
                        tw, th = wh[gy, gx, anchor_idx] * anchors[anchor_idx]
                        cx = (gx + tx) / GRID_SIZE
                        cy = (gy + ty) / GRID_SIZE
                        x1 = np.clip(cx - 0.5 * tw, 0.0, 1.0)
                        y1 = np.clip(cy - 0.5 * th, 0.0, 1.0)
                        x2 = np.clip(cx + 0.5 * tw, 0.0, 1.0)
                        y2 = np.clip(cy + 0.5 * th, 0.0, 1.0)
                        sample_detections.append({"class_id": cls, "score": cls_score, "box": np.asarray([y1, x1, y2, x2], dtype=np.float32)})
            if sample_detections:
                boxes = np.asarray([det["box"] for det in sample_detections], dtype=np.float32)
                scores = np.asarray([det["score"] for det in sample_detections], dtype=np.float32)
                keep = nms(boxes, scores)
                sample_detections = [sample_detections[idx] for idx in keep]
            detections.append(sample_detections[:MAX_DETECTIONS])
        return detections


def create_compiled_model(task_name="detector", dataset_name="coco"):
    if task_name == "classifier":
        num_classes = IMAGENET_NUM_CLASSES if dataset_name == "imagenet" else NUM_CLASSES - 1
        model = build_imagenet_classifier(num_classes=num_classes, name="YOLOv2Classifier")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            steps_per_execution=1,
            run_eagerly=False,
        )
        return model
    num_classes = IMAGENET_NUM_CLASSES if dataset_name == "imagenet" else 20 if dataset_name == "voc" else NUM_CLASSES
    model = YOLOv2(num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        steps_per_execution=1,
        run_eagerly=False,
    )
    # Materialize variables with a real forward call before checkpoint loading.
    model(tf.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 3)), training=False)
    return model
