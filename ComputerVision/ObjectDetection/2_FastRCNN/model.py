import tensorflow as tf

from config import FC_DIM, IMAGENET_NUM_CLASSES, INPUT_SIZE, LEARNING_RATE, ROI_POOL_SIZE, WEIGHT_DECAY
from dataset import NUM_CLASSES


class MaskedROIAccuracy(tf.keras.metrics.Metric):
    """Sparse accuracy restricted to foreground or background ROI targets."""

    def __init__(self, foreground, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.foreground = foreground
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int64)
        predicted = tf.argmax(y_pred, axis=-1, output_type=tf.int64)
        mask = tf.not_equal(
            y_true, 0) if self.foreground else tf.equal(y_true, 0)
        weights = tf.cast(mask, self.dtype)
        if sample_weight is not None:
            weights *= tf.cast(sample_weight, self.dtype)
        self.correct.assign_add(tf.reduce_sum(
            tf.cast(tf.equal(y_true, predicted), self.dtype) * weights))
        self.total.assign_add(tf.reduce_sum(weights))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


def residual_block(inputs, filters, stride, name):
    """Functional residual block so nested summaries and diagrams show every op."""
    regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY)
    shortcut = inputs
    x = tf.keras.layers.Conv2D(
        filters, 3, strides=stride, padding="same", use_bias=False,
        kernel_regularizer=regularizer, name=f"{name}_conv1",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = tf.keras.layers.ReLU(name=f"{name}_relu1")(x)
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False,
        kernel_regularizer=regularizer, name=f"{name}_conv2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, 1, strides=stride, padding="same", use_bias=False,
            kernel_regularizer=regularizer, name=f"{name}_proj_conv",
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(
            name=f"{name}_proj_bn")(shortcut)
    x = tf.keras.layers.Add(name=f"{name}_add")([x, shortcut])
    return tf.keras.layers.ReLU(name=f"{name}_out")(x)


def build_fast_rcnn_backbone(input_shape=(*INPUT_SIZE, 3)):
    """Build the full ResNet-style backbone as a graph-visible nested model."""
    inputs = tf.keras.layers.Input(shape=input_shape, name="backbone_image")
    regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY)
    x = tf.keras.layers.Conv2D(
        64, 7, strides=2, padding="same", use_bias=False,
        kernel_regularizer=regularizer, name="stem_conv",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)
    x = tf.keras.layers.MaxPooling2D(3, 2, padding="same", name="stem_pool")(x)
    for stage, filters, stride in ((2, 64, 1), (3, 128, 2), (4, 256, 2), (5, 512, 2)):
        x = residual_block(x, filters, stride, name=f"stage{stage}_block1")
        x = residual_block(x, filters, 1, name=f"stage{stage}_block2")
    return tf.keras.Model(inputs=inputs, outputs=x, name="FastRCNNBackbone")


class ROIPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=ROI_POOL_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = tuple(pool_size)

    def get_config(self):
        return {**super().get_config(), "pool_size": self.pool_size}

    def call(self, inputs):
        feature_map, rois = inputs

        def crop_single(args):
            fmap, single_rois = args
            box_indices = tf.zeros((tf.shape(single_rois)[0],), dtype=tf.int32)
            pooled = tf.image.crop_and_resize(
                image=tf.expand_dims(fmap, axis=0),
                boxes=single_rois,
                box_indices=box_indices,
                crop_size=self.pool_size,
            )
            return pooled

        pooled = tf.map_fn(
            crop_single,
            (feature_map, rois),
            fn_output_signature=tf.TensorSpec(shape=(
                None, self.pool_size[0], self.pool_size[1], feature_map.shape[-1]), dtype=feature_map.dtype),
        )
        pooled.set_shape(
            (None, None, self.pool_size[0], self.pool_size[1], feature_map.shape[-1]))
        return pooled


def build_imagenet_classifier(num_classes=IMAGENET_NUM_CLASSES, name="FastRCNNClassifier"):
    inputs = tf.keras.layers.Input(shape=(*INPUT_SIZE, 3), name="image")
    backbone = build_fast_rcnn_backbone()
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(
        name="classifier_global_average_pool")(x)
    x = tf.keras.layers.Dense(
        1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY), name="classifier_fc1"
    )(x)
    x = tf.keras.layers.Dropout(0.4, name="classifier_dropout")(x)
    logits = tf.keras.layers.Dense(
        num_classes, name="logits", dtype="float32")(x)
    return tf.keras.Model(inputs=inputs, outputs=logits, name=name)


def build_fast_rcnn_model(input_shape=(*INPUT_SIZE, 3), num_classes=NUM_CLASSES):
    image_input = tf.keras.layers.Input(shape=input_shape, name="image")
    rois_input = tf.keras.layers.Input(shape=(None, 4), name="rois")
    backbone = build_fast_rcnn_backbone(input_shape=input_shape)
    pooled = ROIPooling(name="roi_pooling")(
        inputs=[backbone(image_input), rois_input])
    # Keep the ROI axis intact.  Raw TensorFlow shape/reshape operations cannot
    # consume KerasTensors while building a Functional model under Keras 3.
    roi_features = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Flatten(), name="flatten_rois"
    )(pooled)
    roi_features = tf.keras.layers.Dense(
        FC_DIM, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY), name="roi_fc1"
    )(roi_features)
    roi_features = tf.keras.layers.Dropout(
        0.5, name="roi_dropout1")(roi_features)
    roi_features = tf.keras.layers.Dense(
        FC_DIM, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY), name="roi_fc2"
    )(roi_features)
    roi_features = tf.keras.layers.Dropout(
        0.5, name="roi_dropout2")(roi_features)
    class_logits = tf.keras.layers.Dense(
        num_classes, dtype="float32", name="class_logits")(roi_features)
    bbox_regression = tf.keras.layers.Dense(
        4, dtype="float32", name="bbox_regression")(roi_features)
    return tf.keras.Model(inputs={"image": image_input, "rois": rois_input}, outputs={"class_logits": class_logits, "bbox_regression": bbox_regression}, name="FastRCNN")


def create_compiled_model(task_name="detector", dataset_name="coco", num_classes=None):
    if task_name == "classifier":
        classifier_classes = num_classes or (
            IMAGENET_NUM_CLASSES if dataset_name == "imagenet" else NUM_CLASSES - 1)
        model = build_imagenet_classifier(
            num_classes=classifier_classes, name="FastRCNNClassifier")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy")],
            steps_per_execution=1,
            run_eagerly=False,
        )
        return model
    model = build_fast_rcnn_model(num_classes=num_classes or NUM_CLASSES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            "class_logits": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "bbox_regression": tf.keras.losses.Huber(),
        },
        metrics={
            "class_logits": [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                MaskedROIAccuracy(
                    foreground=True, name="positive_roi_accuracy"),
                MaskedROIAccuracy(foreground=False,
                                  name="background_roi_accuracy"),
            ],
            "bbox_regression": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        },
        steps_per_execution=1,
        run_eagerly=False,
    )
    return model
