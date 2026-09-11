import tensorflow as tf

from config import IMAGENET_NUM_CLASSES, INPUT_SIZE, LEARNING_RATE, LOSS_WEIGHTS, WEIGHT_DECAY
from dataset import NUM_CLASSES, VOC_DET_NUM_CLASSES


def residual_block(inputs, filters, stride, name):
    regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY)
    shortcut = inputs
    x = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, kernel_regularizer=regularizer, name=f"{name}_conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = tf.keras.layers.ReLU(name=f"{name}_relu1")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_regularizer=regularizer, name=f"{name}_conv2")(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False, kernel_regularizer=regularizer, name=f"{name}_proj_conv")(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)
    return tf.keras.layers.ReLU(name=f"{name}_out")(tf.keras.layers.Add(name=f"{name}_add")([x, shortcut]))


def build_rcnn_backbone(input_shape=(*INPUT_SIZE, 3)):
    inputs = tf.keras.layers.Input(shape=input_shape, name="backbone_image")
    regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY)
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False, kernel_regularizer=regularizer, name="stem_conv")(inputs)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)
    x = tf.keras.layers.MaxPooling2D(3, 2, padding="same", name="stem_pool")(x)
    for stage, filters, stride in ((2, 64, 1), (3, 128, 2), (4, 256, 2), (5, 512, 2)):
        x = residual_block(x, filters, stride, f"stage{stage}_block1")
        x = residual_block(x, filters, 1, f"stage{stage}_block2")
    return tf.keras.Model(inputs, x, name="RCNNBackbone")


def build_imagenet_classifier(input_shape=(*INPUT_SIZE, 3), num_classes=IMAGENET_NUM_CLASSES, name="RCNNClassifier"):
    inputs = tf.keras.layers.Input(shape=input_shape, name="image")
    backbone = build_rcnn_backbone(input_shape=input_shape)
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    logits = tf.keras.layers.Dense(num_classes, name="logits", dtype="float32")(x)
    return tf.keras.Model(inputs=inputs, outputs=logits, name=name)


def build_rcnn_model(input_shape=(*INPUT_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = tf.keras.layers.Input(shape=input_shape, name="image")
    backbone = build_rcnn_backbone(input_shape=input_shape)
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    class_logits = tf.keras.layers.Dense(num_classes, name="class_logits", dtype="float32")(x)
    bbox_regression = tf.keras.layers.Dense(4, name="bbox_regression", dtype="float32")(x)
    return tf.keras.Model(inputs=inputs, outputs={"class_logits": class_logits, "bbox_regression": bbox_regression}, name="RCNN")


def compile_rcnn_model(model, learning_rate=LEARNING_RATE):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss={
            "class_logits": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "bbox_regression": tf.keras.losses.Huber(),
        },
        loss_weights=LOSS_WEIGHTS,
        metrics={
            "class_logits": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            "bbox_regression": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        },
        steps_per_execution=1,
        run_eagerly=False,
    )
    return model


def create_compiled_model(task_name="detector", dataset_name="coco"):
    if task_name == "classifier":
        num_classes = IMAGENET_NUM_CLASSES if dataset_name == "imagenet" else NUM_CLASSES - 1
        model = build_imagenet_classifier(num_classes=num_classes, name="RCNNClassifier")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            steps_per_execution=1,
            run_eagerly=False,
        )
        return model
    num_classes = IMAGENET_NUM_CLASSES + 1 if dataset_name == "imagenet" else VOC_DET_NUM_CLASSES if dataset_name == "voc" else NUM_CLASSES
    return compile_rcnn_model(build_rcnn_model(num_classes=num_classes))
