import tensorflow as tf

from config import INPUT_SIZE, LEARNING_RATE, LOSS_WEIGHTS, WEIGHT_DECAY
from dataset import NUM_CLASSES


def build_rcnn_model(input_shape=(*INPUT_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = tf.keras.layers.Input(shape=input_shape, name="image")

    x = tf.keras.layers.Conv2D(
        64, 11, strides=4, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2D(192, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2D(384, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(
        1024,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(
        1024,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    class_logits = tf.keras.layers.Dense(
        num_classes, name="class_logits", dtype="float32")(x)
    bbox_regression = tf.keras.layers.Dense(
        4, name="bbox_regression", dtype="float32")(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs={
            "class_logits": class_logits,
            "bbox_regression": bbox_regression,
        },
        name="RCNN",
    )


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
    )
    return model


def create_compiled_model():
    model = build_rcnn_model()
    return compile_rcnn_model(model)


if __name__ == "__main__":
    model = create_compiled_model()
    model.summary()
