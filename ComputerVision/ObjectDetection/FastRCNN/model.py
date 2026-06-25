import tensorflow as tf

from config import FC_DIM, INPUT_SIZE, LEARNING_RATE, ROI_POOL_SIZE, WEIGHT_DECAY
from dataset import NUM_CLASSES


class ROIPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=ROI_POOL_SIZE):
        super().__init__()
        self.pool_size = tuple(pool_size)

    def call(self, inputs):
        feature_map, rois = inputs
        batch_size = tf.shape(feature_map)[0]
        num_rois = tf.shape(rois)[1]

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
            fn_output_signature=tf.TensorSpec(
                shape=(None, self.pool_size[0], self.pool_size[1], feature_map.shape[-1]),
                dtype=feature_map.dtype,
            ),
        )
        pooled.set_shape((None, None, self.pool_size[0], self.pool_size[1], feature_map.shape[-1]))
        return pooled


def build_fast_rcnn_model(input_shape=(*INPUT_SIZE, 3), num_classes=NUM_CLASSES):
    image_input = tf.keras.layers.Input(shape=input_shape, name="image")
    rois_input = tf.keras.layers.Input(shape=(None, 4), name="rois")

    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", activation="relu")(image_input)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)

    pooled = ROIPooling()(inputs=[x, rois_input])
    num_rois = tf.shape(pooled)[1]
    channels = pooled.shape[-1]

    roi_features = tf.reshape(
        pooled,
        (-1, ROI_POOL_SIZE[0] * ROI_POOL_SIZE[1] * channels),
    )
    roi_features = tf.keras.layers.Dense(
        FC_DIM,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
    )(roi_features)
    roi_features = tf.keras.layers.Dropout(0.5)(roi_features)
    roi_features = tf.keras.layers.Dense(
        FC_DIM,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
    )(roi_features)
    roi_features = tf.keras.layers.Dropout(0.5)(roi_features)

    class_logits = tf.keras.layers.Dense(num_classes, dtype="float32")(roi_features)
    bbox_regression = tf.keras.layers.Dense(4, dtype="float32")(roi_features)

    class_logits = tf.reshape(class_logits, (-1, num_rois, num_classes), name="class_logits")
    bbox_regression = tf.reshape(bbox_regression, (-1, num_rois, 4), name="bbox_regression")

    return tf.keras.Model(
        inputs={"image": image_input, "rois": rois_input},
        outputs={"class_logits": class_logits, "bbox_regression": bbox_regression},
        name="FastRCNN",
    )


def compile_fast_rcnn_model(model, learning_rate=LEARNING_RATE):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "class_logits": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "bbox_regression": tf.keras.losses.Huber(),
        },
        metrics={
            "class_logits": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            "bbox_regression": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        },
    )
    return model


def create_compiled_model():
    return compile_fast_rcnn_model(build_fast_rcnn_model())

