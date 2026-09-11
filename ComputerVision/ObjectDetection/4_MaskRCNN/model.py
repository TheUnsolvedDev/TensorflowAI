import tensorflow as tf

from config import FC_DIM, IMAGENET_NUM_CLASSES, INPUT_SIZE, LEARNING_RATE, MASK_SIZE, ROI_POOL_SIZE, WEIGHT_DECAY
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


def build_mask_rcnn_backbone(input_shape=(*INPUT_SIZE, 3)):
    inputs = tf.keras.layers.Input(shape=input_shape, name="backbone_image")
    regularizer = tf.keras.regularizers.l2(WEIGHT_DECAY)
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False, kernel_regularizer=regularizer, name="stem_conv")(inputs)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)
    x = tf.keras.layers.MaxPooling2D(3, 2, padding="same", name="stem_pool")(x)
    for stage, filters, stride in ((2, 64, 1), (3, 128, 2), (4, 256, 2), (5, 512, 2)):
        x = residual_block(x, filters, stride, f"stage{stage}_block1")
        x = residual_block(x, filters, 1, f"stage{stage}_block2")
    return tf.keras.Model(inputs, x, name="MaskRCNNBackbone")


class ROIPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=ROI_POOL_SIZE):
        super().__init__()
        self.pool_size = tuple(pool_size)

    def call(self, inputs):
        feature_map, rois = inputs

        def crop_single(args):
            fmap, single_rois = args
            box_indices = tf.zeros((tf.shape(single_rois)[0],), dtype=tf.int32)
            return tf.image.crop_and_resize(
                image=tf.expand_dims(fmap, axis=0),
                boxes=single_rois,
                box_indices=box_indices,
                crop_size=self.pool_size,
            )

        pooled = tf.map_fn(
            crop_single,
            (feature_map, rois),
            fn_output_signature=tf.TensorSpec(shape=(None, self.pool_size[0], self.pool_size[1], feature_map.shape[-1]), dtype=feature_map.dtype),
        )
        pooled.set_shape((None, None, self.pool_size[0], self.pool_size[1], feature_map.shape[-1]))
        return pooled


def build_imagenet_classifier(num_classes=IMAGENET_NUM_CLASSES, name="MaskRCNNClassifier"):
    inputs = tf.keras.layers.Input(shape=(*INPUT_SIZE, 3), name="image")
    backbone = build_mask_rcnn_backbone()
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    logits = tf.keras.layers.Dense(num_classes, name="logits", dtype="float32")(x)
    return tf.keras.Model(inputs=inputs, outputs=logits, name=name)


def build_mask_rcnn_model(input_shape=(*INPUT_SIZE, 3), num_classes=NUM_CLASSES):
    image_input = tf.keras.layers.Input(shape=input_shape, name="image")
    rois_input = tf.keras.layers.Input(shape=(None, 4), name="rois")
    feature_map = build_mask_rcnn_backbone(input_shape=input_shape)(image_input)
    pooled = ROIPooling()(inputs=[feature_map, rois_input])
    # Keep the ROI axis intact with Keras layers; raw tf.reshape/tf.shape cannot
    # consume symbolic Keras tensors under Keras 3.
    roi_features = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(pooled)
    roi_features = tf.keras.layers.Dense(FC_DIM, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(roi_features)
    roi_features = tf.keras.layers.Dropout(0.5)(roi_features)
    roi_features = tf.keras.layers.Dense(FC_DIM, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))(roi_features)
    roi_features = tf.keras.layers.Dropout(0.5)(roi_features)
    class_logits = tf.keras.layers.Dense(num_classes, dtype="float32", name="class_logits")(roi_features)
    bbox_regression = tf.keras.layers.Dense(4, dtype="float32", name="bbox_regression")(roi_features)
    mask_features = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"))(pooled)
    mask_features = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"))(mask_features)
    mask_features = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(128, 2, strides=2, activation="relu"))(mask_features)
    mask_features = tf.keras.layers.TimeDistributed(tf.keras.layers.Resizing(*MASK_SIZE, interpolation="bilinear"))(mask_features)
    mask_logits = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(1, 1, dtype="float32"), name="mask_logits")(mask_features)
    return tf.keras.Model(inputs={"image": image_input, "rois": rois_input}, outputs={"class_logits": class_logits, "bbox_regression": bbox_regression, "mask_logits": mask_logits}, name="MaskRCNN")


def create_compiled_model(task_name="detector", dataset_name="coco"):
    if task_name == "classifier":
        num_classes = IMAGENET_NUM_CLASSES if dataset_name == "imagenet" else NUM_CLASSES - 1
        model = build_imagenet_classifier(num_classes=num_classes, name="MaskRCNNClassifier")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            steps_per_execution=1,
            run_eagerly=False,
        )
        return model
    num_classes = IMAGENET_NUM_CLASSES + 1 if dataset_name == "imagenet" else VOC_DET_NUM_CLASSES if dataset_name == "voc" else NUM_CLASSES
    model = build_mask_rcnn_model(num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            "class_logits": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "bbox_regression": tf.keras.losses.Huber(),
            "mask_logits": tf.keras.losses.BinaryCrossentropy(from_logits=True),
        },
        metrics={
            "class_logits": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            "bbox_regression": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
            "mask_logits": [tf.keras.metrics.BinaryAccuracy(name="mask_accuracy", threshold=0.0)],
        },
        steps_per_execution=1,
        run_eagerly=False,
    )
    return model
