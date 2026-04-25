import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import argparse
from config import *

def activation(x, at):
    if at == "RE":
        # ReLU6
        x = tf.keras.activations.relu(x, max_value=6)
    else:
        # Hard swish
        x = x * tf.keras.activations.relu(x, max_value=6) / 6

    return x


def _squeeze(x):
    x_copy = x
    channel = x.shape[-1]
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(channel, activation="relu")(x)
    x = tf.keras.layers.Dense(channel, activation="hard_sigmoid")(x)
    x = tf.keras.layers.Reshape((1, 1, channel))(x)
    x = tf.keras.layers.Multiply()([x_copy, x])
    return x


def bneck(x, filters, kernel, expansion, strides, squeeze, at):
    x_copy = x

    input_shape = x.shape
    tchannel = int(expansion)
    cchannel = int(filters)

    r = strides == 1 and input_shape[3] == filters

    # Expansion convolution
    exp_x = tf.keras.layers.Conv2D(
        tchannel, (1, 1), padding="same", strides=(1, 1))(x)
    exp_x = tf.keras.layers.BatchNormalization(axis=-1)(exp_x)
    exp_x = activation(exp_x, at)

    # Depthwise convolution
    dep_x = tf.keras.layers.DepthwiseConv2D(
        kernel, strides=(strides, strides), depth_multiplier=1, padding="same"
    )(exp_x)
    dep_x = tf.keras.layers.BatchNormalization(axis=-1)(dep_x)
    dep_x = activation(dep_x, at)

    # Squeeze
    if squeeze:
        dep_x = _squeeze(dep_x)

    # Projection convolution
    pro_x = tf.keras.layers.Conv2D(
        cchannel, (1, 1), strides=(1, 1), padding="same")(dep_x)
    pro_x = tf.keras.layers.BatchNormalization(axis=-1)(pro_x)

    x = pro_x

    if r:
        x = tf.keras.layers.Add()([pro_x, x_copy])

    return x, exp_x, dep_x, pro_x


def MobilenetV3_model(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)
    x = tf.keras.layers.BatchNormalization(name='conv1_bn')(inputs)
    x = tf.keras.layers.ReLU(6, name='conv1_relu')(x)

    x = activation(x, "HS")

    x, _, _, _ = bneck(
        x, 16, (3, 3), expansion=16, strides=1, squeeze=False, at="RE"
    )

    # 1/4
    x, _, _, _ = bneck(
        x, 24, (3, 3), expansion=64, strides=2, squeeze=False, at="RE"
    )
    x, _, _, _ = bneck(
        x, 24, (3, 3), expansion=72, strides=1, squeeze=False, at="RE"
    )

    # 1/8
    x, _, _, _ = bneck(
        x, 40, (5, 5), expansion=72, strides=2, squeeze=True, at="RE"
    )
    x, _, _, _ = bneck(
        x, 40, (5, 5), expansion=120, strides=1, squeeze=True, at="RE"
    )
    x_8, _, _, _ = bneck(
        x, 40, (5, 5), expansion=120, strides=1, squeeze=True, at="RE"
    )

    # 1/16
    x, _, _, _ = bneck(
        x_8, 80, (3, 3), expansion=240, strides=2, squeeze=False, at="HS"
    )
    x, _, _, _ = bneck(
        x, 80, (3, 3), expansion=200, strides=1, squeeze=False, at="HS"
    )
    x, _, _, _ = bneck(
        x, 80, (3, 3), expansion=184, strides=1, squeeze=False, at="HS"
    )
    x, _, _, _ = bneck(
        x, 80, (3, 3), expansion=184, strides=1, squeeze=False, at="HS"
    )
    x, _, _, _ = bneck(
        x, 112, (3, 3), expansion=480, strides=1, squeeze=True, at="HS"
    )
    x_16, _, _, _ = bneck(
        x, 112, (3, 3), expansion=672, strides=1, squeeze=True, at="HS"
    )
    # 1/32
    # 13th bottleneck block (C4) https://arxiv.org/pdf/1905.02244v4.pdf p.7
    x, _, _, _ = bneck(
        x_8, 160, (5, 5), expansion=672, strides=2, squeeze=True, at="HS"
    )
    x, _, _, _ = bneck(
        x, 160, (5, 5), expansion=960, strides=1, squeeze=True, at="HS"
    )
    x, _, _, _ = bneck(
        x, 160, (5, 5), expansion=960, strides=1, squeeze=True, at="HS"
    )

    # Layer immediatly before pooling (C5) https://arxiv.org/pdf/1905.02244v4.pdf p.7
    x = tf.keras.layers.Conv2D(960, (1, 1), strides=(1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = activation(x, at="HS")

    # Pooling layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Reshape((1, 1, 960))(x)

    x = tf.keras.layers.Conv2D(1280, (1, 1), padding="same")(x)
    x = activation(x, "HS")
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pool')(x)
    outputs = tf.keras.layers.Dense(units = num_classes, activation = 'softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    model = MobilenetV3_model()
    model.summary()
