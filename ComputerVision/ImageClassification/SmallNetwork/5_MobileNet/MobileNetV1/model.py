import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import argparse
from config import *


def mobilnet_block(x, filters, strides):
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def Mobilenet_model(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = mobilnet_block(x, filters = 64, strides = 1)
    x = mobilnet_block(x, filters = 128, strides = 2)
    x = mobilnet_block(x, filters = 128, strides = 1)
    x = mobilnet_block(x, filters = 256, strides = 2)
    x = mobilnet_block(x, filters = 256, strides = 1)
    x = tf.keras.layers.AvgPool2D(pool_size = 7, strides = 1, data_format='channels_first')(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(units = num_classes, activation = 'softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model



if __name__ == '__main__':
    model = Mobilenet()
    model.summary()
