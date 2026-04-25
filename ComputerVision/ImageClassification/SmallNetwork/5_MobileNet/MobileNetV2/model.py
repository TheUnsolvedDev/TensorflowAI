import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import argparse
from config import *

def expansion_block(x, t, filters, block_id):
    prefix = 'block_{}_'.format(block_id)
    total_filters = t*filters
    x = tf.keras.layers.Conv2D(total_filters, 1, padding='same',
                               use_bias=False, name=prefix + 'expand')(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + 'expand_bn')(x)
    x = tf.keras.layers.ReLU(6, name=prefix + 'expand_relu')(x)
    return x


def depthwise_block(x, stride, block_id):
    prefix = 'block_{}_'.format(block_id)
    x = tf.keras.layers.DepthwiseConv2D(3, strides=(stride, stride), padding='same',
                                        use_bias=False, name=prefix + 'depthwise_conv')(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + 'dw_bn')(x)
    x = tf.keras.layers.ReLU(6, name=prefix + 'dw_relu')(x)
    return x


def projection_block(x, out_channels, block_id):
    prefix = 'block_{}_'.format(block_id)
    x = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1,   padding='same',
                               use_bias=False, name=prefix + 'compress')(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + 'compress_bn')(x)
    return x


def Bottleneck(x, t, filters, out_channels, stride, block_id):
    y = expansion_block(x, t, filters, block_id)
    y = depthwise_block(y, stride, block_id)
    y = projection_block(y, out_channels, block_id)
    if y.shape[-1] == x.shape[-1]:
        y = tf.keras.layers.add([x, y])
    return y


def MobilenetV2_model(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)
    x = tf.keras.layers.ReLU(6, name='conv1_relu')(x)
    # 17 Bottlenecks
    x = depthwise_block(x, stride=1, block_id=1)
    x = projection_block(x, out_channels=16, block_id=1)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=24, stride=2, block_id=2)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=24, stride=1, block_id=3)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=32, stride=2, block_id=4)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=32, stride=1, block_id=5)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=32, stride=1, block_id=6)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=64, stride=2, block_id=7)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=64, stride=1, block_id=8)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=64, stride=1, block_id=9)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=64, stride=1, block_id=10)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=96, stride=1, block_id=11)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=96, stride=1, block_id=12)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=96, stride=1, block_id=13)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=160, stride=2, block_id=14)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=160, stride=1, block_id=15)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=160, stride=1, block_id=16)
    x = Bottleneck(
        x, t=6, filters=x.shape[-1], out_channels=320, stride=1, block_id=17)
    x = tf.keras.layers.Conv2D(
        filters=1280, kernel_size=1, padding='same', use_bias=False, name='last_conv')(x)
    x = tf.keras.layers.BatchNormalization(name='last_bn')(x)
    x = tf.keras.layers.ReLU(6, name='last_relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pool')(x)
    outputs = tf.keras.layers.Dense(units = num_classes, activation = 'softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == '__main__':
    model = MobilenetV2_model()
    model.summary()
