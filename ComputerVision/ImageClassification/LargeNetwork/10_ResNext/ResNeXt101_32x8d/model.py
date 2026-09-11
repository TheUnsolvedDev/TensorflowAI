"""ResNeXt101_32x8d. TensorFlow implementation; no pretrained weights."""
import tensorflow as tf
from config import INPUT_SIZE


def conv_bn(x, filters, kernel, stride=1, groups=1, activation=True, name=None):
    x = tf.keras.layers.Conv2D(filters, kernel, strides=stride, padding='same',
                              groups=groups, use_bias=False, kernel_initializer='he_normal', name=name)(x)
    x = tf.keras.layers.BatchNormalization(momentum=.9, epsilon=1e-5)(x)
    return tf.keras.layers.ReLU()(x) if activation else x


def bottleneck(x, planes, stride, groups, width, se, senet154, name):
    shortcut = x
    if senet154:
        y = conv_bn(x, planes * 2, 1, name=name + '_reduce')
        y = conv_bn(y, planes * 4, 3, stride, groups=64, name=name + '_group')
    else:
        inner = int(planes * width / 64) * groups
        y = conv_bn(x, inner, 1, name=name + '_reduce')
        y = conv_bn(y, inner, 3, stride, groups=groups, name=name + '_group')
    y = conv_bn(y, planes * 4, 1, activation=False, name=name + '_expand')
    if se:
        scale = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(y)
        scale = tf.keras.layers.Conv2D(max(1, planes * 4 // 16), 1, activation='relu')(scale)
        scale = tf.keras.layers.Conv2D(planes * 4, 1, activation='sigmoid')(scale)
        y = tf.keras.layers.Multiply()([y, scale])
    if stride != 1 or x.shape[-1] != planes * 4:
        shortcut = conv_bn(shortcut, planes * 4, 3 if senet154 and stride != 1 else 1,
                           stride, activation=False, name=name + '_shortcut')
    return tf.keras.layers.ReLU()(tf.keras.layers.Add()([shortcut, y]))


def resnext101_32x8d_model(input_shape=tuple(INPUT_SIZE), num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1. / 255)(inputs)
    if False:
        x = conv_bn(x, 64, 3, 2)
        x = conv_bn(x, 64, 3)
        x = conv_bn(x, 128, 3)
    else:
        x = conv_bn(x, 64, 7, 2)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
    for stage, (planes, blocks) in enumerate(zip((64, 128, 256, 512), [3, 4, 23, 3])):
        for block in range(blocks):
            x = bottleneck(x, planes, 2 if stage and block == 0 else 1,
                           32, 8, False, False, f'stage{stage+1}_block{block+1}')
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if False:
        x = tf.keras.layers.Dropout(.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    return tf.keras.Model(inputs, outputs, name='resnext101_32x8d_model')
