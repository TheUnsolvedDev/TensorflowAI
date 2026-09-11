import tensorflow as tf
import numpy as np

from config import *


kernel_init = tf.keras.initializers.glorot_uniform()
bias_init = tf.keras.initializers.Constant(value=0.2)

class LocalResponseNormalization(tf.keras.layers.Layer):
    def __init__(self, alpha=1e-4, beta=.75, depth_radius=2, **kwargs):
        super().__init__(**kwargs)
        self.alpha, self.beta, self.depth_radius = alpha, beta, depth_radius
    def call(self, x):
        return tf.cast(tf.nn.local_response_normalization(tf.cast(x, tf.float32),
                       depth_radius=self.depth_radius, bias=1., alpha=self.alpha, beta=self.beta), x.dtype)
    def get_config(self):
        return {**super().get_config(), 'alpha': self.alpha, 'beta': self.beta, 'depth_radius': self.depth_radius}

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):

    conv_1x1 = tf.keras.layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu',
                                      kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_3x3 = tf.keras.layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu',
                                      kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = tf.keras.layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu',
                                      kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = tf.keras.layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu',
                                      kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = tf.keras.layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu',
                                      kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = tf.keras.layers.MaxPool2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = tf.keras.layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu',
                                       kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = tf.keras.layers.Concatenate(axis=3, name=name)([conv_1x1, conv_3x3, conv_5x5,
                                                             pool_proj])

    return output


def inception1_model(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10, aux_logits=False):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1. / 255)(input_layer)
    x = tf.keras.layers.Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu',
                               name='conv_1_7x7_2', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    x = tf.keras.layers.MaxPool2D(
        (3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3_2')(x)
    x = LocalResponseNormalization()(x)
    
    x = tf.keras.layers.Conv2D(192, (3, 3), padding='same', strides=(
        1, 1), activation='relu', name='conv_2a_3x3_1')(x)
    x = tf.keras.layers.MaxPool2D(
        (3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3_2')(x)
    x = LocalResponseNormalization()(x)
    
    x = inception_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_3a')
    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=192,
                         filters_5x5_reduce=32,
                         filters_5x5=96,
                         filters_pool_proj=64,
                         name='inception_3b')
    x = tf.keras.layers.MaxPool2D(
        (3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3_2')(x)
    x = inception_module(x,
                         filters_1x1=192,
                         filters_3x3_reduce=96,
                         filters_3x3=208,
                         filters_5x5_reduce=16,
                         filters_5x5=48,
                         filters_pool_proj=64,
                         name='inception_4a')
    x1 = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
    x1 = tf.keras.layers.Conv2D(
        128, (1, 1), padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Dense(1024, activation='relu')(x1)
    x1 = tf.keras.layers.Dropout(0.7)(x1)
    x1 = tf.keras.layers.Dense(
        num_classes, activation='softmax', dtype='float32', name='output_1')(x1)
    x = inception_module(x,
                         filters_1x1=160,
                         filters_3x3_reduce=112,
                         filters_3x3=224,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4b')
    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=256,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4c')
    x = inception_module(x,
                         filters_1x1=112,
                         filters_3x3_reduce=144,
                         filters_3x3=288,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4d')
    x2 = tf.keras.layers.AveragePooling2D((5, 5), strides=3)(x)
    x2 = tf.keras.layers.Conv2D(
        128, (1, 1), padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Dense(1024, activation='relu')(x2)
    x2 = tf.keras.layers.Dropout(0.7)(x2)
    x2 = tf.keras.layers.Dense(
        num_classes, activation='softmax', dtype='float32', name='output_2')(x2)
    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_4e')
    x = tf.keras.layers.MaxPool2D(
        (3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3_2')(x)
    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5a')
    x = inception_module(x,
                         filters_1x1=384,
                         filters_3x3_reduce=192,
                         filters_3x3=384,
                         filters_5x5_reduce=48,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5b')
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool_5_3x3_1')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32', name='output')(x)
    model = tf.keras.Model(input_layer, [x, x1, x2] if aux_logits else x, name='inception_v1')
    return model



if __name__ == "__main__":
    model = inception1_model()
    model.summary(expand_nested=True)
    
    tf.keras.utils.plot_model(
        model, to_file=inception1_model.__name__+'.png', show_shapes=True)