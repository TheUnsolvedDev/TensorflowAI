import tensorflow as tf
import numpy as np

from config import *

class LocalResponseNormalization(tf.keras.layers.Layer):
    def __init__(self, alpha=1e-4, beta=.75, depth_radius=2, **kwargs):
        super().__init__(**kwargs)
        self.alpha, self.beta, self.depth_radius = alpha, beta, depth_radius
    def call(self, x):
        return tf.cast(tf.nn.local_response_normalization(tf.cast(x, tf.float32),
                       depth_radius=self.depth_radius, bias=1., alpha=self.alpha, beta=self.beta), x.dtype)
    def get_config(self):
        return {**super().get_config(), 'alpha': self.alpha, 'beta': self.beta, 'depth_radius': self.depth_radius}


def alexnet_model(input_shape=[227, 227, 3], num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1. / 255)(inputs)
    x = tf.keras.layers.ZeroPadding2D(2)(x)
    x = tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4))(x)
    x = LocalResponseNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same")(x)
    x = LocalResponseNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same")(x)

    x = tf.keras.layers.MaxPooling2D(3, strides=2)(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


if __name__ == '__main__':
    model = alexnet_model()
    model.summary()

    tf.keras.utils.plot_model(
        model, to_file=alexnet_model.__name__+'.png', show_shapes=True)