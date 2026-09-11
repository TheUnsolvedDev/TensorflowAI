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


def vgg11_A_LRN_model(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1. / 255)(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                               padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = LocalResponseNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                               padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = LocalResponseNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                               padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                              padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                               padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                               padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                               padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                               padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    models = [
        vgg11_A_LRN_model,]
    
    for model_fn in models:
        model = model_fn()
        model.summary()