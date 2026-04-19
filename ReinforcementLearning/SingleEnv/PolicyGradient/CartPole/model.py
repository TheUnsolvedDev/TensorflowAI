import silence_tensorflow.auto
import tensorflow as tf

from config import *

def policy_network(input_shape=OBS_SHAPE, output_shape=ACTION_SHAPE):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)(inputs)
    x = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)(x)
    logits = tf.keras.layers.Dense(output_shape)(x)
    return tf.keras.Model(inputs=inputs, outputs=logits)