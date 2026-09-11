import tensorflow as tf
from config import *
def policy_network():
    x=tf.keras.Input(shape=OBS_SHAPE); h=tf.keras.layers.Dense(64,activation=tf.nn.leaky_relu)(x); h=tf.keras.layers.Dense(64,activation=tf.nn.leaky_relu)(h); return tf.keras.Model(x,tf.keras.layers.Dense(ACTION_SHAPE)(h))

