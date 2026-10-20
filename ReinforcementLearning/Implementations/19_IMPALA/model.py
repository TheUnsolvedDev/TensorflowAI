import tensorflow as tf
import config


def actor_critic(shape, actions):
    x = tf.keras.Input(shape=shape)
    y = x
    for units in config.HIDDEN_UNITS:
        y = tf.keras.layers.Dense(units, activation="relu")(y)
    return tf.keras.Model(x, [tf.keras.layers.Dense(actions)(y), tf.keras.layers.Dense(1)(y)])
