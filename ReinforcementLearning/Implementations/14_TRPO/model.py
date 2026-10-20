import tensorflow as tf

import config


def policy_model(observation_shape, action_count):
    inputs = tf.keras.Input(shape=observation_shape)
    features = inputs
    for units in config.HIDDEN_UNITS:
        features = tf.keras.layers.Dense(units, activation="tanh")(features)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(action_count)(features))


def value_model(observation_shape):
    inputs = tf.keras.Input(shape=observation_shape)
    features = inputs
    for units in config.HIDDEN_UNITS:
        features = tf.keras.layers.Dense(units, activation="tanh")(features)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(1)(features))
