import tensorflow as tf

import config


def build_model(observation_shape, action_count):
    inputs = tf.keras.Input(shape=observation_shape)
    features = inputs
    for units in config.HIDDEN_UNITS:
        features = tf.keras.layers.Dense(units, activation="relu")(features)
    probabilities = tf.keras.layers.Dense(
        action_count, activation="softmax", name="policy"
    )(features)
    value = tf.keras.layers.Dense(1, name="value")(features)
    return tf.keras.Model(inputs, [probabilities, value])


def build_continuous_model(observation_shape, action_size):
    inputs = tf.keras.Input(shape=observation_shape)
    features = inputs
    for units in config.HIDDEN_UNITS:
        features = tf.keras.layers.Dense(units, activation="relu")(features)
    mean = tf.keras.layers.Dense(action_size, name="mean")(features)
    log_std = tf.keras.layers.Dense(
        action_size,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(-0.5),
        name="log_std",
    )(features)
    value = tf.keras.layers.Dense(1, name="value")(features)
    return tf.keras.Model(inputs, [mean, log_std, value])
