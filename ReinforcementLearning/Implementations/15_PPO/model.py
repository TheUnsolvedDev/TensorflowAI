import tensorflow as tf

import config


def actor_critic_model(observation_shape, action_count):
    inputs = tf.keras.Input(shape=observation_shape)
    features = inputs
    for units in config.HIDDEN_UNITS:
        features = tf.keras.layers.Dense(units, activation="tanh")(features)
    logits = tf.keras.layers.Dense(
        action_count, name="policy_logits")(features)
    value = tf.keras.layers.Dense(1, name="value")(features)
    return tf.keras.Model(inputs, [logits, value])


def gaussian_actor_critic_model(observation_shape, action_size):
    inputs = tf.keras.Input(shape=observation_shape)
    features = inputs
    for units in config.HIDDEN_UNITS:
        features = tf.keras.layers.Dense(units, activation="tanh")(features)
    mean = tf.keras.layers.Dense(action_size, name="policy_mean")(features)
    log_std = tf.keras.layers.Dense(action_size,
                                    bias_initializer=tf.keras.initializers.Constant(-0.5),
                                    name="policy_log_std")(features)
    value = tf.keras.layers.Dense(1, name="value")(features)
    return tf.keras.Model(inputs, [mean, log_std, value])
