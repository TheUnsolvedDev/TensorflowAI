import numpy as np
import tensorflow as tf

import config


def mlp(inputs, hidden=config.HIDDEN_UNITS):
    x = inputs
    for units in hidden:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
    return x


def policy_model(observation_shape, action_count):
    inputs = tf.keras.Input(shape=observation_shape)
    outputs = tf.keras.layers.Dense(action_count, activation="softmax")(mlp(inputs))
    return tf.keras.Model(inputs, outputs)


def continuous_policy_model(observation_shape, action_size):
    inputs = tf.keras.Input(shape=observation_shape)
    features = mlp(inputs)
    mean = tf.keras.layers.Dense(action_size, name="mean")(features)
    log_std = tf.keras.layers.Dense(
        action_size,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(-0.2),
        name="log_std",
    )(features)
    return tf.keras.Model(inputs, [mean, log_std])


def discounted_returns(rewards, gamma):
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns


