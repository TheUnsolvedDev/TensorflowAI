import numpy as np
import tensorflow as tf

import config


def hidden_layers(inputs):
    features = inputs
    for units in config.HIDDEN_UNITS:
        features = tf.keras.layers.Dense(units, activation="relu")(features)
    return features


def build_actor(observation_shape, action_count):
    inputs = tf.keras.Input(shape=observation_shape)
    probabilities = tf.keras.layers.Dense(
        action_count, activation="softmax"
    )(hidden_layers(inputs))
    return tf.keras.Model(inputs, probabilities, name="actor")


def build_continuous_actor(observation_shape, action_size):
    inputs = tf.keras.Input(shape=observation_shape)
    features = hidden_layers(inputs)
    mean = tf.keras.layers.Dense(action_size, name="mean")(features)
    log_std = tf.keras.layers.Dense(
        action_size,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(-0.2),
        name="log_std",
    )(features)
    return tf.keras.Model(inputs, [mean, log_std], name="actor")


def build_critic(observation_shape):
    inputs = tf.keras.Input(shape=observation_shape)
    value = tf.keras.layers.Dense(1)(hidden_layers(inputs))
    return tf.keras.Model(inputs, value, name="critic")


def discounted_returns(rewards, gamma):
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns
