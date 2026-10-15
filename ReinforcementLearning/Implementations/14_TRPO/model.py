from collections import deque

import numpy as np
import tensorflow as tf


def mlp(inputs, hidden=(128, 128)):
    x = inputs
    for units in hidden:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
    return x


def actor_critic_model(observation_shape, action_count):
    inputs = tf.keras.Input(shape=observation_shape)
    features = mlp(inputs)
    policy = tf.keras.layers.Dense(action_count, activation="softmax", name="policy")(features)
    value = tf.keras.layers.Dense(1, name="value")(features)
    return tf.keras.Model(inputs, [policy, value])


def discounted_returns(rewards, dones, gamma):
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running * (1.0 - dones[index])
        returns[index] = running
    return returns


