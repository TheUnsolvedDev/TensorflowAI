from collections import deque

import numpy as np
import tensorflow as tf


def mlp(inputs, hidden=(128, 128)):
    x = inputs
    for units in hidden:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
    return x


def actor_model(observation_shape, action_shape, action_limit):
    inputs = tf.keras.Input(shape=observation_shape)
    actions = tf.keras.layers.Dense(action_shape, activation="tanh")(mlp(inputs))
    outputs = tf.keras.layers.Rescaling(action_limit)(actions)
    return tf.keras.Model(inputs, outputs)


def gaussian_actor_model(observation_shape, action_shape):
    inputs = tf.keras.Input(shape=observation_shape)
    features = mlp(inputs)
    mean = tf.keras.layers.Dense(action_shape, name="mean")(features)
    log_std = tf.keras.layers.Dense(
        action_shape,
        bias_initializer=tf.keras.initializers.Constant(-0.5),
        name="log_std",
    )(features)
    return tf.keras.Model(inputs, [mean, log_std])


def critic_model(observation_shape, action_shape):
    states = tf.keras.Input(shape=observation_shape)
    actions = tf.keras.Input(shape=(action_shape,))
    features = tf.keras.layers.Concatenate()([states, actions])
    outputs = tf.keras.layers.Dense(1)(mlp(features))
    return tf.keras.Model([states, actions], outputs)


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.data = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.data.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.data), batch_size, replace=False)
        batch = [self.data[index] for index in indices]
        return tuple(np.asarray(values) for values in zip(*batch))

    def __len__(self):
        return len(self.data)


def soft_update(target, source, tau):
    for target_weight, source_weight in zip(target.weights, source.weights):
        target_weight.assign((1.0 - tau) * target_weight + tau * source_weight)


def gaussian_log_probability(actions, mean, log_std):
    log_std = tf.clip_by_value(log_std, -20.0, 2.0)
    variance = tf.exp(2.0 * log_std)
    return tf.reduce_sum(
        -0.5 * (((actions - mean) ** 2) / variance + 2.0 * log_std + np.log(2.0 * np.pi)),
        axis=1,
    )
