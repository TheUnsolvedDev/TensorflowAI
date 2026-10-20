import numpy as np
import tensorflow as tf

import config


class ReplayBuffer:
    def __init__(self, observation_shape, seed):
        shape = (config.REPLAY_CAPACITY,) + tuple(observation_shape)
        self.states = np.empty(shape, np.float32)
        self.next_states = np.empty(shape, np.float32)
        self.actions = np.empty(config.REPLAY_CAPACITY, np.int32)
        self.rewards = np.empty(config.REPLAY_CAPACITY, np.float32)
        self.terminals = np.empty(config.REPLAY_CAPACITY, np.float32)
        self.position = self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, terminal):
        i = self.position
        self.states[i], self.actions[i], self.rewards[i] = state, action, reward
        self.next_states[i], self.terminals[i] = next_state, terminal
        self.position = (i + 1) % config.REPLAY_CAPACITY
        self.size = min(self.size + 1, config.REPLAY_CAPACITY)

    def sample(self):
        i = self.rng.integers(self.size, size=config.BATCH_SIZE)
        return (
            self.states[i],
            self.actions[i],
            self.rewards[i],
            self.next_states[i],
            self.terminals[i],
        )

    def __len__(self):
        return self.size


def build_network(observation_shape, action_count):
    inputs = tf.keras.Input(shape=observation_shape)
    x = inputs
    for units in config.HIDDEN_UNITS:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
    outputs = tf.keras.layers.Reshape((action_count, config.QUANTILES))(
        tf.keras.layers.Dense(action_count * config.QUANTILES)(x))
    return tf.keras.Model(inputs, outputs)
