import numpy as np
import tensorflow as tf

import config


class DuelingCombine(tf.keras.layers.Layer):
    def call(self, inputs):
        value, advantage = inputs
        return value + advantage - tf.reduce_mean(advantage, axis=1, keepdims=True)


def build_q_network(observation_shape, action_count):
    inputs = tf.keras.Input(shape=observation_shape)
    features = inputs
    for units in config.HIDDEN_UNITS:
        features = tf.keras.layers.Dense(units, activation="relu")(features)
    if config.USE_DUELING_NETWORK:
        value = tf.keras.layers.Dense(1, name="value")(features)
        advantage = tf.keras.layers.Dense(action_count, name="advantage")(features)
        outputs = DuelingCombine(name="q_values")([value, advantage])
    else:
        outputs = tf.keras.layers.Dense(action_count, name="q_values")(features)
    return tf.keras.Model(inputs, outputs)


class ReplayBuffer:
    def __init__(self, observation_shape, seed):
        shape = (config.REPLAY_CAPACITY,) + tuple(observation_shape)
        self.states = np.empty(shape, dtype=np.float32)
        self.next_states = np.empty(shape, dtype=np.float32)
        self.actions = np.empty(config.REPLAY_CAPACITY, dtype=np.int32)
        self.rewards = np.empty(config.REPLAY_CAPACITY, dtype=np.float32)
        self.terminals = np.empty(config.REPLAY_CAPACITY, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, terminal):
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.terminals[index] = terminal
        self.position = (self.position + 1) % config.REPLAY_CAPACITY
        self.size = min(self.size + 1, config.REPLAY_CAPACITY)

    def sample(self, batch_size):
        indices = self.rng.integers(self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.terminals[indices],
        )

    def __len__(self):
        return self.size

