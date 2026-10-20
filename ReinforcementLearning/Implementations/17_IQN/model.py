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


class IQNModel(tf.keras.Model):
    def __init__(self, observation_shape, action_count):
        super().__init__()
        self.action_count = action_count
        layers = [tf.keras.Input(shape=observation_shape)]
        layers += [
            tf.keras.layers.Dense(units, activation="relu")
            for units in config.HIDDEN_UNITS
        ]
        self.feature = tf.keras.Sequential(layers)
        self.quantile = tf.keras.layers.Dense(
            config.HIDDEN_UNITS[-1], activation="relu")
        self.output_layer = tf.keras.layers.Dense(action_count)

    def call(self, states, quantiles):
        features = self.feature(states)
        frequencies = tf.cast(
            tf.range(1, config.QUANTILE_EMBEDDING + 1), tf.float32)[None, None]
        embedding = tf.cos(np.pi * quantiles[:, :, None] * frequencies)
        embedding = self.quantile(embedding)
        values = self.output_layer(features[:, None, :] * embedding)
        return tf.transpose(values, (0, 2, 1))


def build_network(observation_shape, action_count):
    network = IQNModel(observation_shape, action_count)
    network(tf.zeros((1,) + tuple(observation_shape)),
            tf.ones((1, config.QUANTILE_SAMPLES)))
    return network
