import numpy as np
import tensorflow as tf

import config


def hidden_layers(inputs):
    features = inputs
    for units in config.HIDDEN_UNITS:
        features = tf.keras.layers.Dense(units, activation="relu")(features)
    return features


def actor_model(observation_shape, action_size):
    inputs = tf.keras.Input(shape=observation_shape)
    outputs = tf.keras.layers.Dense(
        action_size, activation="tanh")(hidden_layers(inputs))
    return tf.keras.Model(inputs, outputs)


def gaussian_actor_model(observation_shape, action_size):
    inputs = tf.keras.Input(shape=observation_shape)
    features = hidden_layers(inputs)
    mean = tf.keras.layers.Dense(action_size)(features)
    log_std = tf.keras.layers.Dense(
        action_size, bias_initializer=tf.keras.initializers.Constant(-0.5))(features)
    return tf.keras.Model(inputs, [mean, log_std])


def critic_model(observation_shape, action_size):
    states = tf.keras.Input(shape=observation_shape)
    actions = tf.keras.Input(shape=(action_size,))
    outputs = tf.keras.layers.Dense(1)(hidden_layers(
        tf.keras.layers.Concatenate()([states, actions])))
    return tf.keras.Model([states, actions], outputs)


class ReplayBuffer:
    def __init__(self, observation_shape, action_size, seed):
        self.states = np.empty((config.REPLAY_CAPACITY,) +
                               tuple(observation_shape), np.float32)
        self.actions = np.empty(
            (config.REPLAY_CAPACITY, action_size), np.float32)
        self.rewards = np.empty(config.REPLAY_CAPACITY, np.float32)
        self.next_states = np.empty(
            (config.REPLAY_CAPACITY,) + tuple(observation_shape), np.float32)
        self.terminals = np.empty(config.REPLAY_CAPACITY, np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, terminal):
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.terminals[index] = terminal
        self.position = (self.position + 1) % config.REPLAY_CAPACITY
        self.size = min(self.size + 1, config.REPLAY_CAPACITY)

    def sample(self):
        indices = self.rng.integers(self.size, size=config.BATCH_SIZE)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.terminals[indices],
        )

    def __len__(self):
        return self.size


def soft_update(target, source):
    for target_weight, source_weight in zip(target.weights, source.weights):
        target_weight.assign((1.0 - config.TAU) *
                             target_weight + config.TAU * source_weight)
