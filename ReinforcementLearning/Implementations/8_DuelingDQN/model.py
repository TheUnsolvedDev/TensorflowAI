from collections import deque

import numpy as np
import tensorflow as tf


def mlp(inputs, hidden=(128, 128)):
    x = inputs
    for units in hidden:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
    return x


def q_model(observation_shape, action_count, dueling=False):
    inputs = tf.keras.Input(shape=observation_shape)
    features = mlp(inputs)
    if dueling:
        value = tf.keras.layers.Dense(1)(features)
        advantage = tf.keras.layers.Dense(action_count)(features)
        centered = tf.keras.layers.Lambda(
            lambda value: value - tf.reduce_mean(value, axis=1, keepdims=True)
        )(advantage)
        outputs = tf.keras.layers.Add()([value, centered])
    else:
        outputs = tf.keras.layers.Dense(action_count)(features)
    return tf.keras.Model(inputs, outputs)


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


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity=100000, alpha=0.6):
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done):
        super().add(state, action, reward, next_state, done)
        self.priorities.append(max(self.priorities, default=1.0))

    def sample(self, batch_size, beta=0.4):
        probabilities = np.asarray(self.priorities, dtype=np.float32) ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.data), batch_size, p=probabilities, replace=False)
        batch = [self.data[index] for index in indices]
        weights = (len(self.data) * probabilities[indices]) ** -beta
        weights /= weights.max()
        values = tuple(np.asarray(items) for items in zip(*batch))
        return values, indices, weights.astype(np.float32)

    def update(self, indices, errors):
        for index, error in zip(indices, errors):
            self.priorities[index] = abs(float(error)) + 1e-6


