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
        self.max_priority = 1.0
        if config.USE_PRIORITIZED_REPLAY:
            self.tree_capacity = 1 << (config.REPLAY_CAPACITY - 1).bit_length()
            tree_shape = 2 * self.tree_capacity
            self.sum_tree = np.zeros(tree_shape, dtype=np.float64)
            self.min_tree = np.full(tree_shape, np.inf, dtype=np.float64)

    def _set_priority(self, index, priority):
        node = index + self.tree_capacity
        self.sum_tree[node] = priority
        self.min_tree[node] = priority
        node //= 2
        while node:
            left = node * 2
            self.sum_tree[node] = self.sum_tree[left] + self.sum_tree[left + 1]
            self.min_tree[node] = min(self.min_tree[left], self.min_tree[left + 1])
            node //= 2

    def _set_priorities(self, indices, priorities):
        leaves = np.asarray(indices, dtype=np.int64) + self.tree_capacity
        self.sum_tree[leaves] = priorities
        self.min_tree[leaves] = priorities
        parents = np.unique(leaves // 2)
        while parents.size:
            left = parents * 2
            self.sum_tree[parents] = self.sum_tree[left] + self.sum_tree[left + 1]
            self.min_tree[parents] = np.minimum(self.min_tree[left], self.min_tree[left + 1])
            if parents[0] == 1:
                break
            parents = np.unique(parents // 2)

    def add(self, state, action, reward, next_state, terminal):
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.terminals[index] = terminal
        if config.USE_PRIORITIZED_REPLAY:
            scaled_priority = self.max_priority ** config.PRIORITY_ALPHA
            self._set_priority(index, scaled_priority)
        self.position = (self.position + 1) % config.REPLAY_CAPACITY
        self.size = min(self.size + 1, config.REPLAY_CAPACITY)

    def sample(self, batch_size, beta):
        if config.USE_PRIORITIZED_REPLAY:
            total_priority = self.sum_tree[1]
            segment = total_priority / batch_size
            masses = (np.arange(batch_size) + self.rng.random(batch_size)) * segment
            masses = np.minimum(masses, np.nextafter(total_priority, 0.0))
            nodes = np.ones(batch_size, dtype=np.int64)
            while nodes[0] < self.tree_capacity:
                left = nodes * 2
                left_priorities = self.sum_tree[left]
                go_right = masses >= left_priorities
                masses -= left_priorities * go_right
                nodes = left + go_right
                masses = np.minimum(masses, np.nextafter(self.sum_tree[nodes], 0.0))
            indices = nodes - self.tree_capacity
            smallest = np.finfo(np.float64).tiny
            probabilities = np.maximum(self.sum_tree[nodes] / total_priority, smallest)
            minimum_probability = max(self.min_tree[1] / total_priority, smallest)
            maximum_weight = (self.size * minimum_probability) ** -beta
            weights = np.power(self.size * probabilities, -beta) / maximum_weight
        else:
            indices = self.rng.integers(self.size, size=batch_size)
            weights = np.ones(batch_size, dtype=np.float32)
        batch = (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.terminals[indices],
        )
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices, errors):
        priorities = np.abs(np.asarray(errors)) + config.PRIORITY_EPSILON
        self.max_priority = max(self.max_priority, float(np.max(priorities)))
        scaled_priorities = np.power(priorities, config.PRIORITY_ALPHA).astype(np.float64)
        unique_indices, reverse = np.unique(indices, return_inverse=True)
        unique_priorities = np.full(unique_indices.size, -np.inf, dtype=np.float64)
        np.maximum.at(unique_priorities, reverse, scaled_priorities)
        self._set_priorities(unique_indices, unique_priorities)

    def __len__(self):
        return self.size

