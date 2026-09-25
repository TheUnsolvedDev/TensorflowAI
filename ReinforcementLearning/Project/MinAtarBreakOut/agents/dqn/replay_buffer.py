from collections import deque
import random

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.items = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.items.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.items, batch_size)
        return tuple(zip(*batch))

    def __len__(self):
        return len(self.items)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha=0.6, priority_epsilon=1e-6, seed=None):
        self.capacity = capacity
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.items = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.scaled_priorities = np.zeros(capacity, dtype=np.float64)
        self.tree = np.zeros(capacity + 1, dtype=np.float64)
        self.size = 0
        self.next_index = 0
        self.max_priority = 1.0
        self.total_priority = 0.0
        self.last_indices = np.empty(0, dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    def _add_tree(self, index, value):
        index += 1
        while index <= self.capacity:
            self.tree[index] += value
            index += index & -index

    def __len__(self):
        return self.size

    def _find_prefix(self, values):
        values = np.asarray(values, dtype=np.float64).copy()
        indices = np.zeros(values.shape, dtype=np.int64)
        step = 1 << (self.capacity.bit_length() - 1)
        while step:
            candidates = indices + step
            valid = candidates <= self.capacity
            nodes = self.tree[np.minimum(candidates, self.capacity)]
            take = valid & (nodes <= values)
            indices[take] = candidates[take]
            values[take] -= nodes[take]
            step >>= 1
        return np.minimum(indices, self.size - 1)

    def add(self, state, action, reward, next_state, done):
        index = self.next_index
        old_scaled_priority = self.scaled_priorities[index]
        scaled_priority = self.max_priority**self.alpha
        self.items[index] = (state, action, reward, next_state, done)
        self.priorities[index] = self.max_priority
        self.scaled_priorities[index] = scaled_priority
        delta = scaled_priority - old_scaled_priority
        self._add_tree(index, delta)
        self.total_priority += delta
        self.next_index = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        if batch_size > self.size:
            raise ValueError("batch_size cannot exceed the replay buffer size")
        segment = self.total_priority / batch_size
        values = (np.arange(batch_size) + self.rng.random(batch_size)) * segment
        indices = self._find_prefix(values)
        probabilities = self.scaled_priorities[indices] / self.total_priority
        weights = (self.size * probabilities) ** -beta
        self.last_indices = indices
        batch = [self.items[index] for index in indices]
        return (*zip(*batch), weights / weights.max())

    def update_priorities(self, errors):
        for index, error in zip(self.last_indices, np.asarray(errors).reshape(-1)):
            priority = abs(float(error)) + self.priority_epsilon
            scaled_priority = priority**self.alpha
            delta = scaled_priority - self.scaled_priorities[index]
            self._add_tree(index, delta)
            self.total_priority += delta
            self.priorities[index] = priority
            self.scaled_priorities[index] = scaled_priority
            self.max_priority = max(self.max_priority, priority)
