import random
from collections import deque

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
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.items = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.tree = np.zeros(capacity + 1, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.total = 0.0

    def _add_tree(self, index, value):
        index += 1
        while index <= self.capacity:
            self.tree[index] += value
            index += index & -index

    def _set_priority(self, index, priority):
        priority = float(priority) ** self.alpha
        delta = priority - self.priorities[index]
        self._add_tree(index, delta)
        self.priorities[index] = priority
        self.total += delta

    def _find(self, value):
        index = 0
        step = 1 << (self.capacity.bit_length() - 1)
        while step:
            candidate = index + step
            if candidate <= self.capacity and self.tree[candidate] <= value:
                index = candidate
                value -= self.tree[candidate]
            step >>= 1
        return min(index, self.size - 1)

    def add(self, state, action, reward, next_state, done):
        index = self.position
        self.items[index] = (state, action, reward, next_state, done)
        self._set_priority(index, max(float(self.priorities.max()) ** (1 / self.alpha), 1.0))
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        total = self.total
        values = np.random.random(batch_size) * total
        indices = np.fromiter((self._find(value) for value in values), dtype=np.int64, count=batch_size)
        self.last_indices = indices
        probabilities = self.priorities[indices] / total
        weights = (self.size * probabilities) ** -beta
        weights /= weights.max()
        batch = [self.items[index] for index in indices]
        return (*zip(*batch), weights.tolist())

    def update_priorities(self, errors):
        for index, error in zip(self.last_indices, errors):
            self._set_priority(int(index), abs(float(error)) + 1e-6)

    def __len__(self):
        return self.size
