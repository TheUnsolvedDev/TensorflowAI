import random
from collections import deque


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
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done):
        super().add(state, action, reward, next_state, done)
        self.priorities.append(max(self.priorities, default=1.0))

    def sample(self, batch_size, beta=0.4):
        probabilities = [p ** self.alpha for p in self.priorities]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        indices = random.choices(range(len(self.items)), weights=probabilities, k=batch_size)
        self.last_indices = indices
        batch = [self.items[index] for index in indices]
        weights = [(len(self.items) * probabilities[index]) ** -beta for index in indices]
        maximum = max(weights)
        return (*zip(*batch), [weight / maximum for weight in weights])

    def update_priorities(self, errors):
        for index, error in zip(self.last_indices, errors):
            self.priorities[index] = abs(float(error)) + 1e-6
