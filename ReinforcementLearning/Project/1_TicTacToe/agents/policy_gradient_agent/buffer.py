import numpy as np


class TrajectoryBuffer:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.reset()

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def compute_returns(self):
        returns = []
        G = 0

        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = np.array(returns, dtype=np.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns