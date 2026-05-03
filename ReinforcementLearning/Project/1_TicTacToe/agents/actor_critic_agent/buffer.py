import numpy as np


class A2CBuffer:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.reset()

    def store(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def reset(self):
        self.states, self.actions, self.rewards = [], [], []

    def compute_returns(self):
        G, returns = 0, []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = np.array(returns, dtype=np.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns