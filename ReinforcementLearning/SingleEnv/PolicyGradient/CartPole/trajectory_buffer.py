import numpy as np
from config import GAMMA


class TrajectoryBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.rewards = []

    def store(self, obs, action, reward):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)

    def compute_returns(self):
        returns = []
        G = 0

        for r in reversed(self.rewards):
            G = r + GAMMA * G
            returns.append(G)

        returns.reverse()
        return np.array(returns, dtype=np.float32)

    def get(self):
        obs = np.array(self.obs, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int32)
        returns = self.compute_returns()
        return obs, actions, returns