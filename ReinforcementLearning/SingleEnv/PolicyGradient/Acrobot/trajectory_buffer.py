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
        value = 0
        for reward in reversed(self.rewards):
            value = reward + GAMMA * value
            returns.append(value)
        returns.reverse()
        return np.array(returns, dtype=np.float32)

    def get(self):
        return (
            np.asarray(self.obs, dtype=np.float32),
            np.asarray(self.actions, dtype=np.int32),
            self.compute_returns(),
        )
