import numpy as np


class RandomAgent:
    def act(self, obs, player):
        valid_moves = np.where(obs == 0)[0]
        return np.random.choice(valid_moves)
