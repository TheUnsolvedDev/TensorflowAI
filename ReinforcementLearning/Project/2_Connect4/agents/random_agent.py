import numpy as np


class RandomAgent:
    def __init__(self):
        self.rows = 6
        self.cols = 7

    def act(self, obs, player):
        board = obs.reshape(self.rows, self.cols) if obs.ndim == 1 else obs

        valid_moves = np.where(board[0] == 0)[0]
        valid_moves = list(map(int, valid_moves))

        if len(valid_moves) == 0:
            return 0

        return int(np.random.choice(valid_moves))