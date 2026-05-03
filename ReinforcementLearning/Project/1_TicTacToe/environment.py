import numpy as np
import gymnasium as gym


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, flatten_obs=True):
        super().__init__()
        self.flatten_obs = flatten_obs
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1

        self.action_space = gym.spaces.Discrete(9)
        obs_shape = (9,) if flatten_obs else (3, 3)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=obs_shape, dtype=np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.fill(0)
        self.current_player = 1
        return self._get_obs(), {}

    def step(self, action):
        row, col = divmod(action, 3)

        if self.board[row, col] != 0:
            return self._get_obs(), -1.0, True, False, {"invalid_move": True}

        self.board[row, col] = self.current_player
        winner = self._check_winner()

        if winner is not None:
            reward = 1.0 if winner == self.current_player else -1.0
            return self._get_obs(), reward, True, False, {"winner": winner}

        if np.all(self.board != 0):
            return self._get_obs(), 0.0, True, False, {"draw": True}

        self.current_player *= -1
        return self._get_obs(), 0.0, False, False, {}

    def _get_obs(self):
        return self.board.flatten().copy() if self.flatten_obs else self.board.copy()

    def _check_winner(self):
        b = self.board

        for i in range(3):
            row_sum = np.sum(b[i, :])
            col_sum = np.sum(b[:, i])

            if abs(row_sum) == 3:
                return np.sign(row_sum)
            if abs(col_sum) == 3:
                return np.sign(col_sum)

        diag1 = np.sum(np.diag(b))
        diag2 = np.sum(np.diag(np.fliplr(b)))

        if abs(diag1) == 3:
            return np.sign(diag1)
        if abs(diag2) == 3:
            return np.sign(diag2)

        return None

    def render(self):
        print(self.board)