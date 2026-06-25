import numpy as np
import gymnasium as gym


class Connect4Env(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, flatten_obs=True):
        super().__init__()
        self.rows = 6
        self.cols = 7
        self.flatten_obs = flatten_obs

        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1

        self.action_space = gym.spaces.Discrete(self.cols)

        obs_shape = (self.rows * self.cols,) if flatten_obs else (self.rows, self.cols)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=obs_shape, dtype=np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.fill(0)
        self.current_player = 1
        return self._get_obs(), {}

    def step(self, action):
        col = action

        if self.board[0, col] != 0:
            return self._get_obs(), -1.0, True, False, {"invalid_move": True}

        row = self._get_available_row(col)
        self.board[row, col] = self.current_player

        winner = self._check_winner(row, col)

        if winner is not None:
            reward = 1.0 if winner == self.current_player else -1.0
            return self._get_obs(), reward, True, False, {"winner": winner}

        if np.all(self.board != 0):
            return self._get_obs(), 0.0, True, False, {"draw": True}

        self.current_player *= -1
        return self._get_obs(), 0.0, False, False, {}

    def _get_available_row(self, col):
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, col] == 0:
                return r

    def _get_obs(self):
        return self.board.flatten().copy() if self.flatten_obs else self.board.copy()

    def _check_winner(self, row, col):
        player = self.board[row, col]
        directions = [(0,1), (1,0), (1,1), (1,-1)]

        for dr, dc in directions:
            count = 1

            r, c = row + dr, col + dc
            while self._in_bounds(r, c) and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc

            r, c = row - dr, col - dc
            while self._in_bounds(r, c) and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= 4:
                return player

        return None
    
    def get_valid_actions(self):
        return np.where(self.board[0] == 0)[0]
    
    def get_action_mask(self):
        mask = (self.board[0] == 0).astype(np.int8)
        return mask

    def _in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def render(self):
        print(self.board)