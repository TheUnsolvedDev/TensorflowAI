import numpy as np
import gymnasium as gym


class VectorizedConnect4Env(gym.Env):
    def __init__(self, num_envs=32, flatten_obs=True):
        super().__init__()

        self.B = num_envs
        self.rows = 6
        self.cols = 7
        self.flatten_obs = flatten_obs

        self.board = np.zeros((self.B, self.rows, self.cols), dtype=np.int8)
        self.current_player = np.ones((self.B,), dtype=np.int8)
        self.done = np.zeros((self.B,), dtype=bool)

        self.P = np.zeros(self.B, dtype=np.int64)
        self.M = np.zeros(self.B, dtype=np.int64)

        self.bottom_mask = np.array([(1 << (7 * c))
                                    for c in range(self.cols)], dtype=np.int64)
        self.top_mask = np.array([(1 << (7 * c + 5))
                                 for c in range(self.cols)], dtype=np.int64)

        self.action_space = gym.spaces.Discrete(self.cols)
        obs_shape = (self.rows * self.cols,
                     ) if flatten_obs else (self.rows, self.cols)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.B, *obs_shape), dtype=np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board.fill(0)
        self.current_player.fill(1)
        self.done.fill(False)

        self.P.fill(0)
        self.M.fill(0)

        return self._get_obs(), {}

    def step(self, actions):
        rewards = np.zeros(self.B, dtype=np.float32)

        active = ~self.done
        if not np.any(active):
            return self._get_obs(), rewards, self.done, False, {"actions": actions}

        actions, invalid = self._sanitize_actions(actions, active)

        v_idx = np.where(active)[0]
        cols = actions[v_idx]

        moves = (self.M[v_idx] + self.bottom_mask[cols]) & ~self.M[v_idx]
        self.M[v_idx] |= moves
        self.P[v_idx] ^= self.M[v_idx]

        rows = self._compute_rows(v_idx, cols)
        self.board[v_idx, rows, cols] = self.current_player[v_idx]

        win_mask_local = self._is_win_batch(self.P[v_idx])
        winners_idx = v_idx[win_mask_local]

        self.done[winners_idx] = True
        rewards[winners_idx] = 1.0

        full = np.all(self.board != 0, axis=(1, 2)) & (~self.done)
        self.done[full] = True

        self.current_player[~self.done] *= -1

        rewards[v_idx] -= 0.01 * invalid[v_idx]

        return self._get_obs(), rewards, self.done, False, {"actions": actions}

    def _get_obs(self):
        if self.flatten_obs:
            return self.board.reshape(self.B, -1)
        return self.board.copy()
    
    def board_to_planes(self, canonical_board):
        current = (canonical_board == 1).astype(np.float32)
        opponent = (canonical_board == -1).astype(np.float32)
        valid = (canonical_board[:, 0, :] == 0).astype(np.float32)
        valid = np.repeat(
            valid[:, None, :],
            canonical_board.shape[1],
            axis=1
        )
        player_plane = np.ones_like(current, dtype=np.float32)
        return np.stack([current, opponent, valid, player_plane],axis=-1)

    def get_action_mask(self):
        return (self.board[:, 0, :] == 0).astype(np.int8)

    def get_canonical_obs(self):
        if self.flatten_obs:
            obs = self.board.reshape(self.B, -1)
            return obs * self.current_player[:, None]
        return self.board * self.current_player[:, None, None]

    def _is_win_batch(self, P):
        m = P & (P >> 7)
        win = (m & (m >> 14)) != 0

        m = P & (P >> 6)
        win |= (m & (m >> 12)) != 0

        m = P & (P >> 8)
        win |= (m & (m >> 16)) != 0

        m = P & (P >> 1)
        win |= (m & (m >> 2)) != 0

        return win

    def _compute_rows(self, v_idx, cols):
        col_vals = self.board[v_idx, :, cols]
        empty = (col_vals == 0)
        rows = self.rows - 1 - np.argmax(empty[:, ::-1], axis=1)
        return rows

    def _sanitize_actions(self, actions, active):
        mask = (self.board[:, 0, :] == 0)

        actions = actions.copy()
        invalid = np.zeros(self.B, dtype=bool)

        active_idx = np.where(active)[0]
        active_actions = actions[active_idx]

        invalid_active = mask[active_idx, active_actions] == 0

        if np.any(invalid_active):
            fallback = np.argmax(mask[active_idx], axis=1)
            actions[active_idx[invalid_active]] = fallback[invalid_active]

        invalid[active_idx] = invalid_active

        return actions, invalid
