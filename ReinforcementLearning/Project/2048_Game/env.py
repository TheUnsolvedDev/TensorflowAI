"""A small Gymnasium environment for the 2048 puzzle."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Game2048Env(gym.Env):
    """2048 with actions 0=up, 1=right, 2=down, and 3=left."""

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    TILE_COLORS = {
        0: (205, 193, 180),
        2: (238, 228, 218),
        4: (237, 224, 200),
        8: (242, 177, 121),
        16: (245, 149, 99),
        32: (246, 124, 95),
        64: (246, 94, 59),
        128: (237, 207, 114),
        256: (237, 204, 97),
        512: (237, 200, 80),
        1024: (237, 197, 63),
        2048: (237, 194, 46),
    }

    def __init__(self, render_mode: str | None = None, render_fps: int = 4):
        if render_mode not in {None, "ansi", "human"}:
            raise ValueError("render_mode must be None, 'ansi', or 'human'")

        self.render_mode = render_mode
        self.render_fps = render_fps
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=np.iinfo(np.int32).max, shape=(4, 4), dtype=np.int32
        )
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.window = None
        self.clock = None
        self.window_closed = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.board.fill(0)
        self.score = 0
        self.window_closed = False
        self._add_random_tile()
        self._add_random_tile()
        if self.render_mode == "human":
            self.render()
        return self.board.copy(), self._get_info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}; expected 0, 1, 2, or 3")

        old_board = self.board.copy()
        moved_board, reward = self._move(int(action))
        moved = not np.array_equal(old_board, moved_board)

        if moved:
            self.board = moved_board
            self.score += reward
            self._add_random_tile()

        terminated = not self._can_move()
        if self.render_mode == "human":
            self.render()
        return (
            self.board.copy(),
            float(reward),
            terminated,
            self.window_closed,
            self._get_info(moved),
        )

    def _move(self, action: int) -> tuple[np.ndarray, int]:
        # Rotate the requested direction to the left, merge, then rotate back.
        rotations = {0: 1, 1: 2, 2: 3, 3: 0}[action]
        board = np.rot90(self.board, rotations)
        result = np.zeros_like(board)
        reward = 0

        for row_index, row in enumerate(board):
            values = row[row != 0].tolist()
            merged = []
            index = 0
            while index < len(values):
                if index + 1 < len(values) and values[index] == values[index + 1]:
                    value = values[index] * 2
                    merged.append(value)
                    reward += value
                    index += 2
                else:
                    merged.append(values[index])
                    index += 1
            result[row_index, : len(merged)] = merged

        return np.rot90(result, -rotations), reward

    def _add_random_tile(self) -> None:
        empty_cells = np.argwhere(self.board == 0)
        if len(empty_cells) == 0:
            return
        row, column = empty_cells[self.np_random.integers(len(empty_cells))]
        self.board[row, column] = 4 if self.np_random.random() < 0.1 else 2

    def _can_move(self) -> bool:
        return bool(self._action_mask().any())

    def _action_mask(self) -> np.ndarray:
        board = self.board
        vertical_match = np.any(
            (board[1:, :] != 0) & (board[1:, :] == board[:-1, :])
        )
        horizontal_match = np.any(
            (board[:, 1:] != 0) & (board[:, 1:] == board[:, :-1])
        )
        return np.array(
            [
                vertical_match
                or np.any((board[1:, :] != 0) & (board[:-1, :] == 0)),
                horizontal_match
                or np.any((board[:, :-1] != 0) & (board[:, 1:] == 0)),
                vertical_match
                or np.any((board[:-1, :] != 0) & (board[1:, :] == 0)),
                horizontal_match
                or np.any((board[:, 1:] != 0) & (board[:, :-1] == 0)),
            ],
            dtype=np.int8,
        )

    def _get_info(self, moved: bool | None = None) -> dict:
        info = {
            "score": self.score,
            "max_tile": int(self.board.max()),
            "action_mask": self._action_mask(),
        }
        if moved is not None:
            info["moved"] = moved
        return info

    def render(self):
        if self.render_mode == "human":
            return self._render_human()

        width = max(4, len(str(int(self.board.max()))))
        border = "+" + "+".join(["-" * (width + 2)] * 4) + "+"
        lines = [f"Score: {self.score}", border]
        for row in self.board:
            cells = ["" if value == 0 else str(value) for value in row]
            lines.append("|" + "|".join(f" {cell:>{width}} " for cell in cells) + "|")
            lines.append(border)
        output = "\n".join(lines)
        return output

    def _render_human(self) -> None:
        import pygame

        pygame.init()
        if self.window is None:
            self.window = pygame.display.set_mode((500, 580))
            pygame.display.set_caption("2048 - Random Agent")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.window_closed = True

        self.window.fill((250, 248, 239))
        title_font = pygame.font.Font(None, 52)
        tile_font = pygame.font.Font(None, 44)
        self.window.blit(title_font.render("2048", True, (119, 110, 101)), (25, 18))
        score = title_font.render(f"Score: {self.score}", True, (119, 110, 101))
        self.window.blit(score, (475 - score.get_width(), 18))

        pygame.draw.rect(self.window, (187, 173, 160), (20, 90, 460, 460), border_radius=8)
        for row in range(4):
            for column in range(4):
                value = int(self.board[row, column])
                x, y = 30 + column * 112, 100 + row * 112
                color = self.TILE_COLORS.get(value, (60, 58, 50))
                pygame.draw.rect(self.window, color, (x, y, 102, 102), border_radius=6)
                if value:
                    text_color = (119, 110, 101) if value <= 4 else (249, 246, 242)
                    text = tile_font.render(str(value), True, text_color)
                    self.window.blit(
                        text,
                        (x + (102 - text.get_width()) / 2, y + (102 - text.get_height()) / 2),
                    )

        pygame.display.flip()
        self.clock.tick(self.render_fps)

    def close(self) -> None:
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


gym.register("Game2048-v0", entry_point="env:Game2048Env")
