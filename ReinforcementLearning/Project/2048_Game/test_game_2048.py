import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from env import Game2048Env


class Game2048Test(unittest.TestCase):
    def test_environment_and_merge_rules(self):
        env = Game2048Env()
        check_env(env)

        first, _ = env.reset(seed=7)
        second, _ = env.reset(seed=7)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(np.count_nonzero(first), 2)

        env.board = np.array(
            [[2, 2, 2, 2], [4, 4, 8, 0], [2, 0, 2, 2], [0, 0, 0, 0]],
            dtype=np.int32,
        )
        moved, reward = env._move(3)
        self.assertEqual(reward, 20)
        self.assertEqual(
            moved.tolist(),
            [
                [4, 4, 0, 0],
                [8, 8, 0, 0],
                [4, 2, 0, 0],
                [0, 0, 0, 0],
            ],
        )

        env.board = np.zeros((4, 4), dtype=np.int32)
        env.board[3, 0] = 2
        moved, _ = env._move(0)
        self.assertEqual(moved[0, 0], 2)

        env.board.fill(0)
        env.board[0, 0] = 2
        moved, _ = env._move(1)
        self.assertEqual(moved[0, 3], 2)

        env.board = np.array(
            [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]],
            dtype=np.int32,
        )
        _, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(reward, 0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["moved"])

    def test_action_mask_matches_moves(self):
        env = Game2048Env()
        random = np.random.default_rng(42)
        values = np.array([0, 2, 4, 8, 16], dtype=np.int32)

        for _ in range(1_000):
            env.board = random.choice(values, size=(4, 4))
            expected = [
                not np.array_equal(env.board, env._move(action)[0])
                for action in range(4)
            ]
            self.assertEqual(env._action_mask().tolist(), expected)


if __name__ == "__main__":
    unittest.main()
