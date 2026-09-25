import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    RecordEpisodeStatistics,
    TransformReward,
)

from env.atari_env import make_atari


class BreakoutFireReset(gym.Wrapper):
    """Serve the ball on reset and after a life is lost."""

    def _lives(self):
        return self.unwrapped.ale.lives()

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        observation, _, terminated, truncated, step_info = self.env.step(1)
        if terminated or truncated:
            observation, info = self.env.reset(**kwargs)
        else:
            info.update(step_info)
        self._remaining_lives = self._lives()
        self._fire_next = False
        return observation, info

    def step(self, action):
        if self._fire_next:
            action = 1
            self._fire_next = False
        result = self.env.step(action)
        lives = self._lives()
        if lives < self._remaining_lives and lives > 0:
            self._fire_next = True
        self._remaining_lives = lives
        return result


class ChannelsLast(gym.ObservationWrapper):
    """Convert Gymnasium's stacked (C, H, W) frames to Keras (H, W, C)."""

    def __init__(self, env):
        super().__init__(env)
        channels, height, width = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            0, 255, (height, width, channels), dtype=env.observation_space.dtype
        )

    def observation(self, observation):
        return np.moveaxis(observation, 0, -1)


def training_env(render_mode=None, game="ALE/Breakout-v5"):
    env = make_atari(game, render_mode)
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84)
    env = BreakoutFireReset(env)
    env = FrameStackObservation(env, 4)
    env = ChannelsLast(env)
    env = TransformReward(env, lambda reward: np.clip(reward, -1.0, 1.0))
    return RecordEpisodeStatistics(env)
