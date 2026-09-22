import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    RecordEpisodeStatistics,
    ResizeObservation,
)


class ChannelsLast(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        channels, height, width = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(height, width, channels),
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(1, 2, 0)


class DiscreteActions(gym.ActionWrapper):
    """Map discrete actions to directional and jump/button inputs."""

    def __init__(self, env):
        super().__init__(env)
        available = {button: index for index, button in enumerate(env.env.buttons)}
        self.buttons = [button for button in ("UP", "DOWN", "LEFT", "RIGHT", "A", "B") if button in available]
        self.indices = [available[button] for button in self.buttons]
        self.action_space = gym.spaces.Discrete(len(self.buttons))

    def action(self, action):
        buttons = np.zeros(self.env.action_space.n, dtype=np.uint8)
        buttons[self.indices[int(action)]] = 1
        return buttons


def training_env(render_mode=None, game="SuperMarioBros-Nes-v0"):
    env = __import__("env.mario_env", fromlist=["MarioEnv"]).MarioEnv(game=game, render_mode=render_mode)
    env = DiscreteActions(env)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=False)
    env = FrameStackObservation(env, 4)
    env = ChannelsLast(env)
    return RecordEpisodeStatistics(env)
