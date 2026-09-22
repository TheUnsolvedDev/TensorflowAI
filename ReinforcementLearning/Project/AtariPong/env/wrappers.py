import gymnasium as gym
from gymnasium.wrappers import (
    FrameStackObservation,
    RecordEpisodeStatistics,
)
from env.pong_env import make_env


class ChannelsLast(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        frames, height, width = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, frames), dtype=env.observation_space.dtype
        )

    def observation(self, observation):
        return observation.transpose(1, 2, 0)


def training_env(render_mode=None, game="ALE/Pong-v5"):
    env = make_env(render_mode=render_mode)
    env = FrameStackObservation(env, 4)
    env = ChannelsLast(env)
    return RecordEpisodeStatistics(env)
