import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    RecordEpisodeStatistics,
    ResizeObservation,
    TransformReward,
)


class BreakoutFireReset(gym.Wrapper):
    """Serve the ball on reset and after losing a life."""

    def _lives(self):
        ale = getattr(self.unwrapped, "ale", None)
        return ale.lives() if ale is not None else None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._remaining_lives = self._lives()
        for _ in range(self.np_random.integers(0, 31)):
            observation, _, terminated, truncated, _ = self.env.step(0)
            if terminated or truncated:
                observation, info = self.env.reset(**kwargs)
        observation, reward, terminated, truncated, step_info = self.env.step(1)
        self._remaining_lives = self._lives()
        info.update(step_info)
        return observation, info

    def step(self, action):
        if getattr(self, "_fire_next", False):
            action = 1
            self._fire_next = False
        result = self.env.step(action)
        lives = self._lives()
        if (
            self._remaining_lives is not None
            and lives is not None
            and lives < self._remaining_lives
        ):
            self._fire_next = True
        self._remaining_lives = lives
        return result


class MaxAndSkip(gym.Wrapper):
    """Repeat actions and max-pool the last two raw Atari frames."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
        self._frames = np.zeros((2, *env.observation_space.shape), dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        for index in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if index >= self.skip - 2:
                self._frames[index - (self.skip - 2)] = observation
            if terminated or truncated:
                break
        observation = np.maximum(self._frames[0], self._frames[1])
        return observation, total_reward, terminated, truncated, info


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


def training_env(render_mode=None, game="ALE/Breakout-v5"):
    from env.atari_env import AtariEnv

    env = AtariEnv(game=game, render_mode=render_mode)
    env = BreakoutFireReset(env)
    env = MaxAndSkip(env, 4)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=False)
    env = FrameStackObservation(env, 4)
    env = ChannelsLast(env)
    env = TransformReward(env, lambda reward: np.clip(reward, -1.0, 1.0))
    return RecordEpisodeStatistics(env)
