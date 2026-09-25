import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


class AtariEnv(gym.Env):
    """Small Gymnasium adapter for Atari Breakout."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, game="ALE/Breakout-v5", render_mode=None):
        self.env = gym.make(game, render_mode=render_mode, frameskip=1)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
