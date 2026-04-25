import gymnasium as gym

from config import *

class Environment:
    def __init__(self, name=ENV_NAME, seed=42, render_mode=None):
        self.env = gym.make(name, render_mode=render_mode)
        self.env.reset(seed=seed)

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def render(self):
        self.env.render()