import os
import locale

# Pyglet/X11 needs a locale with a working UTF-8 codeset.
os.environ["LANG"] = "C.UTF-8"
os.environ["LC_ALL"] = "C.UTF-8"
locale.setlocale(locale.LC_ALL, "C.UTF-8")

import gymnasium as gym
import retro


class MarioEnv(gym.Env):
    """Gymnasium adapter for the original Super Mario Bros. NES ROM."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, game="SuperMarioBros-Nes-v0", state=None, render_mode=None):
        self.render_mode = render_mode
        kwargs = {"game": game, "render_mode": render_mode or "rgb_array"}
        if state is not None:
            kwargs["state"] = state
        self.env = retro.make(**kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            state, info = result
        else:
            state, info = result, {}
        if self.render_mode == "human":
            self.render()
        return state, info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            state, reward, terminated, truncated, info = result
            if self.render_mode == "human":
                self.render()
            return state, float(reward), bool(terminated), bool(truncated), info

        state, reward, done, info = result
        if self.render_mode == "human":
            self.render()
        return state, float(reward), bool(done), False, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
