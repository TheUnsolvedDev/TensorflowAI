import gymnasium as gym
import minatar
import numpy as np


class MinAtarEnv(gym.Env):
    """Gymnasium adapter for MinAtar Breakout."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, game="breakout", render_mode=None):
        if game != "breakout":
            raise ValueError("Only the MinAtar breakout game is supported")
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.env = minatar.Environment(game)
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(self.env.num_actions())
        self.observation_space = gym.spaces.Box(0, 1, self.env.state_shape(), dtype=np.bool_)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
        self.env.reset()
        self.render()
        return self.env.state(), {}

    def step(self, action):
        reward, terminated = self.env.act(int(action))
        self.render()
        return self.env.state(), reward, terminated, False, {}

    def render(self):
        if self.render_mode == "human":
            self.env.display_state(1000 // self.metadata["render_fps"])

    def close(self):
        if self.render_mode == "human":
            self.env.close_display()
