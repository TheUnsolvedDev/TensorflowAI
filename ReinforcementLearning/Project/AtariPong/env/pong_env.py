import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing

gym.register_envs(ale_py)


def make_env(render_mode=None):
    env = gym.make(
        "ALE/Pong-v5",
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode=render_mode,
    )
    return AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4)
