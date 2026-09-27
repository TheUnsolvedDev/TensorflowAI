import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


def make_pong(render_mode=None):
    return gym.make(
        "ALE/Pong-v5",
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode=render_mode,
    )
