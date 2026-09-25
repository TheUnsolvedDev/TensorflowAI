import ale_py
import gymnasium as gym


gym.register_envs(ale_py)


def make_atari(game="ALE/Breakout-v5", render_mode=None):
    """Create raw Breakout with frame skipping disabled for preprocessing."""
    return gym.make(
        game,
        render_mode=render_mode,
        frameskip=1,
        repeat_action_probability=0.0,
    )
