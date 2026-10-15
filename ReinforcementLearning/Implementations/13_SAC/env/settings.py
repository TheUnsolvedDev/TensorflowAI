"""Gymnasium environment presets used by training and inference."""

DEFAULT_ENV = "cartpole"

ENVIRONMENTS = {
    "cartpole": {
        "id": "CartPole-v1",
        "action_space": "discrete",
        "max_episode_steps": 500,
    },
    "acrobot": {
        "id": "Acrobot-v1",
        "action_space": "discrete",
        "max_episode_steps": 500,
    },
    "lunar_lander": {
        "id": "LunarLander-v3",
        "action_space": "discrete",
        "max_episode_steps": 1000,
    },
    "pendulum": {
        "id": "Pendulum-v1",
        "action_space": "continuous",
        "max_episode_steps": 200,
    },
    "mountain_car_continuous": {
        "id": "MountainCarContinuous-v0",
        "action_space": "continuous",
        "max_episode_steps": 999,
    },
}


def get_env_settings(name=DEFAULT_ENV):
    """Return a copy of one environment preset."""
    if name not in ENVIRONMENTS:
        choices = ", ".join(ENVIRONMENTS)
        raise ValueError(f"Unknown environment '{name}'. Choose from: {choices}")
    return ENVIRONMENTS[name].copy()
