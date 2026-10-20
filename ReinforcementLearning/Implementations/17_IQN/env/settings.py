DEFAULT_ENV = "CartPole-v1"

ENVIRONMENTS = {
    "cartpole_v1": {"id": "CartPole-v1", "action_space": "discrete"},
    "acrobot_v1": {"id": "Acrobot-v1", "action_space": "discrete"},
    "mountaincar_v0": {"id": "MountainCar-v0", "action_space": "discrete"},
    "lunarlander_v3": {"id": "LunarLander-v3", "action_space": "discrete"},
}


def get_env_settings(name=DEFAULT_ENV):
    if name not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {name}")
    return ENVIRONMENTS[name].copy()
