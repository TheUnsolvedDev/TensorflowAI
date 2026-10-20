DEFAULT_ENV = "Pendulum-v1"

ENVIRONMENTS = {
    "pendulum_v1": {"id": "Pendulum-v1", "action_space": "continuous"},
    "mountaincarcontinuous_v0": {"id": "MountainCarContinuous-v0", "action_space": "continuous"},
    "lunarlandercontinuous_v3": {"id": "LunarLanderContinuous-v3", "action_space": "continuous"},
    "bipedalwalker_v3": {"id": "BipedalWalker-v3", "action_space": "continuous"},
    "halfcheetah_v5": {"id": "HalfCheetah-v5", "action_space": "continuous"},
    "hopper_v5": {"id": "Hopper-v5", "action_space": "continuous"},
    "walker2d_v5": {"id": "Walker2d-v5", "action_space": "continuous"},
    "ant_v5": {"id": "Ant-v5", "action_space": "continuous"},
}


def get_env_settings(name=DEFAULT_ENV):
    if name not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {name}")
    return ENVIRONMENTS[name].copy()
