DEFAULT_ENV = "pendulum"

ENVIRONMENTS = {
    "pendulum": {
        "id": "Pendulum-v1",
        "action_space": "continuous",
        "max_episode_steps": 200},
    "mountain_car_continuous": {
        "id": "MountainCarContinuous-v0",
        "action_space": "continuous",
        "max_episode_steps": 999},
    "lunar_lander_continuous": {
        "id": "LunarLanderContinuous-v3",
        "action_space": "continuous",
        "max_episode_steps": 1000},
    "bipedal_walker": {
        "id": "BipedalWalker-v3",
        "action_space": "continuous",
        "max_episode_steps": 1600},
    "halfcheetah_v5": {
        "id": "HalfCheetah-v5",
        "action_space": "continuous"},
    "hopper_v5": {
        "id": "Hopper-v5",
        "action_space": "continuous"},
    "walker2d_v5": {
        "id": "Walker2d-v5",
        "action_space": "continuous"},
    "ant_v5": {
        "id": "Ant-v5",
        "action_space": "continuous"},
}


def get_env_settings(name=DEFAULT_ENV):
    if name not in ENVIRONMENTS:
        choices = ", ".join(ENVIRONMENTS)
        raise ValueError(
            f"Unknown environment '{name}'. Choose from: {choices}")
    return ENVIRONMENTS[name].copy()
