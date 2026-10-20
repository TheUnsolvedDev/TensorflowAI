DEFAULT_ENV = "cartpole"

ENVIRONMENTS = {
    "cartpole": {
        "id": "CartPole-v1",
        "action_space": "discrete",
        "max_episode_steps": 500},
    "acrobot": {
        "id": "Acrobot-v1",
        "action_space": "discrete",
        "max_episode_steps": 500},
    "mountain_car": {
        "id": "MountainCar-v0",
        "action_space": "discrete",
        "max_episode_steps": 200},
    "lunar_lander": {
        "id": "LunarLander-v3",
        "action_space": "discrete",
        "max_episode_steps": 1000},
    "pendulum": {
        "id": "Pendulum-v1",
        "action_space": "continuous",
                        "max_episode_steps": 200},
    "lunar_lander_continuous": {
        "id": "LunarLanderContinuous-v3",
        "action_space": "continuous",
        "max_episode_steps": 1000},
    "bipedal_walker": {
        "id": "BipedalWalker-v3",
        "action_space": "continuous",
        "max_episode_steps": 1600},
    "halfcheetah": {
        "id": "HalfCheetah-v5",
        "action_space": "continuous",
        "max_episode_steps": 1000},
    "hopper": {
        "id": "Hopper-v5",
        "action_space": "continuous",
        "max_episode_steps": 1000},
    "walker2d": {
        "id": "Walker2d-v5",
        "action_space": "continuous",
        "max_episode_steps": 1000},
    "ant": {
        "id": "Ant-v5",
        "action_space": "continuous",
        "max_episode_steps": 1000},
}


def get_env_settings(name=DEFAULT_ENV):
    if name not in ENVIRONMENTS:
        choices = ", ".join(ENVIRONMENTS)
        raise ValueError(
            f"Unknown environment '{name}'. Choose from: {choices}")
    return ENVIRONMENTS[name].copy()
