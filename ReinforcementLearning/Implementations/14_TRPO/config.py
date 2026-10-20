ALGORITHM_NAME = "TRPO"
ENV_ID = "CartPole-v1"
EPISODES = 1000
TEST_EPISODES = 20
SEED = 42
DEFAULT_GPU = 0
HIDDEN_UNITS = (64, 64)
VALUE_LEARNING_RATE = 1e-3
VALUE_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.97
MAX_KL = 0.01
CG_ITERATIONS = 10
CG_DAMPING = 0.1
BACKTRACK_STEPS = 10
BACKTRACK_COEFFICIENT = 0.8
ENVIRONMENT_CONFIGS = {
    "CartPole-v1": {"EPISODES": 1000},
    "Acrobot-v1": {"EPISODES": 3000, "MAX_KL": 0.005},
    "MountainCar-v0": {"EPISODES": 5000, "GAMMA": 0.995, "MAX_KL": 0.005},
    "LunarLander-v3": {"EPISODES": 3000, "MAX_KL": 0.005},
}


def apply_environment(env_id):
    for name, value in ENVIRONMENT_CONFIGS.get(env_id, {}).items():
        globals()[name] = value
