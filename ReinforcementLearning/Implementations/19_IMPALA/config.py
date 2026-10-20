ALGORITHM_NAME = "IMPALA"
ENV_ID = "CartPole-v1"
EPISODES = 2000
TEST_EPISODES = 20
SEED = 42
DEFAULT_GPU = 0
HIDDEN_UNITS = (128, 128)
LEARNING_RATE = 3e-4
GAMMA = 0.99
VTRACE_LAMBDA = 1.0
RHO_CLIP = 1.0
C_CLIP = 1.0
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.01
ENVIRONMENT_CONFIGS = {"CartPole-v1": {"EPISODES": 2000},
                       "Acrobot-v1": {"EPISODES": 4000},
                       "MountainCar-v0": {"EPISODES": 7500,
                                          "GAMMA": 0.999},
                       "LunarLander-v3": {"EPISODES": 5000}}


def apply_environment(env_id):
    for key, value in ENVIRONMENT_CONFIGS.get(env_id, {}).items():
        globals()[key] = value
