ALGORITHM_NAME = "MuZero"
ENV_ID = "CartPole-v1"
EPISODES = 3000
TEST_EPISODES = 20
SEED = 42
DEFAULT_GPU = 0
HIDDEN = 128
LATENT = 64
LEARNING_RATE = 3e-4
GAMMA = 0.997
SIMULATIONS = 25
UNROLL_STEPS = 5
BATCH_SIZE = 32
REPLAY_CAPACITY = 50000
LEARNING_STARTS = 500
C_PUCT = 1.5
ENVIRONMENT_CONFIGS = {"CartPole-v1": {"EPISODES": 3000},
                       "Acrobot-v1": {"EPISODES": 5000}}


def apply_environment(env_id):
    for key, value in ENVIRONMENT_CONFIGS.get(env_id, {}).items():
        globals()[key] = value
