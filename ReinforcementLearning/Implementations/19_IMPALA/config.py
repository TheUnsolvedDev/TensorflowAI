ALGORITHM_NAME = "IMPALA"
ENV_ID = "CartPole-v1"
EPISODES = 2000
TEST_EPISODES = 20
SEED = 42
DEFAULT_GPU = 0
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "results"
LOG_EVERY = 10
MOVING_AVERAGE_WINDOW = 100
PLOT_DPI = 160
SAVE_TEST_GIF = True
GIF_FPS = 30
GIF_MAX_FRAMES = 300
GIF_TEST_EPISODES = 10
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
