ALGORITHM_NAME = "D4PG"
ENV_ID = "Pendulum-v1"
EPISODES = 1500
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
HIDDEN_UNITS = (256, 256)
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 128
REPLAY_CAPACITY = 1000000
LEARNING_STARTS = 5000
START_STEPS = 5000
EXPLORATION_NOISE = 0.1
ATOM_COUNT = 51
V_MIN = -200.0
V_MAX = 200.0
ENVIRONMENT_CONFIGS = {"Pendulum-v1": {"EPISODES": 1500},
                       "MountainCarContinuous-v0": {"EPISODES": 5000,
                                                    "GAMMA": 0.995},
                       "LunarLanderContinuous-v3": {"EPISODES": 3000},
                       "BipedalWalker-v3": {"EPISODES": 5000},
                       "HalfCheetah-v5": {"EPISODES": 3500,
                                          "LEARNING_STARTS": 25000,
                                          "START_STEPS": 25000,
                                          "V_MIN": -100.0,
                                          "V_MAX": 1500.0},
                       "Hopper-v5": {"EPISODES": 3500,
                                     "LEARNING_STARTS": 25000,
                                     "START_STEPS": 25000,
                                     "V_MIN": -100.0,
                                     "V_MAX": 600.0},
                       "Walker2d-v5": {"EPISODES": 4500,
                                       "LEARNING_STARTS": 25000,
                                       "START_STEPS": 25000,
                                       "V_MIN": -100.0,
                                       "V_MAX": 1000.0},
                       "Ant-v5": {"EPISODES": 5000,
                                  "LEARNING_STARTS": 25000,
                                  "START_STEPS": 25000,
                                  "V_MIN": -500.0,
                                  "V_MAX": 1500.0},
                       }


def apply_environment(env_id):
    for key, value in ENVIRONMENT_CONFIGS.get(env_id, {}).items():
        globals()[key] = value
