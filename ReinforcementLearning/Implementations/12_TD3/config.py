ALGORITHM_NAME = "TD3"
ENV_ID = "Pendulum-v1"
EPISODES = 1000
TEST_EPISODES = 20
SEED = 42
DEFAULT_GPU = 0
HIDDEN_UNITS = (256, 256)
ACTOR_LEARNING_RATE = 3e-4
CRITIC_LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 256
REPLAY_CAPACITY = 1000000
LEARNING_STARTS = 5000
START_STEPS = 5000
EXPLORATION_NOISE = 0.1
TARGET_POLICY_NOISE = 0.2
TARGET_NOISE_CLIP = 0.5
POLICY_DELAY = 2
GRADIENT_CLIP_NORM = 10.0
MOVING_AVERAGE_WINDOW = 100
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "results"
LOG_EVERY = 10
PLOT_DPI = 160
SAVE_TEST_GIF = True
GIF_FPS = 30
GIF_MAX_FRAMES = 300
GIF_TEST_EPISODES = 10
ENABLE_GPU_MEMORY_GROWTH = True
DEFAULT_ENVIRONMENT_CONFIG = {
    "EPISODES": EPISODES,
    "GAMMA": GAMMA,
    "LEARNING_STARTS": LEARNING_STARTS,
    "START_STEPS": START_STEPS,
    "EXPLORATION_NOISE": EXPLORATION_NOISE,
    "ACTOR_LEARNING_RATE": ACTOR_LEARNING_RATE,
    "CRITIC_LEARNING_RATE": CRITIC_LEARNING_RATE}
ENVIRONMENT_CONFIGS = {"Pendulum-v1": {"EPISODES": 1000,
                                       "LEARNING_STARTS": 2000,
                                       "START_STEPS": 2000},
                       "MountainCarContinuous-v0": {"EPISODES": 4000,
                                                    "GAMMA": 0.995,
                                                    "LEARNING_STARTS": 5000,
                                                    "START_STEPS": 10000,
                                                    "EXPLORATION_NOISE": 0.2},
                       "LunarLanderContinuous-v3": {"EPISODES": 2500,
                                                    "LEARNING_STARTS": 10000,
                                                    "START_STEPS": 10000},
                       "BipedalWalker-v3": {"EPISODES": 4000,
                                            "LEARNING_STARTS": 10000,
                                            "START_STEPS": 10000},
                       "HalfCheetah-v5": {"EPISODES": 3000,
                                          "LEARNING_STARTS": 25000,
                                          "START_STEPS": 25000},
                       "Hopper-v5": {"EPISODES": 3000,
                                     "LEARNING_STARTS": 25000,
                                     "START_STEPS": 25000},
                       "Walker2d-v5": {"EPISODES": 4000,
                                       "LEARNING_STARTS": 25000,
                                       "START_STEPS": 25000},
                       "Ant-v5": {"EPISODES": 5000,
                                  "LEARNING_STARTS": 25000,
                                  "START_STEPS": 25000},
                       }


def apply_environment(env_id):
    for name, value in (DEFAULT_ENVIRONMENT_CONFIG | ENVIRONMENT_CONFIGS.get(env_id, {})).items():
        globals()[name] = value
