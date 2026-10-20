ALGORITHM_NAME = "DDPG"
ENV_ID = "Pendulum-v1"
EPISODES = 1000
TEST_EPISODES = 20
SEED = 42
DEFAULT_GPU = 0
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
                                       "START_STEPS": 2000,
                                       "EXPLORATION_NOISE": 0.1},
                       "MountainCarContinuous-v0": {"EPISODES": 5000,
                                                    "GAMMA": 0.995,
                                                    "LEARNING_STARTS": 5000,
                                                    "START_STEPS": 10000,
                                                    "EXPLORATION_NOISE": 0.2},
                       "LunarLanderContinuous-v3": {"EPISODES": 3000,
                                                    "LEARNING_STARTS": 10000,
                                                    "START_STEPS": 10000,
                                                    "EXPLORATION_NOISE": 0.15},
                       "BipedalWalker-v3": {"EPISODES": 5000,
                                            "LEARNING_STARTS": 10000,
                                            "START_STEPS": 10000,
                                            "EXPLORATION_NOISE": 0.1},
                       "HalfCheetah-v5": {"EPISODES": 3000,
                                          "LEARNING_STARTS": 25000,
                                          "START_STEPS": 25000,
                                          "EXPLORATION_NOISE": 0.1},
                       "Reacher-v5": {"EPISODES": 2000,
                                      "LEARNING_STARTS": 10000,
                                      "START_STEPS": 10000,
                                      "EXPLORATION_NOISE": 0.15},
                       }


def apply_environment(env_id):
    for name, value in (DEFAULT_ENVIRONMENT_CONFIG | ENVIRONMENT_CONFIGS.get(env_id, {})).items():
        globals()[name] = value
