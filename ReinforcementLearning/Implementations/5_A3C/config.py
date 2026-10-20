ENV_ID = "CartPole-v1"
EPISODES = 3000
TEST_EPISODES = 100
SEED = 42
DEFAULT_GPU = 0

HIDDEN_UNITS = (128, 128)
LEARNING_RATE = 2.5e-4
RMSPROP_RHO = 0.99
RMSPROP_EPSILON = 1e-5
GAMMA = 0.99
GRADIENT_CLIP_NORM = 10.0
VALUE_LOSS_COEFFICIENT = 0.5
HUBER_DELTA = 1.0
ADVANTAGE_CLIP = 10.0
ENTROPY_COEFFICIENT = 0.005
LOG_STD_MIN = -2.0
LOG_STD_MAX = 1.0
NUM_WORKERS = 4
ROLLOUT_STEPS = 32

CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "results"
LOG_EVERY = 10
MOVING_AVERAGE_WINDOW = 100
PLOT_DPI = 160
SAVE_TEST_GIF = True
GIF_FPS = 30
GIF_MAX_FRAMES = 300
GIF_TEST_EPISODES = 20

ENABLE_GPU_MEMORY_GROWTH = True

DEFAULT_ENVIRONMENT_CONFIG = {
    "EPISODES": EPISODES,
    "GAMMA": GAMMA,
    "ENTROPY_COEFFICIENT": ENTROPY_COEFFICIENT,
    "ROLLOUT_STEPS": ROLLOUT_STEPS,
    "LEARNING_RATE": LEARNING_RATE,
}
ENVIRONMENT_CONFIGS = {"CartPole-v1": {"EPISODES": 1500,
                                       "GAMMA": 0.99,
                                       "ENTROPY_COEFFICIENT": 0.001,
                                       "ROLLOUT_STEPS": 5},
                       "Acrobot-v1": {"EPISODES": 3000,
                                      "GAMMA": 0.99,
                                      "ROLLOUT_STEPS": 20},
                       "MountainCar-v0": {"EPISODES": 7500,
                                          "GAMMA": 0.999,
                                          "ENTROPY_COEFFICIENT": 0.01,
                                          "ROLLOUT_STEPS": 64,
                                          "LEARNING_RATE": 1e-4},
                       "Pendulum-v1": {"EPISODES": 3000,
                                       "GAMMA": 0.99,
                                       "ENTROPY_COEFFICIENT": 0.001,
                                       "ROLLOUT_STEPS": 20,
                                       "LEARNING_RATE": 1e-4},
                       "MountainCarContinuous-v0": {"EPISODES": 7500,
                                                    "GAMMA": 0.999,
                                                    "ENTROPY_COEFFICIENT": 0.01,
                                                    "ROLLOUT_STEPS": 64,
                                                    "LEARNING_RATE": 1e-4},
                       "LunarLander-v3": {"EPISODES": 5000,
                                          "GAMMA": 0.995,
                                          "ROLLOUT_STEPS": 32,
                                          "LEARNING_RATE": 2e-4},
                       "LunarLanderContinuous-v3": {"EPISODES": 5000,
                                                    "GAMMA": 0.995,
                                                    "ROLLOUT_STEPS": 64,
                                                    "LEARNING_RATE": 2e-4},
                       "BipedalWalker-v3": {"EPISODES": 10000,
                                            "GAMMA": 0.99,
                                            "ENTROPY_COEFFICIENT": 0.001,
                                            "ROLLOUT_STEPS": 64,
                                            "LEARNING_RATE": 1e-4},
                       }


def apply_environment(env_id):
    for name, value in (DEFAULT_ENVIRONMENT_CONFIG | ENVIRONMENT_CONFIGS.get(env_id, {})).items():
        globals()[name] = value
