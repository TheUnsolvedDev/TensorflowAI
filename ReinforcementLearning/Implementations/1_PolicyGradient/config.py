ENV_ID = "CartPole-v1"
EPISODES = 3000
TEST_EPISODES = 100
SEED = 42

HIDDEN_UNITS = (128, 128)
LEARNING_RATE = 1e-4
GAMMA = 0.99
GRADIENT_CLIP_NORM = 10.0
NORMALIZE_RETURNS = True

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

DEFAULT_ENVIRONMENT_CONFIG = {"EPISODES": EPISODES, "GAMMA": GAMMA}
ENVIRONMENT_CONFIGS = {
    "CartPole-v1": {"EPISODES": 1500, "GAMMA": 0.99},
    "Acrobot-v1": {"EPISODES": 3000, "GAMMA": 0.99},
    "MountainCar-v0": {"EPISODES": 5000, "GAMMA": 0.995},
    "Pendulum-v1": {"EPISODES": 3000, "GAMMA": 0.99},
    "MountainCarContinuous-v0": {"EPISODES": 5000, "GAMMA": 0.995},
    "LunarLander-v3": {"EPISODES": 5000, "GAMMA": 0.995},
}


def apply_environment(env_id):
    for name, value in (DEFAULT_ENVIRONMENT_CONFIG | ENVIRONMENT_CONFIGS.get(env_id, {})).items():
        globals()[name] = value
