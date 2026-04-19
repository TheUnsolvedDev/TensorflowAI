import os
import datetime
import tensorflow as tf
import gymnasium as gym
import tqdm

from agent import PolicyGradientAgent
from trajectory_buffer import TrajectoryBuffer
from config import *


LOG_DIR = "logs"
VIDEO_DIR = "videos"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)


def make_env(record=False, episode_id=0):
    if record:
        return gym.wrappers.RecordVideo(
            gym.make(ENV_NAME, render_mode="rgb_array"),
            video_folder=VIDEO_DIR,
            episode_trigger=lambda x: True,
            name_prefix=f"episode_{episode_id}"
        )
    else:
        return gym.make(ENV_NAME)


def train():
    agent = PolicyGradientAgent()

    CHECKPOINT_DIR = "checkpoints"
    VIDEO_DIR = "videos"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tf.summary.create_file_writer(f"{LOG_DIR}/{current_time}")

    best_reward = -float("inf")

    for episode in tqdm.tqdm(range(EPOCHS), desc="Training Episodes"):
        env = make_env(record=False, episode_id=episode)
        obs, _ = env.reset()
        buffer = TrajectoryBuffer()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.store(obs, action, reward)
            obs = next_obs
            total_reward += reward
        obs_arr, act_arr, ret_arr = buffer.get()

        agent.update(
            tf.convert_to_tensor(obs_arr),
            tf.convert_to_tensor(act_arr),
            tf.convert_to_tensor(ret_arr)
        )
        improved = total_reward > best_reward

        if improved:
            best_reward = total_reward
            agent.save(f"{CHECKPOINT_DIR}/best.weights.h5")
            record_env = make_env(record=True, episode_id=episode)
            record_obs, _ = record_env.reset()

            done = False
            while not done:
                action = agent.get_action(record_obs)
                record_obs, _, terminated, truncated, _ = record_env.step(action)
                done = terminated or truncated

            record_env.close()

        with writer.as_default():
            tf.summary.scalar("reward", total_reward, step=episode)
            tf.summary.scalar("best_reward", best_reward, step=episode)
            tf.summary.scalar("episode_length", len(buffer.rewards), step=episode)

        env.close()

def setup_gpu(gpu_id):
    gpus = tf.config.list_physical_devices('GPU')
    [tf.config.experimental.set_memory_growth(g, True) for g in gpus]

    if gpu_id == -1:
        print("Using all GPUs")
    elif 0 <= gpu_id < len(gpus):
        tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
        print(f"Using GPU {gpu_id}")
    else:
        print("Invalid GPU ID, using CPU")

if __name__ == "__main__":
    setup_gpu(-1)
    train()