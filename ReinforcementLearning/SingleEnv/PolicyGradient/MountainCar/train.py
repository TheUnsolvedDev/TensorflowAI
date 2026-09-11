import os, datetime
import tensorflow as tf
import gymnasium as gym
import tqdm
from reinforce import REINFORCEAgent
from trajectory_buffer import TrajectoryBuffer
from config import *

LOG_DIR, VIDEO_DIR = "logs", "videos"
os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(VIDEO_DIR, exist_ok=True)

def make_env(record=False, episode_id=0):
    if record:
        return gym.wrappers.RecordVideo(gym.make(ENV_NAME, render_mode="rgb_array"), VIDEO_DIR, episode_trigger=lambda _: True, name_prefix=f"episode_{episode_id}")
    return gym.make(ENV_NAME)

def train():
    agent = REINFORCEAgent(); checkpoint_dir = "checkpoints"; os.makedirs(checkpoint_dir, exist_ok=True)
    writer = tf.summary.create_file_writer(f"{LOG_DIR}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"); best_reward = -float("inf")
    for episode in tqdm.tqdm(range(EPOCHS), desc="Training Episodes"):
        env = make_env(); obs, _ = env.reset(); buffer = TrajectoryBuffer(); done = False; total_reward = 0
        while not done:
            action = agent.get_action(obs); next_obs, reward, term, trunc, _ = env.step(action); done = term or trunc
            buffer.store(obs, action, reward); obs = next_obs; total_reward += reward
        obs_arr, act_arr, ret_arr = buffer.get(); agent.update(tf.convert_to_tensor(obs_arr), tf.convert_to_tensor(act_arr), tf.convert_to_tensor(ret_arr))
        if total_reward > best_reward:
            best_reward = total_reward; agent.save(f"{checkpoint_dir}/best.weights.h5")
            record_env = make_env(True, episode); record_obs, _ = record_env.reset(); record_done = False
            while not record_done:
                record_obs, _, term, trunc, _ = record_env.step(agent.get_action(record_obs)); record_done = term or trunc
            record_env.close()
        with writer.as_default():
            tf.summary.scalar("reward", total_reward, episode); tf.summary.scalar("best_reward", best_reward, episode); tf.summary.scalar("episode_length", len(buffer.rewards), episode)
        env.close()

def setup_gpu(gpu_id):
    gpus = tf.config.list_physical_devices("GPU"); [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
    if gpu_id == -1: print("Using all GPUs")
    elif 0 <= gpu_id < len(gpus): tf.config.set_visible_devices(gpus[gpu_id], "GPU"); print(f"Using GPU {gpu_id}")
    else: print("Invalid GPU ID, using CPU")

if __name__ == "__main__": setup_gpu(-1); train()
