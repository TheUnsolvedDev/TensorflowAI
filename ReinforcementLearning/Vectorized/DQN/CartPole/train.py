import os
import time
import random
import datetime

import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import tqdm
import wandb

from agent import DQN
from environment import VectorizedEnv

from config import *


ALGO_NAME = "DQN"


SWEEP_CONFIG = {

    "method": "bayes",

    "metric": {
        "name": "env/avg_return_100",
        "goal": "maximize"
    },

    "parameters": {

        "lr": {
            "distribution": "log_uniform_values",
            "min": 3e-5,
            "max": 3e-4,
        },

        "gamma": {
            "values": [0.98, 0.99, 0.995]
        },

        "batch_size": {
            "values": [128, 256, 512]
        },

        "buffer_size": {
            "values": [100000, 200000, 300000]
        },

        "epsilon_end": {
            "values": [0.02, 0.05, 0.1]
        },

        "epsilon_decay_steps": {
            "values": [
                800_000,
                1_200_000,
                1_600_000
            ]
        },

        "target_update_freq": {
            "values": [1000, 2500, 5000]
        },
    }
}


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu():

    gpus = tf.config.list_physical_devices("GPU")

    if len(gpus) == 0:
        print("No GPU Found")
        return

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"Using {len(gpus)} GPU(s)")


def gpu_memory_mb():

    try:

        info = tf.config.experimental.get_memory_info("GPU:0")

        current_mb = info["current"] / (1024 ** 2)
        peak_mb = info["peak"] / (1024 ** 2)

        return current_mb, peak_mb

    except:

        return 0.0, 0.0


def linear_epsilon(step, epsilon_end, decay_steps):

    effective_step = step * NUM_ENVS

    fraction = min(
        effective_step / decay_steps,
        1.0
    )

    return EPSILON_START + fraction * (
        epsilon_end - EPSILON_START
    )


def make_dirs():

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def create_writer(run_name):

    log_path = os.path.join(
        LOG_DIR,
        run_name
    )

    return tf.summary.create_file_writer(log_path)


def build_run_name(config):

    run_name = (
        f"{ALGO_NAME}_"
        f"lr{config.lr}_"
        f"bs{config.batch_size}_"
        f"buf{config.buffer_size}_"
        f"eps{config.epsilon_end}_"
        f"gamma{config.gamma}"
    )

    run_name = run_name.replace(".", "_")

    return run_name


def initialize_wandb():

    run = wandb.init()

    config = wandb.config

    wandb.run.name = build_run_name(config)

    return run


def train():

    wandb_run = initialize_wandb()

    config = wandb.config

    set_seed(SEED)

    configure_gpu()

    make_dirs()

    env = VectorizedEnv(
        env_id=ENV_ID,
        num_envs=NUM_ENVS,
        asynchronous=ASYNC_ENV,
        seed=SEED
    )

    obs_shape = env.single_observation_space.shape
    action_dim = env.single_action_space.n

    agent = DQN(
        obs_dim=obs_shape,
        action_dim=action_dim,
        lr=config.lr,
        gamma=config.gamma,
        buffer_size=config.buffer_size
    )

    writer = create_writer(wandb.run.name)

    obs, _ = env.reset()

    episode_returns = np.zeros(
        NUM_ENVS,
        dtype=np.float32
    )

    episode_lengths = np.zeros(
        NUM_ENVS,
        dtype=np.int32
    )

    completed_returns = []
    completed_lengths = []

    best_return = -1e9

    start_time = time.time()

    for global_step in tqdm.tqdm(
        range(1, TOTAL_STEPS + 1)
    ):

        epsilon = linear_epsilon(
            global_step,
            config.epsilon_end,
            config.epsilon_decay_steps
        )

        actions = agent.sample_actions(
            obs,
            epsilon
        )

        next_obs, rewards, terminated, truncated, infos = env.step(actions)

        dones = terminated | truncated

        episode_returns += rewards
        episode_lengths += 1

        done_indices = np.where(dones)[0]

        if len(done_indices) > 0:

            completed_returns.extend(
                episode_returns[done_indices].tolist()
            )

            completed_lengths.extend(
                episode_lengths[done_indices].tolist()
            )

            episode_returns[done_indices] = 0.0
            episode_lengths[done_indices] = 0

        real_next_obs = next_obs.copy()

        if "final_observation" in infos:

            final_obs = infos["final_observation"]

            for i in range(NUM_ENVS):

                if dones[i]:
                    real_next_obs[i] = final_obs[i]

        agent.buffer.add_batch(
            obs.copy(),
            actions,
            rewards,
            real_next_obs,
            terminated,
            truncated
        )

        obs = next_obs

        if global_step > LEARNING_START and global_step % TRAIN_FREQ == 0:

            for _ in range(GRADIENT_STEPS):

                batch = agent.buffer.sample(
                    config.batch_size
                )

                experiences = (

                    batch["obs"],
                    batch["actions"],
                    batch["rewards"],
                    batch["next_obs"],

                    (
                        batch["terminated"] |
                        batch["truncated"]
                    ).astype(np.float32)
                )

                loss = agent.train_step(
                    experiences
                )

        else:

            loss = 0.0

        if global_step % config.target_update_freq == 0:
            agent.update_target()

        if global_step % SAVE_FREQ == 0:

            save_path = os.path.join(
                CHECKPOINT_DIR,
                f"{wandb.run.name}_{global_step}"
            )

            os.makedirs(save_path, exist_ok=True)

            agent.save(save_path)

        if global_step % LOG_FREQ == 0:

            fps = int(
                global_step * NUM_ENVS /
                (time.time() - start_time)
            )

            avg_return = (

                np.mean(completed_returns[-100:])
                if len(completed_returns) > 0
                else 0.0
            )

            avg_length = (

                np.mean(completed_lengths[-100:])
                if len(completed_lengths) > 0
                else 0.0
            )

            q_values = agent._distributed_q(
                obs.astype(np.float32)
            ).numpy()

            q_mean = np.mean(q_values)
            q_max = np.max(q_values)

            gpu_current_mb, gpu_peak_mb = gpu_memory_mb()

            with writer.as_default():

                tf.summary.scalar(
                    "train/loss",
                    loss,
                    step=global_step
                )

                tf.summary.scalar(
                    "env/avg_return_100",
                    avg_return,
                    step=global_step
                )

            wandb.log({

                "train/loss": float(loss),

                "train/epsilon": float(epsilon),

                "train/q_mean": float(q_mean),
                "train/q_max": float(q_max),

                "env/avg_return_100": float(avg_return),

                "env/avg_episode_length_100": float(avg_length),

                "system/fps": fps,

                "system/gpu_memory_current_mb":
                    gpu_current_mb,

                "system/gpu_memory_peak_mb":
                    gpu_peak_mb,

                "buffer/size": len(agent.buffer),

            }, step=global_step)

            if avg_return > best_return:

                best_return = avg_return

                best_path = os.path.join(
                    CHECKPOINT_DIR,
                    "best_model"
                )

                os.makedirs(best_path, exist_ok=True)

                agent.save(best_path)

            print(
                f"Step={global_step} "
                f"Loss={float(loss):.4f} "
                f"Return={avg_return:.2f} "
                f"Epsilon={epsilon:.3f} "
                f"FPS={fps}"
            )

    wandb.finish()

    env.close()


if __name__ == "__main__":

    sweep_name = f"{ENV_ID}_{ALGO_NAME}"

    SWEEP_CONFIG["name"] = sweep_name

    sweep_id = wandb.sweep(

        sweep=SWEEP_CONFIG,

        entity="shuvrajeet",

        project="TensorflowAI",
    )

    print(f"Sweep Name : {sweep_name}")
    print(f"Sweep ID   : {sweep_id}")

    wandb.agent(

        sweep_id,

        function=train,

        count=20
    )