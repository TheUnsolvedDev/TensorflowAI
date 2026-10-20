import config
import agent
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def configure_tensorflow(gpu):
    if gpu < 0:
        tf.config.set_visible_devices([], "GPU")
        return
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        if gpu >= len(gpus):
            raise ValueError(
                f"GPU {gpu} is unavailable; found {len(gpus)} GPU(s)")
        tf.config.set_visible_devices(gpus[gpu], "GPU")
        if config.ENABLE_GPU_MEMORY_GROWTH:
            tf.config.experimental.set_memory_growth(gpus[gpu], True)


def make_agent(environment, seed):
    if not isinstance(
            environment.observation_space,
            gym.spaces.Box) or not isinstance(
            environment.action_space,
            gym.spaces.Box):
        raise TypeError("DDPG requires Box observation and action spaces")
    return agent.DDPGAgent(
        environment.observation_space.shape,
        environment.action_space.low,
        environment.action_space.high,
        seed)


def train(env_id, episodes, seed, checkpoint, log_every, output_dir):
    config.apply_environment(env_id)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    environment = gym.make(env_id)
    policy = make_agent(environment, seed)
    history = []
    best_average = -np.inf
    random_seeds = np.random.default_rng(seed)
    print(f"{'Episode':>8} {'Reward':>10} {'Average':>10} {'Length':>8} {'Loss':>10} {'Best':>10}")
    for episode in range(1, episodes + 1):
        state, _ = environment.reset(
            seed=int(random_seeds.integers(2**31 - 1)))
        total_reward, length, losses = 0.0, 0, []
        done = False
        while not done:
            action = policy.act(state)
            next_state, reward, terminated, truncated, _ = environment.step(
                action)
            policy.remember(state, action, reward, next_state, terminated)
            result = policy.learn()
            if result is not None:
                losses.append(result[0])
            state = next_state
            total_reward += float(reward)
            length += 1
            done = terminated or truncated
        history.append({"episode": episode, "reward": total_reward,
                       "length": length, "loss": float(np.mean(losses)) if losses else np.nan})
        window = [row["reward"]
                  for row in history[-config.MOVING_AVERAGE_WINDOW:]]
        average = float(np.mean(window))
        if average > best_average:
            best_average = average
            policy.save(checkpoint)
        if episode == 1 or episode % log_every == 0:
            print(
                f"{episode:8d} {total_reward:10.2f} {average:10.2f} "
                f"{length:8d} {history[-1]['loss']:10.4f} {best_average:10.2f}"
            )
    environment.close()
    save_results(history, env_id, output_dir)
    return history


def save_results(history, env_id, output_dir):
    target = Path(output_dir) / env_id
    target.mkdir(parents=True, exist_ok=True)
    with (target / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    rewards = np.asarray([row["reward"] for row in history])
    window = min(config.MOVING_AVERAGE_WINDOW, len(rewards))
    moving = np.convolve(rewards, np.ones(window) / window, mode="valid")
    figure, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(rewards, alpha=0.35, label="episode")
    axes[0].plot(np.arange(window - 1, len(rewards)),
                 moving, label=f"{window}-episode average")
    axes[0].set_ylabel("Raw reward")
    axes[0].legend()
    axes[1].plot([row["loss"] for row in history])
    axes[1].set_ylabel("Critic loss")
    axes[1].set_xlabel("Episode")
    figure.tight_layout()
    figure.savefig(target / "training.png", dpi=config.PLOT_DPI)
    plt.close(figure)


def evaluate(env_id, episodes, seed, checkpoint, render=False):
    config.apply_environment(env_id)
    environment = gym.make(env_id, render_mode="human" if render else None)
    policy = make_agent(environment, seed)
    policy.load(checkpoint)
    rewards = []
    for episode in range(episodes):
        state, _ = environment.reset(seed=seed + episode)
        done, total = False, 0.0
        while not done:
            state, reward, terminated, truncated, _ = environment.step(
                policy.act(state, training=False))
            total += float(reward)
            done = terminated or truncated
        rewards.append(total)
        print(f"episode={episode + 1} reward={total:.2f}")
    environment.close()
    print(f"mean_reward={np.mean(rewards):.2f}")


def train_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=config.ENV_ID)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--test-episodes", type=int,
                        default=config.TEST_EPISODES)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--gpu", type=int, default=config.DEFAULT_GPU)
    parser.add_argument("--checkpoint", default="best.weights.h5")
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--log-every", type=int, default=config.LOG_EVERY)
    args = parser.parse_args()
    configure_tensorflow(args.gpu)
    config.apply_environment(args.env)
    train(args.env, args.episodes or config.EPISODES, args.seed,
          args.checkpoint, args.log_every, args.output_dir)
    evaluate(args.env, args.test_episodes, args.seed + 10000, args.checkpoint)


def inference_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=config.ENV_ID)
    parser.add_argument("--episodes", type=int, default=config.TEST_EPISODES)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--gpu", type=int, default=config.DEFAULT_GPU)
    parser.add_argument("--checkpoint", default="best.weights.h5")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    configure_tensorflow(args.gpu)
    evaluate(args.env, args.episodes, args.seed,
             args.checkpoint, not args.no_render)


if __name__ == "__main__":
    train_main()
