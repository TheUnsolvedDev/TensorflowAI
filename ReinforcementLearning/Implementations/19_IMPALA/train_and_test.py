import config
import agent
import tensorflow as tf
import numpy as np
import gymnasium as gym
import argparse
import csv
import os
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def configure_tensorflow(index):
    if index < 0:
        tf.config.set_visible_devices([], "GPU")
        return
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        if index >= len(gpus):
            raise ValueError(f"GPU {index} unavailable; found {len(gpus)}")
        tf.config.set_visible_devices(gpus[index], "GPU")
        tf.config.experimental.set_memory_growth(gpus[index], True)


def make_agent(environment, seed):
    if not isinstance(environment.action_space, gym.spaces.Discrete):
        raise TypeError(f"{config.ALGORITHM_NAME} requires discrete actions")
    return agent.Agent(environment.observation_space.shape, environment.action_space.n, seed)


def checkpoint_path(env_id): return str(
    Path("checkpoints") / f"{env_id}.weights.h5")


def train(env_id, episodes, seed, checkpoint, log_every):
    config.apply_environment(env_id)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)
    environment = gym.make(env_id)
    policy = make_agent(environment, seed)
    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
    history = []
    best = -np.inf
    print(f"{'Episode':>8} {'Reward':>10} {'Average':>10} {'Length':>8} {'Loss':>10}")
    for episode in range(1, episodes + 1):
        state, _ = environment.reset(seed=int(rng.integers(2**31 - 1)))
        done = False
        total = 0.0
        length = 0
        losses = []
        while not done:
            action = policy.act(state)
            next_state, reward, terminated, truncated, _ = environment.step(
                action)
            done = terminated or truncated
            policy.remember(state, action, reward, next_state, terminated)
            state = next_state
            total += float(reward)
            length += 1
        losses.append(policy.learn())
        loss = float(np.mean(losses)) if losses else 0.0
        history.append((episode, total, length, loss))
        average = float(np.mean([x[1] for x in history[-100:]]))
        if episode == 1 or (len(history) >= 100 and average > best):
            best = average
            policy.save(checkpoint)
        if episode == 1 or episode % log_every == 0 or episode == episodes:
            print(
                f"{episode:8d} {total:10.2f} {average:10.2f} {length:8d} {loss:10.4f}")
    environment.close()
    return history


def evaluate(env_id, episodes, seed, checkpoint, render=False):
    environment = gym.make(env_id, render_mode="human" if render else None)
    policy = make_agent(environment, seed)
    policy.load(checkpoint)
    rewards = []
    for episode in range(episodes):
        state, _ = environment.reset(seed=seed + episode)
        done = False
        total = 0.0
        while not done:
            state, reward, terminated, truncated, _ = environment.step(
                policy.act(state, training=False))
            total += float(reward)
            done = terminated or truncated
        rewards.append(total)
        print(f"episode={episode + 1} reward={total:.2f}")
    environment.close()
    print(f"mean_reward={np.mean(rewards):.2f}")
    return rewards


def save_results(env_id, history, evaluation, output_dir):
    target = Path(output_dir) / env_id
    target.mkdir(parents=True, exist_ok=True)
    with (target / "training_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["episode", "reward", "length", "loss"])
        writer.writerows(history)
    with (target / "evaluation_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["episode", "reward"])
        writer.writerows(enumerate(evaluation, 1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rewards = np.asarray([x[1] for x in history])
    window = min(100, len(rewards))
    moving = np.convolve(rewards, np.ones(window) / window, mode="valid")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(rewards, alpha=.35)
    axes[0].plot(np.arange(window - 1, len(rewards)), moving)
    axes[0].set_title("Raw reward")
    axes[1].plot([x[3] for x in history])
    axes[1].set_title("Loss")
    figure.tight_layout()
    figure.savefig(target / "results.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=config.ENV_ID)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--test-episodes", type=int,
                        default=config.TEST_EPISODES)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--gpu", type=int, default=config.DEFAULT_GPU)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    configure_tensorflow(args.gpu)
    config.apply_environment(args.env)
    checkpoint = args.checkpoint or checkpoint_path(args.env)
    history = train(args.env, args.episodes or config.EPISODES,
                    args.seed, checkpoint, args.log_every)
    evaluation = evaluate(args.env, args.test_episodes,
                          args.seed + 10000, checkpoint)
    save_results(args.env, history, evaluation, args.output_dir)


def inference_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=config.ENV_ID)
    parser.add_argument("--episodes", type=int, default=config.TEST_EPISODES)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--gpu", type=int, default=config.DEFAULT_GPU)
    parser.add_argument("--checkpoint")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    configure_tensorflow(args.gpu)
    evaluate(args.env, args.episodes, args.seed,
             args.checkpoint or checkpoint_path(args.env), not args.no_render)


if __name__ == "__main__":
    main()
