import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import gymnasium as gym
import numpy as np
import tensorflow as tf

import agent
import config


def configure_tensorflow():
    if not config.ENABLE_GPU_MEMORY_GROWTH:
        return
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def make_agent(environment):
    config.apply_environment(environment.spec.id)
    if not isinstance(environment.observation_space, gym.spaces.Box):
        raise ValueError("Only vector Box observation spaces are supported")
    if not isinstance(environment.action_space, (gym.spaces.Discrete, gym.spaces.Box)):
        raise ValueError("Only Discrete and Box action spaces are supported")
    return agent.GlobalAgent(environment.observation_space.shape, environment.action_space)


def checkpoint_path(env_id):
    environment_name = env_id.replace("/", "_").replace("-", "_")
    return str(Path(config.CHECKPOINT_DIR) / f"{environment_name}.weights.h5")


def train(env_id, episodes, seed, checkpoint, log_every, num_workers, rollout_steps):
    config.apply_environment(env_id)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    seed_generator = np.random.default_rng(seed)
    sample_environment = gym.make(env_id)
    observation_shape = sample_environment.observation_space.shape
    action_space = sample_environment.action_space
    sample_environment.close()
    policy = agent.GlobalAgent(observation_shape, action_space)
    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
    history = []
    state = {"started": 0, "best_average": -np.inf}
    episode_lock = threading.Lock()
    history_lock = threading.Lock()
    worker_seeds = seed_generator.integers(0, np.iinfo(np.int32).max, size=num_workers)
    print(f"{'Episode':>8} {'Reward':>10} {'Average':>10} {'Length':>8} {'Loss':>10} {'Best Avg':>10}")

    def reserve_episode():
        with episode_lock:
            if state["started"] >= episodes:
                return None
            state["started"] += 1
            return state["started"]

    def worker(worker_id):
        environment = gym.make(env_id)
        worker_seed = int(worker_seeds[worker_id])
        environment_seed_generator = np.random.default_rng(worker_seed)
        local = agent.WorkerAgent(
            observation_shape, action_space, policy, worker_seed
        )
        while True:
            reserved_episode = reserve_episode()
            if reserved_episode is None:
                break
            episode_seed = int(environment_seed_generator.integers(0, np.iinfo(np.int32).max))
            observation, _ = environment.reset(seed=episode_seed)
            total_reward = 0.0
            episode_length = 0
            losses = []
            done = False
            while not done:
                states = []
                actions = []
                rewards = []
                for _ in range(rollout_steps):
                    action = local.act(observation)
                    next_observation, reward, terminated, truncated, _ = environment.step(action)
                    done = terminated or truncated
                    states.append(observation)
                    actions.append(action)
                    rewards.append(reward)
                    observation = next_observation
                    total_reward += reward
                    episode_length += 1
                    if done:
                        break
                losses.append(
                    local.learn(
                        np.asarray(states, dtype=np.float32),
                        np.asarray(actions, dtype=np.float32 if local.continuous else np.int32),
                        np.asarray(rewards, dtype=np.float32),
                        done,
                        observation,
                        policy,
                    )
                )
            loss = float(np.mean(losses))
            with history_lock:
                episode = len(history) + 1
                history.append((episode, total_reward, episode_length, loss))
                average_reward = np.mean([row[1] for row in history[-20:]])
                if len(history) < config.MOVING_AVERAGE_WINDOW or average_reward > state["best_average"]:
                    state["best_average"] = average_reward
                    policy.save(checkpoint)
                if episode == 1 or episode % log_every == 0 or episode == episodes:
                    print(
                        f"{episode:8d} {total_reward:10.2f} {average_reward:10.2f} "
                        f"{episode_length:8d} {loss:10.4f} {state['best_average']:10.2f}"
                    )
        environment.close()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(worker, range(num_workers)))
    return history


def evaluate(env_id, episodes, seed, checkpoint, render=False):
    environment = gym.make(env_id, render_mode="human" if render else None)
    policy = make_agent(environment)
    policy.load(checkpoint)
    rewards = []
    for episode in range(1, episodes + 1):
        state, _ = environment.reset(seed=seed + episode)
        total_reward = 0.0
        done = False
        while not done:
            action = policy.act(state, training=False)
            state, reward, terminated, truncated, _ = environment.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        print(f"episode={episode} reward={total_reward:.2f}")
    environment.close()
    print(f"mean_reward={np.mean(rewards):.2f}")
    return rewards


def save_test_gif(env_id, seed, checkpoint, output_dir):
    import imageio.v2 as imageio

    test_environment = gym.make(env_id)
    policy = make_agent(test_environment)
    policy.load(checkpoint)
    best_seed = seed
    best_reward = -np.inf
    for test_index in range(config.GIF_TEST_EPISODES):
        candidate_seed = seed + test_index
        state, _ = test_environment.reset(seed=candidate_seed)
        total_reward = 0.0
        done = False
        while not done:
            action = policy.act(state, training=False)
            state, reward, terminated, truncated, _ = test_environment.step(action)
            total_reward += reward
            done = terminated or truncated
        if total_reward > best_reward:
            best_reward = total_reward
            best_seed = candidate_seed
    test_environment.close()

    environment = gym.make(env_id, render_mode="rgb_array")
    environment_name = env_id.replace("/", "_").replace("-", "_")
    run_dir = Path(output_dir) / environment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    gif_path = run_dir / "test.gif"
    max_steps = environment.spec.max_episode_steps or config.GIF_MAX_FRAMES
    frame_interval = max(1, int(np.ceil(max_steps / config.GIF_MAX_FRAMES)))
    state, _ = environment.reset(seed=best_seed)
    done = False
    step = 0
    with imageio.get_writer(
        gif_path,
        mode="I",
        duration=1000 / config.GIF_FPS,
        loop=0,
    ) as writer:
        writer.append_data(environment.render())
        while not done:
            action = policy.act(state, training=False)
            state, _, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated
            step += 1
            if step % frame_interval == 0 or done:
                writer.append_data(environment.render())
    environment.close()
    print(f"gif={gif_path} reward={best_reward:.2f} seed={best_seed}")


def save_results(env_id, history, evaluation_rewards, output_dir):
    environment_name = env_id.replace("/", "_").replace("-", "_")
    run_dir = Path(output_dir) / environment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "training_metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "reward", "length", "loss"])
        writer.writerows(history)

    with (run_dir / "evaluation_metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "reward"])
        writer.writerows(enumerate(evaluation_rewards, start=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    training_metrics = np.atleast_1d(
        np.genfromtxt(run_dir / "training_metrics.csv", delimiter=",", names=True)
    )
    evaluation_metrics = np.atleast_1d(
        np.genfromtxt(run_dir / "evaluation_metrics.csv", delimiter=",", names=True)
    )
    episodes = training_metrics["episode"]
    rewards = training_metrics["reward"]
    lengths = training_metrics["length"]
    losses = training_metrics["loss"]
    evaluation_episodes = evaluation_metrics["episode"]
    evaluation_rewards = evaluation_metrics["reward"]
    window = min(config.MOVING_AVERAGE_WINDOW, len(rewards))
    moving_average = np.convolve(rewards, np.ones(window) / window, mode="valid")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(episodes, rewards, color="tab:blue", alpha=0.45, label="Episode reward")
    axes[0, 0].plot(episodes[window - 1 :], moving_average, color="tab:blue", label=f"{window}-episode average")
    axes[0, 0].set_title("Training reward")
    axes[0, 0].legend()
    axes[0, 1].plot(episodes, losses, color="tab:red")
    axes[0, 1].set_title("Actor-value loss")
    axes[1, 0].plot(episodes, lengths, color="tab:green")
    axes[1, 0].set_title("Episode length")
    axes[1, 1].bar(evaluation_episodes, evaluation_rewards, color="tab:purple")
    axes[1, 1].axhline(np.mean(evaluation_rewards), color="black", linestyle="--", label="Mean")
    axes[1, 1].set_title("Evaluation rewards")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("Episode")
        axis.grid(alpha=0.25)
    figure.suptitle(f"A3C on {env_id}")
    figure.tight_layout()
    figure.savefig(run_dir / "results.png", dpi=config.PLOT_DPI)
    plt.close(figure)
    print(f"results={run_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=config.ENV_ID)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--test-episodes", type=int)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--checkpoint")
    parser.add_argument("--log-every", type=int, default=config.LOG_EVERY)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--rollout-steps", type=int)
    args = parser.parse_args()
    config.apply_environment(args.env)
    args.episodes = args.episodes or config.EPISODES
    args.test_episodes = args.test_episodes or config.TEST_EPISODES
    args.rollout_steps = args.rollout_steps or config.ROLLOUT_STEPS
    if min(args.episodes, args.test_episodes, args.log_every, args.num_workers, args.rollout_steps) < 1:
        parser.error("episode counts, logging, workers, and rollout steps must be positive")
    configure_tensorflow()
    checkpoint = args.checkpoint or checkpoint_path(args.env)
    history = train(
        args.env,
        args.episodes,
        args.seed,
        checkpoint,
        args.log_every,
        args.num_workers,
        args.rollout_steps,
    )
    rewards = evaluate(args.env, args.test_episodes, args.seed + 10000, checkpoint)
    save_results(args.env, history, rewards, args.output_dir)
    if config.SAVE_TEST_GIF:
        save_test_gif(args.env, args.seed + 20000, checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
