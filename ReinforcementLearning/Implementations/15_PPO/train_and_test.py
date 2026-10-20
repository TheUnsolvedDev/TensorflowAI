import config
import agent
import tensorflow as tf
import numpy as np
import gymnasium as gym
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def make_agent(environment):
    if isinstance(environment.action_space, gym.spaces.Discrete):
        return agent.PPOAgent(
            environment.observation_space.shape,
            environment.action_space.n)
    if isinstance(environment.action_space, gym.spaces.Box):
        return agent.ContinuousPPOAgent(
            environment.observation_space.shape,
            environment.action_space.low,
            environment.action_space.high)
    raise TypeError(
        f"{config.ALGORITHM_NAME} requires Discrete or Box actions")


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
        tf.config.experimental.set_memory_growth(gpus[gpu], True)


def train(env_id, episodes, seed, checkpoint):
    config.apply_environment(env_id)
    np.random.seed(seed)
    environment = gym.make(env_id)
    agent = make_agent(environment)
    best_reward = -np.inf
    random_seeds = np.random.default_rng(seed)
    for episode in range(1, episodes + 1):
        state, _ = environment.reset(
            seed=int(random_seeds.integers(2**31 - 1)))
        total_reward = 0.0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = environment.step(
                action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
        agent.learn()
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(checkpoint)
        print(
            f"episode={episode} reward={total_reward:.2f} best={best_reward:.2f}")
    environment.close()


def evaluate(env_id, episodes, seed, checkpoint, render=False):
    config.apply_environment(env_id)
    environment = gym.make(env_id, render_mode="human" if render else None)
    agent = make_agent(environment)
    agent.load(checkpoint)
    rewards = []
    for episode in range(episodes):
        state, _ = environment.reset(seed=seed + episode)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.act(state, training=False)
            state, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
        print(f"episode={episode + 1} reward={total_reward:.2f}")
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
    args = parser.parse_args()
    configure_tensorflow(args.gpu)
    config.apply_environment(args.env)
    checkpoint = args.checkpoint
    train(args.env, args.episodes or config.EPISODES, args.seed, checkpoint)
    evaluate(args.env, args.test_episodes, args.seed + 10000, checkpoint)


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
