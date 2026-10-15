import argparse

import gymnasium as gym
import numpy as np

import agent


POLICY_ALGORITHMS = {"policy_gradient", "reinforce", "actor_critic", "a2c", "a3c", "trpo", "ppo"}
CONTINUOUS_ALGORITHMS = {"ddpg", "td3", "sac"}
DQN_VARIANTS = {
    "dqn": "dqn",
    "double_dqn": "double",
    "dueling_dqn": "dueling",
    "per_dqn": "per",
    "rainbow_dqn": "rainbow",
}


def make_agent(algorithm, environment):
    observation_shape = environment.observation_space.shape
    if algorithm in POLICY_ALGORITHMS:
        return agent.PolicyAgent(observation_shape, environment.action_space.n, algorithm)
    if algorithm in CONTINUOUS_ALGORITHMS:
        action_shape = environment.action_space.shape[0]
        action_limit = float(environment.action_space.high[0])
        return agent.ContinuousAgent(observation_shape, action_shape, action_limit, algorithm)
    return agent.DQNAgent(observation_shape, environment.action_space.n, DQN_VARIANTS[algorithm])


def train(algorithm, env_id, episodes, seed, checkpoint):
    np.random.seed(seed)
    environment = gym.make(env_id)
    agent = make_agent(algorithm, environment)
    best_reward = -np.inf
    for episode in range(1, episodes + 1):
        state, _ = environment.reset(seed=seed + episode)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            if algorithm not in POLICY_ALGORITHMS:
                agent.learn()
            state = next_state
            total_reward += reward
        if algorithm in POLICY_ALGORITHMS:
            agent.learn()
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(checkpoint)
        print(f"episode={episode} reward={total_reward:.2f} best={best_reward:.2f}")
    environment.close()


def evaluate(algorithm, env_id, episodes, seed, checkpoint, render=False):
    environment = gym.make(env_id, render_mode="human" if render else None)
    agent = make_agent(algorithm, environment)
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


def train_main(algorithm, default_env):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=default_env)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--test-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="best.weights.h5")
    args = parser.parse_args()
    checkpoint = args.checkpoint
    train(algorithm, args.env, args.episodes, args.seed, checkpoint)
    evaluate(algorithm, args.env, args.test_episodes, args.seed + 10000, checkpoint)


def inference_main(algorithm, default_env):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=default_env)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="best.weights.h5")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    evaluate(algorithm, args.env, args.episodes, args.seed, args.checkpoint, args.render)


if __name__ == "__main__":
    train_main("double_dqn", "CartPole-v1")
