import gymnasium as gym
import tensorflow as tf

from agent import PolicyGradientAgent
from config import *


def simulate(weights_path, episodes=5):
    env = gym.make(ENV_NAME, render_mode="human")

    agent = PolicyGradientAgent()
    agent.load(weights_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(obs)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward

        print(f"Episode {ep} | Reward {total_reward}")

    env.close()


if __name__ == "__main__":
    simulate("checkpoints/policy.weights.h5", episodes=5)