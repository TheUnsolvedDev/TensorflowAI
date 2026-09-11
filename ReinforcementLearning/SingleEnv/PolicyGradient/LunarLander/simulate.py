import gymnasium as gym
from critic_agent import PolicyGradientAgent
from config import *

def simulate(weights_path, episodes=5):
    env = gym.make(ENV_NAME, render_mode="human"); agent = PolicyGradientAgent(); agent.load(weights_path)
    for episode in range(episodes):
        obs, _ = env.reset(); done = False; total_reward = 0
        while not done:
            obs, reward, term, trunc, _ = env.step(agent.get_action(obs)); done = term or trunc; total_reward += reward
        print(f"Episode {episode} | Reward {total_reward}")
    env.close()

if __name__ == "__main__": simulate("checkpoints/policy.weights.h5", episodes=5)
