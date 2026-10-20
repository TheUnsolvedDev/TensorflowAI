import gymnasium as gym
import numpy as np


class LearningRewardWrapper(gym.Wrapper):
    def __init__(self, environment, gamma):
        super().__init__(environment)
        self.gamma = gamma
        self.previous_observation = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.previous_observation = observation
        return observation, info

    def potential(self, observation):
        if self.spec.id in {"MountainCar-v0", "MountainCarContinuous-v0"}:
            position = observation[0]
            height = np.sin(3.0 * position) * 0.45 + 0.55
            return 100.0 * (height - 1.0)
        if self.spec.id == "Acrobot-v1":
            cosine_1, sine_1, cosine_2, sine_2 = observation[:4]
            tip_height = -cosine_1 - (cosine_1 * cosine_2 - sine_1 * sine_2)
            return 5.0 * (tip_height - 1.0)
        return 0.0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action)
        previous_potential = self.potential(self.previous_observation)
        next_potential = 0.0 if terminated else self.potential(observation)
        learning_reward = reward + self.gamma * next_potential - previous_potential
        self.previous_observation = observation
        info["raw_reward"] = reward
        return observation, float(learning_reward), terminated, truncated, info


def make_training_env(env_id, gamma):
    environment = gym.make(env_id)
    if env_id in {"Acrobot-v1", "MountainCar-v0", "MountainCarContinuous-v0"}:
        return LearningRewardWrapper(environment, gamma)
    return environment
