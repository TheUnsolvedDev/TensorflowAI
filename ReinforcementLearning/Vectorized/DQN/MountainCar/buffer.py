import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.obs_shape = obs_shape

        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)

        self.actions = np.zeros((capacity,), dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)

        self.terminated = np.zeros((capacity,), dtype=np.bool_)
        self.truncated = np.zeros((capacity,), dtype=np.bool_)

        self.ptr = 0
        self.size = 0

    def add_batch(self, obs, actions, rewards, next_obs, terminated, truncated):
        batch_size = obs.shape[0]

        indices = (np.arange(batch_size) + self.ptr) % self.capacity

        self.obs[indices] = obs
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_obs[indices] = next_obs
        self.terminated[indices] = terminated
        self.truncated[indices] = truncated

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)

        batch = {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "terminated": self.terminated[indices],
            "truncated": self.truncated[indices]
        }

        return batch

    def __len__(self):
        return self.size
