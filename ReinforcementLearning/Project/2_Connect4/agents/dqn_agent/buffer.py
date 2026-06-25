import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = capacity
        self.obs_dim = obs_dim

        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, *obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_dim), dtype=np.float32)

        self.actions = np.zeros((capacity,), dtype=np.int32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.next_mask = np.zeros((capacity, action_dim), dtype=np.float32)

    def add(self, obs, actions, rewards, next_obs, dones, next_mask):
        B = obs.shape[0]
        idxs = (self.ptr + np.arange(B)) % self.capacity

        self.obs[idxs] = obs
        self.next_obs[idxs] = next_obs
        self.actions[idxs] = actions
        self.rewards[idxs] = rewards
        self.dones[idxs] = dones
        self.next_mask[idxs] = next_mask

        self.ptr = (self.ptr + B) % self.capacity
        self.size = min(self.size + B, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        return {
            "obs": self.obs[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_obs": self.next_obs[idxs],
            "dones": self.dones[idxs],
            "next_mask": self.next_mask[idxs],
        }

    def __len__(self):
        return self.size
