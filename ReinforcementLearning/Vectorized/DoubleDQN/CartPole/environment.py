import gymnasium as gym


def make_env(env_id, seed=None):
    def thunk():
        env = gym.make(env_id)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return thunk


class VectorizedEnv:
    def __init__(self, env_id="CartPole-v1", num_envs=8, asynchronous=False, seed=None):
        self.env_id = env_id
        self.num_envs = num_envs
        self.asynchronous = asynchronous
        self.seed = seed

        env_fns = [
            make_env(env_id, None if seed is None else seed + i)
            for i in range(num_envs)
        ]

        vector_cls = gym.vector.AsyncVectorEnv if asynchronous else gym.vector.SyncVectorEnv

        self.envs = vector_cls(env_fns)

        self.single_observation_space = self.envs.single_observation_space
        self.single_action_space = self.envs.single_action_space

        self.observation_space = self.envs.observation_space
        self.action_space = self.envs.action_space

    def reset(self, seed=None):
        return self.envs.reset(seed=seed)

    def step(self, actions):
        return self.envs.step(actions)

    def close(self):
        self.envs.close()