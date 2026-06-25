import multiprocessing as mp
import numpy as np

_worker_agents = None


def _init_worker(agent_cls, agent_kwargs, num_envs):
    global _worker_agents
    _worker_agents = [agent_cls(**agent_kwargs) for _ in range(num_envs)]


def _act_worker(args):
    i, obs, player, mask = args
    agent = _worker_agents[i]

    a = agent.act(obs, player)

    if mask is not None and hasattr(mask, "__len__"):
        if mask[a] == 0:
            valid = np.where(mask == 1)[0]
            a = valid[0] if len(valid) > 0 else 0

    return i, a

class ParallelVectorAgent:
    def __init__(self, agent_cls, num_envs, num_workers=8, **kwargs):
        self.agent_cls = agent_cls
        self.B = num_envs
        self.pool = mp.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(agent_cls, kwargs, num_envs)
        )

    def act(self, obs, mask=None, player=None):
        tasks = []
        for i in range(self.B):
            p = 1 if player is None else player[i]
            m = None if mask is None else mask[i]
            tasks.append((i, obs[i], p, m))

        results = self.pool.map(_act_worker, tasks)

        actions = np.zeros(self.B, dtype=np.int32)
        for i, a in results:
            actions[i] = a

        return actions