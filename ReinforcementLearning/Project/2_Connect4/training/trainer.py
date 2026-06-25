import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import tqdm

from vectorized_env import VectorizedConnect4Env
from agents.random_agent import RandomAgent
from agents.dqn_agent.agent import DQNAgent
from agents.dqn_agent.buffer import ReplayBuffer
from agents.vector_agent import ParallelVectorAgent


def gpu_mem():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def _unwrap_agent(agent):
    visited = set()
    while hasattr(agent, "agent") and id(agent) not in visited:
        visited.add(id(agent))
        agent = agent.agent
    return agent


def _act_agent(agent, env, obs_canonical, obs_plane, mask, player):
    base_agent = _unwrap_agent(agent)
    mode = getattr(base_agent, "observation_mode", "absolute")
    if mode == "planes_canonical":
        obs = obs_plane
    elif mode == "canonical":
        obs = obs_canonical
    elif mode == "absolute":
        obs = env.board.copy()
    else:
        raise ValueError(f"Unknown observation mode: {mode}")
    return agent.act(obs, mask, player)


def train(agent=None, opponent=None, steps=1000, env=None, eps=0.1, buffer_size=100000, batch_size=256, target_update_freq=2000, update_every=5, burst_training=False):
    gpu_mem()

    if env is None:
        env = VectorizedConnect4Env(num_envs=B, flatten_obs=True)

    if agent is None:
        B = 256
        obs_dim = 42
        action_dim = 7
        agent = DQNAgent(obs_dim, action_dim, lr=1e-3,
                         gamma=0.99, buffer_size=buffer_size)

    if opponent is None:
        opponent = ParallelVectorAgent(RandomAgent, B, num_workers=8)

    obs, _ = env.reset()
    train_start = batch_size * 2

    loss = 0.0

    pbar = tqdm.tqdm(range(steps))
    for step in pbar:
        player = env.current_player.copy()

        obs_c = env.get_canonical_obs()
        obs_plane = env.board_to_planes(obs_c)
        mask = env.get_action_mask()

        # agent_actions = agent.act(obs_plane, mask, eps)
        agent_actions = _act_agent(agent, env, obs_c, obs_plane, mask, player)
        opponent_actions = _act_agent(
            opponent, env, obs_c, obs_plane, mask, player)
        done_mask = env.done.copy()
        agent_actions[done_mask] = 0
        opponent_actions[done_mask] = 0

        actions = np.where(player == 1, agent_actions, opponent_actions)

        next_obs, reward, done, _, _ = env.step(actions)
        next_mask = env.get_action_mask()

        reward_c = reward * player

        next_player = env.current_player.copy()
        next_obs_c = next_obs * next_player[:, None, None]
        next_obs_plane = env.board_to_planes(next_obs_c)

        agent.buffer.add(obs_plane, actions, reward_c,
                         next_obs_plane, done, next_mask)

        eps = max(0.1, eps * 0.99996)

        obs = next_obs

        if not burst_training:
            if (len(agent.buffer) > train_start):
                if step % update_every == 0:
                    batch = agent.buffer.sample(batch_size)
                    loss = agent.train(batch)

        else:
            if step % 200 == 0:
                for _ in range(10000):
                    batch = agent.buffer.sample(batch_size)
                    loss = agent.train(batch)

        if step % target_update_freq == 0:
            agent.update_target()

        pbar.set_postfix(
            loss=f"{float(loss):.5f}",
            eps=f"{eps:.3f}",
            buffer=len(agent.buffer)
        )

    return agent, opponent, env, loss, eps


if __name__ == "__main__":
    pass
