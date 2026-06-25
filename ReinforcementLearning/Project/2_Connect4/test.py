import os
import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from vectorized_env import VectorizedConnect4Env

from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from agents.dueling_dqn_agent.agent import DuelingDQNAgent
from agents.dueling_dqn_agent.wrapper import DQNWrapper

from config import *


def gpu_mem():
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def unwrap_agent(agent):
    visited = set()

    while hasattr(agent, "agent") and id(agent) not in visited:
        visited.add(id(agent))
        agent = agent.agent

    return agent


def act_agent(agent, obs_canonical, obs_plane, mask, player):
    base_agent = unwrap_agent(agent)

    agent_type = getattr(base_agent, "type", "")

    obs = obs_plane if ("DQN" in agent_type or "CNN" in agent_type) else obs_canonical

    try:
        actions = agent.act(obs, mask, player)

    except TypeError:
        actions = agent.act(obs[0], player[0])

    actions = np.asarray(actions, dtype=np.int32).reshape(-1)

    if actions.size == 0:
        actions = np.zeros((1,), dtype=np.int32)

    return actions


def render_board(board):
    symbols = {1: "X", -1: "O", 0: "."}

    print("\nColumns: 0 1 2 3 4 5 6")

    for r in range(6):
        print(" ".join(symbols[int(board[r, c])] for c in range(7)))

    print()


if __name__ == "__main__":
    gpu_mem()

    env = VectorizedConnect4Env(num_envs=1, flatten_obs=False)

    human = RandomAgent()

    dqn = DuelingDQNAgent(OBS_DIM, ACTION_DIM, lr=LR, gamma=GAMMA, buffer_size=BUFFER_SIZE)

    checkpoints = sorted(
        os.listdir("checkpoints"),
        key=lambda x: int(x.split("_")[-1])
    )

    latest = os.path.join("checkpoints", checkpoints[-1])

    print(f"Loading checkpoint: {latest}")

    dqn.load(latest)

    ai = DQNWrapper(dqn)

    _, _ = env.reset()

    done = np.array([False])

    render_board(env.board[0])

    while not np.all(done):
        player = env.current_player.copy()

        obs_canonical = env.get_canonical_obs()
        print("Canonical Obs:", obs_canonical)
        obs_plane = env.board_to_planes(obs_canonical)
        print(obs_plane.shape)

        mask = env.get_action_mask()

        if player[0] == 1:
            actions = act_agent(human, obs_canonical, obs_plane, mask, player)
        else:
            actions = act_agent(ai, obs_canonical, obs_plane, mask, player)

        if actions.shape[0] != env.B:
            actions = np.full((env.B,), actions[0], dtype=np.int32)

        _, reward, done, _, _ = env.step(actions)

        render_board(env.board[0])

        if done[0]:
            if reward[0] == 1.0:
                winner = player[0]

                if winner == 1:
                    print("Human wins")
                else:
                    print("DQN wins")
            else:
                print("Draw")