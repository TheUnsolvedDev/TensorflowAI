import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from tabulate import tabulate

from environment import Connect4Env 

from tournament import TournamentRunner
# from training.trainer import PGTrainer
from agents import random_agent, human_agent, minimax_agent, bitboard_minimax_agent, solver_agent #, policy_gradient_agent, actor_critic_agent


def gpu_mem():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def get_env_specs(env):
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n
    return obs_shape, action_dim


def evaluate_and_format(tournament, agent, random_agent, opponent, depth, games_num=200):
    rows = []

    r1 = tournament.run(agent, random_agent, num_games=games_num)
    r2 = tournament.run(random_agent, agent, num_games=games_num)

    rows.append([
        "Agent vs Random",
        f"{r1['A_wins']} / {r1['B_wins']} / {r1['draws']}",
        f"{r2['B_wins']} / {r2['A_wins']} / {r2['draws']}"
    ])

    r3 = tournament.run(agent, opponent, num_games=games_num)
    r4 = tournament.run(opponent, agent, num_games=games_num)

    rows.append([
        "Agent vs Minimax",
        f"{r3['A_wins']} / {r3['B_wins']} / {r3['draws']}",
        f"{r4['B_wins']} / {r4['A_wins']} / {r4['draws']}"
    ])

    table = tabulate(
        rows,
        headers=[f"Depth {depth}", "First (W/L/D)", "Second (W/L/D)"],
        tablefmt="fancy_grid"
    )

    print(table)


def compute_score(r1, r2, games):
    score1 = (r1["A_wins"] + 0.5 * r1["draws"]) / games
    score2 = (r2["B_wins"] + 0.5 * r2["draws"]) / games

    # equal weighting is cleaner
    return 0.5 * (score1 + score2)


if __name__ == "__main__":
    gpu_mem()

    env = Connect4Env(flatten_obs=True)
    tournament = TournamentRunner(env)

    obs_shape, action_dim = get_env_specs(env)

    agent_random = random_agent.RandomAgent()
    agent_human = human_agent.HumanAgent()
    agent_minimax = [minimax_agent.MinimaxAgent(d) for d in range(1, 7)]
    agent_bitboard_minimax = [bitboard_minimax_agent.BitboardMinimaxAgent(d) for d in range(1, 7)]
    agent_solver_agent = [solver_agent.SolverAgent(d, time_limit=5) for d in range(1, 7)]
    
    r1 = tournament.run(agent_random, agent_random, num_games=1000)
    r2 = tournament.run(agent_bitboard_minimax[5], agent_random, num_games=1000)
    r3 = tournament.run(agent_random, agent_bitboard_minimax[5], num_games=1000)
    r4 = tournament.run(agent_solver_agent[5], agent_random, num_games=1000)
    r5 = tournament.run(agent_random, agent_solver_agent[5], num_games=1000)

    print(r1, r2, r3, r4, r5)
