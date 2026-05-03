import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from tabulate import tabulate
from environment import TicTacToeEnv
from tournament import TournamentRunner
from training.trainer import Trainer
from agents import random_agent, human_agent, minimax_agent, policy_gradient_agent,actor_critic_agent


def gpu_mem():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def evaluate_and_format(tournament, pg_agent, random_agent, minimax_agent, depth, games_num=500):
    rows = []

    r1 = tournament.run(pg_agent, random_agent, num_games=games_num)
    r2 = tournament.run(random_agent, pg_agent, num_games=games_num)

    rows.append([
        "PG vs Random",
        f"{r1['A_wins']} / {r1['B_wins']} / {r1['draws']}",
        f"{r2['A_wins']} / {r2['B_wins']} / {r2['draws']}"
    ])

    r3 = tournament.run(pg_agent, minimax_agent, num_games=games_num)
    r4 = tournament.run(minimax_agent, pg_agent, num_games=games_num)

    rows.append([
        "PG vs Minimax",
        f"{r3['A_wins']} / {r3['B_wins']} / {r3['draws']}",
        f"{r4['A_wins']} / {r4['B_wins']} / {r4['draws']}"
    ])

    table = tabulate(
        rows,
        headers=[f"Matchup (Depth {depth})", "PG First (W/L/D)", "PG Second (W/L/D)"],
        tablefmt="fancy_grid"
    )

    print(table)
    



if __name__ == "__main__":
    gpu_mem()
    env = TicTacToeEnv(flatten_obs=True)
    tournament = TournamentRunner(env)

    agentA = random_agent.RandomAgent()
    agentB = human_agent.HumanAgent()
    agentC = minimax_agent.MinimaxAgent(3)
    agentD = policy_gradient_agent.PolicyGradientAgent(input_shape=(9,), output_shape=9)
    agentE = actor_critic_agent.ActorCriticAgent(input_shape=(9,), output_shape=9)
    # results = tournament.run(agentA, agentC, num_games=1000)

    depth = 0
    history = []
    minimax_agents = [minimax_agent.MinimaxAgent(d) for d in range(1, 7)]
    episodes = 5000 + depth * 5000
    for _ in range(100):
        if depth == 0:
            opponent = agentA
        else:
            opponent = minimax_agents[depth - 1]
            
        trainer = Trainer(env, agentD, opponent)
        agentD = trainer.train(episodes=episodes)
        evaluate_and_format(tournament, agentD, agentA, opponent, depth)
        
        games = 500
        r1 = tournament.run(agentD, opponent, num_games=games)
        r2 = tournament.run(opponent, agentD, num_games=games)

        score1 = (r1["A_wins"] + 0.5 * r1["draws"]) / games
        score2 = (r2["B_wins"] + 0.5 * r2["draws"]) / games

        avg_score = 0.4 * score1 + 0.6 * score2
        history.append(avg_score)

        avg_score_smooth = np.mean(history[-5:])

        print(f"Depth {depth} | Score1 {score1:.2f} | Score2 {score2:.2f} | Avg {avg_score_smooth:.2f}")
        episodes = 5000 + depth * 5000
        if avg_score_smooth > 0.8 and depth < 6:
            depth += 1
            print(f"→ Increasing difficulty to depth {depth}")
        print()
        
