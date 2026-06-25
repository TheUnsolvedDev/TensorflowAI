import multiprocessing as mp
import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import os
import time

from collections import deque
from datetime import datetime
from tabulate import tabulate

from vectorized_env import VectorizedConnect4Env
from tournament import VectorizedTournamentRunner

from agents import random_agent, bitboard_minimax_agent
from agents.vector_agent import ParallelVectorAgent
from agents.dueling_dqn_agent.agent import DuelingDQNAgent
from agents.dueling_dqn_agent.wrapper import DQNWrapper

from training.trainer import train
from config import *


def gpu_mem():
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def fmt(r):
    return f"{r['A_wins']}/{r['B_wins']}/{r['draws']}"


def score(r):
    return (r["A_wins"] + 0.5 * r["draws"]) / r["total_games"]


def combined_score(r1, r2, r3, r4):
    random_score = (score(r1) + score(r2)) * 0.5
    minimax_score = (score(r3) + score(r4)) * 0.5

    return 0.4 * random_score + 0.6 * minimax_score


def build_table(r1, r2, r3, r4, depth):
    return [
        ["DQN vs Random", fmt(r1), fmt(
            r2), f"{r1['invalid_games'] + r2['invalid_games']}", f"{score(r1):.3f} / {score(r2):.3f}"],
        [f"DQN vs Minimax D{depth}", fmt(r3), fmt(
            r4), f"{r3['invalid_games'] + r4['invalid_games']}", f"{score(r3):.3f} / {score(r4):.3f}"]
    ]


class ParallelRandomAgent:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.type = "RandomPallel"

    def act(self, obs, mask=None, player=None):
        batch_size, num_actions = mask.shape
        probs = mask.astype(np.float32)
        counts = probs.sum(axis=1, keepdims=True)
        counts = np.where(counts == 0, 1.0, counts)
        probs = probs / counts
        rand = np.random.rand(batch_size, 1)
        actions = (np.cumsum(probs, axis=1) > rand).argmax(axis=1)
        no_valid = counts[:, 0] == 0
        actions[no_valid] = 0
        return actions.astype(np.int32)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    gpu_mem()

    log_dir = f"logs/connect4/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = tf.summary.create_file_writer(log_dir)
    os.makedirs("checkpoints", exist_ok=True)
    loss_window = deque(maxlen=100)
    random_score_window = deque(maxlen=20)
    minimax_score_window = deque(maxlen=20)

    env = VectorizedConnect4Env(num_envs=NUM_PARALLEL_ENVS, flatten_obs=False)
    eval_env = VectorizedConnect4Env(
        num_envs=NUM_PARALLEL_ENVS, flatten_obs=False)
    tournament = VectorizedTournamentRunner(eval_env)

    current_depth = 5
    agent_random = ParallelVectorAgent(
        random_agent.RandomAgent, NUM_PARALLEL_ENVS, num_workers=8)
    opponent_minimax = ParallelVectorAgent(
        bitboard_minimax_agent.BitboardMinimaxAgent, NUM_PARALLEL_ENVS, num_workers=8, depth=current_depth)
    eval_minimax = ParallelVectorAgent(
        bitboard_minimax_agent.BitboardMinimaxAgent, NUM_PARALLEL_ENVS, num_workers=8, depth=current_depth+3)
    
    a1 = tournament.run(agent_random, opponent_minimax, num_batches=1)
    exit()
    a2 = tournament.run(opponent_minimax, agent_random, num_batches=100)
    a3 = tournament.run(agent_random, agent_random, num_batches=100)
    a4 = tournament.run(opponent_minimax, eval_minimax, num_batches=100)
    a5 = tournament.run(eval_minimax, opponent_minimax, num_batches=100)
    print("Initial Random vs Minimax D1:", fmt(a1), fmt(a2), fmt(a3), fmt(a4), fmt(a5))
    exit()

    eps = EPSILON_START
    agent = DuelingDQNAgent(OBS_DIM, ACTION_DIM, lr=LR,
                     gamma=GAMMA, buffer_size=BUFFER_SIZE)
    agent.q_net.summary()
    opponent_random = ParallelRandomAgent(NUM_PARALLEL_ENVS)

    print("\n===== TRAINING START =====\n")
    for i in range(0, HORIZON + 1, TRAIN_STEPS):
        if i < HORIZON // 4:
            opponent = opponent_random
            opponent_name = "RANDOM"
        else:
            opponent = opponent_minimax
            opponent_name = f"MINIMAX D{current_depth}"

        print(f"\n=== TRAINING AGAINST {opponent_name} | STEP {i} ===")
        start_time = time.time()
        agent, opponent, env, loss, eps = train(agent=agent, opponent=opponent, steps=TRAIN_STEPS, env=env,eps=eps, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ, update_every=UPDATE_EVERY, burst_training=BURST_TRAINING)
        elapsed = time.time() - start_time

        replay_size = len(agent.buffer)

        # loss_window.append(loss)

        wrapped_agent = DQNWrapper(agent)

        r1 = tournament.run(wrapped_agent, agent_random,
                            num_batches=EVAL_INTERVAL)
        r2 = tournament.run(agent_random, wrapped_agent,
                            num_batches=EVAL_INTERVAL)
        r3 = tournament.run(wrapped_agent, eval_minimax,
                            num_batches=EVAL_INTERVAL)
        r4 = tournament.run(eval_minimax, wrapped_agent,
                            num_batches=EVAL_INTERVAL)

        table = build_table(r1, r2, r3, r4, current_depth)
        random_score = (r1["A_wins"] + r2["B_wins"]) / \
            (r1["total_games"] + r2["total_games"])
        minimax_score = (r3["A_wins"] + r4["B_wins"]) / \
            (r3["total_games"] + r4["total_games"])
        combined = combined_score(r1, r2, r3, r4)

        random_score_window.append(random_score)
        minimax_score_window.append(minimax_score)

        if combined > 0.75 and current_depth < 8:
            current_depth += 1
            print(f"\n[Depth Increased] -> {current_depth}")
            opponent_minimax = ParallelVectorAgent(
                bitboard_minimax_agent.BitboardMinimaxAgent, NUM_PARALLEL_ENVS, num_workers=8, depth=current_depth)
            eval_minimax = ParallelVectorAgent(
                bitboard_minimax_agent.BitboardMinimaxAgent, NUM_PARALLEL_ENVS, num_workers=8, depth=current_depth)

        with writer.as_default():
            tf.summary.scalar("train/loss", loss, step=i)
            tf.summary.scalar("train/loss_avg_100",
                              np.mean(loss_window), step=i)
            tf.summary.scalar("train/epsilon", eps, step=i)
            tf.summary.scalar("buffer/size", replay_size, step=i)
            tf.summary.scalar("eval/random_winrate", random_score, step=i)
            tf.summary.scalar("eval/random_winrate_avg",
                              np.mean(random_score_window), step=i)
            tf.summary.scalar("eval/minimax_winrate", minimax_score, step=i)
            tf.summary.scalar("eval/minimax_winrate_avg",
                              np.mean(minimax_score_window), step=i)
            tf.summary.scalar("eval/combined_score", combined, step=i)
            tf.summary.scalar("eval/minimax_depth", current_depth, step=i)
            tf.summary.scalar("system/steps_per_second",
                              TRAIN_STEPS / elapsed, step=i)
            writer.flush()

        print(f"\n[Iteration {i}]")
        print(f"Loss               : {loss:.6f}")
        print(f"Loss Avg (100)     : {np.mean(loss_window):.6f}")
        print(f"Epsilon            : {eps:.4f}")
        print(f"Replay Buffer Size : {replay_size}")
        print(f"Random Score       : {random_score:.4f}")
        print(f"Minimax Score      : {minimax_score:.4f}")
        print(f"Combined Score     : {combined:.4f}")
        print(f"Current Depth      : {current_depth}")
        print(f"Steps/sec          : {TRAIN_STEPS / elapsed:.2f}")
        print(tabulate(table, headers=[
              "Match", "First (W/L/D)", "Second (W/L/D)", "Invalid", "Score"], tablefmt="fancy_grid"))

        if i % 100 == 0 and i > 0:
            ckpt_path = f"checkpoints/step_{i}"
            agent.save(ckpt_path)
            print(f"\n[Checkpoint Saved] {ckpt_path}")

    print("\n===== TRAINING COMPLETE =====\n")

    try:
        agent_random.pool.close()
        agent_random.pool.join()
        opponent_minimax.pool.close()
        opponent_minimax.pool.join()
        eval_minimax.pool.close()
        eval_minimax.pool.join()
    except:
        pass
