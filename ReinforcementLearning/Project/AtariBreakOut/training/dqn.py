import numpy as np
import csv
import gymnasium as gym
import sys
import functools
from datetime import datetime
from pathlib import Path
import tensorflow as tf

import agents.distributed as distributed
import agents.dqn.replay_buffer as replay_buffer
import agents.dqn.trainer as dqn_trainer
import config.dqn_config as dqn_config
import env.wrappers as wrappers
import models.dqn as dqn


def train(
    steps=None,
    model_name="dqn",
    num_envs=None,
    epsilon_decay_steps=None,
    eval_episodes=None,
    eval_every=None,
    game="ALE/Breakout-v5",
    seed=None,
):
    config = dqn_config.DQNConfig()
    if steps is None:
        steps = config.training_transitions
    if num_envs is not None:
        config.num_envs = num_envs
    if epsilon_decay_steps is not None:
        config.epsilon_decay_transitions = epsilon_decay_steps
    if eval_episodes is not None:
        config.evaluation_episodes = eval_episodes
    if eval_every is not None:
        config.evaluation_interval_transitions = eval_every
    if min(steps, config.num_envs, config.evaluation_episodes) < 1:
        raise ValueError("steps, num_envs, and evaluation episodes must be positive")
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
    make_env = functools.partial(wrappers.training_env, game=game)
    env = gym.vector.AsyncVectorEnv(
        [make_env for _ in range(config.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_dir = Path("runs") / model_name / run_id
    model_checkpoint_dir = Path("checkpoints") / model_name
    checkpoint_dir = model_checkpoint_dir / run_id
    best_weights_path = model_checkpoint_dir / "best.weights.h5"
    best_score_path = model_checkpoint_dir / "best_score.txt"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = (log_dir / "metrics.csv").open("w", newline="")
    metrics = csv.writer(metrics_file)
    metrics.writerow(("step", "loss", "average_reward"))
    writer = tf.summary.create_file_writer(str(log_dir))
    strategy = distributed.nccl_strategy()
    with strategy.scope():
        builders = {
            "dqn": dqn,
            "double": __import__("models.double_dqn", fromlist=["build_model"]),
            "dueling": __import__("models.dueling_dqn", fromlist=["build_model"]),
            "per": __import__("models.per", fromlist=["build_model"]),
        }
        model = builders[model_name].build_model(env.single_action_space.n)
        trainer = dqn_trainer.DistributedDQNTrainer(
            model,
            strategy,
            config.gamma,
            config.learning_rate,
            double=model_name == "double",
        )
    model.summary()

    if model_name == "per":
        buffer = replay_buffer.PrioritizedReplayBuffer(
            config.replay_size,
            config.priority_alpha,
            config.priority_epsilon,
            seed,
        )
    else:
        buffer = replay_buffer.ReplayBuffer(config.replay_size)
    states, _ = env.reset(seed=seed)
    best_score = (
        float(best_score_path.read_text()) if best_score_path.exists() else float("-inf")
    )
    environment_steps = 0
    next_target_update = config.target_update_transitions
    next_evaluation = config.evaluation_interval_transitions
    try:
        while environment_steps < steps:
            progress = min(1.0, environment_steps / config.epsilon_decay_transitions)
            epsilon = config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)
            actions = trainer.act(states, epsilon).numpy()
            next_states, rewards, terminated, truncated, _ = env.step(actions)
            for transition in zip(states, actions, rewards, next_states, terminated | truncated):
                buffer.add(*transition)
            states = next_states
            environment_steps += config.num_envs
            if environment_steps >= next_target_update:
                trainer.update_target()
                next_target_update += config.target_update_transitions
            if len(buffer) >= max(config.batch_size, config.warmup_transitions):
                for _ in range(config.updates_per_iteration):
                    if model_name == "per":
                        beta_progress = min(1.0, environment_steps / steps)
                        beta = config.priority_beta_start + beta_progress * (
                            config.priority_beta_end - config.priority_beta_start
                        )
                        batch = buffer.sample(config.batch_size, beta)
                    else:
                        batch = buffer.sample(config.batch_size)
                    loss, errors = trainer.train_batch(
                        *[np.asarray(value) for value in batch]
                    )
                    metrics.writerow((environment_steps, float(loss), ""))
                    if model_name == "per":
                        buffer.update_priorities(errors.numpy())
                if environment_steps % 400 == 0:
                    loss_value = float(loss)
                    metrics_file.flush()
                    percent = 100 * environment_steps / steps
                    print(
                        f"transitions {environment_steps}/{steps} [{percent:6.2f}%] "
                        f"epsilon={epsilon:.3f} loss={loss_value:.5f}",
                        end="\r",
                        flush=True,
                    )
                    with writer.as_default():
                        tf.summary.scalar("loss", loss, step=environment_steps)
                        tf.summary.scalar("epsilon", epsilon, step=environment_steps)
                        writer.flush()
            if environment_steps >= next_evaluation:
                print()
                score = evaluate(
                    trainer,
                    config.num_envs,
                    config.evaluation_episodes,
                    game,
                    seed,
                )
                metrics.writerow((environment_steps, "", score))
                metrics_file.flush()
                print(
                    f"transitions={environment_steps} "
                    f"average evaluation reward={score:.2f}"
                )
                if score > best_score:
                    best_score = score
                    model.save_weights(best_weights_path)
                    best_score_path.write_text(str(best_score))
                    print(f"new best evaluation reward={score:.2f}")
                next_evaluation += config.evaluation_interval_transitions
    finally:
        env.close()
        metrics_file.close()
        writer.close()

    sys.stdout.write("\n")

    weights_path = checkpoint_dir / "breakout.weights.h5"
    model.save_weights(weights_path)
    average_reward = evaluate(
        trainer, config.num_envs, config.evaluation_episodes, game, seed
    )
    if average_reward > best_score:
        best_score = average_reward
        model.save_weights(best_weights_path)
        best_score_path.write_text(str(best_score))
    print(f"average evaluation reward={average_reward:.2f}")
    print(f"saved weights={weights_path}")
    print(f"best evaluation reward={best_score:.2f}")
    print(f"tensorboard logs={log_dir}")
    return average_reward


def evaluate(trainer, num_envs, episodes, game="ALE/Breakout-v5", seed=None):
    make_env = functools.partial(wrappers.training_env, game=game)
    env = gym.vector.AsyncVectorEnv(
        [make_env for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    states, _ = env.reset(seed=seed)
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    completed_rewards = []
    try:
        while len(completed_rewards) < episodes:
            actions = trainer.act(states).numpy()
            states, rewards, terminated, truncated, _ = env.step(actions)
            episode_rewards += rewards
            done = terminated | truncated
            for index in np.flatnonzero(done):
                if len(completed_rewards) < episodes:
                    completed_rewards.append(float(episode_rewards[index]))
                episode_rewards[index] = 0.0
    finally:
        env.close()
    return float(np.mean(completed_rewards))
