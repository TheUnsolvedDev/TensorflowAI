import numpy as np
import gymnasium as gym
import shutil
import sys
import functools
from pathlib import Path
import tensorflow as tf

import agents.distributed as distributed
import agents.dqn.replay_buffer as replay_buffer
import agents.dqn.trainer as dqn_trainer
import config.dqn_config as dqn_config
import env.wrappers as wrappers
import models.dqn as dqn


def train(
    steps=10_000,
    model_name="dqn",
    num_envs=None,
    epsilon_decay_steps=None,
    eval_steps=None,
    eval_every=10_000,
    game="ALE/Breakout-v5",
    seed=None,
):
    config = dqn_config.DQNConfig()
    if num_envs is not None:
        config.num_envs = num_envs
    if epsilon_decay_steps is not None:
        config.epsilon_decay_steps = epsilon_decay_steps
    make_env = functools.partial(wrappers.training_env, game=game)
    env = gym.vector.AsyncVectorEnv([make_env for _ in range(config.num_envs)])
    log_dir = Path("runs") / model_name
    shutil.rmtree(log_dir, ignore_errors=True)
    log_dir.mkdir(parents=True, exist_ok=True)
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

    buffer_type = replay_buffer.PrioritizedReplayBuffer if model_name == "per" else replay_buffer.ReplayBuffer
    buffer = buffer_type(config.replay_size)
    states, _ = env.reset(seed=seed)
    best_score = float("-inf")
    try:
        for step in range(steps):
            progress = min(1.0, step / config.epsilon_decay_steps)
            epsilon = config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)
            actions = trainer.act(states, epsilon).numpy()
            next_states, rewards, terminated, truncated, _ = env.step(actions)
            for transition in zip(states, actions, rewards, next_states, terminated | truncated):
                buffer.add(*transition)
            states = next_states
            if (step + 1) % config.target_update_steps == 0:
                trainer.update_target()
            if len(buffer) >= max(config.batch_size, config.warmup_steps):
                batch = buffer.sample(config.batch_size)
                loss, errors = trainer.train_batch(*[np.asarray(value) for value in batch])
                if model_name == "per":
                    buffer.update_priorities(errors.numpy())
                if step % 100 == 0:
                    percent = 100 * (step + 1) / steps
                    print(
                        f"step {step + 1}/{steps} [{percent:6.2f}%] "
                        f"epsilon={epsilon:.3f} loss={float(loss):.5f}",
                        end="\r",
                        flush=True,
                    )
                    with writer.as_default():
                        tf.summary.scalar("loss", loss, step=step)
                        tf.summary.scalar("epsilon", epsilon, step=step)
                        writer.flush()
            if (step + 1) % eval_every == 0:
                print()
                weights_path = Path("checkpoints") / model_name / f"step_{step + 1}.weights.h5"
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                model.save_weights(weights_path)
                score = evaluate(model, strategy, config.num_envs, eval_steps or eval_every, game)
                print(f"step={step + 1} average evaluation reward={score:.2f}")
                if score > best_score:
                    best_score = score
                    model.save_weights(Path("checkpoints") / model_name / "best.weights.h5")
                    print(f"new best evaluation reward={score:.2f}")
    finally:
        env.close()

    sys.stdout.write("\n")

    weights_path = Path("checkpoints") / model_name / "breakout.weights.h5"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(weights_path)
    average_reward = evaluate(model, strategy, config.num_envs, eval_steps or eval_every, game)
    if average_reward > best_score:
        model.save_weights(Path("checkpoints") / model_name / "best.weights.h5")
        best_score = average_reward
    print(f"average evaluation reward={average_reward:.2f}")
    print(f"saved weights={weights_path}")
    print(f"best evaluation reward={best_score:.2f}")
    print(f"tensorboard logs={log_dir}")
    return average_reward


def evaluate(model, strategy, num_envs, steps, game="ALE/Breakout-v5"):
    make_env = functools.partial(wrappers.training_env, game=game)
    env = gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])
    trainer = dqn_trainer.DistributedDQNTrainer(model, strategy)
    states, _ = env.reset()
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    completed_rewards = []
    try:
        for _ in range(steps):
            actions = trainer.act(states).numpy()
            states, rewards, terminated, truncated, _ = env.step(actions)
            episode_rewards += rewards
            done = terminated | truncated
            for index in np.flatnonzero(done):
                completed_rewards.append(float(episode_rewards[index]))
                episode_rewards[index] = 0.0
    finally:
        env.close()
    return sum(completed_rewards) / len(completed_rewards) if completed_rewards else float(episode_rewards.mean())
