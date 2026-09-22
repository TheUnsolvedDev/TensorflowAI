import argparse
import functools
from pathlib import Path
import agents.evaluate as evaluate
import agents.random_agent as random_agent
import config.random_config as random_config
import env.wrappers as wrappers
import gymnasium as gym
import agents.distributed as distributed
import training.dqn as dqn_training


def latest_weights(model_name, requested=None):
    if requested:
        return requested
    directory = Path("checkpoints") / model_name
    directory.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(
        directory.glob("step_*.weights.h5"),
        key=lambda path: int(path.stem.split("_")[1].split(".")[0]),
    )
    latest = checkpoints[-1] if checkpoints else directory / "pong.weights.h5"
    return str(latest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--game", default="ALE/Pong-v5")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epsilon-decay-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model", choices=("dqn", "double", "dueling", "per"), default="dqn")
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    if args.visualize:
        import agents.dqn.trainer as dqn_trainer
        import models.dqn as dqn
        import models.double_dqn as double_dqn
        import models.dueling_dqn as dueling_dqn
        import models.per as per
        probe = wrappers.training_env(game=args.game, render_mode="human")
        builders = {"dqn": dqn, "double": double_dqn, "dueling": dueling_dqn, "per": per}
        strategy = distributed.nccl_strategy()
        with strategy.scope():
            model = builders[args.model].build_model(probe.action_space.n)
        weights = latest_weights(args.model, args.weights)
        if not Path(weights).exists():
            raise FileNotFoundError(
                f"No checkpoint found for {args.model}. Train it first or pass --weights. "
                f"Expected: {weights}"
            )
        model.load_weights(weights)
        print(f"visualizing weights={weights}")
        agent = dqn_trainer.DistributedDQNTrainer(model, strategy)
        probe.close()
        evaluate.test_agent(
            agent,
            args.episodes,
            lambda: wrappers.training_env(game=args.game, render_mode="human"),
        )
        return

    if args.train:
        dqn_training.train(
            args.steps or 1_000_000,
            args.model,
            args.num_envs,
            args.epsilon_decay_steps,
            args.eval_steps,
            args.eval_every,
            args.game,
        )
        return

    config = random_config.RandomAgentConfig()
    num_envs = args.num_envs or config.num_envs
    steps = args.steps or config.max_steps
    if num_envs < 1 or steps < 1 or args.episodes < 1:
        parser.error("--num-envs, --steps, and --episodes must be positive")

    make_env = functools.partial(wrappers.training_env, game=args.game)
    env = gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])
    agent = random_agent.RandomAgent(env.single_action_space)
    try:
        states, _ = env.reset(seed=args.seed)
        for step in range(steps):
            actions = agent.act_batch(states)
            states, rewards, terminated, truncated, _ = env.step(actions)
            print(f"step={step + 1} rewards={rewards} terminated={terminated}")
        print(f"finished {num_envs} environments")
    finally:
        env.close()


if __name__ == "__main__":
    main()
