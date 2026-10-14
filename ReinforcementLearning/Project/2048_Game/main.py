"""Run the 2048 environment with a random baseline agent."""

import argparse

import gymnasium as gym
import numpy as np

import env  # noqa: F401 - registers Game2048-v0
from agents import RandomAgent


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def run(
    episodes: int = 1,
    seed: int | None = None,
    render: bool = False,
    fps: int = 4,
    num_envs: int = 1,
) -> None:
    if render and num_envs != 1:
        raise ValueError("--render requires --num-envs 1")
    if num_envs > 1:
        run_vectorized(episodes, seed, num_envs)
        return

    game = gym.make(
        "Game2048-v0",
        render_mode="human" if render else None,
        render_fps=fps,
    )
    agent = RandomAgent(game.action_space, seed)

    for episode in range(episodes):
        observation, info = game.reset(seed=None if seed is None else seed + episode)
        terminated = truncated = False

        while not (terminated or truncated):
            observation, _, terminated, truncated, info = game.step(
                agent.act(observation, info)
            )

        print(
            f"Episode {episode + 1}: score={info['score']}, "
            f"max_tile={info['max_tile']}"
        )

    game.close()


def make_game():
    return gym.make("Game2048-v0")


def run_vectorized(episodes: int, seed: int | None, num_envs: int) -> None:
    game = gym.vector.AsyncVectorEnv([make_game] * num_envs)
    agent = RandomAgent(game.single_action_space, seed)
    completed = np.zeros(num_envs, dtype=int)
    observation, info = game.reset(seed=seed)

    while np.any(completed < episodes):
        observation, _, terminated, truncated, info = game.step(
            agent.act(observation, info)
        )
        for index in np.flatnonzero(terminated | truncated):
            if completed[index] >= episodes:
                continue
            completed[index] += 1
            print(
                f"Environment {index + 1}, episode {completed[index]}: "
                f"score={info['score'][index]}, max_tile={info['max_tile'][index]}"
            )

    game.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Gymnasium 2048 game")
    parser.add_argument("--episodes", type=positive_integer, default=1)
    parser.add_argument("--seed", type=int, help="seed for reproducible games")
    parser.add_argument("--render", action="store_true", help="show the graphical board")
    parser.add_argument("--fps", type=positive_integer, default=4, help="visualization speed")
    parser.add_argument(
        "--num-envs",
        type=positive_integer,
        default=1,
        help="parallel environments (uses separate processes)",
    )
    arguments = parser.parse_args()
    if arguments.render and arguments.num_envs != 1:
        parser.error("--render requires --num-envs 1")
    run(
        arguments.episodes,
        arguments.seed,
        arguments.render,
        arguments.fps,
        arguments.num_envs,
    )
