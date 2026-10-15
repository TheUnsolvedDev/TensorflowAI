import argparse

import config
import train_and_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=config.ENV_ID)
    parser.add_argument("--episodes", type=int, default=config.TEST_EPISODES)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    train_and_test.configure_tensorflow()
    checkpoint = args.checkpoint or train_and_test.checkpoint_path(args.env)
    train_and_test.evaluate(args.env, args.episodes, args.seed, checkpoint, render=True)


if __name__ == "__main__":
    main()
