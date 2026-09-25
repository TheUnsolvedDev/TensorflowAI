from gymnasium.wrappers import RecordEpisodeStatistics


def training_env(render_mode=None, game="breakout"):
    from env.minatar_env import MinAtarEnv

    return RecordEpisodeStatistics(MinAtarEnv(game=game, render_mode=render_mode))
