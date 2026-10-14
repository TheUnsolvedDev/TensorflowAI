import numpy as np


class RandomAgent:
    """A random baseline that avoids actions which cannot move a tile."""

    def __init__(self, action_space, seed=None):
        self.action_space = action_space
        self.random = np.random.default_rng(seed)

    def act(self, observation, info):
        masks = np.asarray(info["action_mask"])
        single_environment = masks.ndim == 1
        if single_environment:
            masks = masks[None, :]

        actions = []
        for mask in masks:
            legal_actions = np.flatnonzero(mask)
            actions.append(
                int(self.random.choice(legal_actions)) if len(legal_actions) else 0
            )
        return actions[0] if single_environment else np.asarray(actions)
