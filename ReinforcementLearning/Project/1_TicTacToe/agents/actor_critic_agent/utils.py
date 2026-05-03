import numpy as np


def mask_invalid_actions(probs, state):
    mask = (state == 0).astype(np.float32)
    probs = probs * mask

    if probs.sum() == 0:
        return mask / mask.sum()

    return probs / probs.sum()


def sample_action(probs):
    return np.random.choice(len(probs), p=probs)

def canonicalize(state, player):
    return state * player