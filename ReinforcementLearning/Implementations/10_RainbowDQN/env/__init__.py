"""Shared environment settings for this algorithm."""

from .settings import DEFAULT_ENV, ENVIRONMENTS, get_env_settings
from .reward_wrapper import make_training_env

__all__ = ["DEFAULT_ENV", "ENVIRONMENTS",
           "get_env_settings", "make_training_env"]
