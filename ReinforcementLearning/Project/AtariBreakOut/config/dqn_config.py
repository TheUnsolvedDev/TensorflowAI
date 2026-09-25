from dataclasses import dataclass


@dataclass
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 0.0001
    replay_size: int = 200_000
    batch_size: int = 32
    warmup_transitions: int = 50_000
    num_envs: int = 8
    training_transitions: int = 10_000_000
    updates_per_iteration: int = 2
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_transitions: int = 1_000_000
    target_update_transitions: int = 10_000
    evaluation_interval_transitions: int = 250_000
    evaluation_episodes: int = 10
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_end: float = 1.0
    priority_epsilon: float = 1e-6
