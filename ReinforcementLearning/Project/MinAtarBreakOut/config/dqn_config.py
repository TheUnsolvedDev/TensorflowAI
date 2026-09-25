from dataclasses import dataclass


@dataclass
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 0.00025
    replay_size: int = 100_000
    batch_size: int = 32
    warmup_transitions: int = 10_000
    num_envs: int = 4
    training_transitions: int = 3_000_000
    updates_per_iteration: int = 1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_transitions: int = 500_000
    target_update_transitions: int = 10_000
    evaluation_interval_transitions: int = 50_000
    evaluation_episodes: int = 100
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_end: float = 1.0
    priority_epsilon: float = 1e-6
