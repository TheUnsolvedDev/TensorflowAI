from dataclasses import dataclass


@dataclass
class DQNConfig:
    state_size: int = 4
    action_size: int = 4
    max_steps: int = 10
    gamma: float = 0.99
    learning_rate: float = 0.00025
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    replay_size: int = 100_000
    batch_size: int = 32
    warmup_steps: int = 10_000
    num_envs: int = 32
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 300_000
    target_update_steps: int = 10_000
