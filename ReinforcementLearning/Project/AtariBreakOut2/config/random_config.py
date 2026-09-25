from dataclasses import dataclass


@dataclass
class RandomAgentConfig:
    num_envs: int = 4
    max_steps: int = 10
