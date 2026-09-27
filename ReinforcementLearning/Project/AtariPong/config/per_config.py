from dataclasses import dataclass

import config.dqn_config as dqn_config


@dataclass
class PERConfig(dqn_config.DQNConfig):
    priority_epsilon: float = 0.01
