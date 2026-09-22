import agents.dqn.agent as agent
import agents.dqn.trainer as trainer

DQNAgent = agent.DQNAgent
DistributedDQNTrainer = trainer.DistributedDQNTrainer

__all__ = ["DQNAgent", "DistributedDQNTrainer"]
