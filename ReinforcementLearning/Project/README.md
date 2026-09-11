# Reinforcement-learning game projects

← [Reinforcement learning](../README.md)

| Project | Implemented components | Status |
| --- | --- | --- |
| `1_TicTacToe` | Environment, random/human/minimax agents, tournament, policy-gradient and actor-critic agent packages | Source-backed |
| `2_Connect4` | Environment/vectorized environment, human/search agents, tournament, DQN/Double DQN/dueling DQN packages, trainer and test entrypoint | Source-backed |
| `3_Othello` | Directory only | Scaffold |
| `Chess` | Directory only | Scaffold |

Connect4 Q-networks consume a multi-channel board representation and use
residual blocks. Its agent implementations use replay buffers and TensorFlow
gradient updates; traditional agents provide non-learning baselines. The
projects do not ship trained checkpoints or match-result claims.
