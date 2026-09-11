# Vectorized value-based reinforcement learning

← [Reinforcement learning](../README.md)

`DQN/` and `DoubleDQN/` implement the same environment set with batched
environment wrappers, replay buffers, Q-networks, agents, configurations, and
training entrypoints.

| Algorithm | Target | TensorFlow implementation |
| --- | --- | --- |
| DQN | \(r + \gamma\max_a Q_{\bar\theta}(s',a)\) | Tensor batches, replay sampling, target networks, `GradientTape`, `tf.function` |
| Double DQN | \(r + \gamma Q_{\bar\theta}(s',\arg\max_a Q_\theta(s',a))\) | Separate online/target action roles in the agent update |

`CartPole` and `Acrobot` include test scripts. Training files also contain
TensorBoard summary code. No training score is documented here because the
repository does not retain a benchmark report as source.
