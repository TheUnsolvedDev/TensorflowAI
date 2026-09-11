# Single-environment policy learning

← [Reinforcement learning](../README.md)

The `PolicyGradient/` and `REINFORCE/` tracks repeat the same four Gymnasium
environments—Acrobot, CartPole, LunarLander, and MountainCar—to make the
learning-rule comparison explicit.

| Track | Update idea | Entrypoints |
| --- | --- | --- |
| `PolicyGradient` | Discounted-return policy updates; selected folders also expose critic code | `train.py`, `simulate.py`, `agent.py`, `trajectory_buffer.py` |
| `REINFORCE` | Monte-Carlo return-weighted log-policy updates | `train.py`, `simulate.py`, `agent.py`, `trajectory_buffer.py` |

For a trajectory \(s_t,a_t,r_t\), both tracks are built around discounted
returns \(G_t = \sum_{k\ge0}\gamma^k r_{t+k}\) and an update proportional to
\(\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t\). Runtime settings,
episode counts, and learning rates are defined in each environment's
`config.py`.

The standalone `reinforce.py` files in the policy-gradient folders are
placeholders; do not treat them as alternate runnable implementations.
