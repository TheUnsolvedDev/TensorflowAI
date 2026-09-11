# 🎮 Reinforcement Learning

← [Back to the repository](../README.md)

The reinforcement-learning code is organised by learning rule and execution
style: single-environment policy optimisation, vectorized value learning, and
game-playing projects. Gymnasium environments and local game environments are
runtime dependencies; trained agents and TensorBoard logs are intentionally
untracked.

| Family | Environments / projects | Implementation evidence |
| --- | --- | --- |
| Policy gradient and critic variants | Acrobot, CartPole, LunarLander, MountainCar | Policy networks, trajectory buffers, returns, `GradientTape`, selected critics |
| REINFORCE | Acrobot, CartPole, LunarLander, MountainCar | Monte-Carlo policy-gradient agents and simulator/train scripts |
| Vectorized DQN | Acrobot, CartPole, LunarLander, MountainCar | Q-networks, replay buffers, vectorized environments, compiled updates |
| Vectorized Double DQN | Acrobot, CartPole, LunarLander, MountainCar | Double-Q targets, replay, TensorBoard-capable trainers |
| Games | Tic-Tac-Toe and Connect4 | Search/random/human agents plus policy-gradient, DQN, Double DQN, and dueling-DQN agents |

## Core mapping

For policy methods, trajectories supply discounted returns and a categorical
policy is updated with `tf.GradientTape`. For value methods, replay batches
train Q-networks against bootstrapped targets; Double DQN separates action
selection from target evaluation. Vectorized folders use batched environment
state and `tf.function` in the hot update path.

## Status boundaries

`Project/3_Othello` and `Project/Chess` are directory placeholders in the
current tracked checkout; they are not listed as implemented agents. Several
single-environment `reinforce.py` files are explicit stubs, so documentation
will point readers to the working `agent.py`/`train.py` path instead.
