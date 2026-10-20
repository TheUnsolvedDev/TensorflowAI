# Distributed Distributional DDPG

A compact single-process D4PG learner with a deterministic actor and categorical distributional critic.

This folder is self-contained and uses TensorFlow Functional models or subclassed TensorFlow models where dynamic quantile sampling is required.

## Train

```bash
sh run.sh train --gpu 0
```

Run one environment:

```bash
python train_and_test.py --env Pendulum-v1 --gpu -1
```

## Inference

```bash
sh run.sh inference --env Pendulum-v1 --checkpoint checkpoints/Pendulum-v1.weights.h5
```

Inference renders by default. Add `--no-render` for headless evaluation.

## Supported environments

- `Pendulum-v1`
- `MountainCarContinuous-v0`
- `LunarLanderContinuous-v3`
- `BipedalWalker-v3`
- `HalfCheetah-v5`
- `Hopper-v5`
- `Walker2d-v5`
- `Ant-v5`

The MuJoCo environments require `gymnasium[mujoco]`.

Short smoke tests do not establish convergence or paper-level benchmark performance.

