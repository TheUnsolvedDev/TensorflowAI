# IMPALA

A compact single-machine actor-learner demonstration centered on the V-trace off-policy correction.

This folder is self-contained and uses TensorFlow Functional models or subclassed TensorFlow models where dynamic quantile sampling is required.

## Train

```bash
sh run.sh train --gpu 0
```

Run one environment:

```bash
python train_and_test.py --env CartPole-v1 --gpu -1
```

## Inference

```bash
sh run.sh inference --env CartPole-v1 --checkpoint checkpoints/CartPole-v1.weights.h5
```

Inference renders by default. Add `--no-render` for headless evaluation.

## Supported environments

- `CartPole-v1`
- `Acrobot-v1`
- `MountainCar-v0`
- `LunarLander-v3`

Short smoke tests do not establish convergence or paper-level benchmark performance.

