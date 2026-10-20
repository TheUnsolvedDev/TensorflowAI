# Dreamer

A compact vector-observation Dreamer demonstration with a learned latent world model and imagined actor updates.

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

Short smoke tests do not establish convergence or paper-level benchmark performance.

## Outputs

Each environment writes the same review artifacts:

```text
checkpoints/<environment>.weights.h5
results/<environment>/training_metrics.csv
results/<environment>/evaluation_metrics.csv
results/<environment>/results.png
results/<environment>/test.gif
```

