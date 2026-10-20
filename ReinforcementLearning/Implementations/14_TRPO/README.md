# Trust Region Policy Optimization (TRPO)

A discrete-action implementation of textbook TRPO with GAE, a KL Hessian-vector product, conjugate gradient, trust-region step scaling, and backtracking line search. The value function is fitted separately with Adam.

## Files

```text
14_TRPO/
├── agent.py
├── model.py
├── train_and_test.py
├── inference.py
├── run.sh
└── env/
    ├── __init__.py
    └── settings.py
```

The networks in `model.py` use the TensorFlow Keras Functional API. This folder is self-contained and does not import implementation code from sibling folders or the repository root.

## Requirements

- Python 3.10 or newer
- TensorFlow
- Gymnasium
- NumPy

The default environment is `CartPole-v1`. Use only environments with a compatible observation and action space. `LunarLander-v3` requires Gymnasium's optional Box2D dependencies.

## Training and testing

Run from this folder:

```bash
sh run.sh train
```

Choose the number of training and test episodes:

```bash
sh run.sh train --episodes 300 --test-episodes 10
```

Use another compatible Gymnasium environment:

```bash
sh run.sh train --env CartPole-v1 --episodes 300
```

The best completed moving-average checkpoint is saved as `checkpoints/<environment>.weights.h5`. Supply `--checkpoint` to override it.

## Inference

Evaluate a saved checkpoint:

```bash
sh run.sh inference --env CartPole-v1 --episodes 10
```

Inference renders by default. Disable rendering for headless evaluation:

```bash
sh run.sh inference --env CartPole-v1 --no-render
```

## Interpreter

The launcher defaults to:

```text
/home/shuvrajeet/miniconda3/envs/research/bin/python
```

Override it when necessary:

```bash
PYTHON=python3 sh run.sh train
```

## Validation scope

Compilation and short CPU smoke runs only establish that construction, updates, checkpoint loading, and entry points execute. They do not prove convergence, benchmark performance, or GPU correctness.


## Outputs

Each environment writes the same review artifacts:

```text
checkpoints/<environment>.weights.h5
results/<environment>/training_metrics.csv
results/<environment>/evaluation_metrics.csv
results/<environment>/results.png
results/<environment>/test.gif
```

