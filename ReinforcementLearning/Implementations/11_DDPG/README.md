# Deep Deterministic Policy Gradient (DDPG)

A deterministic continuous-control actor-critic with replay memory and soft target updates.

## Files

```text
11_DDPG/
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

The default environment is `Pendulum-v1`. Use only environments with a compatible observation and action space. `LunarLander-v3` requires Gymnasium's optional Box2D dependencies.

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
sh run.sh train --env Pendulum-v1 --episodes 300
```

The best episode checkpoint is saved as `best.weights.h5` by default. Supply `--checkpoint` to use another path.

## Inference

Evaluate a saved checkpoint:

```bash
sh run.sh inference --checkpoint best.weights.h5 --episodes 10
```

Render evaluation episodes:

```bash
sh run.sh inference --checkpoint best.weights.h5 --render
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

