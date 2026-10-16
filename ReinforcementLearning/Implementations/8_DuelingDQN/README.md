# Dueling Deep Q-Network

A DQN with separate state-value and action-advantage streams combined into Q-values.

## Files

```text
8_DuelingDQN/
├── agent.py
├── config.py
├── model.py
├── train_and_test.py
├── inference.py
├── run.sh
└── env/
    ├── __init__.py
    └── settings.py
```

This folder is self-contained. Its Q-network uses the TensorFlow Keras Functional API, and action-value inference plus Bellman gradient updates are compiled with `tf.function`. GPU memory growth is enabled before model construction when GPUs are available.

Training uses a compact replay buffer backed by preallocated NumPy arrays and direct vectorized uniform sampling. It also uses Huber loss, global gradient clipping, a linearly decayed epsilon, periodic hard target updates, and random environment seeds at episode resets. Truncation ends an episode but does not incorrectly erase the Bellman bootstrap.

## Environments

The launcher trains these discrete-action, vector-observation environments:

- `CartPole-v1`
- `Acrobot-v1`
- `MountainCar-v0`
- `LunarLander-v3`

`ENVIRONMENT_CONFIGS` in `config.py` provides environment-specific episode budgets, discount factors, replay warmup, epsilon decay, and target update intervals. LunarLander requires Gymnasium's Box2D dependencies.

## Requirements

- Python 3.10 or newer
- TensorFlow
- Gymnasium
- NumPy
- Matplotlib
- ImageIO

## Training

Train every configured environment sequentially:

```bash
sh run.sh train
```

Override run-level values for every environment:

```bash
sh run.sh train --episodes 500 --test-episodes 20 --log-every 25
```

Train one environment directly:

```bash
python train_and_test.py --env CartPole-v1 --episodes 1500
```

Each environment receives its own checkpoint under `checkpoints/`. Checkpoints are selected by the best recent average reward rather than one lucky episode.

## Results

Metrics remain in memory during training and are written afterward:

```text
results/<environment>/
├── training_metrics.csv
├── evaluation_metrics.csv
├── results.png
└── test.gif
```

The figure reports reward, moving reward average, TD loss, episode length, epsilon, replay size, and test rewards. The GIF records the best deterministic test episode.

## Inference

Human-rendered inference loads the selected environment checkpoint:

```bash
sh run.sh inference --env CartPole-v1 --episodes 10
```

The launcher defaults to `/home/shuvrajeet/miniconda3/envs/research/bin/python`. Override it with `PYTHON=python3` when needed.

## GPU selection

Use one physical GPU for training or inference:

```bash
sh run.sh train --gpu 0
sh run.sh train --gpu 1
sh run.sh inference --env CartPole-v1 --gpu 0
```

Use `--gpu -1` to force CPU execution. GPU 0 is the default, and an unavailable index fails before model construction.

## Validation scope

Compilation and short runtime checks establish construction, replay sampling, compiled updates, checkpoint loading, metrics, plots, and GIF generation. They do not prove convergence or GPU performance.

