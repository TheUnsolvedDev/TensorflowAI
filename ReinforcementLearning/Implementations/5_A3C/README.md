# Asynchronous Advantage Actor-Critic (A3C)

An asynchronous advantage actor-critic implementation with independent workers and one global network.

## Files

```text
5_A3C/
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

`model.py` defines one shared actor-value network with the TensorFlow Keras Functional API. Independent workers run separate environments and keep local copies of this network. Every worker collects a short rollout, computes n-step returns and advantage gradients, applies those gradients asynchronously to the global network without a global update lock, and synchronizes its local weights.

Both the action-probability forward pass and gradient calculation are compiled with `tf.function`. Huber value loss, clipped normalized advantages, entropy regularization, and global gradient clipping keep updates stable. GPU memory growth is enabled before model construction when GPUs are available. This folder is self-contained and does not import implementation code from sibling folders or the repository root.

Default environment, model, optimizer, training, checkpoint, logging, plotting, and TensorFlow settings are centralized in `config.py`. Command-line arguments can override the run-specific defaults without editing the file.

Training uses one agent run. A generator initialized by `--seed` produces a fresh random environment seed for every explicit episode reset, and A3C workers receive independent random streams. This exposes the agent to varied initial states. The streams are repeatable, although asynchronous worker scheduling is not guaranteed to be bitwise deterministic. Evaluation and GIF seeds remain separate from training.

`ENVIRONMENT_CONFIGS` supplies environment-specific episode, discount, entropy, and rollout settings without changing the A3C update.

Acrobot and both MountainCar variants use potential-based rewards only for learning. Logged and evaluated rewards remain the original Gymnasium rewards. Four workers are used by default.

## Requirements

- Python 3.10 or newer
- TensorFlow
- Gymnasium
- NumPy
- ImageIO

The default environment is `CartPole-v1`. Vector Box observations with either Discrete or bounded Box actions are supported. Discrete actions use a categorical policy; continuous actions use a bounded Gaussian policy. Box2D environments require Gymnasium's optional Box2D dependencies.

## Training and testing

Run all configured environments sequentially:

```bash
sh run.sh train
```

The launcher trains:

- `CartPole-v1`
- `Acrobot-v1`
- `MountainCar-v0`
- `Pendulum-v1`
- `MountainCarContinuous-v0`
- `LunarLander-v3`
- `LunarLanderContinuous-v3`
- `BipedalWalker-v3`

Choose the number of training and test episodes:

```bash
sh run.sh train --episodes 300 --test-episodes 10
```

Terminal progress is printed every 10 episodes by default. Change the interval without changing the training update frequency:

```bash
sh run.sh train --episodes 300 --log-every 25
```

Run one compatible environment directly when needed:

```bash
python train_and_test.py --env CartPole-v1 --episodes 300
```

Choose the asynchronous worker count and rollout length:

```bash
sh run.sh train --num-workers 4 --rollout-steps 64
```

Each environment receives one global actor-value checkpoint under `checkpoints/`, such as `checkpoints/CartPole_v1.weights.h5`. Supply `--checkpoint` when running `train_and_test.py` directly to use another path.

## Metrics and graphs

CSV metrics and graphs are enabled by default. Metrics stay in memory while the agent trains, then the CSV files are written after training and evaluation finish. The final graph reads its values from those CSV files. Each environment has one results directory that is overwritten by its next completed run:

```text
results/<environment>/
├── training_metrics.csv
├── evaluation_metrics.csv
├── results.png
└── test.gif
```

The default final figure contains training reward, the configured moving reward average, combined actor-value loss, episode length, and evaluation rewards. Checkpoints are selected by recent average reward instead of one lucky episode. After evaluation, `test.gif` records the highest-reward deterministic episode from `GIF_TEST_EPISODES` candidate seeds. GIF settings are available in `config.py`. Select another results root with `--output-dir`:

```bash
sh run.sh train --output-dir experiment_results
```

## Inference

Evaluate a saved checkpoint:

```bash
sh run.sh inference --env CartPole-v1 --episodes 10
```

Inference automatically loads the selected environment's checkpoint and always opens the Gymnasium human renderer.

## Interpreter

The launcher defaults to:

```text
/home/shuvrajeet/miniconda3/envs/research/bin/python
```

Override it when necessary:

```bash
PYTHON=python3 sh run.sh train
```

## GPU selection

Use one physical GPU for training or inference:

```bash
sh run.sh train --gpu 0
sh run.sh train --gpu 1
sh run.sh inference --env CartPole-v1 --gpu 0
```

Use `--gpu -1` to force CPU execution. GPU 0 is the default, and an unavailable index fails before model construction.

## Validation scope

The exact validation performed for this implementation is reported with the completed code. Short smoke runs establish construction, concurrent updates, checkpoint loading, and entry-point behavior; they do not by themselves prove convergence or benchmark performance.

