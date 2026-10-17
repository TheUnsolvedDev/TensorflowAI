# Advantage Actor-Critic (A2C)

A synchronous advantage actor-critic implementation using parallel environments and fixed-length rollouts.

## Files

```text
4_A2C/
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

The shared actor-value network in `model.py` uses the TensorFlow Keras Functional API. Eight synchronous environments collect short environment-specific rollouts. Bootstrapped n-step returns train the value head, while the policy head uses standardized return-minus-value advantages across each rollout batch. Action selection and batched RMSprop updates are compiled with `tf.function`; Huber value loss, clipped detached advantages, entropy regularization, and global gradient clipping keep updates stable. GPU memory growth is enabled before model construction when GPUs are available. This folder is self-contained and does not import implementation code from sibling folders or the repository root.

Default environment, model, optimizer, training, checkpoint, logging, plotting, and TensorFlow settings are centralized in `config.py`. Command-line arguments can override the run-specific defaults without editing the file.

Training uses one agent run. A generator initialized by `--seed` produces a fresh random environment seed for every explicit episode reset. Vector environments and A3C workers receive independent random streams. This exposes the agent to varied initial states while keeping the full run reproducible. Evaluation and GIF seeds remain separate from training.

`ENVIRONMENT_CONFIGS` supplies environment-specific episode, discount, entropy, and rollout settings without changing the A2C update.

Acrobot and both MountainCar variants use potential-based rewards only for learning. Logged and evaluated rewards remain the original Gymnasium rewards.

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

Change the synchronous rollout configuration when needed:

```bash
sh run.sh train --num-envs 8 --rollout-steps 64
```

Run one compatible environment directly when needed:

```bash
python train_and_test.py --env CartPole-v1 --episodes 300
```

Each environment receives its own checkpoint under `checkpoints/`, such as `checkpoints/CartPole_v1.weights.h5`. Supply `--checkpoint` when running `train_and_test.py` directly to use another path.

## Metrics and graphs

CSV metrics and graphs are enabled by default. Metrics stay in memory while the agent trains, then the CSV files are written after training and evaluation finish. The final graph reads its values from those CSV files. Each environment has one results directory that is overwritten by its next completed run:

```text
results/<environment>/
├── training_metrics.csv
├── evaluation_metrics.csv
├── results.png
└── test.gif
```

The default final figure contains training reward, the configured moving reward average, policy loss, episode length, and evaluation rewards. Checkpoints are selected by recent average reward instead of one lucky episode. After evaluation, `test.gif` records the highest-reward deterministic episode from `GIF_TEST_EPISODES` candidate seeds. GIF settings are available in `config.py`. Select another results root with `--output-dir`:

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

Compilation and short CPU smoke runs only establish that construction, updates, checkpoint loading, and entry points execute. They do not prove convergence, benchmark performance, or GPU correctness.

