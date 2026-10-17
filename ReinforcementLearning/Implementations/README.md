# TensorFlow Reinforcement Learning Implementations

A collection of self-contained reinforcement-learning algorithms implemented with TensorFlow and Gymnasium. Each algorithm lives in its own folder. The standardized folders keep their model, agent, configuration, training loop, inference entry point, launcher, and environment settings together so they can be studied independently.

## Algorithm catalogue

| Folder | Algorithm | Action spaces | Current status |
|---|---|---|---|
| `1_PolicyGradient` | Policy Gradient | Discrete and continuous | Standardized |
| `2_REINFORCE` | REINFORCE | Discrete and continuous | Standardized |
| `3_ActorCritic` | Actor-Critic | Discrete and continuous | Standardized |
| `4_A2C` | Advantage Actor-Critic | Discrete and continuous | Standardized |
| `5_A3C` | Asynchronous Advantage Actor-Critic | Discrete and continuous | Standardized |
| `6_DQN` | Deep Q-Network | Discrete | Standardized |
| `7_DoubleDQN` | Double DQN | Discrete | Standardized |
| `8_DuelingDQN` | Dueling DQN | Discrete | Standardized |
| `9_PrioritizedExperienceReplayDQN` | Prioritized Replay DQN | Discrete | Standardized |
| `10_RainbowDQN` | Rainbow DQN | Discrete | Standardized |
| `11_DDPG` | Deep Deterministic Policy Gradient | Continuous | Starter implementation |
| `12_TD3` | Twin Delayed DDPG | Continuous | Starter implementation |
| `13_SAC` | Soft Actor-Critic | Continuous | Starter implementation |
| `14_TRPO` | Trust Region Policy Optimization | Policy optimization | Starter implementation |
| `15_PPO` | Proximal Policy Optimization | Policy optimization | Starter implementation |

The standardized implementations are folders 1–10. Policy Gradient and REINFORCE intentionally demonstrate the same Monte Carlo estimator as baseline and named textbook presentations. Rainbow combines Double DQN, Dueling DQN, prioritized replay, three-step returns, C51 distributional learning, and noisy linear layers.

## Shared structure

The standardized folders use this layout:

```text
<implementation>/
├── agent.py
├── config.py
├── model.py
├── train_and_test.py
├── inference.py
├── run.sh
├── README.md
└── env/
    ├── __init__.py
    └── settings.py
```

They provide TensorFlow Functional models, compiled `tf.function` hot paths, GPU memory growth, random seeds at environment resets, environment-specific defaults, human-rendered inference, moving-average checkpoints, CSV metrics, result plots, and test GIFs.

The DQN family uses preallocated NumPy replay storage. DQN, Double DQN, and Dueling DQN use direct uniform sampling. PER DQN and Rainbow use sum/min segment trees, stratified priority sampling, and importance weights.

For folders 6–10, Acrobot and MountainCar use potential-based rewards during training. Original Gymnasium rewards remain the source of checkpoint selection, CSV metrics, plots, GIF scores, and evaluation results.

## Supported environments

Folders 1–5 support both discrete and continuous vector environments:

| Environment | Action space |
|---|---|
| `CartPole-v1` | Discrete |
| `Acrobot-v1` | Discrete |
| `MountainCar-v0` | Discrete |
| `Pendulum-v1` | Continuous |
| `MountainCarContinuous-v0` | Continuous |
| `LunarLander-v3` | Discrete |
| `LunarLanderContinuous-v3` | Continuous |
| `BipedalWalker-v3` | Continuous |

Folders 6–10 support the discrete environments `CartPole-v1`, `Acrobot-v1`, `MountainCar-v0`, and `LunarLander-v3`.

Box2D environments require the corresponding optional Gymnasium dependencies.

For implementations 1–5, Acrobot and both MountainCar variants use potential-based rewards during training to expose progress before terminal success. Checkpoints, CSV metrics, plots, GIFs, and evaluation always use the original Gymnasium reward.

| Environment | Supported by 1–5 | Expected baseline fit |
|---|---:|---|
| CartPole, Acrobot | Yes | Good |
| MountainCar | Yes | Difficult; shaped training reward |
| Pendulum | Yes | Moderate |
| MountainCarContinuous | Yes | Difficult; shaped training reward |
| LunarLander | Yes | Moderate |
| LunarLanderContinuous | Actor-Critic, A2C, A3C | Difficult |
| BipedalWalker | Actor-Critic, A2C, A3C | Difficult; PPO, TD3, or SAC preferred |

## Requirements

- Python 3.10 or newer
- TensorFlow
- Gymnasium
- NumPy
- Matplotlib
- ImageIO

The launchers default to:

```text
/home/shuvrajeet/miniconda3/envs/research/bin/python
```

Use another interpreter when needed:

```bash
PYTHON=python3 sh run.sh train
```

## Training

Enter an implementation folder and launch training:

```bash
cd 6_DQN
sh run.sh train
```

The standardized launcher trains all environments supported by that implementation sequentially. Run only one environment directly through the Python entry point:

```bash
python train_and_test.py --env CartPole-v1 --episodes 1500
```

Common overrides can be forwarded through the launcher:

```bash
sh run.sh train --episodes 500 --test-episodes 20 --log-every 25
```

## GPU selection

Folders 1–10 select one physical GPU before model construction. GPU 0 is the default:

```bash
sh run.sh train --gpu 0
sh run.sh train --gpu 1
```

Force CPU execution with:

```bash
sh run.sh train --gpu -1
```

An unavailable GPU index produces an error instead of silently using another device.

## Inference

Inference loads the environment-specific checkpoint and enables human rendering:

```bash
sh run.sh inference --env CartPole-v1 --episodes 10 --gpu 0
```

## Outputs

Each trained environment writes its artifacts under the implementation folder:

```text
checkpoints/
└── <environment>.weights.h5

results/<environment>/
├── training_metrics.csv
├── evaluation_metrics.csv
├── results.png
└── test.gif
```

Training metrics remain in memory during the training loop and are written afterward so plotting and GIF creation do not interrupt gradient updates.

## Validation scope

Short runtime checks cover model construction, compiled updates, replay sampling, checkpoints, CSV output, plots, GIF creation, environment interfaces, and launcher routing. These checks do not establish convergence for every algorithm/environment pair or prove GPU performance.
