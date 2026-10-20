# 2. Improved DDPM

TensorFlow implementation of *Improved Denoising Diffusion Probabilistic
Models* built on the folder-1 DDPM baseline.

## What Is Improved

- The Functional UNet predicts both noise and reverse-process variance (`2C`
  output channels).
- Learned-range variance interpolates between posterior variance and beta.
- The hybrid objective combines epsilon-prediction MSE with the variational
  lower bound: `L_simple + 0.001 * L_vlb`.
- The VLB noise/mean branch is stop-gradient, so it trains the variance head
  without degrading the epsilon objective.
- Timestep scale/shift conditioning replaces the baseline additive bias.
- Loss-second-moment timestep sampling focuses training on difficult
  timesteps while importance weights keep the objective unbiased.
- The timestep sampler state is checkpointed and restored.
- A correctly respaced 50-step ancestral sampler uses the learned variance;
  it does not incorrectly skip ordinary DDPM transitions.
- The Functional architecture diagram is overwritten at `denoiser.png` when a
  run starts; this graph export does not execute a dummy forward pass.

The model remains unconditional and fully convolutional. It does not contain
unrelated text, CLIP, latent-diffusion, ControlNet, or consistency-model paths.

## Defaults

| Setting | Value |
|---|---:|
| Epochs | 100 |
| Training diffusion steps | 200 |
| Sampling steps | 50 |
| Noise schedule | cosine |
| Global batch size | 32 |
| Optimizer | AdamW |
| Learning rate | `2e-4` |
| Weight decay | `1e-4` |
| Gradient clipping | `1.0` |
| Hybrid VLB weight | `0.001` |
| Checkpoint/sample interval | 5 epochs |

MNIST, Fashion-MNIST, CIFAR-10 and CIFAR-100 use 32px. CelebA and bedroom use
96px. Every dataset defaults to global batch size 32.

## Run

```bash
python train_and_test.py --gpu -1 --type mnist --reset
python train_and_test.py --gpu -1 --type celeba --reset
```

`--gpu -1` selects the first two visible GPUs with NCCL. Generated-only 4x4
sample grids and checkpoints are saved every five epochs, while epoch loss is
appended to `history.json` through Keras callbacks.

Resume a compatible run:

```bash
python train_and_test.py --gpu -1 --type mnist --resume
RESUME=1 DATASETS="mnist celeba" bash run.sh
```

Run implementation checks without loading a dataset:

```bash
python train_and_test.py --gpu -1 --type mnist --verify
```

The default launcher order is MNIST, CelebA, bedroom, Fashion-MNIST, CIFAR-10,
then CIFAR-100. Images are streamed and prefetched without a disk cache.

This implementation follows the learned-variance, hybrid-objective and
importance-sampling ideas from Nichol and Dhariwal (2021), adapted to the
smaller local-training baseline used in this repository.
