# 1. DiffusionProbabilisticModels

Minimal trainable DDPM with a multiscale residual convolutional UNet. It uses
two GroupNorm residual blocks per resolution, resize-convolution upsampling, a
dilated 8x8 bottleneck, and no attention layers.

## What This Folder Implements

Implements unconditional epsilon-prediction DDPM training with a cosine noise
schedule. The model predicts the noise added to an image and samples with the
exact reverse DDPM transition. It is a from-scratch computer-vision model, not
a text-conditioned or pretrained diffusion pipeline.

## Architecture

The `conv_unet_v2` denoiser is fully convolutional:

- two residual blocks at every encoder and decoder resolution
- Group Normalization, SiLU/Swish activation, and dropout in each residual block
- sinusoidal timestep embeddings injected into every residual block
- strided convolutions for downsampling
- bilinear resize followed by a 3x3 convolution for upsampling, avoiding
  transposed-convolution checkerboard artifacts
- skip connections between matching encoder and decoder resolutions
- an 8x8 bottleneck for 128px inputs with dilation rates `1, 2, 4, 1`
- zero-initialized residual/output convolutions for stable early optimization
- no spatial or self-attention layers

Channel widths are resolution-dependent:

| Input | Resolution path | Parameters |
|---|---|---:|
| 32px | `64 -> 128 -> 256` | about 9.9M |
| 128px | `64 -> 128 -> 256 -> 256 -> 384` | about 33M |

## Training Configuration

| Setting | Value |
|---|---:|
| Epochs | 100 |
| Diffusion steps | 200 |
| Noise schedule | cosine |
| Objective | noise-prediction MSE |
| Optimizer | AdamW |
| Learning rate | `2e-4` |
| Weight decay | `1e-4` |
| Gradient clipping | norm `1.0` |
| Dropout | `0.1` |
| Checkpoint interval | 5 epochs |
| Sample interval | 5 epochs |

Optimizer-managed EMA is intentionally disabled. Keras' periodic EMA overwrite
crosses a synchronization boundary inside this custom `MirroredStrategy`
training loop and fails under NCCL. The denoiser therefore samples from the
normally trained weights.

## Files

- `config.py` for dataset roots, diffusion settings, and conservative multi-GPU defaults
- `dataset.py` for folder-local dataset loading and preprocessing
- `model.py` for the standalone architecture and training logic for this algorithm only
- `train_and_test.py` for CLI, MirroredStrategy setup, resume logic, checkpoints, and sample generation
- `run.sh` for repeated local runs

## Supported Datasets

- `mnist`
- `fashion_mnist`
- `cifar10`
- `celeba`
- `bedroom`

MNIST, Fashion-MNIST, and CIFAR-10 default to 32px with batch size 32. CelebA
and bedroom default to 128px with batch size 8.

| Dataset | Resolution | Global batch size |
|---|---:|---:|
| MNIST | 32x32 | 32 |
| Fashion-MNIST | 32x32 | 32 |
| CIFAR-10 | 32x32 | 32 |
| CelebA | 128x128 | 8 |
| Bedroom | 128x128 | 8 |

## Run

```bash
python train_and_test.py --gpu -1 --type mnist --reset
```

`--gpu -1` requires both GPUs and uses NCCL mirrored training.

`--reset` deletes only the selected run directory before training, so MNIST begins with fresh weights, history, and samples. Generated samples are saved every five epochs under `logs/mnist/DiffusionProbabilisticModels/diffusion/samples/`; each grid shows reference images above generated images.

Architecture v2 is incompatible with older weights. Start its first run with
`--reset`; later runs can use `--resume` normally.

Resume training with either of the repo-style flags:

```bash
python train_and_test.py --gpu 0 --type mnist --continue
python train_and_test.py --gpu 0 --type mnist --resume
```

For the shell launcher, fresh training is the default; resume explicitly with `RESUME=1`:

```bash
DATASETS=mnist bash run.sh --reset
DATASETS="celeba bedroom" bash run.sh --reset
RESUME=1 DATASETS=mnist bash run.sh
```

Run the folder-local DDPM regression checks without training:

```bash
python train_and_test.py --gpu -1 --type mnist --verify-sampler
```

## Practical Notes

- This folder is standalone and does not import runtime helpers from sibling folders.
- Images are streamed and prefetched; no dataset cache is written to disk.
- Folder images are center-cropped to a square, resized with antialiasing, and
  randomly flipped horizontally during training.
- Weights, training state, and sample grids are saved every five epochs.
- Sampling uses the exact DDPM reverse chain over every configured diffusion timestep, with posterior variance and a deterministic final (`t=0`) step.
- A flat noise-prediction loss does not by itself mean training has stopped;
  generated sample grids are the primary quality check.
- Multi-GPU setup follows the same practical `MirroredStrategy(...NcclAllReduce())` pattern used in neighboring repo families.
- The implementation is scoped for local training and architecture recognizability rather than full paper-scale reproduction.
