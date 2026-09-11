# 1. DiffusionProbabilisticModels

Minimal trainable DDPM forward/reverse diffusion baseline with a compact convolutional denoiser.

## What This Folder Implements

Implements the original diffusion idea in repo-scale form rather than a large modern production stack.

## Files

- `config.py` for dataset roots, diffusion settings, cache behavior, and conservative multi-GPU defaults
- `dataset.py` for folder-local dataset loading and preprocessing
- `model.py` for the standalone architecture and training logic for this algorithm only
- `train_and_test.py` for CLI, MirroredStrategy setup, resume logic, checkpoints, and sample generation
- `run.sh` for repeated local runs

## Supported Datasets

- `mnist`
- `fashion_mnist`
- `cifar10`

## Run

```bash
python train_and_test.py --gpu -1 --type mnist --mode diffusion --reset
```

`--reset` deletes only the selected run directory before training, so MNIST begins with fresh weights, history, and samples. The generated samples are saved under `logs/mnist/DiffusionProbabilisticModels/diffusion/samples/` at epochs 5, 10, ..., 50; each grid shows reference images above generated images.

Resume training with either of the repo-style flags:

```bash
python train_and_test.py --gpu 0 --type mnist --mode diffusion --continue
python train_and_test.py --gpu 0 --type mnist --mode diffusion --resume
```

For the shell launcher, fresh training is the default; resume explicitly with `RESUME=1`:

```bash
DATASETS=mnist MODE=diffusion bash run.sh --reset
RESUME=1 DATASETS=mnist MODE=diffusion bash run.sh
```

Run the folder-local DDPM regression checks without training:

```bash
python train_and_test.py --gpu -1 --type mnist --mode diffusion --verify-sampler
```

## Practical Notes

- This folder is standalone and does not import runtime helpers from sibling folders.
- Sampling uses the exact DDPM reverse chain over every configured diffusion timestep, with posterior variance and a deterministic final (`t=0`) step.
- Multi-GPU setup follows the same practical `MirroredStrategy(...NcclAllReduce())` pattern used in neighboring repo families.
- The implementation is scoped for local training and architecture recognizability rather than full paper-scale reproduction.
