# 4. DDIM

Deterministic denoising implicit sampler built on a DDPM-style denoiser.

## What This Folder Implements

Focuses on fewer-step deterministic sampling while keeping the standalone training surface.

## Files

- `config.py` for dataset roots, diffusion settings, cache behavior, and conservative multi-GPU defaults
- `dataset.py` for folder-local dataset loading and preprocessing
- `model.py` for the standalone architecture and training logic for this algorithm only
- `train_and_test.py` for CLI, MirroredStrategy setup, resume logic, checkpoints, and sample generation
- `run.sh` for repeated local runs

## Supported Datasets

- `cifar10`
- `cifar100`
- `celeba`

## Run

```bash
python train_and_test.py --gpu -1 --type cifar10 --mode diffusion
```

Resume training with either of the repo-style flags:

```bash
python train_and_test.py --gpu 0 --type cifar10 --mode diffusion --continue
python train_and_test.py --gpu 0 --type cifar10 --mode diffusion --resume
```

## Practical Notes

- This folder is standalone and does not import runtime helpers from sibling folders.
- Multi-GPU setup follows the same practical `MirroredStrategy(...NcclAllReduce())` pattern used in neighboring repo families.
- The implementation is scoped for local training and architecture recognizability rather than full paper-scale reproduction.
