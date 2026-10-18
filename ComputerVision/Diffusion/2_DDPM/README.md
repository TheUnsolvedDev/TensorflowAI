# DDPM

An unconditional image DDPM with epsilon prediction, a multiscale residual convolutional UNet, a 1,000-step cosine noise schedule, and ancestral sampling. It uses no attention layers.

Supported datasets: `mnist`, `fashion_mnist`, `cifar10`, `cifar100`, `celeba`, and `bedroom`.

Run one dataset using all visible GPUs:

```bash
python train_and_test.py --type cifar10 --gpu -1 --epochs 200
```

MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 default to 32px with batch size
32. CelebA and bedroom default to 128px with batch size 8. Select 32, 64, 128,
or 256 explicitly when needed:

```bash
python train_and_test.py --type celeba --image-size 128 --gpu -1 --epochs 200
python train_and_test.py --type bedroom --image-size 256 --batch-size 1 --gpu -1 --epochs 200
```

The shell launcher selects datasets through `DATASETS`:

```bash
DATASETS=celeba sh run.sh --reset
DATASETS=bedroom sh run.sh --reset
DATASETS="celeba bedroom" sh run.sh --reset
```

`--gpu N` selects one GPU; `--gpu -1` uses all visible GPUs. Resume the current run with `--resume`; use `--reset` to remove its checkpoint and state. Weights are stored under `logs/<dataset>/DDPM/diffusion/`. The saved state still prevents incompatible image sizes or schedules from being resumed.

Images are streamed and prefetched; no dataset cache is written to disk.

The `run.sh` script trains the datasets listed by `DATASETS` (default: `mnist fashion_mnist cifar10 cifar100 celeba`). Set `EPOCHS`, `GPU`, and `PYTHON_BIN` to override its defaults. The script starts a fresh v2 run; add `--resume` after its options to continue an interrupted run.
