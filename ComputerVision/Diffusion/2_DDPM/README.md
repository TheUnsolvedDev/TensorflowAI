# DDPM

An unconditional image DDPM with epsilon prediction, a 1,000-step linear noise schedule, and ancestral sampling.

Supported datasets: `mnist`, `fashion_mnist`, `cifar10`, `cifar100`, and `celeba`.

Run one dataset using all visible GPUs:

```bash
python train_and_test.py --type cifar10 --gpu -1 --epochs 200
```

`--gpu N` selects one GPU; `--gpu -1` uses all visible GPUs. Resume the current run with `--resume`; use `--reset` to remove its checkpoint and state. New weights are stored under `logs/<dataset>/DDPM_v2/diffusion/` so they do not conflict with earlier runs.

The `run.sh` script trains the datasets listed by `DATASETS` (default: `mnist fashion_mnist cifar10 cifar100 celeba`). Set `EPOCHS`, `GPU`, and `PYTHON_BIN` to override its defaults. The script starts a fresh v2 run; add `--resume` after its options to continue an interrupted run.
