# Initial GAN

← [GAN family](../README.md)

This folder introduces adversarial training with generator and discriminator
models, local dataset loading, and `train_and_test.py`. The minimax objective
trains \(G(z)\) to fool \(D(x)\), while \(D\) separates real from generated
samples. TensorFlow `GradientTape` updates the two networks separately. Dataset
and output settings live in `config.py`; generated samples are local-only.

## Training

Run all four configured datasets, using all visible GPUs and resuming any
existing run:

```bash
sh run.sh
```

The launcher defaults to the `research` Conda environment. Override it when
needed with `PYTHON_BIN=/path/to/python sh run.sh`.

Run one dataset on CPU for a short check:

```bash
python train_and_test.py --type mnist --gpu -2 --epochs 1
```

`BATCH_SIZE` is the global batch size and must be divisible by the number of
GPUs. Checkpoints include both models and both Adam optimizers. The history also
records `sample_stddev`; a warning is printed when generated-batch diversity is
low enough to suggest mode collapse. This detector is a warning, not a guarantee
of image quality, so inspect the saved grids and use FID for formal comparisons.
