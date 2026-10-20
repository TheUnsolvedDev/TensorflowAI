# 7. DALLE2

> **Current status:** architecture/configuration scaffold. The dataset and
> training entrypoints contain explicit placeholders; no completed run is
> claimed by this documentation.

CLIP-style local embedding alignment plus diffusion prior and decoder stages.

## What This Folder Implements

Keeps the recognizable DALL-E 2 prior/decoder shape with lightweight local encoders.

## Files

- `config.py` for dataset roots, diffusion settings, cache behavior, and conservative multi-GPU defaults
- `dataset.py` for folder-local dataset loading and preprocessing
- `model.py` for the standalone architecture and training logic for this algorithm only
- `train_and_test.py` for CLI, MirroredStrategy setup, resume logic, checkpoints, and sample generation
- `run.sh` for repeated local runs

## Supported Datasets

- `coco`
- `flickr30k`

## Run

```bash
python train_and_test.py --gpu -1 --type coco --mode all
```

Resume training with either of the repo-style flags:

```bash
python train_and_test.py --gpu 0 --type coco --mode all --continue
python train_and_test.py --gpu 0 --type coco --mode all --resume
```

## Practical Notes

- This folder is standalone and does not import runtime helpers from sibling folders.
- Multi-GPU setup follows the same practical `MirroredStrategy(...NcclAllReduce())` pattern used in neighboring repo families.
- The implementation is scoped for local training and architecture recognizability rather than full paper-scale reproduction.
