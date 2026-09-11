# Sequence Models

Each folder keeps a separate streaming `dataset.py` with `BaseTextDataset` and
AG News, DBPedia, and IMDB adapters. Files are reopened per pass, validation is
assigned deterministically, vocabulary metadata may be reused, and encoded
examples are never cached.

```bash
cd 3_GRU
bash run.sh ag_news --smoke
```

Use `--gpu -1` for all visible GPUs, or a numeric GPU index for one device.
Training uses NCCL `MirroredStrategy` where supported, then falls back safely.
`--steps-per-epoch` and `--validation-steps` bound large experiments.
