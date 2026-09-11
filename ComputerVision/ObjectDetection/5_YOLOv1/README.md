# YOLOv1 From Scratch on COCO 2017

With `--full-reports`, detector training writes `logs/<dataset>/YOLOv1/inspection/epoch_###.png`: a fixed 4x4 validation grid with red `GT <class>` boxes and green `<class>: <probability>` predictions. Classifier-only runs do not create these grids.

## Runtime defaults

The default detector contract is 256x256, grid 4x4, and conservative global batch size 8. Set `DETECTOR_BATCH_SIZE=N` only after measuring a real training step; it must divide evenly across visible replicas. Training streams data without `Dataset.cache()`, enables memory growth, and automatically selects NCCL MirroredStrategy for two or more visible GPUs with one-GPU/CPU fallback. `--resume` preserves incompatible checkpoints and fails rather than overwriting them.

Standalone folder with local files only:

- `config.py`
- `dataset.py`
- `model.py`
- `train.py`
- `test.py`

Default dataset root is COCO 2017 under your `Dataset/coco` tree. This implementation uses:

- `448x448` image input
- `7x7` prediction grid
- scratch-built grid target encoding for COCO boxes
- local custom training loop and inference decoder

Train:

```bash
python train.py --gpu 0
```

Infer:

```bash
python test.py --gpu 0 --num_images 5
```
# Transient bounded cache

Training caches parsed annotations and at most 2,048 deterministic target payloads in one local `cache/active` generation. JPEG decoding, augmentation, shuffling, batching, and prefetch remain streaming over the full dataset; cache capacity never limits enumeration or coverage. Use `--no-cache`, `--rebuild-cache`, or `--cache-max-samples N`. Dataset-scoped checkpoints warm-start at epoch 1 by default; `--continue` requires checkpoint and saved state, while legacy `--resume` is an alias. `--reset` cannot be combined with continue and replaces selected run artifacts in place. TensorBoard and JSON history are written by default, while `--full-reports` enables the expanded model summary and diagram.
