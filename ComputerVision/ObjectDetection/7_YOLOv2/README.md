# YOLOv2 From Scratch on COCO 2017

With `--full-reports`, detector training writes `logs/<dataset>/YOLOv2/inspection/epoch_###.png`: a fixed 4x4 validation grid with red `GT <class>` boxes and green `<class>: <probability>` predictions. Classifier-only runs do not create these grids.

## Runtime defaults

The default detector contract is 256x256, grid 8x8, proportionally scaled anchors, and conservative global batch size 8. Set `DETECTOR_BATCH_SIZE=N` only after measuring a real training step; it must divide evenly across visible replicas. Data streaming uses shuffle/batch/prefetch without cache. Memory growth is enabled before automatic NCCL multi-GPU selection, with one-GPU/CPU fallback. `--resume` does not replace incompatible checkpoints.

Standalone local folder with its own:

- `config.py`
- `dataset.py`
- `model.py`
- `train.py`
- `test.py`

This version moves from YOLOv1’s plain grid regression to anchor-based prediction on a `13x13` feature map. COCO target generation, anchor matching, training loss, and inference decoding are all local to this folder.
# Transient bounded cache

Training caches parsed annotations and at most 2,048 deterministic target payloads in one local `cache/active` generation. JPEG decoding, augmentation, shuffling, batching, and prefetch remain streaming over the full dataset; cache capacity never limits enumeration or coverage. Use `--no-cache`, `--rebuild-cache`, or `--cache-max-samples N`. Dataset-scoped checkpoints warm-start at epoch 1 by default; `--continue` requires checkpoint and saved state, while legacy `--resume` is an alias. `--reset` cannot be combined with continue and replaces selected run artifacts in place. TensorBoard and JSON history are written by default, while `--full-reports` enables the expanded model summary and diagram.
