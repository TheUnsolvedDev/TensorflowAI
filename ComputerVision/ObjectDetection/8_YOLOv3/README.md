# YOLOv3 From Scratch on COCO 2017

With `--full-reports`, detector training writes `logs/<dataset>/YOLOv3/inspection/epoch_###.png`: a fixed 4x4 validation grid with red `GT <class>` boxes and green `<class>: <probability>` predictions. Classifier-only runs do not create these grids.

## Runtime defaults

The default detector contract is 256x256, scales 8/16/32, proportionally scaled pixel anchors, and conservative global batch size 8. Set `DETECTOR_BATCH_SIZE=N` only after measuring a real training step; it must divide evenly across visible replicas. Training is streaming rather than cached and automatically uses NCCL MirroredStrategy for two or more GPUs, with one-GPU/CPU fallback. Incompatible `--resume` checkpoints are preserved and reported as errors.

Standalone local folder with local COCO parsing, multi-scale target encoding, model heads, training, and inference.

The implementation keeps the repo’s usual detector shape:

- `config.py`
- `dataset.py`
- `model.py`
- `train.py`
- `test.py`

This folder uses three YOLO-style prediction scales and local anchor groups without depending on any sibling folder.
# Transient bounded cache

Training caches parsed annotations and at most 2,048 deterministic target payloads in one local `cache/active` generation. JPEG decoding, augmentation, shuffling, batching, and prefetch remain streaming over the full dataset; cache capacity never limits enumeration or coverage. Use `--no-cache`, `--rebuild-cache`, or `--cache-max-samples N`. Dataset-scoped checkpoints warm-start at epoch 1 by default; `--continue` requires checkpoint and saved state, while legacy `--resume` is an alias. `--reset` cannot be combined with continue and replaces selected run artifacts in place. TensorBoard and JSON history are written by default, while `--full-reports` enables the expanded model summary and diagram.
