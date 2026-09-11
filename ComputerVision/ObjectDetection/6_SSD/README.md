# SSD From Scratch on COCO 2017

With `--full-reports`, detector training writes `logs/<dataset>/SSD/inspection/epoch_###.png`: a fixed 4x4 validation grid with red `GT <class>` boxes and green `<class>: <probability>` predictions. Classifier-only runs do not create these grids.

## Runtime defaults

The default detector contract is 256x256 with 16/8/4 feature maps and conservative global batch size 8. Set `DETECTOR_BATCH_SIZE=N` only after measuring a real training step; it must divide evenly across visible replicas. Training uses a streaming shuffle/batch/prefetch pipeline, automatic NCCL MirroredStrategy when multiple GPUs are visible, and a one-GPU/CPU fallback. `--resume` preserves incompatible checkpoints and fails clearly.

Standalone SSD folder with local:

- prior generation
- COCO target matching
- classification/regression heads
- train and test entrypoints

The implementation is kept educational and self-contained while following the repo’s usual detector folder structure.
# Transient bounded cache

Training caches parsed annotations and at most 2,048 deterministic target payloads in one local `cache/active` generation. JPEG decoding, augmentation, shuffling, batching, and prefetch remain streaming over the full dataset; cache capacity never limits enumeration or coverage. Use `--no-cache`, `--rebuild-cache`, or `--cache-max-samples N`. Dataset-scoped checkpoints warm-start at epoch 1 by default; `--continue` requires checkpoint and saved state, while legacy `--resume` is an alias. `--reset` cannot be combined with continue and replaces selected run artifacts in place. TensorBoard and JSON history are written by default, while `--full-reports` enables the expanded model summary and diagram.
