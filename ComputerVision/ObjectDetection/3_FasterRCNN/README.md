# Faster R-CNN

Every Faster R-CNN detector epoch writes `logs/<dataset>/FasterRCNN/inspection/epoch_###.png`: a fixed 4x4 validation grid with red `GT <class>` boxes and green `<class>: <probability>` predictions. Classifier-only runs do not create these grids; `--full-reports` is only for expanded architecture reports.

The default detector contract is 256x256 with global batch size 4, a conservative setting for a 16 GiB GPU. Detector datasets repeat and use fixed-size batches (`drop_remainder=True`), so the custom Python-loop training step receives a statically sized per-replica batch under NCCL `MirroredStrategy`. The detector remains Functional: its reusable Functional ROI head consumes the already-computed batch feature map during ROI loss calculation, avoiding a full backbone graph per image. GPU memory growth is set before automatic NCCL `MirroredStrategy` selection for two or more visible GPUs; one-GPU and CPU execution remain supported fallbacks. The trainer reports visible GPU count, strategy, replicas, global batch size, and per-replica batch size. Dataset-scoped checkpoints warm-start by default at epoch 1; `--continue` requires matching weights and valid epoch state, while legacy `--resume` is an alias. `--reset` is mutually exclusive with continue and replaces only the selected run artifacts in place. Incompatible checkpoints fail clearly and are preserved.

## Lowering VRAM without changing Faster R-CNN into cached-proposal training

The RPN, proposal generation, ROI pooling, and their gradients remain on GPU. JPEG decode, annotations, metadata caching, shuffle, and prefetch remain in the streaming CPU input pipeline. The executable detector is a Functional Keras model with image and ROI inputs; dynamic anchors, proposal/target sampling, losses, metrics, optimization, and decoded inference are held by the separate trainer helper.

Gradient accumulators are transient optimizer state and are deliberately not stored in the existing H5 detector checkpoint format. A resumed run restores the detector weights and starts its next accumulation window empty; completed optimizer updates are preserved normally.

COCO 2017, Pascal VOC 2012, and ImageNet CLS-LOC are isolated detector runs. Use `python train.py --detector voc` for VOC; it trains only against the labelled `train.txt`/`val.txt` splits and creates a 21-class detector head.
# Transient bounded cache

Training caches parsed annotations and deterministic CPU target preparation in one local `cache/active` generation. JPEG decoding, augmentation, shuffling, batching, and prefetch remain streaming. The active generation is atomically replaced when its input fingerprint changes and is removed in a `finally` block when training exits. Use `--no-cache`, `--rebuild-cache`, or `--cache-max-samples N`; a cache capacity never reduces sample enumeration or shuffled coverage. Each run writes dataset-scoped TensorBoard events, `history.json`, `training_state.json`, and `validation_metrics.json`; `--full-reports` additionally writes the model diagram and summary.
