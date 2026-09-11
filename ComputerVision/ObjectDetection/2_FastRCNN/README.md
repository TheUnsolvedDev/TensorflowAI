# Fast R-CNN

The default detector contract is 256x256 with conservative global batch size 8. Set `DETECTOR_BATCH_SIZE=N` only after measuring a real training step; it must divide evenly across visible replicas. COCO 2017, Pascal VOC 2012, and ImageNet CLS-LOC run as separate detectors with their own background-inclusive class heads (81, 21, and 1001 classes), caches, checkpoints, logs, and inference outputs. VOC uses only its labelled `train.txt` and `val.txt` splits; its unlabeled test split is never fitted. Flickr30k is not a detection option here because the local checkout provides captions rather than boxes.

Detector training defaults to 100 epochs (`--epochs N` overrides it); the optional classifier keeps its 20-epoch default. `train.sh` runs `coco`, then `voc`, then `imagenet` sequentially. Set `DETECTORS="voc"` or run `python train.py --detector voc` for a targeted run. GPU memory growth is configured before automatic NCCL `MirroredStrategy` selection for two or more GPUs, with one-GPU/CPU fallback and replica batch reporting. Default training loads a compatible dataset/task checkpoint but starts the requested epoch budget at epoch 0; with no checkpoint, it starts from fresh weights. Pass `--continue` to require both the matching checkpoint and its `training_state.json`, then continue toward `--epochs`. `--resume` is retained as an alias for `--continue`. Pass `--reset` to remove the selected dataset/task log artifacts and train from newly initialized weights; it cannot be combined with `--continue`. Startup always reports the selected mode plus exact checkpoint/state paths. Every detector epoch writes `logs/<dataset>/FastRCNN/inspection/epoch_###.png`: a fixed 4x4 grid with red `GT <class>` boxes and green `<class>: <probability>` predictions. Classifier-only runs do not create these grids. `--full-reports` enables TensorBoard, model diagrams, and fixed-subset COCO metrics for COCO only.

# Persistent full proposal cache

By default detector training builds or reuses complete, split-isolated proposal arrays under `cache/FastRCNN/<dataset>/<split>/<fingerprint>/` for the best steady-state throughput. The cache holds proposals only—not decoded JPEGs—and its full arrays are pinned to host memory so GPU 0 does not retain the complete proposal split; `tf.data` streams image decode/resize/augmentation and sends only per-replica batches to the GPUs.

Use the normal workflow:

```bash
python train.py --detector coco --prepare-cache
python train.py --detector coco --epochs 100
python train.py --detector coco --continue --epochs 100
python train.py --detector coco --reset --epochs 100
DETECTORS="voc" ./train.sh
```

`config.py` is the default source for `CACHE_ENABLED`, `CACHE_REBUILD`, and `CACHE_MAX_SAMPLES`; `train.py` passes those values explicitly to each dataset/cache instance. The startup `[cache config]` line reports the effective values. `--rebuild-cache`, `--no-cache`, `--cache-max-samples N`, and `--cache-workers N` override them only for that invocation. The displayed `records=` value is the number of annotated images in the split, not `CACHE_MAX_SAMPLES`. `--cache-max-samples N` must cover the complete selected split; `0` selects the same lazy fallback.

Cold preparation uses a bounded process pool with four workers by default. It keeps at most two tasks per worker in flight; workers only decode images and generate proposals, while the parent alone updates the cache arrays, manifest, and atomic promotion. Set `--cache-workers 1` for serial preparation. Startup reports worker count, images per second, ETA, and any failing sample; epoch progress reports steps per second. Worker count does not affect cache fingerprints, so a partially built cache resumes with any worker count.
