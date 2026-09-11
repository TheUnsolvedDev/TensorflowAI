# Mask R-CNN From Scratch

With `--full-reports`, detector training writes `logs/<dataset>/MaskRCNN/inspection/epoch_###.png`: a fixed 4x4 validation grid retaining native mask overlays, with red `GT <class>` boxes and green `<class>: <probability>` predictions. Classifier-only runs do not create these grids.

## Runtime defaults

Detector training uses 256x256 inputs and conservative global batch size 2 because ROI and mask activations are large. Set `DETECTOR_BATCH_SIZE=N` only after measuring a real training step; it must divide evenly across visible replicas. ROI/mask targets are regenerated each pass in the streaming shuffle/batch/prefetch pipeline; there is no persistent proposal cache. Memory growth precedes automatic NCCL multi-GPU selection, with safe one-GPU/CPU fallback. `--resume` never deletes or replaces an incompatible checkpoint.

This folder is standalone and follows the same repo pattern as the other object detectors:

COCO 2017, Pascal VOC 2012, and ImageNet CLS-LOC run separately. `python train.py --detector voc` uses VOC's labelled `train.txt` and `val.txt` splits and creates a 21-class detector head; because VOC exposes boxes rather than instance masks, its mask targets are box masks.

- `config.py`
- `dataset.py`
- `model.py`
- `train.py`
- `test.py`
- `utils.py`

Default dataset target is COCO 2017:

```text
coco/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
└── val2017/
```

The implementation is educational and scratch-built:

- region proposals from selective search when available, otherwise grid fallback
- ROI pooling classification and box regression heads
- instance mask head on top of pooled ROI features
- COCO detection + segmentation target generation inside the folder itself

Run training:

```bash
python train.py --gpu 0
```

Run inference:

```bash
python test.py --gpu 0 --num_images 5
```
# Transient bounded cache

Training starts immediately: `cache/active` contains a manifest and an initially empty proposal index, while expensive proposal payloads are added only when an image is reached. JPEG pixels are never cached; augmentation, shuffling, batching, and prefetch remain streaming over the full dataset. The opportunistic proposal cache is capped at 2,048 images by default and does not restrict training coverage; use `--cache-max-samples N` (`0` disables proposal caching), `--no-cache`, or `--rebuild-cache`. ImageNet cache invalidation uses annotation-root metadata, so use `--rebuild-cache` after nested XML edits. The active generation is atomically replaced and removed in a `finally` block when training exits.

Dataset-scoped matching checkpoints warm-start at epoch 1 by default. `--continue` requires matching weights and saved epoch state; legacy `--resume` remains an alias. `--reset` is mutually exclusive with continue and replaces only the selected task/dataset artifact directory in place. TensorBoard and JSON history remain under the dataset-scoped log directory.
