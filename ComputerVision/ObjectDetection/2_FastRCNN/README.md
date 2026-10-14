# Fast R-CNN

This implementation trains its backbone from scratch. Detector images are resized to 512x512 and the stride-16 backbone produces a 32x32 feature map. The default global batch is 4; set `DETECTOR_BATCH_SIZE=N` only after measuring a real training step, and keep it divisible by the number of visible replicas.

COCO 2017, Pascal VOC 2012, and ImageNet CLS-LOC have separate class heads, proposal caches, checkpoints, logs, and inference outputs. VOC uses its labelled `train.txt` and `val.txt` splits.

## Required selective search

Fast R-CNN requires the OpenCV contrib build. The program stops if selective search is missing instead of silently training with low-recall grid proposals.

```bash
/home/shuvrajeet/miniconda3/envs/research/bin/python -m pip uninstall opencv-python
/home/shuvrajeet/miniconda3/envs/research/bin/python -m pip install opencv-contrib-python==4.13.0.92
```

Measure proposal recall before building a complete cache. On the local validation check, fast mode reached 0.77 COCO recall@0.5 and finished the complete cache at 3.2 images/second; quality mode reached 0.83 but was much slower. Fast mode and 4,000 proposal slots are therefore the practical defaults. Set `FAST_RCNN_PROPOSAL_MODE=quality` only when the added recall is worth rebuilding the cache.

```bash
cd /home/shuvrajeet/Documents/TensorflowAI/ComputerVision/ObjectDetection/2_FastRCNN
PYTHON=/home/shuvrajeet/miniconda3/envs/research/bin/python
$PYTHON train.py --detector coco --proposal-report
$PYTHON train.py --detector voc --proposal-report
FAST_RCNN_PROPOSAL_MODE=quality $PYTHON train.py --detector coco --proposal-report
```

The proposal mode, OpenCV version, input size, and sampling settings are part of the cache fingerprint. Old caches remain untouched and are not reused after these changes.

## Training and evaluation

Training uses 64 ROIs per image, at most 25% foreground, hard background proposals from the 0.1-0.5 IoU interval, and exact ground-truth boxes only in the training sampler. Validation and inference use genuine selective-search proposals. Horizontal flips transform the image, ROIs, and regression targets together.

Run independent COCO and VOC training jobs sequentially with one command:

```bash
sh run.sh --reset --epochs 100
```

Arguments are forwarded to both runs. If the COCO run fails, the VOC run is not started.

```bash
$PYTHON train.py --detector coco --prepare-cache
$PYTHON train.py --detector coco --reset --epochs 100
$PYTHON train.py --detector coco --continue --epochs 100
$PYTHON train.py --detector coco --evaluate

$PYTHON train.py --detector voc --prepare-cache
$PYTHON train.py --detector voc --reset --epochs 100
$PYTHON train.py --detector voc --evaluate
```

Use `--reset` for the first run with the 512x512 stride-16 architecture. Detector training evaluates a fixed 200-image validation subset every epoch and keeps one best checkpoint: COCO uses subset mAP and VOC/ImageNet use AP50. Inference scores each foreground class independently, applies class-wise NMS, and returns at most 100 detections per image.

Persistent proposal caches live under `cache/FastRCNN/<dataset>/<split>/<fingerprint>/`. Use `--rebuild-cache` only to replace the currently selected generation, `--no-cache` for lazy proposal generation, and `--cache-workers 1` for serial preparation. `--full-reports` adds TensorBoard and model diagrams; validation metrics and 4x4 inspection grids are always produced for detectors.
