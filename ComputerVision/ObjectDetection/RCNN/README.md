# R-CNN From Scratch on COCO 2017

This directory contains a TensorFlow implementation of a classic R-CNN style object detector trained on the COCO 2017 detection dataset.

The pipeline follows the original high-level R-CNN idea:

1. Generate region proposals for an image.
2. Crop each proposed region and resize it to a fixed resolution.
3. Run a CNN classifier and bounding-box regressor on each crop.
4. Decode regressed boxes and apply NMS at inference time.

This implementation is written in the same modular structure used across the repository:

- `config.py`
- `dataset.py`
- `model.py`
- `train.py`
- `test.py`
- `utils.py`

## Directory Structure

```text
RCNN/
├── config.py
├── dataset.py
├── model.py
├── train.py
├── test.py
├── utils.py
└── README.md
```

## What This Implementation Does

- Uses COCO 2017 `train2017` for training and `val2017` for validation/inference by default.
- Loads annotations with `pycocotools`.
- Maps COCO category ids to contiguous training ids with `background=0`.
- Generates proposal-level samples from full images.
- Trains a dual-head network:
  - classification head for object category prediction
  - box regression head for proposal refinement
- Supports:
  - CPU
  - single GPU
  - multi-GPU training with `MirroredStrategy + NcclAllReduce`
- Enables TensorFlow GPU memory growth to reduce early full-memory reservation.
- Saves:
  - weights
  - training history
  - epoch state
  - inference visualizations

## Dataset Layout

The code expects a COCO 2017 layout like this:

```text
coco/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ...
└── val2017/
    ├── 000000000139.jpg
    ├── 000000000285.jpg
    └── ...
```

By default `config.py` searches these roots:

```python
DATASET_PATHS = [
    "/home/shuvrajeet/Documents/Dataset",
    "/mnt/storage/da24d402/Documents/Dataset",
    "/storage/nas/da24d402/Documents/Dataset",
]
```

and then tries these COCO directories:

```python
COCO_ROOT_CANDIDATES = [
    os.path.join(DATASET_PATH, "coco"),
    os.path.join(DATASET_PATH, "COCO"),
    os.path.join(DATASET_PATH, "coco2017"),
    os.path.join(DATASET_PATH, "ObjectDetection", "coco"),
]
```

If your dataset lives somewhere else, change `COCO_ROOT_CANDIDATES` in [config.py](./config.py).

## File Overview

### `config.py`

Contains all central runtime and training configuration:

- dataset root discovery
- input size
- batch size
- epochs
- learning rate
- proposal limits
- IoU thresholds
- inference thresholds
- checkpoint and log paths

Important defaults:

```python
INPUT_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
MAX_PROPOSALS = 2000
TRAIN_PROPOSALS_PER_IMAGE = 128
VAL_PROPOSALS_PER_IMAGE = 64
POSITIVE_IOU_THRESHOLD = 0.5
NEGATIVE_IOU_THRESHOLD = 0.3
```

### `dataset.py`

Responsible for:

- loading COCO annotations
- filtering crowd annotations
- filtering tiny boxes using `MIN_BOX_SIZE`
- converting COCO category ids into contiguous class indices
- generating proposal-level training samples
- creating TensorFlow datasets

Key classes and functions:

- `COCO_CLASSES`
- `collect_coco_samples(...)`
- `RCNNProposalDataset`
- `COCOInferenceDataset`
- `build_train_dataset(...)`
- `build_val_dataset(...)`

### `model.py`

Defines a compact AlexNet-style R-CNN network:

- convolutional backbone
- fully connected feature extractor
- classification head
- bounding box regression head

Outputs:

- `class_logits`
- `bbox_regression`

### `utils.py`

Contains shared detection utilities:

- IoU computation
- proposal deduplication
- region proposal generation
- crop and resize
- box encoding and decoding
- NMS
- drawing detections

Proposal generation behavior:

- If OpenCV selective search is available through `cv2.ximgproc`, it is used.
- Otherwise the code falls back to a deterministic multi-scale grid proposal generator.

This fallback keeps the full pipeline runnable even when OpenCV contrib modules are not installed.

### `train.py`

Handles end-to-end training:

- runtime / GPU setup
- memory growth
- single GPU / multi-GPU strategy selection
- dataset construction
- resume from checkpoint
- training callbacks
- checkpoint saving
- history saving
- epoch-state saving

### `test.py`

Handles inference:

- load trained weights
- proposal generation
- crop classification and box regression
- box decoding
- per-class NMS
- visualization saving

## Dependencies

Core Python packages used by this implementation:

- `tensorflow`
- `numpy`
- `opencv-python` or `opencv-contrib-python`
- `pycocotools`
- `matplotlib`

If you want true selective search, use an OpenCV build that exposes:

```python
cv2.ximgproc.segmentation.createSelectiveSearchSegmentation
```

Without that, the fallback grid proposal generator will be used.

## Multi-GPU and Memory Behavior

Both training and inference scripts enable GPU memory growth:

```python
tf.config.experimental.set_memory_growth(device, True)
```

Training runtime behavior:

- `--gpu -1`
  - uses all visible GPUs
  - if more than one GPU is visible, uses:
    - `tf.distribute.MirroredStrategy`
    - `tf.distribute.NcclAllReduce()`
- `--gpu 0`
  - restricts execution to GPU 0
- no GPU available
  - falls back to CPU

Additional cleanup:

- `tf.keras.backend.clear_session()` before model construction
- `tf.keras.backend.clear_session()` after training/inference
- `gc.collect()` after training/inference

This helps reduce memory retention across runs.

## Training

### Basic Training

Run from the repository root:

```bash
python ObjectDetection/RCNN/train.py --gpu 0
```

Use all visible GPUs:

```bash
python ObjectDetection/RCNN/train.py --gpu -1
```

Resume training:

```bash
python ObjectDetection/RCNN/train.py --gpu 0 --resume
```

Override epochs:

```bash
python ObjectDetection/RCNN/train.py --gpu 0 --epochs 5
```

### Training Outputs

Training saves artifacts under:

```text
ObjectDetection/RCNN/logs/coco2017/RCNN/
```

Expected files:

```text
logs/coco2017/RCNN/
├── rcnn.weights.h5
├── history.json
├── training_state.json
└── inference/
```

### Saved Training State

- `rcnn.weights.h5`
  - latest saved weights
- `history.json`
  - recorded metrics per epoch
- `training_state.json`
  - last completed epoch index for resume support

## Inference

### Single Image

```bash
python ObjectDetection/RCNN/test.py --gpu 0 --image /path/to/image.jpg
```

### Dataset Split Inference

```bash
python ObjectDetection/RCNN/test.py --gpu 0 --num_images 5
```

By default this uses `val2017`.

### Inference Outputs

Rendered outputs are saved in:

```text
ObjectDetection/RCNN/logs/coco2017/RCNN/inference/
```

The script also prints top detections in the terminal:

```text
person          score=0.913 box=[...]
car             score=0.874 box=[...]
```

## Model Details

This is not Fast R-CNN or Faster R-CNN.

It is a simpler classic R-CNN style pipeline:

- proposals are generated externally
- each proposal is cropped independently
- each crop is passed through the network
- regression is class-agnostic in the current version

Backbone shape:

- conv
- pool
- conv
- pool
- conv
- conv
- conv
- pool
- flatten
- dense
- dropout
- dense
- dropout
- classification head
- bbox regression head

## Proposal Sampling Logic

For each image:

1. Generate region proposals.
2. Append ground-truth boxes so positives are always available.
3. Compute IoU between proposals and GT boxes.
4. Sample:
   - positives with IoU `>= POSITIVE_IOU_THRESHOLD`
   - negatives with IoU `< NEGATIVE_IOU_THRESHOLD`
5. Train classification on all sampled proposals.
6. Train bbox regression only on positive proposals.

Regression loss is masked for background proposals using sample weights.

## Important Configuration Knobs

You will likely want to tune these first:

### Proposal count

```python
MAX_PROPOSALS = 2000
TRAIN_PROPOSALS_PER_IMAGE = 128
VAL_PROPOSALS_PER_IMAGE = 64
```

Higher values can improve proposal coverage but increase runtime and memory cost sharply.

### Batch size

```python
BATCH_SIZE = 32
```

If GPU memory is tight, reduce this.

### Input size

```python
INPUT_SIZE = (224, 224)
```

Increasing this can improve feature detail but also increases memory and compute.

### Inference thresholds

```python
SCORE_THRESHOLD = 0.4
NMS_IOU_THRESHOLD = 0.3
INFERENCE_TOPK = 300
```

These directly affect the number and quality of final detections.

## Practical Notes

This implementation is intentionally educational and from-scratch in style.

That means:

- it is much slower than Faster R-CNN style detectors
- proposal generation can dominate runtime
- training on full COCO can take a long time
- inference is proposal-by-proposal and not end-to-end optimized

In the current setup, full COCO training is heavy because:

- `train2017` is large
- each image generates many proposal crops
- each crop is processed independently

This implementation is best viewed as:

- a from-scratch educational detector
- a stepping stone toward Fast R-CNN / Faster R-CNN
- a baseline for experimenting with proposal sampling and classification/regression behavior

## Known Limitations

- No class-specific box regression head.
- No ROI Pooling / ROI Align.
- No Region Proposal Network.
- No mixed precision path yet.
- No mAP evaluation script yet.
- No proposal caching yet.
- No COCO metrics export yet.
- Inference is not optimized for very large evaluation runs.

## Recommended Next Improvements

If you want to extend this implementation, the highest-value next steps are:

1. Add proposal caching to disk.
2. Add a smaller training subset option for faster debugging.
3. Add mixed precision training.
4. Add class-specific bounding box regression.
5. Add COCO evaluation metrics with `pycocotools`.
6. Move toward Fast R-CNN with shared feature extraction.
7. Move toward Faster R-CNN with an RPN.

## Quick Start

### 1. Check dataset path

Make sure `config.py` resolves `COCO_ROOT` correctly.

### 2. Train

```bash
python ObjectDetection/RCNN/train.py --gpu 0
```

### 3. Resume if needed

```bash
python ObjectDetection/RCNN/train.py --gpu 0 --resume
```

### 4. Run inference

```bash
python ObjectDetection/RCNN/test.py --gpu 0 --image /path/to/image.jpg
```

## Script Entry Points

Training help:

```bash
python ObjectDetection/RCNN/train.py --help
```

Inference help:

```bash
python ObjectDetection/RCNN/test.py --help
```

## Summary

This folder contains a complete scratch-style R-CNN pipeline for COCO 2017 with:

- modular project structure
- proposal-based training
- classification + box regression heads
- checkpoint/resume support
- inference visualization
- GPU memory growth
- single GPU support
- multi-GPU `NcclAllReduce` support

It is built for clarity and experimentation first, not production speed.
