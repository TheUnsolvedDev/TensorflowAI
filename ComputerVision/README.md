# 👁️ Computer Vision

← [Back to the repository](../README.md)

This area contains image generation, classification, object detection, and
robustness experiments implemented with TensorFlow model components and local
training scripts. Families are organised by architecture rather than by a
shared framework; read each folder's `config.py` before running it because
dataset roots and runtime controls are local.

## Families

| Family | Scope | Status | Documentation |
| --- | --- | --- | --- |
| Diffusion | Ten numbered denoising/generative-model folders | Mixed: source exists; several dataset/train entrypoints are explicit scaffolds | [Open](Diffusion/README.md) |
| GANs | Eleven adversarial image-generation variants | Source-backed, configuration-driven training folders | [Open](GenerativeAdvesarialNetworks/README.md) |
| Image classification | Small/large CNNs, compression/distillation, ViT, and robustness scripts | Architecture code and local dataset/training paths | [Open](ImageClassification/README.md) |
| Object detection | R-CNN variants, Mask R-CNN, YOLOv1–v3, and SSD | Detector/train/test code with COCO, VOC, and ImageNet-related adapters | [Open](ObjectDetection/README.md) |

## TensorFlow execution model

The source uses `tf.keras` building blocks extensively, with custom
`tf.keras.layers.Layer` and `tf.keras.Model` classes for several generative and
detection systems. Custom optimisation paths use `tf.GradientTape`; datasets
use `tf.data` where streaming and augmentation are needed. Selected folders
configure `MirroredStrategy` when multiple GPUs are visible, while ordinary
Keras execution remains usable on a single GPU or CPU.

## Artifact policy

Training outputs—including logs, checkpoints, samples, TensorBoard events, and
cached detector targets—are local artifacts. They are not evidence of a
reproducible benchmark in the tracked source and should not be committed.
