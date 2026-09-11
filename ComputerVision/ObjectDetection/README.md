# 🎯 Object Detection

← [Computer Vision](../README.md) · [Repository](../../README.md)

The detection projects implement detector-specific TensorFlow models and local
train/test tooling. They share a practical pattern: dataset adapters,
configuration-driven roots, model/loss code, training loops, inspection, and
bounded local caches. COCO, Pascal VOC, and ImageNet-related paths appear in
the source; availability is a local prerequisite, not a bundled dataset.

| Detector family | Implementation | Notes |
| --- | --- | --- |
| R-CNN | [`1_RCNN`](1_RCNN/README.md) | Region-based detection pipeline |
| Fast R-CNN | [`2_FastRCNN`](2_FastRCNN/README.md) | ROI classification/regression path |
| Faster R-CNN | [`3_FasterRCNN`](3_FasterRCNN/README.md) | RPN, proposal, ROI and detector training code |
| Mask R-CNN | [`4_MaskRCNN`](4_MaskRCNN/README.md) | Detection/mask-oriented extension |
| YOLO | [`5_YOLOv1`](5_YOLOv1/README.md), [`7_YOLOv2`](7_YOLOv2/README.md), [`8_YOLOv3`](8_YOLOv3/README.md) | Grid/anchor-style one-stage detectors |
| SSD | [`6_SSD`](6_SSD/README.md) | Multi-scale one-stage detector |

## Runtime and validation

Several detectors use `tf.data`, custom `tf.keras.layers.Layer`/models,
`tf.GradientTape`, and device-aware distribution setup. Some folders include
test entry points; Fast R-CNN also contains `tests/test_run_controls.py`.
Generated TensorBoard reports, visual inspections, checkpoints, and caches are
run artifacts—not shipped benchmark results.
