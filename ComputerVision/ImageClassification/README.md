# 🖼️ Image Classification Architectures

← [Computer Vision](../README.md) · [Repository](../../README.md)

The image-classification area contains architecture-first TensorFlow projects.
Most runnable variants pair `config.py`, `dataset.py`, `model.py`, and
`train_and_test.py`; configuration selects supported local datasets and
training/runtime options. The pages below group depth and scale variants so
the documentation follows the implementation families rather than duplicating
the same workflow.

| Track | Families | Documentation |
| --- | --- | --- |
| Small-network and efficiency-focused | LeNet-5, ZFNet, SqueezeNet, XNOR-Net, MobileNet, ShuffleNet, knowledge distillation, deep compression, FractalNet, MLP-Mixer, PolyNet, Xception | [Small networks](SmallNetwork/readme.md) |
| Large convolutional and attention models | AlexNet, VGG, Network in Network, Inception, ResNet, HighwayNet, DenseNet, residual attention, SENet, ResNeXt, CapsuleNet, Vision Transformer | [Large networks](LargeNetwork/readme.md) |
| Robustness | Adversarial saliency maps, black-box attacks, FGSM, iterative least-likely method | [Robustness](Robustness/README.md) |

## Shared mechanics

The family implementations construct architectures from TensorFlow/Keras
layers instead of `tf.keras.applications`. Input pipelines, data roots,
augmentation, optimizer choices, and output directories are controlled per
folder; not every variant has an identical set of datasets or a retained
experiment report. Follow the selected folder's scripts and configuration.
