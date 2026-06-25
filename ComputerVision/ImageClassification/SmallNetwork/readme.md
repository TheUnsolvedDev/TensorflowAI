# Small Network Image Classification

This directory contains smaller or efficiency-oriented image classification architectures. Compared with the `LargeNetwork` directory, these models generally focus more on compactness, mobile deployment, compression, or architectural efficiency.

It also includes a few methodology-oriented folders such as knowledge distillation and deep compression, so this directory is not only “small CNNs” in the narrow sense.

## What is in this directory

Model families and methods present in the repo:

1. `1_Lenet5`
2. `2_ZFNet`
3. `3_SqueezeNet`
4. `4_XNOR-Net`
5. `5_MobileNet`
6. `6_ShuffleNet`
7. `7_KnowledgeDistillation`
8. `8_DeepCompression`
9. `9_FractalNet`
10. `10_MLP-Mixer`
11. `11_PolyNet`
12. `12_XceptionNet`

## What is actually runnable

Runnable implementations with the usual training files:

- `1_Lenet5`
- `2_ZFNet`
- `3_SqueezeNet/SqueezeNetNoBypass`
- `3_SqueezeNet/SqueezeNetSimpleBypass`
- `3_SqueezeNet/SqueezeNetComplexBypass`
- `4_XNOR-Net`
- `5_MobileNet/MobileNetV1`
- `5_MobileNet/MobileNetV2`
- `5_MobileNet/MobileNetV3`
- `6_ShuffleNet/ShuffleNetV1`
- `6_ShuffleNet/ShuffleNetV2`
- `7_KnowledgeDistillation`
- `8_DeepCompression`
- `12_XceptionNet`

Present in the directory but not currently wired up like the runnable folders above:

- `9_FractalNet`
- `10_MLP-Mixer`
- `11_PolyNet`

## Shared project pattern

Most runnable subprojects follow a repeated scaffold:

- `config.py`
- `dataset.py`
- `model.py`
- `train_and_test.py`
- `run.sh`
- log directories with histories, TensorBoard outputs, and prediction visualizations

Typical training behavior:

- multi-dataset support
- TensorFlow / Keras training loops
- early stopping
- TensorBoard logging
- reduce-on-plateau learning rate scheduling
- history and prediction plots written to per-run log folders

## Datasets supported in code

The common dataset scaffold in this directory supports:

- `mnist`
- `fashion_mnist`
- `cifar10`
- `cifar100`
- `skin_cancer`
- `cassava_leaf_disease`
- `chest_xray`
- `crop_disease`

These are the same broad dataset groups used by much of the larger classification directory, which makes cross-model comparisons easier.

## Implemented architecture families

### 1. LeNet-5

Folder: `1_Lenet5`

This is the smallest and most classical CNN in the directory. It is useful as a compact baseline and as the student model in some of the compression-oriented work.

Why it matters:

- historically important early CNN
- simple baseline for small-image tasks
- easy reference point for knowledge distillation and compression experiments

### 2. ZFNet

Folder: `2_ZFNet`

ZFNet is an AlexNet-era CNN refinement that became known for improved visualization and better understanding of convolutional feature hierarchies.

Why it matters:

- bridges older large-kernel CNNs and later cleaner deep architectures
- useful for comparing early convolution design choices

### 3. SqueezeNet

Folder: `3_SqueezeNet`

Implemented variants:

- `SqueezeNetNoBypass`
- `SqueezeNetSimpleBypass`
- `SqueezeNetComplexBypass`

SqueezeNet focuses on reducing parameter count while maintaining useful performance. The core idea is the Fire module, which uses squeeze and expand stages to limit expensive computation.

Why it matters:

- compact architecture with much smaller parameter footprint
- good case study for model size efficiency
- the bypass variants let you compare connectivity design inside the same family

### 4. XNOR-Net

Folder: `4_XNOR-Net`

This is the binary / low-precision efficiency-oriented model in the directory. XNOR-style architectures target much cheaper computation by binarizing activations and/or weights.

Why it matters:

- efficiency-first design
- relevant when deployment cost matters more than maximum accuracy

### 5. MobileNet

Folder: `5_MobileNet`

Implemented variants:

- `MobileNetV1`
- `MobileNetV2`
- `MobileNetV3`

These models are designed for efficient classification using depthwise separable convolutions and later architectural refinements.

Why they matter:

- mobile and edge deployment
- excellent compact baselines
- clear evolution from simple efficient CNNs to more optimized modern lightweight networks

### 6. ShuffleNet

Folder: `6_ShuffleNet`

Implemented variants:

- `ShuffleNetV1`
- `ShuffleNetV2`

ShuffleNet focuses on computational efficiency using grouped operations and channel shuffling.

Why it matters:

- efficiency at low compute budgets
- complements MobileNet by exploring a different lightweight design strategy

### 7. Knowledge Distillation

Folder: `7_KnowledgeDistillation`

This folder is methodological rather than a single architecture. It trains:

- a teacher model
- a smaller student model from scratch
- a student model via distillation

In the current code:

- the teacher is AlexNet-based
- the student is LeNet-based
- distillation uses a `Distiller` wrapper with teacher/student losses and temperature scaling

Why it matters:

- shows how to transfer knowledge from a larger model into a smaller one
- useful when you want a compact deployable model with better performance than plain small-model training

### 8. Deep Compression

Folder: `8_DeepCompression`

This folder focuses on model compression ideas rather than only designing a new backbone from scratch.

Why it matters:

- studies reducing model size and cost after training
- fits naturally with the efficiency theme of this directory

### 9. FractalNet

Folder: `9_FractalNet`

This folder exists, but it is not currently set up in the same runnable pattern as the main implemented subprojects.

Interpret it as:

- present in the repo
- not currently exposed through the same standard training scaffold

### 10. MLP-Mixer

Folder: `10_MLP-Mixer`

This folder is present, but it does not currently appear to have the same runnable project structure as the other active implementations.

Conceptually, it belongs to the “non-convolutional image classifier” family where token mixing and channel mixing are handled by MLP blocks instead of convolutions.

### 11. PolyNet

Folder: `11_PolyNet`

This folder is present, but like `MLP-Mixer` and `FractalNet`, it is not currently organized as a standard runnable subproject in this tree.

### 12. XceptionNet

Folder: `12_XceptionNet`

This is the most modern fully runnable architecture in the small-network directory. Xception extends depthwise separable convolution ideas into a stronger architecture than earlier lightweight CNNs.

Why it matters:

- efficient convolutional design
- good bridge between compact networks and stronger modern CNN performance

## Typical training flow

Most runnable small-network trainers follow the same pattern:

1. pick a dataset with `--type`
2. pick a GPU with `--gpu`
3. build tf.data datasets
4. compile the model with Adam and categorical cross-entropy
5. train with early stopping, TensorBoard, and LR reduction
6. evaluate on the test set
7. save `history.json`, prediction grids, and aggregate plots

Representative command:

```bash
python train_and_test.py --gpu 0 --type cifar10
```

Examples:

```bash
python train_and_test.py --gpu 0 --type mnist
python train_and_test.py --gpu 0 --type fashion_mnist
python train_and_test.py --gpu 0 --type chest_xray
python train_and_test.py --gpu 0 --type crop_disease
```

## Special folders

### Knowledge Distillation

`7_KnowledgeDistillation` differs from the others because it trains multiple models in one workflow:

- teacher
- distilled student
- student trained from scratch

This makes it one of the most practically useful folders if your goal is model compression rather than just architecture study.

### Deep Compression

`8_DeepCompression` is also more method-oriented than architecture-oriented. It belongs in this directory because its target is efficient classification under tighter resource constraints.

## Configuration patterns

The small-network folders are not perfectly uniform, but the general pattern is:

- image classification with resized inputs
- moderate batch sizes
- Adam optimization
- categorical cross-entropy for supervised classification

Some folders are closer to older-style CNN experiments, while others are clearly motivated by deployment efficiency.

## Practical notes about this codebase

- This directory mixes compact architectures with compression/training methods.
- A few folders are present as placeholders or partial experiments rather than fully runnable implementations.
- Many subfolders contain committed logs, so the repo stores both source code and experiment outputs together.
- The repeated trainer structure makes it easy to compare different small models on the same datasets.

## Suggested reading order

If you want to understand the progression of small and efficient classifiers in this repo:

1. `1_Lenet5`
2. `2_ZFNet`
3. `3_SqueezeNet`
4. `4_XNOR-Net`
5. `5_MobileNet`
6. `6_ShuffleNet`
7. `7_KnowledgeDistillation`
8. `8_DeepCompression`
9. `12_XceptionNet`

Treat `9_FractalNet`, `10_MLP-Mixer`, and `11_PolyNet` as partial/present folders unless you plan to flesh them out further.
