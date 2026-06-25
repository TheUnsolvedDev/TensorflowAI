# Large Network Image Classification

This directory contains larger image classification architectures implemented in TensorFlow. The collection is organized roughly as a historical progression from early deep CNNs to residual, attention, and transformer-based models.

The codebase is best understood as a model zoo of self-contained experiments rather than a single unified package.

## What is in this directory

Implemented families:

1. `1_AlexNet`
2. `2_VGGNet`
3. `3_NetworkInNetwork`
4. `4_InceptionNet`
5. `5_ResNet`
6. `6_HighwayNet`
7. `7_DenseNet`
8. `8_ResidualAttentionNet`
9. `9_SENet`
10. `10_ResNext`
11. `11_CapsuleNetwork`
12. `12_VisionTransformer`

## What is actually runnable

Runnable implementations with the usual `config.py`, `dataset.py`, `train_and_test.py`, and often `run.sh`:

- `1_AlexNet`
- `2_VGGNet/VGG11`
- `2_VGGNet/VGG11_LRN`
- `2_VGGNet/VGG13`
- `2_VGGNet/VGG16C`
- `2_VGGNet/VGG16D`
- `2_VGGNet/VGG19`
- `3_NetworkInNetwork`
- `4_InceptionNet/InceptionV1`
- `4_InceptionNet/InceptionV2`
- `4_InceptionNet/InceptionV3`
- `4_InceptionNet/InceptionV4`
- `5_ResNet/ResNet18`
- `5_ResNet/ResNet34`
- `5_ResNet/ResNet50`
- `5_ResNet/ResNet101`
- `5_ResNet/ResNet152`
- `6_HighwayNet`
- `7_DenseNet/DenseNet121`
- `7_DenseNet/DenseNet169`
- `7_DenseNet/DenseNet201`
- `7_DenseNet/DenseNet264`
- `8_ResidualAttentionNet/ResidualAttentionNet56`
- `8_ResidualAttentionNet/ResidualAttentionNet92`
- `9_SENet`
- `10_ResNext`
- `12_VisionTransformer`

Present but not structured like the other runnable subprojects:

- `11_CapsuleNetwork`

## Shared project pattern

Most runnable folders follow the same pattern:

- `config.py` defines input size, batch size, epochs, and learning rate
- `dataset.py` builds tf.data pipelines for multiple datasets
- `model.py` defines the architecture
- `train_and_test.py` handles training, validation, evaluation, plotting, and logging
- `run.sh` is often included for convenience

Typical outputs:

- TensorBoard logs
- `history.json`
- prediction grids such as `predictions.png`
- aggregate accuracy plots across datasets
- generated architecture diagrams from `plot_model`

## Datasets supported in code

The common dataset loaders in this directory support:

- `mnist`
- `fashion_mnist`
- `cifar10`
- `cifar100`
- `skin_cancer`
- `cassava_leaf_disease`
- `chest_xray`
- `crop_disease`

In practice, the image-based medical/agriculture datasets are read from local dataset roots, while MNIST/CIFAR are loaded via `tf.keras.datasets`.

## Implemented architecture families

### 1. AlexNet

Folder: `1_AlexNet`

A classic early deep CNN. Large kernels and stacked convolution blocks make it a useful historical starting point for modern image classification.

Good for:

- understanding the jump from shallow CNNs to large-scale deep vision models
- comparing older convolution design against later residual and efficient models

### 2. VGGNet

Folder: `2_VGGNet`

Implemented variants:

- `VGG11`
- `VGG11_LRN`
- `VGG13`
- `VGG16C`
- `VGG16D`
- `VGG19`

VGG-style models rely on repeated small `3x3` convolutions and very uniform stage design. They are heavy in parameter count but conceptually simple.

Why they matter:

- easy to reason about
- strong baseline for “plain deep CNN” design
- useful for comparing depth scaling without residual connections

### 3. Network in Network

Folder: `3_NetworkInNetwork`

This architecture replaces simple linear convolutional filters with small learned subnetworks, often through `1x1` convolutions.

Why it matters:

- pushes feature abstraction deeper inside each block
- an important step toward later bottleneck and inception-style designs

### 4. InceptionNet

Folder: `4_InceptionNet`

Implemented variants:

- `InceptionV1`
- `InceptionV2`
- `InceptionV3`
- `InceptionV4`

These models use parallel branches within a block to capture multiple receptive-field scales at once.

Why they matter:

- more expressive multi-scale feature extraction
- an important branch in CNN design before residual networks became dominant

### 5. ResNet

Folder: `5_ResNet`

Implemented variants:

- `ResNet18`
- `ResNet34`
- `ResNet50`
- `ResNet101`
- `ResNet152`

ResNet introduces residual skip connections, allowing much deeper optimization than plain stacked CNNs.

Why they matter:

- foundational modern vision architecture
- strong baseline for deeper supervised classification
- provides the template for many later families

### 6. HighwayNet

Folder: `6_HighwayNet`

Highway networks use learned gates to regulate how much transformed information versus carried-forward information passes through each block.

Why they matter:

- historically important precursor to residual connections
- useful for understanding gated depth before ResNet became the standard

### 7. DenseNet

Folder: `7_DenseNet`

Implemented variants:

- `DenseNet121`
- `DenseNet169`
- `DenseNet201`
- `DenseNet264`

DenseNet connects each block to all later blocks in the same dense stage, encouraging feature reuse and stronger gradient flow.

Why they matter:

- efficient feature reuse
- improved gradient propagation
- reduced need to relearn similar low-level features repeatedly

### 8. Residual Attention Network

Folder: `8_ResidualAttentionNet`

Implemented variants:

- `ResidualAttentionNet56`
- `ResidualAttentionNet92`

These models combine residual backbones with attention modules that emphasize informative spatial or feature responses.

Why they matter:

- introduces explicit attention into CNN classification
- bridges standard residual models and more structured attention-based networks

### 9. SENet

Folder: `9_SENet`

Squeeze-and-Excitation networks recalibrate channel responses by learning which feature channels should be emphasized or suppressed.

Why they matter:

- lightweight performance-oriented channel attention
- widely reused in later CNN families

### 10. ResNext

Folder: `10_ResNext`

ResNext extends the residual idea with grouped transformations, often described through cardinality rather than just depth or width.

Why they matter:

- stronger representational power without only scaling depth
- practical middle ground between plain residual blocks and more elaborate branch-heavy modules

### 11. Capsule Network

Folder: `11_CapsuleNetwork`

This folder exists in the repo, but it does not currently follow the same runnable structure as the other large-network implementations.

Interpret it as:

- present in the repo
- not yet integrated into the same training scaffold as the others

### 12. Vision Transformer

Folder: `12_VisionTransformer`

This is the transformer-based image classifier in the directory. Instead of relying purely on convolutions, it treats the image as a sequence of patch embeddings and applies transformer layers for classification.

Why it matters:

- marks the shift from CNN-dominant classification to transformer-based vision models
- useful for comparing convolutional inductive bias against patch-token modeling

## Typical training flow

Most large-network trainers follow this pattern:

1. choose a dataset with `--type`
2. choose a GPU with `--gpu`
3. build tf.data pipelines
4. compile the selected model with Adam and categorical cross-entropy
5. train with early stopping, TensorBoard, and learning-rate reduction
6. evaluate on the test set
7. save history and visualization artifacts

Representative command pattern:

```bash
python train_and_test.py --gpu 0 --type cifar10
```

Common dataset examples:

```bash
python train_and_test.py --gpu 0 --type mnist
python train_and_test.py --gpu 0 --type cifar100
python train_and_test.py --gpu 0 --type chest_xray
python train_and_test.py --gpu 0 --type skin_cancer
```

## Configuration patterns

Most of the classical CNN folders use roughly:

- `INPUT_SIZE = [224, 224, 3]`
- `BATCH_SIZE = 64`
- `EPOCHS = 10`
- `LEARNING_RATE = 1e-4`

Some later folders differ:

- `SENet` and `ResNext` use `64x64` input and longer training defaults
- `VisionTransformer` uses `224x224` but a separate config profile
- a few subfolders contain more experimental or inconsistent config values

## Practical notes about this codebase

- The directory mixes polished runnable subprojects with a few partial or experimental folders.
- Logging artifacts are committed in many subdirectories, so these folders contain both code and experiment history.
- Dataset path handling is not perfectly uniform across all subfolders.
- The training scripts are very similar across many models, which makes the directory easy to extend but also means there is some duplication.

## Suggested reading order

If you want to understand the progression of large image classifiers in this repo:

1. `1_AlexNet`
2. `2_VGGNet`
3. `3_NetworkInNetwork`
4. `4_InceptionNet`
5. `5_ResNet`
6. `6_HighwayNet`
7. `7_DenseNet`
8. `8_ResidualAttentionNet`
9. `9_SENet`
10. `10_ResNext`
11. `12_VisionTransformer`

`11_CapsuleNetwork` should be treated separately because it is not wired into the same common training structure.
