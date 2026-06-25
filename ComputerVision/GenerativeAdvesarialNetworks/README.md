# Generative Adversarial Networks

This directory contains a progression of GAN implementations in TensorFlow, starting from a simple fully connected GAN and moving toward more stable and expressive variants such as DCGAN, LSGAN, WGAN, WGAN-GP, and a conditional GAN.

The folders are ordered as a learning path:

1. `1_InitialGAN`
2. `2_DeepConvolutionalGAN`
3. `3_LeastSquareGAN`
4. `4_WassersteinGAN`
5. `5_WassersteinGANGP`
6. `6_ConditionalGAN`

## Common structure

Most subprojects follow the same layout:

- `config.py` for hyperparameters
- `dataset.py` for loading and preprocessing datasets
- `model.py` for the generator, discriminator, and training logic
- `train_and_test.py` as the main training entry point
- `run.sh` in most folders for convenience
- `logs/` for saved weights, samples, training state, and history

Common defaults across the implementations:

- base image size: `32x32`
- latent dimension: `64`
- generator learning rate: `1e-4`
- discriminator learning rate: `1e-4`
- normalization to `[-1, 1]`
- sample image grids saved during training

For larger image datasets such as CelebA or anime faces, some implementations switch to larger input resolution and a larger latent space.

## Implemented GAN variants

### 1. Initial GAN

Folder: `1_InitialGAN`

This is the most basic implementation in the collection. The generator and discriminator are primarily dense networks:

- generator: latent vector -> dense layers -> reshaped image
- discriminator: flattened image -> dense layers -> real/fake score

Training details:

- standard binary cross-entropy GAN loss
- spectral normalization is used in the discriminator dense blocks
- small Gaussian noise is added to real and generated images during discriminator training

This version is useful as a baseline because it shows the original adversarial setup without convolution-heavy architecture changes.

### 2. Deep Convolutional GAN

Folder: `2_DeepConvolutionalGAN`

This version moves from fully connected image synthesis to a convolutional generator/discriminator design. The architecture uses residual-style upsampling and downsampling blocks:

- generator grows images from a learned `4x4` feature map using residual upsampling blocks
- discriminator reduces images using residual downsampling blocks and global pooling

Training details:

- binary cross-entropy objective
- one-sided label smoothing for real images
- Gaussian instance noise added during discriminator training

This is the first implementation in the directory that is structurally closer to practical image GANs.

### 3. Least Squares GAN

Folder: `3_LeastSquareGAN`

LSGAN keeps a very similar convolutional architecture to the DCGAN-style implementation, but changes the adversarial objective:

- replaces binary cross-entropy with mean squared error
- pushes real predictions toward `1` and fake predictions toward `0`

Why this matters:

- the least-squares objective can provide smoother gradients
- it often reduces vanishing-gradient behavior compared with the original GAN loss

This implementation is useful when the standard GAN objective becomes unstable or produces weak discriminator gradients.

### 4. Wasserstein GAN

Folder: `4_WassersteinGAN`

This implementation changes the optimization target from classification loss to Wasserstein distance estimation:

- generator loss: maximize critic score on fake samples
- critic loss: maximize the score gap between real and fake samples

Training details:

- uses a critic rather than a probability discriminator
- applies weight clipping to critic parameters
- trains the critic more times than the generator (`N_DISC_STEP = 3`)

Why it exists:

- WGAN is intended to improve training stability
- critic scores are often more meaningful than BCE losses for monitoring progress

This is the classic pre-gradient-penalty Wasserstein formulation.

### 5. Wasserstein GAN with Gradient Penalty

Folder: `5_WassersteinGANGP`

This variant improves on WGAN by replacing weight clipping with gradient penalty:

- interpolates between real and fake images
- penalizes the critic when gradient norm deviates from `1`
- uses `LAMBDA_GP = 10.0`

Why it matters:

- more stable than clipped WGAN in many settings
- avoids the capacity and optimization issues caused by weight clipping

In this repo, the architecture is very close to the WGAN version, so the main conceptual difference is the critic regularization.

### 6. Conditional GAN

Folder: `6_ConditionalGAN`

This implementation extends the WGAN-GP-style setup by conditioning both generator and discriminator on labels:

- generator takes noise and a label vector
- discriminator takes an image and the corresponding label vector
- labels are merged into the model through learned dense projections

Supported conditioning modes in the code:

- one-hot conditioning for datasets like MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100
- multi-label conditioning for CelebA using attribute annotations

This is the most expressive GAN implementation in the directory because it allows controlled generation rather than unconditional sampling.

## Datasets supported in code

Across these folders, the dataset loaders support combinations of:

- `mnist`
- `fashion_mnist`
- `cifar10`
- `cifar100`
- `celeba`
- `anime_faces` in the unconditional variants

Notes:

- unconditional GAN folders generally treat all images as belonging to one distribution
- `6_ConditionalGAN` uses labels explicitly
- CelebA in the conditional implementation uses attribute vectors rather than single-class labels

## Practical differences between the folders

| Folder | Main change | Loss style | Conditioning |
| --- | --- | --- | --- |
| `1_InitialGAN` | Dense baseline GAN | BCE | No |
| `2_DeepConvolutionalGAN` | Convolutional / residual image generator | BCE | No |
| `3_LeastSquareGAN` | Least-squares objective | MSE | No |
| `4_WassersteinGAN` | Wasserstein critic + weight clipping | Wasserstein | No |
| `5_WassersteinGANGP` | Gradient penalty instead of clipping | Wasserstein + GP | No |
| `6_ConditionalGAN` | Label-conditioned generation | Wasserstein + GP style critic training | Yes |

## How to run

From inside any GAN subfolder, the common training pattern is:

```bash
python train_and_test.py --gpu -1 --type cifar10
```

Common dataset examples:

```bash
python train_and_test.py --gpu 0 --type mnist
python train_and_test.py --gpu 0 --type fashion_mnist
python train_and_test.py --gpu 0 --type cifar10
python train_and_test.py --gpu 0 --type celeba
```

Resume flags are not fully uniform:

- most unconditional folders use `--continue`
- `6_ConditionalGAN` uses `--resume`

## Outputs produced during training

Each implementation saves a mix of:

- `generator.weights.h5`
- `discriminator.weights.h5`
- `training_state.json`
- `history.json`
- generated sample grids under `logs/.../samples/`
- a final generated grid after training

Some folders also include architecture diagrams such as `generator.png` and `discriminator.png`.

## Suggested learning order

If you want to study the evolution of GAN training in this repo, the best order is:

1. `1_InitialGAN` to understand the original adversarial setup
2. `2_DeepConvolutionalGAN` to see why convolutional generators work better for images
3. `3_LeastSquareGAN` to understand how changing the loss affects stability
4. `4_WassersteinGAN` to move from classifier-based discrimination to critic-based training
5. `5_WassersteinGANGP` to see the more stable Wasserstein formulation
6. `6_ConditionalGAN` to add controllable generation

## Notes on the current codebase

- The directory name is spelled `GenerativeAdvesarialNetworks` in the repo, so commands and paths need to use that exact spelling.
- The implementations are research/learning code and save many artifacts directly into local `logs/` folders.
- There is shared design reuse across later folders, especially from DCGAN -> LSGAN -> WGAN -> WGAN-GP.

If you want, the next useful step would be to add per-folder README files with exact architecture notes, supported datasets, and example results for each GAN individually.
