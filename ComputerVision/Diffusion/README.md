# Diffusion Models

This directory contains standalone diffusion-family implementations in the same repo style used by the neighboring computer-vision families: serially numbered folders, no shared runtime helpers, local dataset routing, local checkpoints, and TensorFlow/Keras-first multi-GPU training entrypoints.

## Folder Order

1. `1_DiffusionProbabilisticModels`
2. `2_DDPM`
3. `3_ImprovedDDPM`
4. `4_DDIM`
5. `5_GLIDE`
6. `6_Imagen`
7. `7_DALLE2`
8. `8_StableDiffusion`
9. `9_ControlNet`
10. `10_ConsistencyModels`

## Common Structure

Each folder is standalone and includes:

- `config.py` for dataset roots and hyperparameters
- `dataset.py` for fully local dataset loading and preprocessing
- `model.py` for the architecture and training logic of that folder only
- `train_and_test.py` for CLI, multi-GPU setup, checkpoints, resume, and sampling
- `run.sh` for repeated local runs
- `README.md` for scope and dataset notes

## Practical Scope

These folders are designed to be recognizable implementations rather than full paper-scale reproductions. The code keeps the repo's from-scratch style, conservative defaults, local caption parsing, and practical multi-GPU support.

Text-conditioned folders default to local COCO captions and support Flickr30k as a lighter fallback. `ControlNet` uses on-the-fly edge conditioning, and latent-diffusion folders keep their VAE and latent UNet local to each folder rather than sharing code.
