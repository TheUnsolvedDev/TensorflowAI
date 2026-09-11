# Progressive GAN

← [GAN family](../README.md)

The ProgressiveGAN folder contains an adversarial TensorFlow model and a local
training/dataset pipeline intended for staged image generation. Generator and
discriminator use separate optimisers defined in `config.py`; `GradientTape`
drives their updates. The tracked project contains source only, not a trained
progression or benchmark.
