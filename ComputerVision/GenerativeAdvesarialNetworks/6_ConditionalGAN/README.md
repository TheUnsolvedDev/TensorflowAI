# Conditional GAN

← [GAN family](../README.md)

Conditional GAN training supplies class/condition information to both generator
and discriminator so sampling can target a requested category. The local model,
dataset, configuration, and trainer implement this conditioning path with
TensorFlow tensors and adversarial gradients.
