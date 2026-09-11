# Wasserstein GAN

← [GAN family](../README.md)

The WGAN implementation uses a critic-style objective rather than a
probability discriminator. `model.py` exposes custom TensorFlow model/layer
logic, `GradientTape`, `tf.function`, and `tf.data` usage. The practical goal
is to optimise a critic score gap between real and generated batches; training
details and dataset selection remain folder-local.
