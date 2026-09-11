# Deep Convolutional GAN

← [GAN family](../README.md)

The DCGAN folder applies convolutional generator/discriminator components to
image synthesis. Its model and trainer contain explicit compiled TensorFlow
training paths (`tf.function`) and gradient-based adversarial updates. Per-step
cost is dominated by convolutional feature maps; batch size, dataset and
checkpoint/sample cadence are configuration-driven.
