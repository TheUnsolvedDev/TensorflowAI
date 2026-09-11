# CycleGAN

← [GAN family](../README.md)

CycleGAN learns mappings between two domains using paired generators and
discriminators. In addition to adversarial loss, cycle consistency encourages
\(F(G(x))\approx x\) and \(G(F(y))\approx y\). The folder contains TensorFlow
model, dataset, configuration, and training code; local data availability is
required.
