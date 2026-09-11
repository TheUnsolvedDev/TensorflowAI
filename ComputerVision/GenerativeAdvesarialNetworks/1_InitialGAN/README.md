# Initial GAN

← [GAN family](../README.md)

This folder introduces adversarial training with generator and discriminator
models, local dataset loading, and `train_and_test.py`. The minimax objective
trains \(G(z)\) to fool \(D(x)\), while \(D\) separates real from generated
samples. TensorFlow `GradientTape` updates the two networks separately. Dataset
and output settings live in `config.py`; generated samples are local-only.
