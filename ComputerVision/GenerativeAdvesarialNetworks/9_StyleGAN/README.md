# StyleGAN

← [GAN family](../README.md)

The StyleGAN folder implements a style-oriented generator path and adversarial
training in custom TensorFlow code. `model.py` uses `GradientTape` and compiled
training functions; `dataset.py` supplies data and `train_and_test.py` drives
the configuration-controlled run. No numerical comparison is retained.
