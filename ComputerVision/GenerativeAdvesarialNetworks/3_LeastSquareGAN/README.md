# Least Squares GAN

← [GAN family](../README.md)

LSGAN replaces binary cross-entropy adversarial terms with least-squares
targets, reducing saturation in the discriminator signal. The custom TensorFlow
model and training loop are in `model.py` and `train_and_test.py`; local
configuration selects data and run settings. No tracked FID or visual result is
claimed.
