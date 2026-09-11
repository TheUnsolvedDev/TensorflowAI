# SRGAN

← [GAN family](../README.md)

SRGAN is the super-resolution member of the GAN collection. `config.py`
declares content and adversarial loss weights, while `SRGAN` in `model.py`
optimises generator/discriminator components with Adam. The objective combines
content reconstruction and a small adversarial term; source images and results
are local prerequisites.
