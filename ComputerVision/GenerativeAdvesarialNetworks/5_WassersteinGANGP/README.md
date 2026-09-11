# WGAN with Gradient Penalty

← [GAN family](../README.md)

This variant adds a gradient-norm regulariser on interpolated samples to enforce
the critic's Lipschitz constraint without weight clipping. The key term is
\(\lambda(\lVert\nabla_{\hat{x}}D(\hat{x})\rVert_2-1)^2\), computed through
TensorFlow gradients. The repository retains implementation code, not a result
benchmark.
