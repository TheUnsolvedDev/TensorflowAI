# GloVe

← [Word embeddings](../README.md)

`GloVeResidual` and `build_model` implement a learned count-factorisation path.
The intended objective fits embedding dot products and biases to log
co-occurrence counts with a weighting function. The local dataset code supplies
count-derived pairs; the trainer is distribution-aware when multiple GPUs are
visible. Results and corpus files are local-only.
