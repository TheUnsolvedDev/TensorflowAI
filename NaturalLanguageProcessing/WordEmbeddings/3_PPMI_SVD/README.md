# PPMI + SVD

← [Word embeddings](../README.md)

`ppmi_svd` in `model.py` converts a co-occurrence matrix into a positive
pointwise-mutual-information representation and factorises it. For counts
\(C\), PPMI keeps \(\max(0,\log\frac{p(i,j)}{p(i)p(j)})\); truncated SVD then
forms low-dimensional factors. This is a matrix-factorisation baseline, not a
gradient-trained TensorFlow model. Runtime and memory grow with vocabulary and
the nonzero/dense matrix representation.
