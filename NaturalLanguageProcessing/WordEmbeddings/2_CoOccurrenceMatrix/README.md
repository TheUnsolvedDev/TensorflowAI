# Co-Occurrence Matrix

← [Word embeddings](../README.md)

This folder constructs context-count statistics rather than training a neural
embedding model. `dataset.py` prepares token/context pairs, `model.py` exposes
`sparse_matrix`, and `train_and_test.py` runs the transformation. The central
object is \(C_{ij}\), the number of times context token \(j\) occurs near token
\(i\). Space is governed by stored nonzero pairs; dense materialisation costs
\(O(|V|^2)\). No benchmark artifact is retained.
