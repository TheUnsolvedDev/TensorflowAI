# One-Hot Encoding

← [Word embeddings](../README.md)

`model.py` builds the baseline embedding/classification model; `dataset.py`
creates context examples and `train_and_test.py` runs the local experiment.
One-hot vectors represent vocabulary items as rows of the identity matrix, so
memory is \(O(|V|)\) per dense vector. The implementation uses TensorFlow
categorical operations and Keras model layers. Configure corpus path, batch
size, epochs, and learning rate in `config.py`; no retained result table exists.
