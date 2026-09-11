# Word2Vec CBOW

← [Word embeddings](../README.md)

The CBOW folder maps a context window to its centre token. `build_model` in
`model.py` combines context embeddings before a vocabulary prediction layer;
`dataset.py` creates windows and `train_and_test.py` controls training. The
objective is categorical likelihood of the target token given its context.
Tensor cost is dominated by embedding lookup and vocabulary projection. Local
hyperparameters live in `config.py`; no tracked training metric is available.
