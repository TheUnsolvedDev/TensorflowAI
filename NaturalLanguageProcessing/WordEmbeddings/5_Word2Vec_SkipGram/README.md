# Word2Vec Skip-Gram

← [Word embeddings](../README.md)

The Skip-Gram implementation reverses the CBOW prediction direction: a centre
token predicts neighbouring tokens. `dataset.py` forms centre/context examples,
`model.py` defines `build_model`, and `train_and_test.py` executes the local
training path. It uses Keras/TensorFlow embeddings and categorical prediction;
per-example cost depends on context examples and output-vocabulary projection.
