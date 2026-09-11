# FastText-style Embeddings

← [Word embeddings](../README.md)

This implementation's `build_model` and dataset path extend word embedding
experiments with subword-aware representation. Character n-gram composition is
useful for sharing information among morphologically related or rare words.
The folder uses TensorFlow/Keras training code and a local corpus configuration;
it has no tracked benchmark report.
