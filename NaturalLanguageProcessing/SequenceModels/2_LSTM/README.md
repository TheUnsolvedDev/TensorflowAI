# LSTM

← [Sequence models](../README.md)

`build_lstm` provides an embedding-to-LSTM-to-classifier architecture. LSTM
gates regulate information and gradient flow through the cell state. The model
is assembled through TensorFlow/Keras, but its shared dataset adapter contains
an explicit stub, so no dataset result is claimed.
