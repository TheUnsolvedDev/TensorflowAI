# 🔁 Sequence Models

← [NLP](../README.md) · [Repository](../../README.md)

This track compares recurrent sequence classifiers sharing a local configuration
and training shape: embedding input, recurrent encoder, classifier head, and
cross-entropy optimisation.

| Folder | Architecture | TensorFlow construction | Dataset status |
| --- | --- | --- | --- |
| [`1_VanillaRNN`](1_VanillaRNN/README.md) | Vanilla RNN | `build_vanilla_rnn` | Adapter base is incomplete |
| [`2_LSTM`](2_LSTM/README.md) | LSTM | `build_lstm` | Adapter base is incomplete |
| [`3_GRU`](3_GRU/README.md) | GRU | `build_gru` | Adapter base is incomplete |
| [`4_BidirectionalRNN`](4_BidirectionalRNN/README.md) | Bidirectional RNN | `build_bidirectional_rnn` | Adapter base is incomplete |
| [`5_StackedRNN`](5_StackedRNN/README.md) | Stacked RNN | `build_stacked_rnn` | Adapter base is incomplete |

The source references AG News, DBPedia, and IMDB adapters and uses `tf.data`
in dataset code. However, each `dataset.py` contains an explicit
`NotImplementedError`; architecture pages therefore describe the model but do
not claim an end-to-end verified dataset run.
