# GRU

← [Sequence models](../README.md)

`build_gru` uses an embedding layer, GRU encoder, and classification head.
Update and reset gates provide a lighter gated recurrence than an LSTM. The
folder has a training script and `MirroredStrategy`-aware code, while its data
adapter remains explicitly incomplete.
