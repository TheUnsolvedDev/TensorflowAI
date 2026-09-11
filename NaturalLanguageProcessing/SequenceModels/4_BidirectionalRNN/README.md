# Bidirectional RNN

← [Sequence models](../README.md)

`build_bidirectional_rnn` combines forward and backward recurrent encodings so
each classification representation can use both left and right context. This
requires the whole input sequence and is therefore not causal inference. The
local dataset adapter is a scaffold.
