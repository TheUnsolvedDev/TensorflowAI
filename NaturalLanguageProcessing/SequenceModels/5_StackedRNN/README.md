# Stacked RNN

← [Sequence models](../README.md)

`build_stacked_rnn` composes multiple recurrent layers to increase abstraction
depth. Its complexity remains sequential in token length, with activation
memory proportional to batch, length, hidden size, and recurrent depth. The
model source exists; the corresponding dataset adapter is not complete.
