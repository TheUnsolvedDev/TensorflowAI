# Basic Encoder–Decoder

← [Seq2Seq](../README.md)

`build_encoder`, `build_decoder`, and `build_seq2seq_model` construct a
recurrent encoder-decoder. The encoder state conditions the decoder, which
predicts target tokens autoregressively. Configuration supplies lengths,
embedding dimensions, units, batch size, and learning rate. Dataset adapters
are explicit scaffolds, so no translation score is claimed.
