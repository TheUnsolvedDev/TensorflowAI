# Transformer Seq2Seq

← [Seq2Seq](../README.md)

Custom `PositionalEncoding`, `EncoderLayer`, `DecoderLayer`, and
`FeedForwardNetwork` classes implement the encoder-decoder Transformer path.
Self-attention and cross-attention make the dominant attention cost quadratic
in sequence length. The trainer contains `tf.data` and distribution-aware
paths, but the dataset adapter has an explicit unimplemented method.
