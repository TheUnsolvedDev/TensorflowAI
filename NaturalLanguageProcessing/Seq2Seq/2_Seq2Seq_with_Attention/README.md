# Seq2Seq with Attention

← [Seq2Seq](../README.md)

The model adds `build_cross_attention` to the recurrent encoder-decoder path.
For decoder state \(q_t\) and encoder states \(h_i\), attention computes a
weighted context \(\sum_i \alpha_{ti}h_i\). The source includes an attention
inference model, but the local dataset implementation is incomplete.
