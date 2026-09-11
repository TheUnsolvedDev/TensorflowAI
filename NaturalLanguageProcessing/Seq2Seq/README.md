# 🔄 Sequence-to-Sequence Models

← [NLP](../README.md) · [Repository](../../README.md)

The sequence-to-sequence folders explore encoder-decoder translation/generation
mechanics with explicit TensorFlow architecture builders and local configuration
files.

| Folder | Algorithm | Key source component | Dataset status |
| --- | --- | --- | --- |
| [`1_BasicEncoderDecoder`](1_BasicEncoderDecoder/README.md) | Recurrent encoder-decoder | `build_seq2seq_model` | Adapter contains a stub |
| [`2_Seq2Seq_with_Attention`](2_Seq2Seq_with_Attention/README.md) | Cross-attention seq2seq | `build_cross_attention` | Adapter contains a stub |
| [`3_CopyMechanism`](3_CopyMechanism/README.md) | Pointer-generator | `PointerGenerator` | Adapter contains a stub |
| [`4_CoverageModel`](4_CoverageModel/README.md) | Coverage attention | `CoverageAttention` | Adapter contains a stub |
| [`5_TransformerSeq2Seq`](5_TransformerSeq2Seq/README.md) | Transformer encoder-decoder | custom encoder/decoder layers | Adapter contains a stub |

Configurations define local dataset roots, batch sizes, epochs, and learning
rates. The model code is useful for tracing architecture construction, but the
dataset layers need completion before the folders can be presented as complete
end-to-end experiments.
