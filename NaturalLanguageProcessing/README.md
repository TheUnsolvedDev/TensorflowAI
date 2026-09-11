# 💬 Natural Language Processing

← [Back to the repository](../README.md)

This area progresses from token representations and recurrent classifiers to
sequence-to-sequence systems and a full local language-model project. Each
numbered source-backed folder is intentionally small enough to inspect;
configuration and data paths are local rather than globally packaged.

| Area | Implemented source-backed families | Documentation |
| --- | --- | --- |
| Word representations | One-hot encoding, co-occurrence, PPMI-SVD, CBOW, Skip-Gram, GloVe, FastText | [Open](WordEmbeddings/README.md) |
| Sequence classification | Vanilla RNN, LSTM, GRU, bidirectional RNN, stacked RNN | [Open](SequenceModels/README.md) |
| Seq2Seq | Basic encoder-decoder, attention, copy mechanism, coverage, Transformer seq2seq | [Open](Seq2Seq/README.md) |
| End-to-end project | Reddit scraping, corpus preparation, BPE, TFRecords, decoder-only Transformer, evaluation, sampling | [RedditStory](Project/RedditStory/README.md) |

## Scaffold directories

`AttentionMechanisms/`, `DiffusionText/`, `GenerativeModels/`,
`PretrainedLanguageModels/`, and `Transformers/` are present as numbered
learning-roadmap directories but contain no tracked Python implementation files
in this checkout. They are intentionally not described as implemented models.

## TensorFlow techniques

The source uses `tf.keras.Model` and custom layers for sequence and attention
architectures; text input paths use `tf.data` where a dataset implementation is
present. Several trainers use `tf.distribute.MirroredStrategy`, while the
RedditStory project additionally has a TFRecord pipeline, explicit causal mask
logic, `GradientTape` training, and optional XLA controls.

## Current limitations

The sequence-model and seq2seq dataset base classes contain explicit
`NotImplementedError` paths for adapters. Their pages distinguish the model
architecture from the incomplete dataset wiring, and no benchmark result is
claimed without a retained report.
