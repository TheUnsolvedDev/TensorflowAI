# Transformer Seq2Seq

← [Seq2Seq](../README.md)

Custom `PositionalEncoding`, `EncoderLayer`, `DecoderLayer`, and
`FeedForwardNetwork` classes implement the encoder-decoder Transformer path.
Self-attention and cross-attention make the dominant attention cost quadratic
in sequence length. The trainer provides masked attention and loss, a
Transformer warmup learning-rate schedule, resumable checkpoints, and
distribution-aware `tf.data` input. Datasets with published train/validation
files preserve those splits; ManyThings and Cornell use a deterministic split.
`--gpu -1` uses every visible GPU with NCCL, while `--gpu N` uses one GPU
without collective communication. Configured batch sizes are global batches.

`run.sh` trains the compact, high-signal datasets first and leaves the most
expensive CNN/DailyMail summarization run until last. Checkpoints produced by
the older oversized attention-head implementation are not shape-compatible
with this corrected model.

Each dataset configuration also sets its own epoch budget. `--epochs N`
overrides that budget when a longer or shorter experiment is intentional.
