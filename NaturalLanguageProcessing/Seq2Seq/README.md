# Seq2Seq

Each model folder has an independent `BaseSeq2SeqDataset` plus concrete local
adapters for WMT English-French, WMT English-German, ManyThings French-English,
Cornell dialogue, CNN/DailyMail, and WikiLarge. Source records are reopened for
vectorizer adaptation and every epoch; only compact vectorizer metadata is
saved.

```bash
cd 2_Seq2Seq_with_Attention
bash run.sh manythings_english_french --smoke
```

The five implementations are basic encoder-decoder, attention, copy mechanism,
coverage model, and Transformer. Multi-GPU uses memory growth plus NCCL
`MirroredStrategy` when available. Do not use `run.sh` to infer GPU readiness:
verify the visible device count and execute a bounded smoke run first.
