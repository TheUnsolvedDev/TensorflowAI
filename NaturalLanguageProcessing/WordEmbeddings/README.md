# Word Embeddings

Every numbered folder is standalone. The gradient-trained folders use memory
growth and `MirroredStrategy` with NCCL when multiple GPUs are visible; the
matrix-only Co-Occurrence and PPMI-SVD demonstrations intentionally run as
bounded local transformations rather than distributed gradient training.

Run from a model folder:

```bash
bash run.sh shakespeare.txt --smoke
```

`1_OneHotEncoding`, `4_Word2Vec_CBOW`, `5_Word2Vec_SkipGram`, `6_GloVe`, and
`7_FastText` produce model artifacts only after training. Corpus examples are
read repeatedly from source; no `tf.data` or disk preprocessing cache is used.
