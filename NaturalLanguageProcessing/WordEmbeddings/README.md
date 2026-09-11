# 🔤 Word Embeddings

← [NLP](../README.md) · [Repository](../../README.md)

These standalone folders move from explicit sparse representations to learned
dense embeddings. Matrix-only demonstrations are deliberately bounded local
transforms; learned models use TensorFlow/Keras models and their local training
scripts.

| Folder | Algorithm | Core operation | Status |
| --- | --- | --- | --- |
| [`1_OneHotEncoding`](1_OneHotEncoding/README.md) | One-hot baseline | Categorical vector representation | Source-backed |
| [`2_CoOccurrenceMatrix`](2_CoOccurrenceMatrix/README.md) | Co-occurrence | Sparse count matrix | Source-backed |
| [`3_PPMI_SVD`](3_PPMI_SVD/README.md) | PPMI + SVD | Reweighted matrix factorisation | Source-backed |
| [`4_Word2Vec_CBOW`](4_Word2Vec_CBOW/README.md) | CBOW | Context-to-target prediction | Source-backed |
| [`5_Word2Vec_SkipGram`](5_Word2Vec_SkipGram/README.md) | Skip-Gram | Target-to-context prediction | Source-backed |
| [`6_GloVe`](6_GloVe/README.md) | GloVe | Weighted log-count regression | Source-backed |
| [`7_FastText`](7_FastText/README.md) | FastText-style model | Subword-aware token representation | Source-backed |

Run from a selected folder with its local command, for example
`bash run.sh shakespeare.txt --smoke`. Corpus and output paths are local; no
training metrics are retained as repository-wide benchmarks.
