# Coverage-Aware Seq2Seq

← [Seq2Seq](../README.md)

`CoverageAttention` tracks previously allocated attention mass to discourage
repeated focus on the same source positions. The architecture maps this state
into the attention score computation. It is source-backed model code with an
incomplete dataset adapter and no retained evaluation results.
