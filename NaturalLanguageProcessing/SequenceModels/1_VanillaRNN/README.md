# Vanilla RNN

← [Sequence models](../README.md)

`build_vanilla_rnn` constructs embedding, recurrent, and classification layers.
The recurrence carries hidden state through token positions, with sequential
time dependence limiting parallelism across sequence length. `dataset.py` is
currently incomplete, so this documents the architecture rather than a proven
end-to-end training run.
