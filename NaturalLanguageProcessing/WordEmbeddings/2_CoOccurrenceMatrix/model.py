import tensorflow as tf

def sparse_matrix(vocab_size, indices, values):
    """Build an in-memory sparse co-occurrence matrix; it is never cached to disk."""
    return tf.sparse.reorder(tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=(vocab_size, vocab_size)))
