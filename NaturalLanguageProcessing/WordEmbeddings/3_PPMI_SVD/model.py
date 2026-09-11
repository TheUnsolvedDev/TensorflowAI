import tensorflow as tf

def ppmi_svd(cooccurrence, embedding_dim):
    matrix = tf.convert_to_tensor(cooccurrence, tf.float32)
    total = tf.reduce_sum(matrix); row = tf.reduce_sum(matrix, axis=1, keepdims=True); column = tf.reduce_sum(matrix, axis=0, keepdims=True)
    ppmi = tf.maximum(tf.math.log(tf.maximum(matrix * total / tf.maximum(row * column, 1e-12), 1e-12)), 0.0)
    singular, left, _ = tf.linalg.svd(ppmi, full_matrices=False)
    width = tf.minimum(tf.shape(singular)[0], embedding_dim)
    return left[:, :width] * tf.sqrt(singular[:width])
