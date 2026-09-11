import tensorflow as tf
from config import WINDOW_SIZE

def build_model(vocab_size, embedding_dim=128):
    context = tf.keras.Input((2 * WINDOW_SIZE,), dtype=tf.int32, name="context_ids")
    embeddings = tf.keras.layers.Embedding(vocab_size, embedding_dim, name="input_embedding")(context)
    pooled = tf.keras.layers.GlobalAveragePooling1D(name="context_mean")(embeddings)
    output = tf.keras.layers.Dense(vocab_size, activation="softmax", name="target_distribution", dtype="float32")(pooled)
    return tf.keras.Model(context, output, name="word2vec_cbow")
