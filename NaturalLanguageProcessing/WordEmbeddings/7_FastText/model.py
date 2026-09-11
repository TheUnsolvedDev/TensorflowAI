import tensorflow as tf
from config import MAX_SUBWORDS, NGRAM_BUCKETS

def build_model(vocab_size, embedding_dim=128):
    word_id = tf.keras.Input((), dtype=tf.int32, name="word_id")
    subword_ids = tf.keras.Input((MAX_SUBWORDS,), dtype=tf.int32, name="subword_ids")
    word = tf.keras.layers.Embedding(vocab_size, embedding_dim, name="word_embedding")(word_id)
    subwords = tf.keras.layers.Embedding(NGRAM_BUCKETS, embedding_dim, mask_zero=True, name="subword_embedding")(subword_ids)
    subword_mean = tf.keras.layers.GlobalAveragePooling1D()(subwords)
    combined = tf.keras.layers.Add()([word, subword_mean])
    output = tf.keras.layers.Dense(vocab_size, activation="softmax", dtype="float32", name="context_distribution")(combined)
    return tf.keras.Model({"word_id": word_id, "subword_ids": subword_ids}, output, name="fasttext_skipgram")
