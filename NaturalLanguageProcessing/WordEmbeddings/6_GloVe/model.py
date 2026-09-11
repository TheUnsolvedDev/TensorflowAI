import tensorflow as tf
from config import ALPHA, X_MAX

class GloVeResidual(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.word = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.context = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.word_bias = tf.keras.layers.Embedding(vocab_size, 1)
        self.context_bias = tf.keras.layers.Embedding(vocab_size, 1)
    def call(self, inputs):
        center, context, count = inputs
        prediction = tf.reduce_sum(self.word(center) * self.context(context), axis=-1)
        prediction += tf.squeeze(self.word_bias(center), -1) + tf.squeeze(self.context_bias(context), -1)
        weight = tf.pow(tf.minimum(count / X_MAX, 1.0), ALPHA)
        return tf.sqrt(weight) * (prediction - tf.math.log(tf.maximum(count, 1e-6)))

def build_model(vocab_size, embedding_dim=128):
    center = tf.keras.Input((), dtype=tf.int32, name="center")
    context = tf.keras.Input((), dtype=tf.int32, name="context")
    count = tf.keras.Input((), dtype=tf.float32, name="count")
    return tf.keras.Model({"center": center, "context": context, "count": count}, GloVeResidual(vocab_size, embedding_dim)([center, context, count]), name="glove")
