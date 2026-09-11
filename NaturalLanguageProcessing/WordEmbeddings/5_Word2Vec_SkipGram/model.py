import tensorflow as tf
def build_model(vocab_size,embedding_dim=128):
 x=tf.keras.Input(shape=(),dtype=tf.int32);e=tf.keras.layers.Embedding(vocab_size,embedding_dim)(x);return tf.keras.Model(x,tf.keras.layers.Dense(vocab_size,activation='softmax')(e),name='SkipGram')
