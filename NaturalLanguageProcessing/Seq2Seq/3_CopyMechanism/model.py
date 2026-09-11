import tensorflow as tf


class PointerGenerator(tf.keras.layers.Layer):
    """Mixes vocabulary generation with attention mass copied from source ids."""
    def __init__(self, target_vocab_size, **kwargs):
        super().__init__(**kwargs); self.target_vocab_size = target_vocab_size
        self.gate = tf.keras.layers.Dense(1, activation="sigmoid")
        self.projection = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        decoder_states, encoder_states, source_ids = inputs
        scores = tf.einsum("btd,bsd->bts", decoder_states, encoder_states)
        source_mask = tf.cast(tf.not_equal(source_ids, 0), scores.dtype)[:, None, :]
        attention = tf.nn.softmax(scores + (1.0 - source_mask) * tf.cast(-1e9, scores.dtype), axis=-1)
        context = tf.einsum("bts,bsd->btd", attention, encoder_states)
        generated = tf.nn.softmax(self.projection(tf.concat([decoder_states, context], axis=-1)), axis=-1)
        copied = tf.einsum("bts,bsv->btv", attention, tf.one_hot(source_ids, self.target_vocab_size, dtype=attention.dtype))
        probability = self.gate(tf.concat([decoder_states, context], axis=-1))
        return probability * generated + (1.0 - probability) * copied


def build_copy_seq2seq_model(source_vocab_size, target_vocab_size, source_max_length=32,
                              target_max_length=32, embedding_dim=128, units=128):
    source = tf.keras.Input((source_max_length,), dtype=tf.int64, name="encoder_inputs")
    target = tf.keras.Input((target_max_length - 1,), dtype=tf.int64, name="decoder_inputs")
    source_embedding = tf.keras.layers.Embedding(source_vocab_size, embedding_dim, mask_zero=True)(source)
    encoder_output, state = tf.keras.layers.GRU(units, return_sequences=True, return_state=True, name="encoder")(source_embedding)
    target_embedding = tf.keras.layers.Embedding(target_vocab_size, embedding_dim, mask_zero=True)(target)
    decoder_output = tf.keras.layers.GRU(units, return_sequences=True, name="decoder")(target_embedding, initial_state=state)
    output = PointerGenerator(target_vocab_size, name="pointer_generator")([decoder_output, encoder_output, source])
    return tf.keras.Model([source, target], output, name="copy_mechanism_seq2seq")
