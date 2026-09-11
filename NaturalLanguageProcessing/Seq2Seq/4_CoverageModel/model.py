import tensorflow as tf


class CoverageAttention(tf.keras.layers.Layer):
    """Additive coverage penalty discourages repeated attention to source tokens."""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.encoder_projection = tf.keras.layers.Dense(units, use_bias=False)
        self.decoder_projection = tf.keras.layers.Dense(units, use_bias=False)
        self.coverage_projection = tf.keras.layers.Dense(units, use_bias=False)
        self.score_projection = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, inputs):
        encoder_states, decoder_states, source_ids = inputs
        encoder = self.encoder_projection(encoder_states)[:, None, :, :]
        decoder = self.decoder_projection(decoder_states)[:, :, None, :]
        raw_scores = self.score_projection(tf.nn.tanh(encoder + decoder))[..., 0]
        initial = tf.nn.softmax(raw_scores, axis=-1)
        coverage = tf.cumsum(initial, axis=1, exclusive=True)
        penalty = tf.reduce_sum(self.coverage_projection(coverage[..., None]), axis=-1)
        mask = tf.cast(tf.not_equal(source_ids, 0), raw_scores.dtype)[:, None, :]
        attention = tf.nn.softmax(raw_scores - penalty + (1.0 - mask) * tf.cast(-1e9, raw_scores.dtype), axis=-1)
        return tf.einsum("bts,bsd->btd", attention, encoder_states)


def build_coverage_seq2seq_model(source_vocab_size, target_vocab_size, source_max_length=32,
                                  target_max_length=32, embedding_dim=128, units=128):
    source = tf.keras.Input((source_max_length,), dtype=tf.int64, name="encoder_inputs")
    target = tf.keras.Input((target_max_length - 1,), dtype=tf.int64, name="decoder_inputs")
    source_embedding = tf.keras.layers.Embedding(source_vocab_size, embedding_dim, mask_zero=True)(source)
    encoder_output, state = tf.keras.layers.GRU(units, return_sequences=True, return_state=True, name="encoder")(source_embedding)
    target_embedding = tf.keras.layers.Embedding(target_vocab_size, embedding_dim, mask_zero=True)(target)
    decoder_output = tf.keras.layers.GRU(units, return_sequences=True, name="decoder")(target_embedding, initial_state=state)
    context = CoverageAttention(units, name="coverage_attention")([encoder_output, decoder_output, source])
    output = tf.keras.layers.Dense(target_vocab_size, activation="softmax", name="token_distribution")(tf.concat([decoder_output, context], axis=-1))
    return tf.keras.Model([source, target], output, name="coverage_seq2seq")
