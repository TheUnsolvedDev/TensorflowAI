# model.py

import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, max_length, d_model):
        super().__init__()

        positions = tf.range(max_length, dtype=tf.float32)[:, tf.newaxis]
        dimensions = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]

        angle_rates = 1.0 / tf.pow(10000.0, (2 * (dimensions // 2)) / tf.cast(d_model, tf.float32))

        angle_rads = positions * angle_rates

        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        positional_encoding = tf.concat([sines, cosines], axis=-1)
        positional_encoding = positional_encoding[tf.newaxis, ...]

        self.positional_encoding = tf.cast(positional_encoding, tf.float32)

    def call(self, x):
        sequence_length = tf.shape(x)[1]
        return x + self.positional_encoding[:, :sequence_length, :]


class FeedForwardNetwork(tf.keras.layers.Layer):

    def __init__(self, d_model, dff, dropout_rate):
        super().__init__()

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation="relu"),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(d_model)
        ])

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        ffn_output = self.ffn(x, training=training)
        return self.layer_norm(x + ffn_output)


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, dropout_rate):
        super().__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = FeedForwardNetwork(d_model=d_model, dff=dff, dropout_rate=dropout_rate)

    def call(self, x, training=False):

        attention_output = self.mha(query=x, value=x, key=x, training=training)

        attention_output = self.dropout(attention_output, training=training)

        x = self.layer_norm(x + attention_output)

        return self.ffn(x, training=training)


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, dropout_rate):
        super().__init__()

        self.self_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

        self.cross_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)

        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)

        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = FeedForwardNetwork(d_model=d_model, dff=dff, dropout_rate=dropout_rate)

    def call(self, x, encoder_outputs, training=False):

        self_attention_output = self.self_attention(query=x, value=x, key=x, use_causal_mask=True, training=training)

        self_attention_output = self.dropout_1(self_attention_output, training=training)

        x = self.layer_norm_1(x + self_attention_output)

        cross_attention_output, attention_scores = self.cross_attention(query=x, value=encoder_outputs, key=encoder_outputs, return_attention_scores=True, training=training)

        cross_attention_output = self.dropout_2(cross_attention_output, training=training)

        x = self.layer_norm_2(x + cross_attention_output)

        x = self.ffn(x, training=training)

        return x, attention_scores


def build_transformer_encoder(encoder_inputs, vocab_size, max_length, d_model, num_heads, dff, num_layers, dropout_rate):

    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model, name="encoder_embedding")(encoder_inputs)

    x = PositionalEncoding(max_length=max_length, d_model=d_model)(x)

    x = tf.keras.layers.Dropout(dropout_rate)(x)

    for i in range(num_layers):
        x = EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)(x)

    return x


def build_transformer_decoder(decoder_inputs, encoder_outputs, vocab_size, max_length, d_model, num_heads, dff, num_layers, dropout_rate):

    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model, name="decoder_embedding")(decoder_inputs)

    x = PositionalEncoding(max_length=max_length, d_model=d_model)(x)

    x = tf.keras.layers.Dropout(dropout_rate)(x)

    attention_scores = None

    for i in range(num_layers):
        x, attention_scores = DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)(x, encoder_outputs)

    return x, attention_scores


def build_transformer_seq2seq_model(source_vocab_size, target_vocab_size, source_max_length=128, target_max_length=128, d_model=256, num_heads=8, dff=1024, num_encoder_layers=4, num_decoder_layers=4, dropout_rate=0.1):

    encoder_inputs = tf.keras.Input(shape=(source_max_length,), dtype=tf.int32, name="encoder_inputs")

    decoder_inputs = tf.keras.Input(shape=(target_max_length - 1,), dtype=tf.int32, name="decoder_inputs")

    encoder_outputs = build_transformer_encoder(encoder_inputs=encoder_inputs, vocab_size=source_vocab_size, max_length=source_max_length, d_model=d_model, num_heads=num_heads, dff=dff, num_layers=num_encoder_layers, dropout_rate=dropout_rate)

    decoder_outputs, attention_scores = build_transformer_decoder(decoder_inputs=decoder_inputs, encoder_outputs=encoder_outputs, vocab_size=target_vocab_size, max_length=target_max_length, d_model=d_model, num_heads=num_heads, dff=dff, num_layers=num_decoder_layers, dropout_rate=dropout_rate)

    outputs = tf.keras.layers.Dense(target_vocab_size, activation="softmax", name="output_projection")(decoder_outputs)

    return tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[outputs, attention_scores], name="transformer_seq2seq")


if __name__ == "__main__":

    model = build_transformer_seq2seq_model(source_vocab_size=30000, target_vocab_size=30000, source_max_length=128, target_max_length=128, d_model=256, num_heads=8, dff=1024, num_encoder_layers=4, num_decoder_layers=4)

    model.summary(expand_nested=True, show_trainable=True)

    tf.keras.utils.plot_model(model, to_file="transformer_seq2seq.png", show_shapes=True, show_dtype=True, show_layer_names=True, expand_nested=True, show_layer_activations=True)

    encoder_input = tf.random.uniform(shape=(2, 128), minval=0, maxval=30000, dtype=tf.int32)

    decoder_input = tf.random.uniform(shape=(2, 127), minval=0, maxval=30000, dtype=tf.int32)

    outputs = model([encoder_input, decoder_input])

    print("\nToken Output Shape :", outputs[0].shape)

    print("\nAttention Shape :", outputs[1].shape)