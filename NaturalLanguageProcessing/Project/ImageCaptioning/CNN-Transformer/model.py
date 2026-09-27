"""Scratch CNN encoder with a causal Transformer caption decoder."""
import tensorflow as tf

from config import (CNN_FILTERS, CNN_KERNEL_SIZE, CNN_STRIDE, DECODER, DROPOUT,
                    IMAGE_CHANNELS, IMAGE_SIZE, MAX_LENGTH, PIXEL_MAX_VALUE,
                    TRANSFORMER_DIM, TRANSFORMER_FF_DIM, TRANSFORMER_HEADS,
                    TRANSFORMER_LAYERS)

tf.keras.backend.set_image_data_format("channels_last")


def build_cnn(spatial=False):
    inputs = tf.keras.Input((*IMAGE_SIZE, IMAGE_CHANNELS))
    x = tf.keras.layers.Rescaling(1.0 / PIXEL_MAX_VALUE)(inputs)
    for filters in CNN_FILTERS:
        x = tf.keras.layers.Conv2D(
            filters, CNN_KERNEL_SIZE, strides=CNN_STRIDE,
            padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    if spatial:
        x = tf.keras.layers.Reshape((-1, CNN_FILTERS[-1]), name="image_tokens")(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs, x, name="cnn_encoder")


class TokenPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size):
        super().__init__(name="token_position_embedding")
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, TRANSFORMER_DIM)
        self.position_embedding = tf.keras.layers.Embedding(MAX_LENGTH, TRANSFORMER_DIM)

    def call(self, tokens):
        positions = tf.range(tf.shape(tokens)[1])
        return self.token_embedding(tokens) + self.position_embedding(positions)


class TransformerDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, number):
        super().__init__(name=f"transformer_decoder_{number}")
        key_dim = TRANSFORMER_DIM // TRANSFORMER_HEADS
        self.self_attention = tf.keras.layers.MultiHeadAttention(
            TRANSFORMER_HEADS, key_dim, dropout=DROPOUT, name="causal_attention")
        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            TRANSFORMER_HEADS, key_dim, dropout=DROPOUT, name="image_attention")
        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(TRANSFORMER_FF_DIM, activation="gelu"),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Dense(TRANSFORMER_DIM),
        ], name="feed_forward")
        self.norms = [tf.keras.layers.LayerNormalization(epsilon=1e-6)
                      for _ in range(3)]
        self.dropouts = [tf.keras.layers.Dropout(DROPOUT) for _ in range(3)]

    def call(self, inputs, training=None):
        x, image_memory, tokens = inputs
        length = tf.shape(tokens)[1]
        causal = tf.linalg.band_part(tf.ones((length, length), tf.bool), -1, 0)
        padding = tf.not_equal(tokens, 0)[:, None, :]
        attention_mask = causal[None, :, :] & padding
        attended = self.self_attention(
            x, x, attention_mask=attention_mask, training=training)
        x = self.norms[0](x + self.dropouts[0](attended, training=training))
        attended = self.cross_attention(x, image_memory, training=training)
        x = self.norms[1](x + self.dropouts[1](attended, training=training))
        forwarded = self.feed_forward(x, training=training)
        return self.norms[2](x + self.dropouts[2](forwarded, training=training))


def build_model(vocab_size, decoder=DECODER):
    if decoder.lower() != "transformer":
        raise ValueError("decoder must be 'transformer'")
    image = tf.keras.Input((*IMAGE_SIZE, IMAGE_CHANNELS), name="image")
    tokens = tf.keras.Input((None,), dtype=tf.int32, name="tokens")
    features = build_cnn(spatial=True)(image)
    memory = tf.keras.layers.Dense(
        TRANSFORMER_DIM, activation="gelu", name="image_projection")(features)
    x = TokenPositionEmbedding(vocab_size)(tokens)
    x = tf.keras.layers.Dropout(DROPOUT, name="embedding_dropout")(x)
    for number in range(1, TRANSFORMER_LAYERS + 1):
        x = TransformerDecoderBlock(number)([x, memory, tokens])
    logits = tf.keras.layers.Dense(vocab_size, name="logits")(x)
    return tf.keras.Model(
        {"image": image, "tokens": tokens}, logits,
        name="cnn_transformer_captioner")


def verify_captioner(model, vocab_size):
    output = model({
        "image": tf.zeros((1, *IMAGE_SIZE, IMAGE_CHANNELS)),
        "tokens": tf.ones((1, 2), tf.int32),
    }, training=False)
    if output.shape != (1, 2, vocab_size):
        raise RuntimeError(
            f"Caption model check failed: expected (1, 2, {vocab_size}), got {output.shape}")


if __name__ == "__main__":
    model = build_model(100)
    verify_captioner(model, 100)
    print("CNN-Transformer model check passed")
