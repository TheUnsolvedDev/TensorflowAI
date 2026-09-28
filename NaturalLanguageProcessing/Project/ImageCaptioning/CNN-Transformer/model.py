"""Scratch CNN encoder with a causal Transformer caption decoder."""
import tensorflow as tf

from config import (DECODER, DROPOUT, IMAGE_CHANNELS, IMAGE_GRID_SIZE,
                    IMAGE_SIZE, MAX_LENGTH, PIXEL_MAX_VALUE, RESNET_BLOCKS,
                    RESNET_FILTERS,
                    TRANSFORMER_DIM, TRANSFORMER_FF_DIM, TRANSFORMER_HEADS,
                    TRANSFORMER_LAYERS)

tf.keras.backend.set_image_data_format("channels_last")


def build_cnn(spatial=False):
    inputs = tf.keras.Input((*IMAGE_SIZE, IMAGE_CHANNELS))
    x = tf.keras.layers.Rescaling(1.0 / PIXEL_MAX_VALUE)(inputs)
    x = tf.keras.layers.Conv2D(
        RESNET_FILTERS[0], 7, strides=2, padding="same",
        use_bias=False, name="stem_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)
    x = tf.keras.layers.MaxPool2D(3, strides=2, padding="same", name="stem_pool")(x)

    for stage, (filters, blocks) in enumerate(
            zip(RESNET_FILTERS, RESNET_BLOCKS), start=1):
        for block in range(1, blocks + 1):
            stride = 2 if stage > 1 and block == 1 else 1
            shortcut = x
            x = tf.keras.layers.Conv2D(
                filters, 3, strides=stride, padding="same", use_bias=False,
                name=f"stage{stage}_block{block}_conv1")(x)
            x = tf.keras.layers.BatchNormalization(
                name=f"stage{stage}_block{block}_bn1")(x)
            x = tf.keras.layers.ReLU(name=f"stage{stage}_block{block}_relu1")(x)
            x = tf.keras.layers.Conv2D(
                filters, 3, padding="same", use_bias=False,
                name=f"stage{stage}_block{block}_conv2")(x)
            x = tf.keras.layers.BatchNormalization(
                name=f"stage{stage}_block{block}_bn2")(x)
            if stride != 1 or shortcut.shape[-1] != filters:
                shortcut = tf.keras.layers.Conv2D(
                    filters, 1, strides=stride, use_bias=False,
                    name=f"stage{stage}_block{block}_shortcut_conv")(shortcut)
                shortcut = tf.keras.layers.BatchNormalization(
                    name=f"stage{stage}_block{block}_shortcut_bn")(shortcut)
            x = tf.keras.layers.Add(name=f"stage{stage}_block{block}_add")([x, shortcut])
            x = tf.keras.layers.ReLU(name=f"stage{stage}_block{block}_out")(x)
    if spatial:
        x = tf.keras.layers.Reshape((-1, x.shape[-1]), name="image_tokens")(x)
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


class ImagePositionEmbedding(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="image_position_embedding")

    def call(self, image_tokens):
        rows, columns = IMAGE_GRID_SIZE
        quarter = TRANSFORMER_DIM // 4
        frequencies = tf.exp(
            -tf.math.log(10000.0) * tf.cast(tf.range(quarter), tf.float32) /
            tf.cast(max(quarter - 1, 1), tf.float32))
        row_angles = tf.cast(tf.repeat(tf.range(rows), columns), tf.float32)[:, None] * frequencies
        column_angles = tf.cast(tf.tile(tf.range(columns), [rows]), tf.float32)[:, None] * frequencies
        positions = tf.concat([
            tf.sin(row_angles), tf.cos(row_angles),
            tf.sin(column_angles), tf.cos(column_angles),
        ], axis=1)
        return image_tokens + positions[None, :, :]


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
    memory = ImagePositionEmbedding()(memory)
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
