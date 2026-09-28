"""Scratch CNN encoder with an attentive LSTM/GRU caption decoder."""
import tensorflow as tf

from config import (DECODER, DROPOUT, EMBEDDING_DIM, HIDDEN_DIM,
                    IMAGE_CHANNELS, IMAGE_SIZE, PIXEL_MAX_VALUE,
                    RESNET_BLOCKS, RESNET_FILTERS)

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


def build_model(vocab_size, decoder=DECODER):
    decoder = decoder.lower()
    if decoder not in {"lstm", "gru"}:
        raise ValueError("decoder must be 'lstm' or 'gru'")

    image = tf.keras.Input((*IMAGE_SIZE, IMAGE_CHANNELS), name="image")
    tokens = tf.keras.Input((None,), dtype=tf.int32, name="tokens")
    features = build_cnn(spatial=True)(image)
    pooled = tf.keras.layers.GlobalAveragePooling1D(name="image_pool")(features)

    # Padding is already excluded and normalized by dataset sample weights.
    x = tf.keras.layers.Embedding(
        vocab_size, EMBEDDING_DIM, mask_zero=False, name="token_embedding")(tokens)
    if decoder == "lstm":
        h = tf.keras.layers.Dense(HIDDEN_DIM, activation="tanh", name="image_h")(pooled)
        c = tf.keras.layers.Dense(HIDDEN_DIM, activation="tanh", name="image_c")(pooled)
        x = tf.keras.layers.LSTM(HIDDEN_DIM, return_sequences=True, dropout=DROPOUT, name="decoder")(x, initial_state=[h, c])
    else:
        state = tf.keras.layers.Dense(HIDDEN_DIM, activation="tanh", name="image_state")(pooled)
        x = tf.keras.layers.GRU(HIDDEN_DIM, return_sequences=True, dropout=DROPOUT, name="decoder")(x, initial_state=state)
    memory = tf.keras.layers.Dense(HIDDEN_DIM, name="image_projection")(features)
    context = tf.keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=HIDDEN_DIM // 4, dropout=DROPOUT,
        name="image_attention")(x, memory)
    x = tf.keras.layers.Concatenate(name="decoder_context")([x, context])
    logits = tf.keras.layers.Dense(vocab_size, name="logits")(x)
    return tf.keras.Model({"image": image, "tokens": tokens}, logits, name=f"cnn_{decoder}_captioner")


def verify_captioner(model, vocab_size):
    output = model({
        "image": tf.zeros((1, *IMAGE_SIZE, IMAGE_CHANNELS)),
        "tokens": tf.ones((1, 2), tf.int32),
    }, training=False)
    if output.shape != (1, 2, vocab_size):
        raise RuntimeError(f"Caption model check failed: expected (1, 2, {vocab_size}), got {output.shape}")


if __name__ == "__main__":
    model = build_model(100)
    output = model({"image": tf.zeros((2, *IMAGE_SIZE, IMAGE_CHANNELS)), "tokens": tf.ones((2, 9), tf.int32)})
    assert output.shape == (2, 9, 100)
    print("Model check passed:", output.shape)
