"""CNN image encoder and LSTM/GRU caption decoder, all trained from scratch."""
import tensorflow as tf

from config import (CNN_FILTERS, CNN_KERNEL_SIZE, CNN_STRIDE, DECODER, DROPOUT,
                    EMBEDDING_DIM, HIDDEN_DIM, IMAGE_CHANNELS, IMAGE_SIZE,
                    PIXEL_MAX_VALUE)

tf.keras.backend.set_image_data_format("channels_last")


def build_cnn():
    inputs = tf.keras.Input((*IMAGE_SIZE, IMAGE_CHANNELS))
    x = tf.keras.layers.Rescaling(1.0 / PIXEL_MAX_VALUE)(inputs)
    for filters in CNN_FILTERS:
        x = tf.keras.layers.Conv2D(
            filters, CNN_KERNEL_SIZE, strides=CNN_STRIDE, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs, x, name="cnn_encoder")


def build_model(vocab_size, decoder=DECODER):
    decoder = decoder.lower()
    if decoder not in {"lstm", "gru"}:
        raise ValueError("decoder must be 'lstm' or 'gru'")

    image = tf.keras.Input((*IMAGE_SIZE, IMAGE_CHANNELS), name="image")
    tokens = tf.keras.Input((None,), dtype=tf.int32, name="tokens")
    features = build_cnn()(image)

    x = tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True, name="token_embedding")(tokens)
    if decoder == "lstm":
        h = tf.keras.layers.Dense(HIDDEN_DIM, activation="tanh", name="image_h")(features)
        c = tf.keras.layers.Dense(HIDDEN_DIM, activation="tanh", name="image_c")(features)
        x = tf.keras.layers.LSTM(HIDDEN_DIM, return_sequences=True, dropout=DROPOUT, name="decoder")(x, initial_state=[h, c])
    else:
        state = tf.keras.layers.Dense(HIDDEN_DIM, activation="tanh", name="image_state")(features)
        x = tf.keras.layers.GRU(HIDDEN_DIM, return_sequences=True, dropout=DROPOUT, name="decoder")(x, initial_state=state)
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
