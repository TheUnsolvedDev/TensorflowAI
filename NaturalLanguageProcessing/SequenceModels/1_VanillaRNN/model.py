# model.py

import tensorflow as tf


def build_vanilla_rnn(vocab_size, num_classes, embedding_dim=128, hidden_dim=128, dropout=0.2):
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    x = tf.keras.layers.Embedding(
        vocab_size, embedding_dim, mask_zero=True, name="embedding")(inputs)
    x = tf.keras.layers.SimpleRNN(hidden_dim, dropout=dropout, name="rnn")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="classifier")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs,
                           name="VanillaRNNClassifier")
    return model


if __name__ == "__main__":
    vocab_size = 20000
    num_classes = 4
    max_length = 128
    model = build_vanilla_rnn(vocab_size=vocab_size, num_classes=num_classes,
                              embedding_dim=256, hidden_dim=256, dropout=0.3)
    model.summary()
    dummy_input = tf.random.uniform(
        (32, max_length), minval=0, maxval=vocab_size, dtype=tf.int32)
    output = model(dummy_input)
    print("\nInput Shape :", dummy_input.shape)
    print("Output Shape:", output.shape)
