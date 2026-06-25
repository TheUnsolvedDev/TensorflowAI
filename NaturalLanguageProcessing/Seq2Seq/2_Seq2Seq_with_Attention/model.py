# model.py

import tensorflow as tf


def build_encoder(encoder_embedding, encoder_units, decoder_units, num_encoder):
    encoder_x = encoder_embedding
    for i in range(num_encoder):
        encoder_x, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            encoder_units, return_sequences=True, return_state=True), name=f"encoder_lstm_{i+1}")(encoder_x)
    state_h = tf.keras.layers.Dense(decoder_units, activation="tanh", name="encoder_to_decoder_h")(
        tf.keras.layers.Concatenate(name="encoder_state_h")([forward_h, backward_h]))
    state_c = tf.keras.layers.Dense(decoder_units, activation="tanh", name="encoder_to_decoder_c")(
        tf.keras.layers.Concatenate(name="encoder_state_c")([forward_c, backward_c]))
    return encoder_x, state_h, state_c


def build_decoder(decoder_embedding, state_h, state_c, decoder_units, num_decoder):
    decoder_x = decoder_embedding
    for i in range(num_decoder):
        if i == 0:
            decoder_x = tf.keras.layers.LSTM(
                decoder_units, return_sequences=True, name=f"decoder_lstm_{i+1}")(decoder_x, initial_state=[state_h, state_c])
        else:
            decoder_x = tf.keras.layers.LSTM(
                decoder_units, return_sequences=True, name=f"decoder_lstm_{i+1}")(decoder_x)
        decoder_x = tf.keras.layers.LayerNormalization(
            name=f"decoder_layer_norm_{i+1}")(decoder_x)
    return decoder_x


def build_cross_attention(encoder_outputs, decoder_outputs, decoder_units):
    encoder_attention = tf.keras.layers.Dense(
        decoder_units, activation="tanh", name="encoder_attention_projection")(encoder_outputs)
    decoder_outputs = tf.keras.layers.Lambda(
        lambda x: x, mask=lambda inputs, mask: None, name="decoder_mask_remove")(decoder_outputs)
    encoder_attention = tf.keras.layers.Lambda(
        lambda x: x, mask=lambda inputs, mask: None, name="encoder_mask_remove")(encoder_attention)
    attention_output, attention_scores = tf.keras.layers.Attention(name="cross_attention")(
        [decoder_outputs, encoder_attention],
        return_attention_scores=True
    )
    decoder_context = tf.keras.layers.Concatenate(
        name="attention_concat")([decoder_outputs, attention_output])
    decoder_context = tf.keras.layers.LayerNormalization(
        name="attention_layer_norm")(decoder_context)
    decoder_context = tf.keras.layers.Dense(
        decoder_units, activation="tanh", name="attention_output_dense")(decoder_context)
    return decoder_context, attention_scores


def build_seq2seq_model(source_vocab_size, target_vocab_size, source_max_length=128, target_max_length=128, embedding_dim=512, encoder_units=256, decoder_units=512, num_encoder=2, num_decoder=1):
    encoder_inputs = tf.keras.Input(
        shape=(source_max_length,), dtype=tf.int32, name="encoder_inputs")
    encoder_embedding = tf.keras.layers.Embedding(
        input_dim=source_vocab_size, output_dim=embedding_dim, mask_zero=False, name="encoder_embedding")(encoder_inputs)
    encoder_outputs, state_h, state_c = build_encoder(
        encoder_embedding=encoder_embedding, encoder_units=encoder_units, decoder_units=decoder_units, num_encoder=num_encoder)
    decoder_inputs = tf.keras.Input(
        shape=(target_max_length - 1,), dtype=tf.int32, name="decoder_inputs")
    decoder_embedding = tf.keras.layers.Embedding(
        input_dim=target_vocab_size, output_dim=embedding_dim, mask_zero=False, name="decoder_embedding")(decoder_inputs)
    decoder_outputs = build_decoder(decoder_embedding=decoder_embedding, state_h=state_h,
                                    state_c=state_c, decoder_units=decoder_units, num_decoder=num_decoder)
    decoder_context, attention_scores = build_cross_attention(
        encoder_outputs=encoder_outputs, decoder_outputs=decoder_outputs, decoder_units=decoder_units)
    decoder_outputs = tf.keras.layers.Dense(
        target_vocab_size, activation="softmax", name="output_projection")(decoder_context)
    return tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[decoder_outputs, attention_scores], name="attention_seq2seq_model")


if __name__ == "__main__":

    model = build_seq2seq_model(
        source_vocab_size=30000,
        target_vocab_size=30000,
        source_max_length=128,
        target_max_length=128,
        embedding_dim=256,
        encoder_units=256,
        decoder_units=512
    )

    model.summary(expand_nested=True, show_trainable=True)

    tf.keras.utils.plot_model(
        model,
        to_file="attention_seq2seq_model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        expand_nested=True,
        show_layer_activations=True
    )

    encoder_input = tf.random.uniform(
        shape=(2, 128), minval=0, maxval=30000, dtype=tf.int32)

    decoder_input = tf.random.uniform(
        shape=(2, 127), minval=0, maxval=30000, dtype=tf.int32)

    outputs = model([encoder_input, decoder_input])

    print("\nOutput Shape :", outputs[0].shape)
    print("\nOutput Shape :", outputs[1].shape)
