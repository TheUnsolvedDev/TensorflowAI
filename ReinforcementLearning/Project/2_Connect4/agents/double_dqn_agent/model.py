import tensorflow as tf


def residual_block(x, filters):
    residual = x

    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Add()([x, residual])

    x = tf.keras.layers.ReLU()(x)

    return x


def build_q_network(board_shape=(6, 7, 4), action_dim=7, filters=64, num_blocks=4):
    inputs = tf.keras.Input(shape=board_shape, dtype=tf.float32)

    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    for _ in range(num_blocks):
        x = residual_block(x, filters)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)

    outputs = tf.keras.layers.Dense(
        action_dim,
        kernel_initializer=tf.keras.initializers.RandomUniform(-1e-3, 1e-3)
    )(x)

    return tf.keras.Model(inputs, outputs)