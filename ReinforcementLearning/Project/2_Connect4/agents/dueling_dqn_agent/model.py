import tensorflow as tf


# def build_q_network(obs_dim, action_dim):
#     inputs = tf.keras.Input(shape=(obs_dim,), dtype=tf.float32)

#     x = tf.keras.layers.Dense(256, activation="relu")(inputs)
#     x = tf.keras.layers.LayerNormalization()(x)
#     x = tf.keras.layers.Dense(256, activation="relu")(x)
#     x = tf.keras.layers.LayerNormalization()(x)
#     v = tf.keras.layers.Dense(128, activation="relu")(x)
#     v = tf.keras.layers.Dense(
#         1,
#         kernel_initializer=tf.keras.initializers.RandomUniform(-1e-3, 1e-3)
#     )(v)
#     a = tf.keras.layers.Dense(128, activation="relu")(x)
#     a = tf.keras.layers.Dense(
#         action_dim,
#         kernel_initializer=tf.keras.initializers.RandomUniform(-1e-3, 1e-3)
#     )(a)
#     a_mean = tf.reduce_mean(a, axis=1, keepdims=True)
#     q = v + (a - a_mean)
#     return tf.keras.Model(inputs, q)



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

    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    for _ in range(num_blocks):
        x = residual_block(x, filters)

    x = tf.keras.layers.Flatten()(x)

    value = tf.keras.layers.Dense(256, activation="relu")(x)
    value = tf.keras.layers.LayerNormalization()(value)

    value = tf.keras.layers.Dense(
        1,
        kernel_initializer=tf.keras.initializers.RandomUniform(-1e-3, 1e-3)
    )(value)

    advantage = tf.keras.layers.Dense(256, activation="relu")(x)
    advantage = tf.keras.layers.LayerNormalization()(advantage)

    advantage = tf.keras.layers.Dense(
        action_dim,
        kernel_initializer=tf.keras.initializers.RandomUniform(-1e-3, 1e-3)
    )(advantage)

    advantage_mean = tf.keras.layers.Lambda(
        lambda a: tf.reduce_mean(a, axis=1, keepdims=True)
    )(advantage)

    outputs = tf.keras.layers.Add()([
        value,
        tf.keras.layers.Subtract()([advantage, advantage_mean])
    ])

    return tf.keras.Model(inputs, outputs)
