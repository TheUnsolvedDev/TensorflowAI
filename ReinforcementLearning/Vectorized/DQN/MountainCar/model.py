import tensorflow as tf


def build_q_network(obs_shape, action_dim, hidden_dims=(256, 256)):
    inputs = tf.keras.layers.Input(shape=obs_shape, dtype=tf.float32)
    x = tf.keras.layers.LayerNormalization()(inputs)
    for hidden_dim in hidden_dims:
        x = tf.keras.layers.Dense(hidden_dim)(x)
        x = tf.keras.layers.ReLU()(x)
    outputs = tf.keras.layers.Dense(action_dim)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)