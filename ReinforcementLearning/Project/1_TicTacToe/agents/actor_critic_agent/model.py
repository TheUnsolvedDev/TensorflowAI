import tensorflow as tf


def build_actor_critic(input_dim, n_actions):
    inputs = tf.keras.layers.Input(shape=(input_dim,))

    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    policy = tf.keras.layers.Dense(n_actions, activation='softmax')(x)
    value = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, [policy, value])