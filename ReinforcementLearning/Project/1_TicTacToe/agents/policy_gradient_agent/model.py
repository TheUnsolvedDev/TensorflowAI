import tensorflow as tf


def build_policy_network(input_dim, n_actions):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    
    outputs = tf.keras.layers.Dense(n_actions, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)