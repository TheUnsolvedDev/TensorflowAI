import tensorflow as tf


def breakout_cnn(action_size, input_shape=(10, 10, 4), name="breakout_cnn"):
    inputs = tf.keras.Input(shape=input_shape, name="screen")
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(action_size, name="q_values")(x)
    return tf.keras.Model(inputs, outputs, name=name)
