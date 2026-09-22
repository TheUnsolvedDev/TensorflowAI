import tensorflow as tf


def pong_cnn(action_size, input_shape=(84, 84, 4), name="pong_cnn"):
    inputs = tf.keras.Input(shape=input_shape, name="screen")
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(action_size, name="q_values")(x)
    return tf.keras.Model(inputs, outputs, name=name)
