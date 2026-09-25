import tensorflow as tf


class MeanAdvantage(tf.keras.layers.Layer):
    def call(self, values):
        return tf.reduce_mean(values, axis=1, keepdims=True)


def build_model(action_size, input_shape=(10, 10, 4)):
    inputs = tf.keras.Input(shape=input_shape, name="screen")
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    value = tf.keras.layers.Dense(1, name="value")(x)
    advantage = tf.keras.layers.Dense(action_size, name="advantage")(x)
    mean_advantage = MeanAdvantage(name="mean_advantage")(advantage)
    q_values = value + (advantage - mean_advantage)
    return tf.keras.Model(inputs, q_values, name="dueling_dqn")
