import tensorflow as tf
import config


def build_models(shape, actions):
    observation = tf.keras.Input(shape=shape)
    latent = tf.keras.layers.Dense(config.LATENT, activation="tanh")(
        tf.keras.layers.Dense(config.HIDDEN, activation="relu")(observation))
    representation = tf.keras.Model(observation, latent)
    z = tf.keras.Input(shape=(config.LATENT,))
    prediction_hidden = tf.keras.layers.Dense(
        config.HIDDEN, activation="relu")(z)
    prediction = tf.keras.Model(z, [tf.keras.layers.Dense(actions)(
        prediction_hidden), tf.keras.layers.Dense(1)(prediction_hidden)])
    a = tf.keras.Input(shape=(actions,))
    transition = tf.keras.layers.Dense(config.HIDDEN, activation="relu")(
        tf.keras.layers.Concatenate()([z, a]))
    dynamics = tf.keras.Model([z, a], [tf.keras.layers.Dense(
        config.LATENT, activation="tanh")(transition), tf.keras.layers.Dense(1)(transition)])
    return representation, dynamics, prediction
