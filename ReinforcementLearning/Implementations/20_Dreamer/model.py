import tensorflow as tf
import config


def build_models(shape, actions):
    observation = tf.keras.Input(shape=shape)
    latent = tf.keras.layers.Dense(
        config.LATENT, activation="tanh")(observation)
    encoder = tf.keras.Model(observation, latent)
    z = tf.keras.Input(shape=(config.LATENT,))
    a = tf.keras.Input(shape=(actions,))
    joined = tf.keras.layers.Concatenate()([z, a])
    hidden = tf.keras.layers.Dense(config.HIDDEN, activation="elu")(joined)
    dynamics = tf.keras.Model([z, a], tf.keras.layers.Dense(
        config.LATENT, activation="tanh")(hidden))
    decoder = tf.keras.Model(z, tf.keras.layers.Dense(shape[0])(
        tf.keras.layers.Dense(config.HIDDEN, activation="elu")(z)))
    reward = tf.keras.Model([z, a], tf.keras.layers.Dense(1)(hidden))
    actor = tf.keras.Model(z, tf.keras.layers.Dense(actions)(
        tf.keras.layers.Dense(config.HIDDEN, activation="elu")(z)))
    value = tf.keras.Model(z, tf.keras.layers.Dense(1)(
        tf.keras.layers.Dense(config.HIDDEN, activation="elu")(z)))
    return encoder, dynamics, decoder, reward, actor, value
