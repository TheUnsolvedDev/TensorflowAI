import numpy as np
import tensorflow as tf
import config


def actor_model(shape, actions):
    x = tf.keras.Input(shape=shape)
    y = x
    for units in config.HIDDEN_UNITS:
        y = tf.keras.layers.Dense(units, activation="relu")(y)
    return tf.keras.Model(x, tf.keras.layers.Dense(actions, activation="tanh")(y))


def critic_model(shape, actions):
    s = tf.keras.Input(shape=shape)
    a = tf.keras.Input(shape=(actions,))
    y = tf.keras.layers.Concatenate()([s, a])
    for units in config.HIDDEN_UNITS:
        y = tf.keras.layers.Dense(units, activation="relu")(y)
    return tf.keras.Model([s, a], tf.keras.layers.Dense(config.ATOM_COUNT, activation="softmax")(y))


class ReplayBuffer:
    def __init__(self, shape, actions, seed):
        n = config.REPLAY_CAPACITY
        self.s = np.empty((n,) + tuple(shape), np.float32)
        self.ns = np.empty_like(self.s)
        self.a = np.empty((n, actions), np.float32)
        self.r = np.empty(n, np.float32)
        self.t = np.empty(n, np.float32)
        self.p = self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, ns, t):
        i = self.p
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.ns[i] = ns
        self.t[i] = t
        self.p = (i + 1) % config.REPLAY_CAPACITY
        self.size = min(self.size + 1, config.REPLAY_CAPACITY)

    def sample(self):
        i = self.rng.integers(self.size, size=config.BATCH_SIZE)
        return self.s[i], self.a[i], self.r[i], self.ns[i], self.t[i]

    def __len__(self):
        return self.size


def soft_update(target, source):
    for x, y in zip(target.weights, source.weights):
        x.assign((1 - config.TAU) * x + config.TAU * y)
