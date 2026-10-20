import numpy as np
import tensorflow as tf
import config
import model


class Agent:
    def __init__(self, shape, low, high, seed=42):
        self.low = np.asarray(low, np.float32)
        self.high = np.asarray(high, np.float32)
        self.scale = tf.constant((self.high - self.low) / 2)
        self.bias = tf.constant((self.high + self.low) / 2)
        n = self.low.size
        self.actor = model.actor_model(shape, n)
        self.target_actor = model.actor_model(shape, n)
        self.critic = model.critic_model(shape, n)
        self.target_critic = model.critic_model(shape, n)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        self.ao = tf.keras.optimizers.Adam(config.ACTOR_LEARNING_RATE)
        self.co = tf.keras.optimizers.Adam(config.CRITIC_LEARNING_RATE)
        self.replay = model.ReplayBuffer(shape, n, seed)
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.support = tf.linspace(
            config.V_MIN, config.V_MAX, config.ATOM_COUNT)
        self.delta = (config.V_MAX - config.V_MIN) / (config.ATOM_COUNT - 1)

    def _scale(self, x):
        return self.bias + self.scale * x

    @tf.function(reduce_retracing=True)
    def _act(self, s):
        return self._scale(self.actor(s, training=False))[0]

    def act(self, state, training=True):
        self.steps += int(training)
        if training and self.steps <= config.START_STEPS:
            return self.rng.uniform(self.low, self.high).astype(np.float32)
        a = self._act(tf.convert_to_tensor(
            np.asarray(state, np.float32)[None])).numpy()
        if training:
            a += self.rng.normal(0, config.EXPLORATION_NOISE,
                                 a.shape) * np.asarray(self.scale)
        return np.clip(a, self.low, self.high).astype(np.float32)

    def remember(self, *transition):
        self.replay.add(*transition)

    @tf.function(reduce_retracing=True)
    def _train(self, s, a, r, ns, t):
        nd = self.target_critic([ns, self._scale(self.target_actor(ns))])
        tz = tf.clip_by_value(
            r[:, None] + config.GAMMA * (1 - t[:, None]) * self.support, config.V_MIN, config.V_MAX)
        b = (tz - config.V_MIN) / self.delta
        lo = tf.cast(tf.floor(b), tf.int32)
        hi = tf.cast(tf.math.ceil(b), tf.int32)
        eq = tf.cast(lo == hi, tf.float32)
        target = tf.reduce_sum(nd[:, :, None] *
                               (tf.one_hot(lo, config.ATOM_COUNT) *
                                (tf.cast(hi, tf.float32) -
                                 b +
                                 eq)[:, :, None] +
                                tf.one_hot(hi, config.ATOM_COUNT) *
                                (b -
                                   tf.cast(lo, tf.float32))[:, :, None]), axis=1)
        with tf.GradientTape() as tape:
            pred = self.critic([s, a], training=True)
            cl = -tf.reduce_mean(tf.reduce_sum(tf.stop_gradient(target)
                                 * tf.math.log(pred + 1e-8), axis=1))
        self.co.apply_gradients(zip(tape.gradient(
            cl, self.critic.trainable_variables), self.critic.trainable_variables))
        with tf.GradientTape() as tape:
            dist = self.critic([s, self._scale(self.actor(s, training=True))])
            al = -tf.reduce_mean(tf.reduce_sum(dist * self.support, axis=1))
        self.ao.apply_gradients(zip(tape.gradient(
            al, self.actor.trainable_variables), self.actor.trainable_variables))
        model.soft_update(self.target_actor, self.actor)
        model.soft_update(self.target_critic, self.critic)
        return cl

    def learn(self):
        if len(self.replay) < max(config.BATCH_SIZE, config.LEARNING_STARTS):
            return None
        return float(self._train(*[tf.convert_to_tensor(x) for x in self.replay.sample()]))

    def save(self, path):
        self.actor.save_weights(path)

    def load(self, path):
        self.actor.load_weights(path)
