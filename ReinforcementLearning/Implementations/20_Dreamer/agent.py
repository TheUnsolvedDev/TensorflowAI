import numpy as np
import tensorflow as tf
import config
import model


class Agent:
    def __init__(self, shape, actions, seed=42):
        self.actions = actions
        models = model.build_models(shape, actions)
        self.encoder, self.dynamics, self.decoder = models[:3]
        self.reward_model, self.actor, self.value = models[3:]
        self.world_variables = (
            self.encoder.trainable_variables
            + self.dynamics.trainable_variables
            + self.decoder.trainable_variables
            + self.reward_model.trainable_variables
        )
        self.world_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)
        self.actor_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)
        self.value_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)
        n = config.REPLAY_CAPACITY
        self.s = np.empty((n,) + tuple(shape), np.float32)
        self.ns = np.empty_like(self.s)
        self.a = np.empty(n, np.int32)
        self.r = np.empty(n, np.float32)
        self.t = np.empty(n, np.float32)
        self.position = self.size = self.steps = 0
        self.rng = np.random.default_rng(seed)

    @tf.function(reduce_retracing=True)
    def _act(self, s):
        latent = self.encoder(s, training=False)
        return tf.nn.softmax(self.actor(latent, training=False))[0]

    def act(self, state, training=True):
        p = self._act(tf.convert_to_tensor(
            np.asarray(state, np.float32)[None])).numpy()
        return int(self.rng.choice(self.actions, p=p)) if training else int(np.argmax(p))

    def remember(self, s, a, r, ns, t):
        i = self.position
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.ns[i] = ns
        self.t[i] = t
        self.position = (i + 1) % config.REPLAY_CAPACITY
        self.size = min(self.size + 1, config.REPLAY_CAPACITY)
        self.steps += 1

    @tf.function(reduce_retracing=True)
    def _train(self, s, a, r, ns, t):
        onehot = tf.one_hot(a, self.actions)
        with tf.GradientTape() as tape:
            z = self.encoder(s, training=True)
            next_z = tf.stop_gradient(self.encoder(ns, training=False))
            predicted_z = self.dynamics([z, onehot], training=True)
            reconstruction = self.decoder(z, training=True)
            predicted_reward = tf.squeeze(
                self.reward_model([z, onehot], training=True), 1)
            world_loss = tf.reduce_mean(tf.square(predicted_z - next_z)) + tf.reduce_mean(
                tf.square(reconstruction - s)) + tf.reduce_mean(tf.square(predicted_reward - r))
        variables = self.encoder.trainable_variables + self.dynamics.trainable_variables + \
            self.decoder.trainable_variables + self.reward_model.trainable_variables
        self.world_optimizer.apply_gradients(
            zip(tape.gradient(world_loss, variables), variables))
        start = tf.stop_gradient(self.encoder(s, training=False))
        with tf.GradientTape() as tape:
            latent = start
            returns = tf.zeros(tf.shape(s)[0])
            discount = 1.0
            for _ in range(config.IMAGINATION_HORIZON):
                probabilities = tf.nn.softmax(
                    self.actor(latent, training=True))
                returns += discount * \
                    tf.squeeze(self.reward_model(
                        [latent, probabilities], training=False), 1)
                latent = self.dynamics([latent, probabilities], training=False)
                discount *= config.GAMMA
            returns += discount * \
                tf.squeeze(self.value(latent, training=False), 1)
            actor_loss = -tf.reduce_mean(returns)
        self.actor_optimizer.apply_gradients(zip(tape.gradient(
            actor_loss, self.actor.trainable_variables), self.actor.trainable_variables))
        with tf.GradientTape() as tape:
            target = tf.stop_gradient(r + config.GAMMA * (1 - t) * tf.squeeze(
                self.value(self.encoder(ns, training=False), training=False), 1))
            value_loss = tf.reduce_mean(
                tf.square(target - tf.squeeze(self.value(start, training=True), 1)))
        self.value_optimizer.apply_gradients(zip(tape.gradient(
            value_loss, self.value.trainable_variables), self.value.trainable_variables))
        return world_loss + actor_loss + value_loss

    def learn(self):
        if self.size < max(config.BATCH_SIZE, config.LEARNING_STARTS):
            return None
        i = self.rng.integers(self.size, size=config.BATCH_SIZE)
        return float(self._train(*[tf.convert_to_tensor(x)
                     for x in [self.s[i], self.a[i], self.r[i], self.ns[i], self.t[i]]]))

    def save(self, path):
        self.actor.save_weights(path)
        self.encoder.save_weights(
            path.replace(".weights.h5", ".encoder.weights.h5")
        )

    def load(self, path):
        self.actor.load_weights(path)
        self.encoder.load_weights(
            path.replace(".weights.h5", ".encoder.weights.h5")
        )
