import numpy as np
import tensorflow as tf

import config
import model


class Agent:
    def __init__(self, observation_shape, action_count, seed=42):
        self.action_count = action_count
        self.online = model.build_network(observation_shape, action_count)
        self.target = model.build_network(observation_shape, action_count)
        self.target.set_weights(self.online.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)
        self.replay = model.ReplayBuffer(observation_shape, seed)
        self.rng = np.random.default_rng(seed)
        self.steps = self.updates = 0

    @property
    def epsilon(self):
        progress = min(self.steps / config.EPSILON_DECAY_STEPS, 1.0)
        return config.EPSILON_START + progress * (config.EPSILON_END - config.EPSILON_START)

    @tf.function(reduce_retracing=True)
    def _act(self, state):
        taus = tf.random.uniform((tf.shape(state)[0], config.QUANTILE_SAMPLES))
        return tf.reduce_mean(self.online(state, taus), axis=2)[0]

    def act(self, state, training=True):
        if training:
            self.steps += 1
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_count))
        return int(tf.argmax(self._act(tf.convert_to_tensor(np.asarray(state, np.float32)[None]))))

    def remember(self, state, action, reward, next_state, terminal):
        self.replay.add(state, action, reward, next_state, terminal)

    @tf.function(reduce_retracing=True)
    def _train(self, states, actions, rewards, next_states, terminals):
        batch = tf.shape(states)[0]
        taus = tf.random.uniform((batch, config.QUANTILE_SAMPLES))
        target_taus = tf.random.uniform((batch, config.QUANTILE_SAMPLES))
        next_actions = tf.argmax(tf.reduce_mean(self.online(
            next_states, taus), axis=2), axis=1, output_type=tf.int32)
        next_quantiles = tf.gather(self.target(
            next_states, target_taus), next_actions, batch_dims=1)
        targets = tf.stop_gradient(
            rewards[:, None] + config.GAMMA * (1.0 - terminals[:, None]) * next_quantiles)
        with tf.GradientTape() as tape:
            predicted = tf.gather(self.online(
                states, taus), actions, batch_dims=1)
            errors = targets[:, None, :] - predicted[:, :, None]
            huber = tf.where(tf.abs(errors) <= config.KAPPA, 0.5 * tf.square(errors),
                             config.KAPPA * (tf.abs(errors) - 0.5 * config.KAPPA))
            loss = tf.reduce_mean(tf.abs(
                taus[:, :, None] - tf.cast(errors < 0.0, tf.float32)) * huber / config.KAPPA)
        gradients = tape.gradient(loss, self.online.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(
            gradients, config.GRADIENT_CLIP_NORM)
        self.optimizer.apply_gradients(
            zip(gradients, self.online.trainable_variables))
        return loss

    def learn(self):
        if len(self.replay) < max(config.BATCH_SIZE, config.LEARNING_STARTS):
            return None
        loss = self._train(*[tf.convert_to_tensor(x)
                           for x in self.replay.sample()])
        self.updates += 1
        if self.updates % config.TARGET_UPDATE_FREQUENCY == 0:
            self.target.set_weights(self.online.get_weights())
        return float(loss)

    def save(self, path):
        self.online.save_weights(path)

    def load(self, path):
        self.online.load_weights(path)
        self.target.set_weights(self.online.get_weights())
