import numpy as np
import tensorflow as tf

import config
import model


class SACAgent:
    def __init__(self, observation_shape, action_low, action_high, seed=42):
        self.action_low = np.asarray(action_low, np.float32)
        self.action_high = np.asarray(action_high, np.float32)
        self.scale = tf.constant((self.action_high - self.action_low) / 2.0)
        self.bias = tf.constant((self.action_high + self.action_low) / 2.0)
        self.log_scale = tf.reduce_sum(tf.math.log(self.scale))
        action_size = self.action_low.size
        self.actor = model.gaussian_actor_model(observation_shape, action_size)
        self.critics = [model.critic_model(
            observation_shape, action_size) for _ in range(2)]
        self.targets = [model.critic_model(
            observation_shape, action_size) for _ in range(2)]
        for target, source in zip(self.targets, self.critics):
            target.set_weights(source.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(
            config.ACTOR_LEARNING_RATE)
        self.critic_optimizers = [tf.keras.optimizers.Adam(
            config.CRITIC_LEARNING_RATE) for _ in range(2)]
        self.log_alpha = tf.Variable(0.0, trainable=True)
        self.alpha_optimizer = tf.keras.optimizers.Adam(
            config.ALPHA_LEARNING_RATE)
        self.target_entropy = tf.constant(-float(action_size))
        self.replay = model.ReplayBuffer(observation_shape, action_size, seed)
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    def _sample(self, states, deterministic=False):
        mean, log_std = self.actor(states, training=not deterministic)
        log_std = tf.clip_by_value(
            log_std, config.LOG_STD_MIN, config.LOG_STD_MAX)
        raw = mean if deterministic else mean + \
            tf.exp(log_std) * tf.random.normal(tf.shape(mean))
        squashed = tf.tanh(raw)
        actions = self.bias + self.scale * squashed
        log_prob = -0.5 * (tf.square((raw - mean) / (tf.exp(log_std) +
                           1e-8)) + 2.0 * log_std + np.log(2.0 * np.pi))
        log_prob = tf.reduce_sum(log_prob, axis=1) - tf.reduce_sum(
            tf.math.log(1.0 - tf.square(squashed) + 1e-6), axis=1) - self.log_scale
        return actions, log_prob

    @tf.function(reduce_retracing=True)
    def _act(self, states, deterministic):
        return self._sample(states, deterministic)[0][0]

    def act(self, state, training=True):
        self.steps += int(training)
        if training and self.steps <= config.START_STEPS:
            return self.rng.uniform(self.action_low, self.action_high).astype(np.float32)
        state = tf.convert_to_tensor(np.asarray(state, np.float32)[None])
        return self._act(state, not training).numpy().astype(np.float32)

    def remember(self, state, action, reward, next_state, terminal):
        self.replay.add(state, action, reward, next_state, terminal)

    @tf.function(reduce_retracing=True)
    def _train_step(self, states, actions, rewards, next_states, terminals):
        next_actions, next_log_prob = self._sample(next_states)
        next_q = tf.minimum(tf.squeeze(self.targets[0]([next_states, next_actions]), 1), tf.squeeze(
            self.targets[1]([next_states, next_actions]), 1))
        next_q -= tf.exp(self.log_alpha) * next_log_prob
        targets = tf.stop_gradient(
            rewards + config.GAMMA * (1.0 - terminals) * next_q)
        losses = []
        for critic, optimizer in zip(self.critics, self.critic_optimizers):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(
                    tf.square(targets - tf.squeeze(critic([states, actions], training=True), 1)))
            optimizer.apply_gradients(zip(tape.gradient(
                loss, critic.trainable_variables), critic.trainable_variables))
            losses.append(loss)
        with tf.GradientTape() as tape:
            policy_actions, log_prob = self._sample(states)
            q = tf.minimum(tf.squeeze(self.critics[0]([states, policy_actions]), 1), tf.squeeze(
                self.critics[1]([states, policy_actions]), 1))
            actor_loss = tf.reduce_mean(tf.exp(self.log_alpha) * log_prob - q)
        self.actor_optimizer.apply_gradients(zip(tape.gradient(
            actor_loss, self.actor.trainable_variables), self.actor.trainable_variables))
        with tf.GradientTape() as tape:
            alpha_loss = - \
                tf.reduce_mean(
                    self.log_alpha * tf.stop_gradient(log_prob + self.target_entropy))
        self.alpha_optimizer.apply_gradients(
            zip(tape.gradient(alpha_loss, [self.log_alpha]), [self.log_alpha]))
        for target, source in zip(self.targets, self.critics):
            model.soft_update(target, source)
        return tf.add_n(losses) / 2.0, actor_loss, tf.exp(self.log_alpha)

    def learn(self):
        if len(self.replay) < max(config.BATCH_SIZE, config.LEARNING_STARTS):
            return None
        batch = [tf.convert_to_tensor(value, tf.float32)
                 for value in self.replay.sample()]
        return tuple(float(value) for value in self._train_step(*batch))

    def save(self, path):
        self.actor.save_weights(path)

    def load(self, path):
        self.actor.load_weights(path)
