import numpy as np
import tensorflow as tf

import config
import model


class TD3Agent:
    def __init__(self, observation_shape, action_low, action_high, seed=42):
        self.action_low = np.asarray(action_low, np.float32)
        self.action_high = np.asarray(action_high, np.float32)
        self.scale = tf.constant((self.action_high - self.action_low) / 2.0)
        self.bias = tf.constant((self.action_high + self.action_low) / 2.0)
        action_size = self.action_low.size
        self.actor = model.actor_model(observation_shape, action_size)
        self.target_actor = model.actor_model(observation_shape, action_size)
        self.critics = [model.critic_model(
            observation_shape, action_size) for _ in range(2)]
        self.targets = [model.critic_model(
            observation_shape, action_size) for _ in range(2)]
        self.target_actor.set_weights(self.actor.get_weights())
        for target, source in zip(self.targets, self.critics):
            target.set_weights(source.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(
            config.ACTOR_LEARNING_RATE)
        self.critic_optimizers = [tf.keras.optimizers.Adam(
            config.CRITIC_LEARNING_RATE) for _ in range(2)]
        self.replay = model.ReplayBuffer(observation_shape, action_size, seed)
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.updates = 0

    def _scale(self, action):
        return self.bias + self.scale * action

    @tf.function(reduce_retracing=True)
    def _act(self, states):
        return self._scale(self.actor(states, training=False))[0]

    def act(self, state, training=True):
        self.steps += int(training)
        if training and self.steps <= config.START_STEPS:
            return self.rng.uniform(self.action_low, self.action_high).astype(np.float32)
        action = self._act(tf.convert_to_tensor(
            np.asarray(state, np.float32)[None])).numpy()
        if training:
            action += self.rng.normal(0.0, config.EXPLORATION_NOISE,
                                      action.shape) * (self.action_high - self.action_low) / 2.0
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)

    def remember(self, state, action, reward, next_state, terminal):
        self.replay.add(state, action, reward, next_state, terminal)

    @tf.function(reduce_retracing=True)
    def _critic_step(self, states, actions, rewards, next_states, terminals):
        normalized = self.target_actor(next_states, training=False)
        noise = tf.random.normal(
            tf.shape(normalized), stddev=config.TARGET_POLICY_NOISE
        )
        noise = tf.clip_by_value(
            noise, -config.TARGET_NOISE_CLIP, config.TARGET_NOISE_CLIP
        )
        next_actions = self._scale(
            tf.clip_by_value(normalized + noise, -1.0, 1.0))
        next_q = tf.minimum(tf.squeeze(self.targets[0]([next_states, next_actions]), 1), tf.squeeze(
            self.targets[1]([next_states, next_actions]), 1))
        targets = tf.stop_gradient(
            rewards + config.GAMMA * (1.0 - terminals) * next_q)
        losses = []
        for critic, optimizer in zip(self.critics, self.critic_optimizers):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(
                    tf.square(targets - tf.squeeze(critic([states, actions], training=True), 1)))
            gradients = tape.gradient(loss, critic.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(
                gradients, config.GRADIENT_CLIP_NORM)
            optimizer.apply_gradients(
                zip(gradients, critic.trainable_variables))
            losses.append(loss)
        return tf.add_n(losses) / 2.0

    @tf.function(reduce_retracing=True)
    def _actor_step(self, states):
        with tf.GradientTape() as tape:
            loss = - \
                tf.reduce_mean(self.critics[0](
                    [states, self._scale(self.actor(states, training=True))]))
        gradients = tape.gradient(loss, self.actor.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(
            gradients, config.GRADIENT_CLIP_NORM)
        self.actor_optimizer.apply_gradients(
            zip(gradients, self.actor.trainable_variables))
        model.soft_update(self.target_actor, self.actor)
        for target, source in zip(self.targets, self.critics):
            model.soft_update(target, source)
        return loss

    def learn(self):
        if len(self.replay) < max(config.BATCH_SIZE, config.LEARNING_STARTS):
            return None
        batch = [tf.convert_to_tensor(value, tf.float32)
                 for value in self.replay.sample()]
        critic_loss = self._critic_step(*batch)
        self.updates += 1
        actor_loss = np.nan
        if self.updates % config.POLICY_DELAY == 0:
            actor_loss = float(self._actor_step(batch[0]))
        return float(critic_loss), actor_loss

    def save(self, path):
        self.actor.save_weights(path)

    def load(self, path):
        self.actor.load_weights(path)
        self.target_actor.set_weights(self.actor.get_weights())
