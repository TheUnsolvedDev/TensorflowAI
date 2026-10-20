import numpy as np
import tensorflow as tf

import config
import model


class DDPGAgent:
    def __init__(self, observation_shape, action_low, action_high, seed=42):
        self.action_low = np.asarray(action_low, np.float32)
        self.action_high = np.asarray(action_high, np.float32)
        self.scale = tf.constant((self.action_high - self.action_low) / 2.0)
        self.bias = tf.constant((self.action_high + self.action_low) / 2.0)
        action_size = self.action_low.size
        self.actor = model.actor_model(observation_shape, action_size)
        self.target_actor = model.actor_model(observation_shape, action_size)
        self.critic = model.critic_model(observation_shape, action_size)
        self.target_critic = model.critic_model(observation_shape, action_size)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(
            config.ACTOR_LEARNING_RATE)
        self.critic_optimizer = tf.keras.optimizers.Adam(
            config.CRITIC_LEARNING_RATE)
        self.replay = model.ReplayBuffer(observation_shape, action_size, seed)
        self.rng = np.random.default_rng(seed)
        self.steps = 0

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
    def _train_step(self, states, actions, rewards, next_states, terminals):
        next_actions = self._scale(
            self.target_actor(next_states, training=False))
        next_q = tf.squeeze(self.target_critic(
            [next_states, next_actions], training=False), 1)
        targets = tf.stop_gradient(
            rewards + config.GAMMA * (1.0 - terminals) * next_q)
        with tf.GradientTape() as tape:
            q = tf.squeeze(self.critic([states, actions], training=True), 1)
            critic_loss = tf.reduce_mean(tf.square(targets - q))
        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(
            gradients, config.GRADIENT_CLIP_NORM)
        self.critic_optimizer.apply_gradients(
            zip(gradients, self.critic.trainable_variables))
        with tf.GradientTape() as tape:
            policy_actions = self._scale(self.actor(states, training=True))
            actor_loss = - \
                tf.reduce_mean(self.critic(
                    [states, policy_actions], training=False))
        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(
            gradients, config.GRADIENT_CLIP_NORM)
        self.actor_optimizer.apply_gradients(
            zip(gradients, self.actor.trainable_variables))
        model.soft_update(self.target_actor, self.actor)
        model.soft_update(self.target_critic, self.critic)
        return critic_loss, actor_loss

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
        self.target_actor.set_weights(self.actor.get_weights())
