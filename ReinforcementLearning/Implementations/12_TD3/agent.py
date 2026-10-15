import numpy as np
import tensorflow as tf

import model


class ContinuousAgent:
    def __init__(self, observation_shape, action_shape, action_limit, variant="ddpg", gamma=0.99):
        self.variant = variant
        self.gamma = gamma
        self.tau = 0.005
        self.action_limit = float(action_limit)
        self.action_shape = action_shape
        if variant == "sac":
            self.actor = model.gaussian_actor_model(observation_shape, action_shape)
            self.actor_target = None
        else:
            self.actor = model.actor_model(observation_shape, action_shape, self.action_limit)
            self.actor_target = model.actor_model(observation_shape, action_shape, self.action_limit)
        self.critic = model.critic_model(observation_shape, action_shape)
        self.critic_target = model.critic_model(observation_shape, action_shape)
        self.second_critic = model.critic_model(observation_shape, action_shape) if variant in {"td3", "sac"} else None
        self.second_target = model.critic_model(observation_shape, action_shape) if variant in {"td3", "sac"} else None
        if self.actor_target is not None:
            self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())
        if self.second_critic is not None:
            self.second_target.set_weights(self.second_critic.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(3e-4)
        self.critic_optimizer = tf.keras.optimizers.Adam(3e-4)
        self.second_optimizer = tf.keras.optimizers.Adam(3e-4) if self.second_critic is not None else None
        self.buffer = model.ReplayBuffer()
        self.steps = 0

    def sample_sac_action(self, states):
        mean, log_std = self.actor(states, training=True)
        log_std = tf.clip_by_value(log_std, -20.0, 2.0)
        noise = tf.random.normal(tf.shape(mean))
        raw_action = mean + tf.exp(log_std) * noise
        squashed = tf.tanh(raw_action)
        action = squashed * self.action_limit
        log_probability = model.gaussian_log_probability(raw_action, mean, log_std)
        correction = tf.reduce_sum(tf.math.log(1.0 - tf.square(squashed) + 1e-6), axis=1)
        return action, log_probability - correction

    def act(self, state, training=True):
        state_batch = np.asarray(state, np.float32)[None]
        if self.variant == "sac":
            if training:
                action, _ = self.sample_sac_action(state_batch)
                action = action.numpy()[0]
            else:
                mean, _ = self.actor(state_batch, training=False)
                action = (tf.tanh(mean) * self.action_limit).numpy()[0]
        else:
            action = self.actor(state_batch, training=False).numpy()[0]
        if training and self.variant != "sac":
            scale = 0.1
            action += np.random.normal(0.0, scale * self.action_limit, self.action_shape)
        return np.clip(action, -self.action_limit, self.action_limit)

    def remember(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def learn(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = tf.convert_to_tensor(states, tf.float32)
        actions = tf.convert_to_tensor(actions, tf.float32)
        next_states = tf.convert_to_tensor(next_states, tf.float32)
        rewards = tf.convert_to_tensor(rewards, tf.float32)
        dones = tf.convert_to_tensor(dones, tf.float32)
        if self.variant == "sac":
            next_actions, next_log_probability = self.sample_sac_action(next_states)
        else:
            next_actions = self.actor_target(next_states, training=False)
        if self.variant == "td3":
            noise = tf.clip_by_value(tf.random.normal(tf.shape(next_actions), stddev=0.2), -0.5, 0.5)
            next_actions = tf.clip_by_value(next_actions + noise, -self.action_limit, self.action_limit)
        next_value = tf.squeeze(self.critic_target([next_states, next_actions]), 1)
        if self.second_target is not None:
            second_value = tf.squeeze(self.second_target([next_states, next_actions]), 1)
            next_value = tf.minimum(next_value, second_value)
        if self.variant == "sac":
            next_value -= 0.2 * next_log_probability
        targets = rewards + self.gamma * (1.0 - dones) * next_value
        with tf.GradientTape() as tape:
            values = tf.squeeze(self.critic([states, actions], training=True), 1)
            critic_loss = tf.reduce_mean(tf.square(targets - values))
        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))
        if self.second_critic is not None:
            with tf.GradientTape() as tape:
                values = tf.squeeze(self.second_critic([states, actions], training=True), 1)
                second_loss = tf.reduce_mean(tf.square(targets - values))
            gradients = tape.gradient(second_loss, self.second_critic.trainable_variables)
            self.second_optimizer.apply_gradients(zip(gradients, self.second_critic.trainable_variables))
        self.steps += 1
        if self.variant != "td3" or self.steps % 2 == 0:
            with tf.GradientTape() as tape:
                if self.variant == "sac":
                    predicted_actions, log_probability = self.sample_sac_action(states)
                    first_q = self.critic([states, predicted_actions], training=False)
                    second_q = self.second_critic([states, predicted_actions], training=False)
                    actor_loss = tf.reduce_mean(0.2 * log_probability - tf.squeeze(tf.minimum(first_q, second_q), 1))
                else:
                    predicted_actions = self.actor(states, training=True)
                    actor_loss = -tf.reduce_mean(self.critic([states, predicted_actions], training=False))
            gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
            if self.actor_target is not None:
                model.soft_update(self.actor_target, self.actor, self.tau)
            model.soft_update(self.critic_target, self.critic, self.tau)
            if self.second_critic is not None:
                model.soft_update(self.second_target, self.second_critic, self.tau)
        return float(critic_loss)

    def save(self, path):
        self.actor.save_weights(path)

    def load(self, path):
        self.actor.load_weights(path)
        if self.actor_target is not None:
            self.actor_target.set_weights(self.actor.get_weights())
