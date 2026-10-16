from pathlib import Path

import numpy as np
import tensorflow as tf

import config
import model


class ActorCriticAgent:
    def __init__(self, observation_shape, action_space):
        self.gamma = config.GAMMA
        self.continuous = hasattr(action_space, "low")
        self.action_size = int(np.prod(action_space.shape)) if self.continuous else action_space.n
        if self.continuous:
            low = tf.constant(action_space.low, tf.float32)
            high = tf.constant(action_space.high, tf.float32)
            self.action_scale = (high - low) / 2.0
            self.action_bias = (high + low) / 2.0
            self.actor = model.build_continuous_actor(observation_shape, self.action_size)
        else:
            self.actor = model.build_actor(observation_shape, self.action_size)
        self.critic = model.build_critic(observation_shape)
        self.actor_optimizer = tf.keras.optimizers.Adam(config.ACTOR_LEARNING_RATE)
        self.critic_optimizer = tf.keras.optimizers.Adam(config.CRITIC_LEARNING_RATE)

    @tf.function(reduce_retracing=True)
    def _act(self, state):
        return self.actor(tf.expand_dims(tf.cast(state, tf.float32), 0), training=False)

    def act(self, state, training=True):
        output = self._act(state)
        if not self.continuous:
            probabilities = output.numpy()[0]
            action = np.random.choice(self.action_size, p=probabilities) if training else np.argmax(probabilities)
            return int(action)
        mean, log_std = output
        raw = mean.numpy()[0]
        if training:
            raw += np.random.normal(size=self.action_size) * np.exp(np.clip(log_std.numpy()[0], -5.0, 2.0))
        return (np.tanh(raw) * self.action_scale.numpy() + self.action_bias.numpy()).astype(np.float32)

    def policy_terms(self, output, actions):
        if not self.continuous:
            selected = tf.gather(output, tf.cast(actions, tf.int32), batch_dims=1)
            probabilities = tf.clip_by_value(output, 1e-8, 1.0)
            entropy = -tf.reduce_mean(tf.reduce_sum(probabilities * tf.math.log(probabilities), axis=1))
            return tf.math.log(selected + 1e-8), entropy
        mean, log_std = output
        log_std = tf.clip_by_value(log_std, -5.0, 2.0)
        squashed = tf.clip_by_value((actions - self.action_bias) / self.action_scale, -0.999999, 0.999999)
        raw = tf.atanh(squashed)
        normal_log_probability = -0.5 * (
            tf.square((raw - mean) / tf.exp(log_std)) + 2.0 * log_std + np.log(2.0 * np.pi)
        )
        log_probability = tf.reduce_sum(
            normal_log_probability - tf.math.log(1.0 - tf.square(squashed) + 1e-6), axis=1
        )
        entropy = tf.reduce_mean(tf.reduce_sum(log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=1))
        return log_probability, entropy

    @tf.function(reduce_retracing=True)
    def train_step(self, state, action, reward, next_state, terminated):
        state = tf.expand_dims(tf.cast(state, tf.float32), 0)
        next_state = tf.expand_dims(tf.cast(next_state, tf.float32), 0)
        action = tf.expand_dims(action, 0)
        next_value = tf.squeeze(self.critic(next_state, training=False), axis=1)
        td_target = reward + self.gamma * (1.0 - terminated) * next_value
        with tf.GradientTape() as actor_tape:
            output = self.actor(state, training=True)
            value = tf.squeeze(self.critic(state, training=False), axis=1)
            td_error = tf.clip_by_value(
                tf.stop_gradient(td_target - value), -config.ADVANTAGE_CLIP, config.ADVANTAGE_CLIP
            )
            log_probability, entropy = self.policy_terms(output, action)
            actor_loss = -tf.reduce_mean(log_probability * td_error) - config.ENTROPY_COEFFICIENT * entropy
        with tf.GradientTape() as critic_tape:
            value = tf.squeeze(self.critic(state, training=True), axis=1)
            critic_loss = tf.keras.losses.huber(tf.stop_gradient(td_target), value, delta=config.HUBER_DELTA)
        actor_gradients = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_gradients, _ = tf.clip_by_global_norm(actor_gradients, config.GRADIENT_CLIP_NORM)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_gradients, _ = tf.clip_by_global_norm(critic_gradients, config.GRADIENT_CLIP_NORM)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        return actor_loss + config.VALUE_LOSS_COEFFICIENT * critic_loss

    def learn(self, state, action, reward, next_state, terminated):
        action_dtype = tf.float32 if self.continuous else tf.int32
        loss = self.train_step(
            tf.convert_to_tensor(state, tf.float32),
            tf.convert_to_tensor(action, action_dtype),
            tf.convert_to_tensor(reward, tf.float32),
            tf.convert_to_tensor(next_state, tf.float32),
            tf.convert_to_tensor(float(terminated), tf.float32),
        )
        return float(loss.numpy())

    def critic_path(self, path):
        path = Path(path)
        name = path.name.removesuffix(".weights.h5")
        return path.with_name(f"{name}.critic.weights.h5")

    def save(self, path):
        self.actor.save_weights(path)
        self.critic.save_weights(self.critic_path(path))

    def load(self, path):
        self.actor.load_weights(path)
        critic_path = self.critic_path(path)
        if critic_path.exists():
            self.critic.load_weights(critic_path)

