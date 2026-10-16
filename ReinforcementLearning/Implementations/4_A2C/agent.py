import numpy as np
import tensorflow as tf

import config
import model


class A2CAgent:
    def __init__(self, observation_shape, action_space):
        self.gamma = config.GAMMA
        self.continuous = hasattr(action_space, "low")
        self.action_size = int(np.prod(action_space.shape)) if self.continuous else action_space.n
        if self.continuous:
            low = tf.constant(action_space.low, tf.float32)
            high = tf.constant(action_space.high, tf.float32)
            self.action_scale = (high - low) / 2.0
            self.action_bias = (high + low) / 2.0
            self.model = model.build_continuous_model(observation_shape, self.action_size)
        else:
            self.model = model.build_model(observation_shape, self.action_size)
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=config.LEARNING_RATE, rho=config.RMSPROP_RHO, epsilon=config.RMSPROP_EPSILON
        )

    @tf.function(reduce_retracing=True)
    def _act(self, states, training):
        output = self.model(tf.cast(states, tf.float32), training=False)
        if not self.continuous:
            probabilities = output[0]
            if training:
                return tf.squeeze(tf.random.categorical(tf.math.log(probabilities + 1e-8), 1), axis=1)
            return tf.argmax(probabilities, axis=1, output_type=tf.int64)
        mean, log_std = output[0], tf.clip_by_value(output[1], -5.0, 2.0)
        raw = mean + tf.random.normal(tf.shape(mean)) * tf.exp(log_std) if training else mean
        return tf.tanh(raw) * self.action_scale + self.action_bias

    def act(self, states, training=True):
        states = np.asarray(states, dtype=np.float32)
        single_state = states.ndim == 1
        if single_state:
            states = states[None]
        actions = self._act(states, training).numpy()
        if single_state:
            return actions[0].astype(np.float32) if self.continuous else int(actions[0])
        return actions.astype(np.float32) if self.continuous else actions

    def policy_terms(self, output, actions):
        if not self.continuous:
            probabilities = tf.clip_by_value(output[0], 1e-8, 1.0)
            selected = tf.gather(probabilities, tf.cast(actions, tf.int32), batch_dims=1)
            entropy = -tf.reduce_mean(tf.reduce_sum(probabilities * tf.math.log(probabilities), axis=1))
            return tf.math.log(selected), entropy
        mean, log_std = output[0], tf.clip_by_value(output[1], -5.0, 2.0)
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
    def train_step(self, states, actions, returns):
        with tf.GradientTape() as tape:
            output = self.model(states, training=True)
            values = tf.squeeze(output[-1], axis=1)
            advantages = tf.stop_gradient(returns - values)
            advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
            advantages = tf.clip_by_value(advantages, -config.ADVANTAGE_CLIP, config.ADVANTAGE_CLIP)
            log_probability, entropy = self.policy_terms(output, actions)
            actor_loss = -tf.reduce_mean(log_probability * advantages)
            critic_loss = tf.keras.losses.huber(returns, values, delta=config.HUBER_DELTA)
            loss = actor_loss + config.VALUE_LOSS_COEFFICIENT * critic_loss - config.ENTROPY_COEFFICIENT * entropy
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, config.GRADIENT_CLIP_NORM)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def learn(self, states, actions, rewards, terminated, truncated, next_states):
        flat_next_states = next_states.reshape((-1,) + next_states.shape[2:])
        output = self.model(tf.convert_to_tensor(flat_next_states, tf.float32), training=False)
        next_values = output[-1].numpy()[:, 0].reshape(rewards.shape)
        running = next_values[-1]
        returns = np.zeros_like(rewards, dtype=np.float32)
        for step in range(len(rewards) - 1, -1, -1):
            running = np.where(terminated[step] > 0, 0.0, running)
            running = np.where(truncated[step] > 0, next_values[step], running)
            running = rewards[step] + self.gamma * running
            returns[step] = running
        flat_states = tf.convert_to_tensor(states.reshape((-1,) + states.shape[2:]), tf.float32)
        if self.continuous:
            flat_actions = tf.convert_to_tensor(actions.reshape(-1, self.action_size), tf.float32)
        else:
            flat_actions = tf.convert_to_tensor(actions.reshape(-1), tf.int32)
        loss = self.train_step(flat_states, flat_actions, tf.convert_to_tensor(returns.reshape(-1), tf.float32))
        return float(loss.numpy())

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)

