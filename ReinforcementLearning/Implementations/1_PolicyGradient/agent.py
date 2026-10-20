import numpy as np
import tensorflow as tf

import config
import model


class PolicyAgent:
    def __init__(self, observation_shape, action_space, gamma=None):
        self.gamma = config.GAMMA if gamma is None else gamma
        self.continuous = hasattr(action_space, "low")
        self.action_size = int(np.prod(action_space.shape)
                               ) if self.continuous else action_space.n
        if self.continuous:
            low = tf.constant(action_space.low, tf.float32)
            high = tf.constant(action_space.high, tf.float32)
            self.action_scale = (high - low) / 2.0
            self.action_bias = (high + low) / 2.0
            self.model = model.continuous_policy_model(
                observation_shape, self.action_size)
        else:
            self.model = model.policy_model(
                observation_shape, self.action_size)
        self.optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)
        self.states, self.actions, self.rewards = [], [], []

    @tf.function(reduce_retracing=True)
    def _act(self, state):
        return self.model(tf.expand_dims(tf.cast(state, tf.float32), 0), training=False)

    def act(self, state, training=True):
        output = self._act(state)
        if not self.continuous:
            probabilities = output.numpy()[0]
            action = np.random.choice(
                self.action_size, p=probabilities) if training else np.argmax(probabilities)
            return int(action)
        mean, log_std = output
        raw = mean.numpy()[0]
        if training:
            raw += np.random.normal(size=self.action_size) * np.exp(
                np.clip(log_std.numpy()[0], config.LOG_STD_MIN, config.LOG_STD_MAX))
        return (
            np.tanh(raw) *
            self.action_scale.numpy() +
            self.action_bias.numpy()).astype(
            np.float32)

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def continuous_log_probability(self, mean, log_std, actions):
        log_std = tf.clip_by_value(
            log_std, config.LOG_STD_MIN, config.LOG_STD_MAX)
        squashed = tf.clip_by_value(
            (actions - self.action_bias) / self.action_scale, -0.999999, 0.999999)
        raw = tf.atanh(squashed)
        log_probability = -0.5 * (
            tf.square((raw - mean) / tf.exp(log_std)) +
            2.0 * log_std + np.log(2.0 * np.pi)
        )
        correction = tf.math.log(1.0 - tf.square(squashed) + 1e-6)
        return tf.reduce_sum(log_probability - correction, axis=1)

    @tf.function(reduce_retracing=True)
    def train_step(self, states, actions, returns):
        with tf.GradientTape() as tape:
            output = self.model(states, training=True)
            if self.continuous:
                log_probability = self.continuous_log_probability(
                    output[0], output[1], actions)
            else:
                selected = tf.gather(output, tf.cast(
                    actions, tf.int32), batch_dims=1)
                log_probability = tf.math.log(selected + 1e-8)
            loss = -tf.reduce_mean(log_probability * returns)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(
            gradients, config.GRADIENT_CLIP_NORM)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        return loss

    def learn(self):
        if not self.states:
            return 0.0
        returns = model.discounted_returns(self.rewards, self.gamma)
        if config.NORMALIZE_RETURNS:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        action_dtype = tf.float32 if self.continuous else tf.int32
        loss = self.train_step(
            tf.convert_to_tensor(self.states, tf.float32),
            tf.convert_to_tensor(self.actions, action_dtype),
            tf.convert_to_tensor(returns, tf.float32),
        )
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        return float(loss.numpy())

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)
