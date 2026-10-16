import numpy as np
import tensorflow as tf

import config
import model


def action_settings(action_space):
    continuous = hasattr(action_space, "low")
    size = int(np.prod(action_space.shape)) if continuous else action_space.n
    if not continuous:
        return continuous, size, None, None
    low = tf.constant(action_space.low, tf.float32)
    high = tf.constant(action_space.high, tf.float32)
    return continuous, size, (high - low) / 2.0, (high + low) / 2.0


class GlobalAgent:
    def __init__(self, observation_shape, action_space):
        self.continuous, self.action_size, self.action_scale, self.action_bias = action_settings(action_space)
        if self.continuous:
            self.model = model.build_continuous_model(observation_shape, self.action_size)
        else:
            self.model = model.build_model(observation_shape, self.action_size)
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=config.LEARNING_RATE, rho=config.RMSPROP_RHO, epsilon=config.RMSPROP_EPSILON
        )
        self.optimizer.build(self.model.trainable_variables)

    @tf.function(reduce_retracing=True)
    def _act(self, state):
        return self.model(tf.expand_dims(tf.cast(state, tf.float32), 0), training=False)

    def act(self, state, training=True):
        output = self._act(state)
        if not self.continuous:
            probabilities = output[0].numpy()[0]
            action = np.random.choice(self.action_size, p=probabilities) if training else np.argmax(probabilities)
            return int(action)
        mean, log_std = output[0].numpy()[0], np.clip(output[1].numpy()[0], -5.0, 2.0)
        raw = mean + np.random.normal(size=self.action_size) * np.exp(log_std) if training else mean
        return (np.tanh(raw) * self.action_scale.numpy() + self.action_bias.numpy()).astype(np.float32)

    def apply_gradients(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return self.model.get_weights()

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)


class WorkerAgent:
    def __init__(self, observation_shape, action_space, global_agent, seed):
        self.gamma = config.GAMMA
        self.rng = np.random.default_rng(seed)
        self.continuous, self.action_size, self.action_scale, self.action_bias = action_settings(action_space)
        if self.continuous:
            self.model = model.build_continuous_model(observation_shape, self.action_size)
        else:
            self.model = model.build_model(observation_shape, self.action_size)
        self.model.set_weights(global_agent.model.get_weights())

    @tf.function(reduce_retracing=True)
    def _act(self, state):
        return self.model(tf.expand_dims(tf.cast(state, tf.float32), 0), training=False)

    def act(self, state):
        output = self._act(state)
        if not self.continuous:
            probabilities = output[0].numpy()[0]
            return int(self.rng.choice(self.action_size, p=probabilities))
        mean = output[0].numpy()[0]
        log_std = np.clip(output[1].numpy()[0], -5.0, 2.0)
        raw = mean + self.rng.normal(size=self.action_size) * np.exp(log_std)
        return (np.tanh(raw) * self.action_scale.numpy() + self.action_bias.numpy()).astype(np.float32)

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
    def compute_gradients(self, states, actions, returns):
        with tf.GradientTape() as tape:
            output = self.model(states, training=True)
            values = tf.squeeze(output[-1], axis=1)
            advantages = tf.stop_gradient(returns - values)
            advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
            advantages = tf.clip_by_value(advantages, -config.ADVANTAGE_CLIP, config.ADVANTAGE_CLIP)
            log_probability, entropy = self.policy_terms(output, actions)
            actor_loss = -tf.reduce_mean(log_probability * advantages)
            value_loss = tf.keras.losses.huber(returns, values, delta=config.HUBER_DELTA)
            loss = actor_loss + config.VALUE_LOSS_COEFFICIENT * value_loss - config.ENTROPY_COEFFICIENT * entropy
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, config.GRADIENT_CLIP_NORM)
        return gradients, loss

    def learn(self, states, actions, rewards, terminated, next_state, global_agent):
        if terminated:
            running = 0.0
        else:
            output = self.model(tf.expand_dims(tf.convert_to_tensor(next_state, tf.float32), 0), training=False)
            running = float(output[-1][0, 0])
        returns = np.zeros(len(rewards), dtype=np.float32)
        for index in range(len(rewards) - 1, -1, -1):
            running = rewards[index] + self.gamma * running
            returns[index] = running
        action_dtype = tf.float32 if self.continuous else tf.int32
        gradients, loss = self.compute_gradients(
            tf.convert_to_tensor(states, tf.float32),
            tf.convert_to_tensor(actions, action_dtype),
            tf.convert_to_tensor(returns, tf.float32),
        )
        self.model.set_weights(global_agent.apply_gradients(gradients))
        return float(loss.numpy())

