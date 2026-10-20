import numpy as np
import tensorflow as tf

import config
import model


class PPOAgent:
    def __init__(self, observation_shape, action_count):
        self.action_count = action_count
        self.network = model.actor_critic_model(
            observation_shape, action_count)
        self.optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)
        self.rng = np.random.default_rng(config.SEED)
        self.action_dtype = np.int32
        self.clear()

    @tf.function(reduce_retracing=True)
    def _act(self, states):
        logits, values = self.network(states, training=False)
        return tf.nn.softmax(logits), tf.squeeze(values, 1)

    def act(self, state, training=True):
        probabilities, value = self._act(
            tf.convert_to_tensor(np.asarray(state, np.float32)[None]))
        probabilities = probabilities.numpy()[0]
        action = int(self.rng.choice(self.action_count, p=probabilities)
                     ) if training else int(np.argmax(probabilities))
        if training:
            self.old_log_probs.append(
                float(np.log(probabilities[action] + 1e-8)))
            self.values.append(float(value[0]))
        return action

    def remember(self, state, action, reward, next_state, terminal):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(float(terminal))
        self.next_state = next_state

    @tf.function(reduce_retracing=True)
    def _update(self, states, actions, old_log_probs, returns, advantages):
        with tf.GradientTape() as tape:
            logits, values = self.network(states, training=True)
            log_probs = - \
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=actions, logits=logits)
            ratio = tf.exp(log_probs - old_log_probs)
            clipped = tf.clip_by_value(
                ratio, 1.0 - config.CLIP_RATIO, 1.0 + config.CLIP_RATIO)
            policy_loss = - \
                tf.reduce_mean(tf.minimum(
                    ratio * advantages, clipped * advantages))
            value_loss = tf.reduce_mean(
                tf.square(returns - tf.squeeze(values, 1)))
            probabilities = tf.nn.softmax(logits)
            entropy = - \
                tf.reduce_mean(tf.reduce_sum(probabilities *
                               tf.nn.log_softmax(logits), axis=1))
            loss = policy_loss + config.VALUE_COEFFICIENT * \
                value_loss - config.ENTROPY_COEFFICIENT * entropy
        gradients = tape.gradient(loss, self.network.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(
            gradients, config.MAX_GRADIENT_NORM)
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_variables))
        return loss

    def learn(self):
        if not self.states:
            return 0.0
        bootstrap = 0.0 if self.terminals[-1] else float(self._act(
            tf.convert_to_tensor(np.asarray(self.next_state, np.float32)[None]))[1][0])
        advantages = np.zeros(len(self.rewards), np.float32)
        gae, next_value = 0.0, bootstrap
        for index in range(len(self.rewards) - 1, -1, -1):
            mask = 1.0 - self.terminals[index]
            delta = self.rewards[index] + config.GAMMA * \
                mask * next_value - self.values[index]
            gae = delta + config.GAMMA * config.GAE_LAMBDA * mask * gae
            advantages[index], next_value = gae, self.values[index]
        returns = advantages + np.asarray(self.values, np.float32)
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)
        arrays = [
            np.asarray(
                self.states, np.float32), np.asarray(
                self.actions, self.action_dtype), np.asarray(
                self.old_log_probs, np.float32), returns, advantages]
        losses = []
        for _ in range(config.UPDATE_EPOCHS):
            order = self.rng.permutation(len(self.states))
            for start in range(0, len(order), config.MINIBATCH_SIZE):
                indices = order[start:start + config.MINIBATCH_SIZE]
                losses.append(float(self._update(
                    *[tf.convert_to_tensor(array[indices]) for array in arrays])))
        self.clear()
        return float(np.mean(losses))

    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.terminals, self.old_log_probs, self.values = [], [], []
        self.next_state = None

    def save(self, path):
        self.network.save_weights(path)

    def load(self, path):
        self.network.load_weights(path)


class ContinuousPPOAgent(PPOAgent):
    def __init__(self, observation_shape, action_low, action_high):
        self.action_low = np.asarray(action_low, np.float32)
        self.action_high = np.asarray(action_high, np.float32)
        self.scale = tf.constant((self.action_high - self.action_low) / 2.0)
        self.bias = tf.constant((self.action_high + self.action_low) / 2.0)
        self.log_scale = tf.reduce_sum(tf.math.log(self.scale))
        self.action_count = self.action_low.size
        self.network = model.gaussian_actor_critic_model(
            observation_shape, self.action_count)
        self.optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)
        self.rng = np.random.default_rng(config.SEED)
        self.action_dtype = np.float32
        self.clear()

    def _distribution(self, states, training=False):
        mean, log_std, values = self.network(states, training=training)
        return mean, tf.clip_by_value(log_std, -5.0, 2.0), tf.squeeze(values, 1)

    @tf.function(reduce_retracing=True)
    def _act(self, states):
        mean, _, values = self._distribution(states)
        return self.bias + self.scale * tf.tanh(mean), values

    def _log_probability(self, actions, mean, log_std):
        normalized = tf.clip_by_value(
            (actions - self.bias) / self.scale, -0.999999, 0.999999)
        raw = tf.atanh(normalized)
        gaussian = -0.5 * (tf.square((raw - mean) / tf.exp(log_std)
                                     ) + 2.0 * log_std + np.log(2.0 * np.pi))
        correction = tf.reduce_sum(
            tf.math.log(1.0 - tf.square(normalized) + 1e-6), axis=1
        )
        return tf.reduce_sum(gaussian, axis=1) - correction - self.log_scale

    def act(self, state, training=True):
        states = tf.convert_to_tensor(np.asarray(state, np.float32)[None])
        mean, log_std, value = self._distribution(states)
        raw = mean + tf.exp(log_std) * \
            tf.random.normal(tf.shape(mean)) if training else mean
        action = self.bias + self.scale * tf.tanh(raw)
        if training:
            self.old_log_probs.append(
                float(self._log_probability(action, mean, log_std)[0]))
            self.values.append(float(value[0]))
        return np.clip(action.numpy()[0], self.action_low, self.action_high).astype(np.float32)

    @tf.function(reduce_retracing=True)
    def _update(self, states, actions, old_log_probs, returns, advantages):
        with tf.GradientTape() as tape:
            mean, log_std, values = self._distribution(states, training=True)
            log_probs = self._log_probability(actions, mean, log_std)
            ratio = tf.exp(log_probs - old_log_probs)
            clipped = tf.clip_by_value(
                ratio, 1.0 - config.CLIP_RATIO, 1.0 + config.CLIP_RATIO)
            policy_loss = - \
                tf.reduce_mean(tf.minimum(
                    ratio * advantages, clipped * advantages))
            value_loss = tf.reduce_mean(tf.square(returns - values))
            entropy = tf.reduce_mean(tf.reduce_sum(
                log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=1))
            loss = policy_loss + config.VALUE_COEFFICIENT * \
                value_loss - config.ENTROPY_COEFFICIENT * entropy
        gradients = tape.gradient(loss, self.network.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(
            gradients, config.MAX_GRADIENT_NORM)
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_variables))
        return loss
