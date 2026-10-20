import numpy as np
import tensorflow as tf

import config
import model


class TRPOAgent:
    def __init__(self, observation_shape, action_count):
        self.action_count = action_count
        self.policy = model.policy_model(observation_shape, action_count)
        self.value = model.value_model(observation_shape)
        self.value_optimizer = tf.keras.optimizers.Adam(
            config.VALUE_LEARNING_RATE)
        self.rng = np.random.default_rng(config.SEED)
        self.clear()

    @tf.function(reduce_retracing=True)
    def _act(self, states):
        return tf.nn.softmax(
            self.policy(
                states, training=False)), tf.squeeze(
            self.value(
                states, training=False), 1)

    def act(self, state, training=True):
        probabilities, value = self._act(
            tf.convert_to_tensor(np.asarray(state, np.float32)[None]))
        probabilities = probabilities.numpy()[0]
        action = int(self.rng.choice(self.action_count, p=probabilities)
                     ) if training else int(np.argmax(probabilities))
        if training:
            self.old_probabilities.append(probabilities)
            self.values.append(float(value[0]))
        return action

    def remember(self, state, action, reward, next_state, terminal):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(float(terminal))
        self.next_state = next_state

    def _flat(self, tensors):
        return tf.concat([tf.reshape(tensor, [-1]) for tensor in tensors], 0)

    def _parameters(self):
        return self._flat(self.policy.trainable_variables)

    def _assign(self, flat):
        offset = 0
        for variable in self.policy.trainable_variables:
            size = int(tf.size(variable))
            variable.assign(tf.reshape(
                flat[offset:offset + size], variable.shape))
            offset += size

    def _surrogate(self, states, actions, old_probabilities, advantages):
        probabilities = tf.nn.softmax(self.policy(states, training=True))
        selected = tf.gather(probabilities, actions, batch_dims=1)
        old_selected = tf.gather(old_probabilities, actions, batch_dims=1)
        return tf.reduce_mean(selected / (old_selected + 1e-8) * advantages)

    def _kl(self, states, old_probabilities):
        probabilities = tf.nn.softmax(self.policy(states, training=True))
        log_ratio = tf.math.log(old_probabilities + 1e-8) - tf.math.log(
            probabilities + 1e-8
        )
        return tf.reduce_mean(
            tf.reduce_sum(old_probabilities * log_ratio, axis=1)
        )

    def _hessian_vector_product(self, states, old_probabilities, vector):
        with tf.GradientTape() as outer:
            with tf.GradientTape() as inner:
                kl = self._kl(states, old_probabilities)
            gradient = self._flat(inner.gradient(
                kl, self.policy.trainable_variables))
            product = tf.reduce_sum(gradient * vector)
        return self._flat(outer.gradient(product, self.policy.trainable_variables)
                          ) + config.CG_DAMPING * vector

    def _conjugate_gradient(self, product, b):
        x = tf.zeros_like(b)
        residual = tf.identity(b)
        direction = tf.identity(b)
        residual_dot = tf.reduce_sum(residual * residual)
        for _ in range(config.CG_ITERATIONS):
            product_direction = product(direction)
            alpha = residual_dot / \
                (tf.reduce_sum(direction * product_direction) + 1e-8)
            x += alpha * direction
            residual -= alpha * product_direction
            new_dot = tf.reduce_sum(residual * residual)
            if float(new_dot) < 1e-10:
                break
            direction = residual + new_dot / (residual_dot + 1e-8) * direction
            residual_dot = new_dot
        return x

    @tf.function(reduce_retracing=True)
    def _value_step(self, states, returns):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(
                tf.square(returns - tf.squeeze(self.value(states, training=True), 1)))
        self.value_optimizer.apply_gradients(zip(tape.gradient(
            loss, self.value.trainable_variables), self.value.trainable_variables))
        return loss

    def learn(self):
        if not self.states:
            return 0.0
        states = tf.convert_to_tensor(np.asarray(self.states, np.float32))
        actions = tf.convert_to_tensor(self.actions, tf.int32)
        old_probabilities = tf.convert_to_tensor(
            np.asarray(self.old_probabilities, np.float32))
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
        returns = tf.convert_to_tensor(
            advantages + np.asarray(self.values, np.float32))
        advantages = tf.convert_to_tensor(
            (advantages - advantages.mean()) / (advantages.std() + 1e-8))
        with tf.GradientTape() as tape:
            objective = self._surrogate(
                states, actions, old_probabilities, advantages)
        gradient = self._flat(tape.gradient(
            objective, self.policy.trainable_variables))
        direction = self._conjugate_gradient(lambda vector: self._hessian_vector_product(
            states, old_probabilities, vector), gradient)
        curvature = tf.reduce_sum(
            direction * self._hessian_vector_product(states, old_probabilities, direction))
        full_step = direction * \
            tf.sqrt(2.0 * config.MAX_KL / (curvature + 1e-8))
        old_parameters = self._parameters()
        old_objective = float(objective)
        accepted = False
        for step in range(config.BACKTRACK_STEPS):
            self._assign(old_parameters +
                         (config.BACKTRACK_COEFFICIENT ** step) * full_step)
            new_objective = float(self._surrogate(
                states, actions, old_probabilities, advantages))
            if new_objective > old_objective and float(
                    self._kl(states, old_probabilities)) <= config.MAX_KL:
                accepted = True
                break
        if not accepted:
            self._assign(old_parameters)
        losses = [float(self._value_step(states, returns))
                  for _ in range(config.VALUE_EPOCHS)]
        self.clear()
        return float(np.mean(losses))

    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.terminals, self.old_probabilities, self.values = [], [], []
        self.next_state = None

    def save(self, path):
        self.policy.save_weights(path)

    def load(self, path):
        self.policy.load_weights(path)
