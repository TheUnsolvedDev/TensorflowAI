import numpy as np
import tensorflow as tf

import config
import model


class DQNAgent:
    def __init__(self, observation_shape, action_count, seed):
        self.action_count = action_count
        self.rng = np.random.default_rng(seed)
        self.online = model.build_q_network(observation_shape, action_count)
        self.target = model.build_q_network(observation_shape, action_count)
        self.target.set_weights(self.online.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)
        self.replay = model.ReplayBuffer(observation_shape, seed + 1)
        self.environment_steps = 0
        self.training_steps = 0

    @property
    def epsilon(self):
        progress = min(self.environment_steps /
                       config.EPSILON_DECAY_STEPS, 1.0)
        return config.EPSILON_START + progress * (config.EPSILON_END - config.EPSILON_START)

    @tf.function(reduce_retracing=True)
    def _q_values(self, state):
        state = tf.expand_dims(tf.cast(state, tf.float32), axis=0)
        return self.online(state, training=False)[0]

    def act(self, state, training=True):
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_count))
        return int(tf.argmax(self._q_values(state)).numpy())

    def remember(self, state, action, reward, next_state, terminal):
        self.replay.add(state, action, reward, next_state, terminal)
        self.environment_steps += 1

    @tf.function(
        input_signature=[
            tf.TensorSpec((None, None), tf.float32),
            tf.TensorSpec((None,), tf.int32),
            tf.TensorSpec((None,), tf.float32),
            tf.TensorSpec((None, None), tf.float32),
            tf.TensorSpec((None,), tf.float32),
        ]
    )
    def _train_step(self, states, actions, rewards, next_states, terminals):
        next_target_values = self.target(next_states, training=False)
        next_online_values = self.online(next_states, training=False)
        next_actions = tf.argmax(
            next_online_values, axis=1, output_type=tf.int32)
        next_values = tf.gather(
            next_target_values, next_actions, batch_dims=1)
        targets = tf.stop_gradient(
            rewards + config.GAMMA * (1.0 - terminals) * next_values)
        with tf.GradientTape() as tape:
            q_values = self.online(states, training=True)
            selected_values = tf.gather(q_values, actions, batch_dims=1)
            errors = targets - selected_values
            absolute_errors = tf.abs(errors)
            quadratic = tf.minimum(absolute_errors, config.HUBER_DELTA)
            losses = 0.5 * \
                tf.square(quadratic) + config.HUBER_DELTA * \
                (absolute_errors - quadratic)
            loss = tf.reduce_mean(losses)
        gradients = tape.gradient(loss, self.online.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(
            gradients, config.GRADIENT_CLIP_NORM)
        self.optimizer.apply_gradients(
            zip(gradients, self.online.trainable_variables))
        return loss

    def learn(self):
        if len(self.replay) < max(config.BATCH_SIZE, config.LEARNING_STARTS):
            return None
        states, actions, rewards, next_states, terminals = self.replay.sample(
            config.BATCH_SIZE)
        loss = self._train_step(
            tf.convert_to_tensor(states, tf.float32),
            tf.convert_to_tensor(actions, tf.int32),
            tf.convert_to_tensor(rewards, tf.float32),
            tf.convert_to_tensor(next_states, tf.float32),
            tf.convert_to_tensor(terminals, tf.float32),
        )
        self.training_steps += 1
        if self.training_steps % config.TARGET_UPDATE_FREQUENCY == 0:
            self.target.set_weights(self.online.get_weights())
        return float(loss.numpy())

    def save(self, path):
        self.online.save_weights(path)

    def load(self, path):
        self.online.load_weights(path)
        self.target.set_weights(self.online.get_weights())
