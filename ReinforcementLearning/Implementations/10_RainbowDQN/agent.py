from collections import deque

import numpy as np
import tensorflow as tf

import config
import model


class DQNAgent:
    def __init__(self, observation_shape, action_count, seed):
        self.action_count = action_count
        self.online = model.build_q_network(observation_shape, action_count)
        self.target = model.build_q_network(observation_shape, action_count)
        self.target.set_weights(self.online.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)
        self.replay = model.ReplayBuffer(observation_shape, seed + 1)
        self.n_step_buffer = deque()
        self.support = tf.linspace(config.V_MIN, config.V_MAX, config.ATOM_COUNT)
        self.environment_steps = 0
        self.training_steps = 0

    @property
    def epsilon(self):
        return 0.0

    @tf.function(reduce_retracing=True)
    def _q_values(self, state, training):
        distribution = self.online(tf.expand_dims(tf.cast(state, tf.float32), 0), training=training)[0]
        return tf.reduce_sum(distribution * self.support, axis=1)

    def act(self, state, training=True):
        return int(tf.argmax(self._q_values(state, training)).numpy())

    def remember(self, state, action, reward, next_state, terminal, episode_end=False):
        self.n_step_buffer.append((state, action, reward, next_state, terminal))
        if len(self.n_step_buffer) >= config.N_STEP:
            self._store_n_step()
        if episode_end:
            while self.n_step_buffer:
                self._store_n_step()
        self.environment_steps += 1

    def _store_n_step(self):
        total_reward = 0.0
        final_next_state = self.n_step_buffer[0][3]
        final_terminal = False
        steps = 0
        for _, _, reward, next_state, terminal in self.n_step_buffer:
            total_reward += (config.GAMMA ** steps) * reward
            final_next_state = next_state
            final_terminal = terminal
            steps += 1
            if terminal or steps == config.N_STEP:
                break
        state, action = self.n_step_buffer[0][:2]
        self.replay.add(state, action, total_reward, final_next_state, final_terminal, config.GAMMA ** steps)
        self.n_step_buffer.popleft()

    @tf.function(
        input_signature=[
            tf.TensorSpec((None, None), tf.float32),
            tf.TensorSpec((None,), tf.int32),
            tf.TensorSpec((None,), tf.float32),
            tf.TensorSpec((None, None), tf.float32),
            tf.TensorSpec((None,), tf.float32),
            tf.TensorSpec((None,), tf.float32),
            tf.TensorSpec((None,), tf.float32),
        ]
    )
    def _train_step(self, states, actions, rewards, next_states, terminals, discounts, weights):
        next_online = self.online(next_states, training=True)
        next_values = tf.reduce_sum(next_online * self.support, axis=2)
        next_actions = tf.argmax(next_values, axis=1, output_type=tf.int32)
        next_target = self.target(next_states, training=True)
        next_probabilities = tf.gather(next_target, next_actions, batch_dims=1)
        target_support = rewards[:, None] + discounts[:, None] * (1.0 - terminals[:, None]) * self.support
        target_support = tf.clip_by_value(target_support, config.V_MIN, config.V_MAX)
        positions = (target_support - config.V_MIN) / config.ATOM_DELTA
        lower = tf.cast(tf.floor(positions), tf.int32)
        upper = tf.cast(tf.math.ceil(positions), tf.int32)
        equal = tf.cast(tf.equal(lower, upper), tf.float32)
        projected = tf.reduce_sum(
            next_probabilities[:, :, None]
            * (
                tf.one_hot(lower, config.ATOM_COUNT)
                * (tf.cast(upper, tf.float32) - positions + equal)[:, :, None]
                + tf.one_hot(upper, config.ATOM_COUNT)
                * (positions - tf.cast(lower, tf.float32))[:, :, None]
            ),
            axis=1,
        )
        projected = tf.stop_gradient(projected)
        with tf.GradientTape() as tape:
            distribution = self.online(states, training=True)
            selected = tf.gather(distribution, actions, batch_dims=1)
            sample_losses = -tf.reduce_sum(projected * tf.math.log(selected + 1e-8), axis=1)
            loss = tf.reduce_mean(weights * sample_losses)
        gradients = tape.gradient(loss, self.online.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, config.GRADIENT_CLIP_NORM)
        self.optimizer.apply_gradients(zip(gradients, self.online.trainable_variables))
        return loss, sample_losses

    def learn(self):
        if len(self.replay) < max(config.BATCH_SIZE, config.LEARNING_STARTS):
            return None
        progress = min(self.training_steps / config.PRIORITY_BETA_STEPS, 1.0)
        beta = config.PRIORITY_BETA_START + progress * (1.0 - config.PRIORITY_BETA_START)
        batch, indices, weights = self.replay.sample(config.BATCH_SIZE, beta)
        states, actions, rewards, next_states, terminals, discounts = batch
        loss, priorities = self._train_step(
            tf.convert_to_tensor(states, tf.float32),
            tf.convert_to_tensor(actions, tf.int32),
            tf.convert_to_tensor(rewards, tf.float32),
            tf.convert_to_tensor(next_states, tf.float32),
            tf.convert_to_tensor(terminals, tf.float32),
            tf.convert_to_tensor(discounts, tf.float32),
            tf.convert_to_tensor(weights, tf.float32),
        )
        self.replay.update_priorities(indices, priorities.numpy())
        self.training_steps += 1
        if self.training_steps % config.TARGET_UPDATE_FREQUENCY == 0:
            self.target.set_weights(self.online.get_weights())
        return float(loss.numpy())

    def save(self, path):
        self.online.save_weights(path)

    def load(self, path):
        self.online.load_weights(path)
        self.target.set_weights(self.online.get_weights())
