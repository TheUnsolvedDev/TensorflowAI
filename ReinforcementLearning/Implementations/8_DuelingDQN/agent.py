import numpy as np
import tensorflow as tf

import model


class DQNAgent:
    def __init__(self, observation_shape, action_count, variant="dqn", gamma=0.99):
        self.variant = variant
        self.gamma = gamma
        self.action_count = action_count
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        dueling = variant in {"dueling", "rainbow"}
        self.model = model.q_model(observation_shape, action_count, dueling)
        self.target = model.q_model(observation_shape, action_count, dueling)
        self.target.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.buffer = model.PrioritizedReplayBuffer() if variant in {"per", "rainbow"} else model.ReplayBuffer()
        self.steps = 0

    def act(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_count)
        values = self.model(np.asarray(state, np.float32)[None], training=False)
        return int(tf.argmax(values[0]))

    def remember(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def learn(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return 0.0
        prioritized = self.variant in {"per", "rainbow"}
        if prioritized:
            batch, indices, weights = self.buffer.sample(batch_size)
        else:
            batch = self.buffer.sample(batch_size)
            weights = np.ones(batch_size, np.float32)
        states, actions, rewards, next_states, dones = batch
        next_online = self.model(next_states, training=False)
        next_target = self.target(next_states, training=False)
        if self.variant in {"double", "rainbow"}:
            next_actions = tf.argmax(next_online, axis=1)
            next_values = tf.gather(next_target, next_actions, batch_dims=1)
        else:
            next_values = tf.reduce_max(next_target, axis=1)
        targets = rewards + self.gamma * (1.0 - dones.astype(np.float32)) * next_values.numpy()
        with tf.GradientTape() as tape:
            values = self.model(states, training=True)
            chosen = tf.gather(values, actions.astype(np.int32), batch_dims=1)
            errors = targets - chosen
            losses = tf.keras.losses.huber(targets, chosen)
            loss = tf.reduce_mean(losses * weights)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        if prioritized:
            self.buffer.update(indices, errors.numpy())
        self.steps += 1
        if self.steps % 100 == 0:
            self.target.set_weights(self.model.get_weights())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss)

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)
        self.target.set_weights(self.model.get_weights())


