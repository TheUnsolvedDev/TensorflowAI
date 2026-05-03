import numpy as np
import tensorflow as tf

from .model import build_policy_network
from .buffer import TrajectoryBuffer
from .utils import mask_invalid_actions, sample_action, canonicalize


class PolicyGradientAgent:
    def __init__(self, input_shape, output_shape, learning_rate=0.01):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.model = build_policy_network(
            self.input_shape[0], self.output_shape)
        self.buffer = TrajectoryBuffer()
        self.model.summary()

    @tf.function
    def _act(self, obs):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        obs = tf.expand_dims(obs, axis=0)   # (1, 9)
        return self.model(obs)

    def act(self, obs, player):
        canonical_state = canonicalize(obs, player).astype(np.float32)
        probs = self._act(canonical_state).numpy().flatten()
        probs = mask_invalid_actions(probs, canonical_state)
        action = sample_action(probs)
        self.buffer.store(canonical_state, action, 0.0)
        return action

    def store_reward(self, reward):
        self.buffer.rewards[-1] = reward

    @tf.function
    def _learn(self, states, actions, returns):
        actions = tf.cast(actions, tf.int32)
        with tf.GradientTape() as tape:
            probs = self.model(states)
            indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions],axis=1)
            selected_probs = tf.gather_nd(probs, indices)
            loss = -tf.reduce_mean(tf.math.log(selected_probs + 1e-8) * returns)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def learn(self):
        states = np.array(self.buffer.states, dtype=np.float32)
        actions = np.array(self.buffer.actions, dtype=np.int32)
        returns = self.buffer.compute_returns()
        loss = self._learn(states, actions, returns)
        self.buffer.reset()
        return loss.numpy()
