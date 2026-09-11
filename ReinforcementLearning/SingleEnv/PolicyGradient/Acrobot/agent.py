import tensorflow as tf
import numpy as np

from config import *
from model import policy_network


class PolicyGradientAgent:
    def __init__(self):
        self.policy = policy_network()
        self.optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    @tf.function
    def model_act(self, obs):
        obs = tf.expand_dims(obs, axis=0)
        logits = self.policy(obs)
        action = tf.random.categorical(logits, 1)
        return tf.squeeze(action, axis=1)

    def get_action(self, obs):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        action = self.model_act(obs)
        return int(action.numpy()[0])

    def compute_returns(self, rewards):
        returns = []
        value = 0
        for reward in reversed(rewards):
            value = reward + GAMMA * value
            returns.append(value)
        returns.reverse()
        return np.array(returns, dtype=np.float32)

    def save(self, path):
        self.policy.save_weights(path)

    def load(self, path):
        self.policy(tf.zeros((1, *OBS_SHAPE)))
        self.policy.load_weights(path)

    @tf.function
    def update(self, obs, actions, returns):
        with tf.GradientTape() as tape:
            logits = self.policy(obs)
            log_probs = tf.nn.log_softmax(logits)
            selected = tf.reduce_sum(
                tf.one_hot(actions, depth=ACTION_SHAPE) * log_probs, axis=1
            )
            loss = -tf.reduce_mean(selected * returns)
        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
