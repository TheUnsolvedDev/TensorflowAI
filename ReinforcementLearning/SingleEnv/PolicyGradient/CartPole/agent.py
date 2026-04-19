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
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.append(G)
        returns.reverse()
        return np.array(returns, dtype=np.float32)
    
    def save(self, path):
        self.policy.save_weights(path)

    def load(self, path):
        dummy = tf.zeros((1, *OBS_SHAPE))
        self.policy(dummy)  # build model
        self.policy.load_weights(path)

    @tf.function
    def update(self, obs, actions, returns):
        with tf.GradientTape() as tape:
            logits = self.policy(obs)
            log_probs = tf.nn.log_softmax(logits)

            action_one_hot = tf.one_hot(actions, depth=ACTION_SHAPE)
            selected_log_probs = tf.reduce_sum(action_one_hot * log_probs, axis=1)

            loss = -tf.reduce_mean(selected_log_probs * returns)

        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))