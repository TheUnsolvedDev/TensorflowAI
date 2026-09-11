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
        logits = self.policy(tf.expand_dims(obs, 0))
        return tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    def get_action(self, obs):
        return int(self.model_act(tf.convert_to_tensor(obs, dtype=tf.float32)).numpy()[0])
    def compute_returns(self, rewards):
        values = []
        value = 0
        for reward in reversed(rewards):
            value = reward + GAMMA * value
            values.append(value)
        return np.asarray(values[::-1], dtype=np.float32)
    def save(self, path): self.policy.save_weights(path)
    def load(self, path):
        self.policy(tf.zeros((1, *OBS_SHAPE)))
        self.policy.load_weights(path)
    @tf.function
    def update(self, obs, actions, returns):
        with tf.GradientTape() as tape:
            log_probs = tf.nn.log_softmax(self.policy(obs))
            selected = tf.reduce_sum(tf.one_hot(actions, ACTION_SHAPE) * log_probs, axis=1)
            loss = -tf.reduce_mean(selected * returns)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.policy.trainable_variables), self.policy.trainable_variables))
