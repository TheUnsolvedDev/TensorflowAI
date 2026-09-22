import tensorflow as tf


class DQNAgent:
    def __init__(self, model, strategy):
        self.model = model
        self.strategy = strategy

    def act(self, states):
        return tf.argmax(self.model(states, training=False), axis=-1)
