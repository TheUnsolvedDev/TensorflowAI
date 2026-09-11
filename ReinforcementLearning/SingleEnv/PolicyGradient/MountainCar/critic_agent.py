import tensorflow as tf
from config import *
from model import policy_network

def value_network(input_shape=OBS_SHAPE):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)(inputs)
    x = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)(x)
    return tf.keras.Model(inputs=inputs, outputs=tf.keras.layers.Dense(1)(x))

class PolicyGradientAgent:
    def __init__(self):
        self.policy = policy_network(); self.critic = value_network()
        self.policy_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
        self.critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LEARNING_RATE)
    @tf.function
    def model_act(self, obs):
        return tf.squeeze(tf.random.categorical(self.policy(tf.expand_dims(obs, 0)), 1), axis=1)
    def get_action(self, obs): return int(self.model_act(tf.convert_to_tensor(obs, tf.float32)).numpy()[0])
    def save(self, path):
        self.policy.save_weights(path.replace('.weights.h5', '.policy.weights.h5'))
        self.critic.save_weights(path.replace('.weights.h5', '.critic.weights.h5'))
    def load(self, path):
        dummy=tf.zeros((1,*OBS_SHAPE)); self.policy(dummy); self.critic(dummy)
        self.policy.load_weights(path.replace('.weights.h5', '.policy.weights.h5'))
        self.critic.load_weights(path.replace('.weights.h5', '.critic.weights.h5'))
    @tf.function
    def update(self, obs, actions, returns):
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            selected=tf.reduce_sum(tf.one_hot(actions,ACTION_SHAPE)*tf.nn.log_softmax(self.policy(obs)),axis=1)
            values=tf.squeeze(self.critic(obs),axis=1)
            advantages=tf.stop_gradient(returns-values)
            actor_loss=-tf.reduce_mean(selected*advantages); critic_loss=tf.reduce_mean(tf.square(returns-values))
        self.policy_optimizer.apply_gradients(zip(actor_tape.gradient(actor_loss,self.policy.trainable_variables),self.policy.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_tape.gradient(critic_loss,self.critic.trainable_variables),self.critic.trainable_variables))
        return actor_loss, critic_loss

