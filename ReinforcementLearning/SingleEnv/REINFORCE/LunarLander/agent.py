import tensorflow as tf
from config import *
from model import policy_network
class REINFORCEAgent:
    def __init__(self): self.policy=policy_network(); self.optimizer=tf.keras.optimizers.Adam(LEARNING_RATE)
    @tf.function
    def model_act(self,obs): return tf.squeeze(tf.random.categorical(self.policy(tf.expand_dims(obs,0)),1),axis=1)
    def get_action(self,obs): return int(self.model_act(tf.convert_to_tensor(obs,tf.float32)).numpy()[0])
    def save(self,path): self.policy.save_weights(path)
    def load(self,path): self.policy(tf.zeros((1,*OBS_SHAPE))); self.policy.load_weights(path)
    @tf.function
    def update(self,obs,actions,returns):
        with tf.GradientTape() as tape:
            selected=tf.reduce_sum(tf.one_hot(actions,ACTION_SHAPE)*tf.nn.log_softmax(self.policy(obs)),axis=1); loss=-tf.reduce_mean(selected*returns)
        self.optimizer.apply_gradients(zip(tape.gradient(loss,self.policy.trainable_variables),self.policy.trainable_variables))

