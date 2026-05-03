import numpy as np
import tensorflow as tf

from .model import build_actor_critic
from .buffer import A2CBuffer
from .utils import canonicalize, mask_invalid_actions, sample_action


class ActorCriticAgent:
    def __init__(self, input_shape, output_shape, lr=1e-3, gamma=0.99):
        self.model = build_actor_critic(input_shape[0], output_shape)
        self.opt = tf.keras.optimizers.Adam(lr)
        self.buffer = A2CBuffer(gamma)
        
    @tf.function
    def _act(self, obs):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        obs = tf.expand_dims(obs, axis=0)   # (1, 9)
        return self.model(obs)

    def act(self, obs, player):
        canonical_state = canonicalize(obs, player).astype(np.float32)
        probs, _ = self._act(canonical_state)
        probs = mask_invalid_actions(probs.numpy().flatten(), canonical_state)
        action = sample_action(probs)
        self.buffer.store(canonical_state, action, 0.0)
        return action
    
    def store_reward(self, r):
        self.buffer.rewards[-1] = r

    @tf.function
    def _learn(self, states, actions, returns):
        actions = tf.cast(actions, tf.int32)

        with tf.GradientTape() as tape:
            probs, values = self.model(states)
            values = tf.squeeze(values, axis=1)

            idx = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
            logp = tf.math.log(tf.gather_nd(probs, idx) + 1e-8)
            adv = returns - values
            actor_loss = -tf.reduce_mean(logp * adv)
            critic_loss = tf.reduce_mean(tf.square(adv))
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1))
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def learn(self):
        s = np.array(self.buffer.states, dtype=np.float32)
        a = np.array(self.buffer.actions, dtype=np.int32)
        r = self.buffer.compute_returns()

        loss = self._learn(s, a, r)
        self.buffer.reset()
        return loss.numpy()