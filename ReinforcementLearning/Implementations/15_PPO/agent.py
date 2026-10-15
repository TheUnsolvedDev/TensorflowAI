import numpy as np
import tensorflow as tf

import model


class PolicyAgent:
    def __init__(self, observation_shape, action_count, algorithm="reinforce", gamma=0.99):
        self.algorithm = algorithm
        self.gamma = gamma
        self.action_count = action_count
        if algorithm in {"policy_gradient", "reinforce"}:
            self.model = model.policy_model(observation_shape, action_count)
            self.value = None
        else:
            self.model = model.actor_critic_model(observation_shape, action_count)
            self.value = self.model
        self.optimizer = tf.keras.optimizers.Adam(3e-4)
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.old_probabilities = []

    def act(self, state, training=True):
        state_batch = np.asarray(state, dtype=np.float32)[None]
        output = self.model(state_batch, training=False)
        probabilities = output[0] if isinstance(output, (list, tuple)) else output
        probabilities = probabilities.numpy()[0]
        action = np.random.choice(self.action_count, p=probabilities) if training else int(np.argmax(probabilities))
        if training:
            self.old_probabilities.append(probabilities[action])
        return int(action)

    def remember(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(float(done))

    def learn(self):
        if not self.states:
            return 0.0
        states = tf.convert_to_tensor(self.states, tf.float32)
        actions = tf.convert_to_tensor(self.actions, tf.int32)
        returns = model.discounted_returns(self.rewards, self.dones, self.gamma)
        if self.algorithm == "policy_gradient":
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        with tf.GradientTape() as tape:
            output = self.model(states, training=True)
            if isinstance(output, (list, tuple)):
                probabilities, values = output
                values = tf.squeeze(values, axis=1)
                advantages = tf.stop_gradient(returns - values)
                selected = tf.gather(probabilities, actions, batch_dims=1)
                policy_loss = -tf.reduce_mean(tf.math.log(selected + 1e-8) * advantages)
                value_loss = tf.reduce_mean(tf.square(returns - values))
                loss = policy_loss + 0.5 * value_loss
                if self.algorithm == "ppo":
                    old = tf.convert_to_tensor(self.old_probabilities, tf.float32)
                    ratio = selected / (old + 1e-8)
                    clipped = tf.clip_by_value(ratio, 0.8, 1.2)
                    loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped * advantages)) + 0.5 * value_loss
                if self.algorithm == "trpo":
                    old = tf.convert_to_tensor(self.old_probabilities, tf.float32)
                    ratio = selected / (old + 1e-8)
                    kl = tf.reduce_mean(old * tf.math.log((old + 1e-8) / (selected + 1e-8)))
                    loss = -tf.reduce_mean(ratio * advantages) + 10.0 * tf.square(tf.maximum(kl - 0.01, 0.0))
            else:
                selected = tf.gather(output, actions, batch_dims=1)
                loss = -tf.reduce_mean(tf.math.log(selected + 1e-8) * returns)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.old_probabilities.clear()
        return float(loss)

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)


