import numpy as np
import tensorflow as tf
import config
import model


class Agent:
    def __init__(self, shape, actions, seed=42):
        self.actions = actions
        self.network = model.actor_critic(shape, actions)
        self.optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)
        self.rng = np.random.default_rng(seed)
        self.clear()

    @tf.function(reduce_retracing=True)
    def _act(self, s):
        logits, value = self.network(s, training=False)
        return tf.nn.softmax(logits), tf.squeeze(value, 1)

    def act(self, state, training=True):
        probabilities, value = self._act(
            tf.convert_to_tensor(np.asarray(state, np.float32)[None]))
        p = probabilities.numpy()[0]
        action = int(self.rng.choice(self.actions, p=p)
                     ) if training else int(np.argmax(p))
        if training:
            self.behavior.append(p[action])
            self.values.append(float(value[0]))
        return action

    def remember(self, s, a, r, ns, t):
        self.states.append(s)
        self.chosen.append(a)
        self.rewards.append(r)
        self.terminals.append(float(t))
        self.next_state = ns

    @tf.function(reduce_retracing=True)
    def _train(self, states, actions, rewards, terminals, behavior, old_values, bootstrap):
        logits, values = self.network(states, training=False)
        probabilities = tf.nn.softmax(logits)
        target_prob = tf.gather(probabilities, actions, batch_dims=1)
        rho = tf.minimum(config.RHO_CLIP, target_prob / (behavior + 1e-8))
        c = tf.minimum(config.C_CLIP, target_prob / (behavior + 1e-8))
        next_values = tf.concat([old_values[1:], bootstrap[None]], 0)
        deltas = rho * (rewards + config.GAMMA * (1 - terminals)
                        * next_values - old_values)
        accumulator = tf.constant(0.0)
        vs = tf.TensorArray(tf.float32, tf.shape(rewards)[0])
        for i in tf.range(tf.shape(rewards)[0] - 1, -1, -1):
            accumulator = deltas[i] + config.GAMMA * \
                (1 - terminals[i]) * c[i] * accumulator
            vs = vs.write(i, old_values[i] + accumulator)
        targets = vs.stack()
        next_vs = tf.concat([targets[1:], bootstrap[None]], 0)
        advantages = tf.stop_gradient(
            rho * (rewards + config.GAMMA * (1 - terminals) * next_vs - old_values))
        with tf.GradientTape() as tape:
            new_logits, new_values = self.network(states, training=True)
            logp = - \
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=actions, logits=new_logits)
            p = tf.nn.softmax(new_logits)
            entropy = - \
                tf.reduce_mean(tf.reduce_sum(
                    p * tf.nn.log_softmax(new_logits), axis=1))
            policy_loss = -tf.reduce_mean(logp * advantages)
            value_loss = tf.reduce_mean(
                tf.square(tf.stop_gradient(targets) - tf.squeeze(new_values, 1)))
            loss = policy_loss + config.VALUE_COEFFICIENT * \
                value_loss - config.ENTROPY_COEFFICIENT * entropy
        self.optimizer.apply_gradients(zip(tape.gradient(
            loss, self.network.trainable_variables), self.network.trainable_variables))
        return loss

    def learn(self):
        if not self.states:
            return 0.0
        bootstrap = tf.constant(0.0) if self.terminals[-1] else self._act(
            tf.convert_to_tensor(np.asarray(self.next_state, np.float32)[None]))[1][0]
        args = [
            tf.convert_to_tensor(x) for x in [
                np.asarray(
                    self.states, np.float32), np.asarray(
                    self.chosen, np.int32), np.asarray(
                    self.rewards, np.float32), np.asarray(
                        self.terminals, np.float32), np.asarray(
                            self.behavior, np.float32), np.asarray(
                                self.values, np.float32)]]
        loss = float(self._train(*args, bootstrap))
        self.clear()
        return loss

    def clear(self):
        self.states = []
        self.chosen = []
        self.rewards = []
        self.terminals = []
        self.behavior = []
        self.values = []
        self.next_state = None

    def save(self, path):
        self.network.save_weights(path)

    def load(self, path):
        self.network.load_weights(path)
