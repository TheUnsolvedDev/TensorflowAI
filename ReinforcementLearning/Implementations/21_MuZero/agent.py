import numpy as np
import tensorflow as tf
import config
import model


class Node:
    def __init__(self, latent, prior=1.0, reward=0.0):
        self.latent = latent
        self.prior = float(prior)
        self.reward = float(reward)
        self.visits = 0
        self.value_sum = 0.0
        self.children = {}

    @property
    def value(self):
        return self.value_sum / self.visits if self.visits else 0.0


class Agent:
    def __init__(self, shape, actions, seed=42):
        self.actions = actions
        self.representation, self.dynamics, self.prediction = model.build_models(
            shape, actions)
        self.optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE)
        self.rng = np.random.default_rng(seed)
        n = config.REPLAY_CAPACITY
        self.s = np.empty((n,) + tuple(shape), np.float32)
        self.ns = np.empty_like(self.s)
        self.a = np.empty(n, np.int32)
        self.r = np.empty(n, np.float32)
        self.t = np.empty(n, np.float32)
        self.pi = np.empty((n, actions), np.float32)
        self.position = self.size = 0
        self.last_policy = np.ones(actions, np.float32) / actions

    def _expand(self, node):
        logits, value = self.prediction(node.latent[None], training=False)
        priors = tf.nn.softmax(logits)[0].numpy()
        for action, prior in enumerate(priors):
            onehot = tf.one_hot([action], self.actions)
            latent, reward = self.dynamics(
                [node.latent[None], onehot], training=False)
            node.children[action] = Node(latent[0], prior, float(reward[0, 0]))
        return float(value[0, 0])

    def _search(self, state):
        latent = self.representation(tf.convert_to_tensor(
            np.asarray(state, np.float32)[None]), training=False)[0]
        root = Node(latent)
        self._expand(root)
        for _ in range(config.SIMULATIONS):
            node = root
            path = [root]
            actions = []
            while node.children:
                scale = np.sqrt(node.visits + 1)
                action = max(
                    node.children,
                    key=lambda a: node.children[a].value +
                    config.C_PUCT *
                    node.children[a].prior *
                    scale /
                    (
                        1 +
                        node.children[a].visits))
                actions.append(action)
                node = node.children[action]
                path.append(node)
                if node.visits == 0:
                    break
            value = self._expand(node) if not node.children else node.value
            for parent, action, child in zip(
                    reversed(path[:-1]), reversed(actions), reversed(path[1:])):
                value = child.reward + config.GAMMA * value
                child.value_sum += value
                child.visits += 1
            root.visits += 1
        visits = np.asarray(
            [root.children[a].visits for a in range(self.actions)], np.float32)
        return visits / (visits.sum() + 1e-8)

    def act(self, state, training=True):
        self.last_policy = self._search(state)
        return int(
            self.rng.choice(
                self.actions,
                p=self.last_policy)) if training else int(
            np.argmax(
                self.last_policy))

    def remember(self, s, a, r, ns, t):
        i = self.position
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.ns[i] = ns
        self.t[i] = t
        self.pi[i] = self.last_policy
        self.position = (i + 1) % config.REPLAY_CAPACITY
        self.size = min(self.size + 1, config.REPLAY_CAPACITY)

    @tf.function(reduce_retracing=True)
    def _train(self, s, a, r, ns, t, pi):
        variables = self.representation.trainable_variables + \
            self.dynamics.trainable_variables + self.prediction.trainable_variables
        with tf.GradientTape() as tape:
            z = self.representation(s, training=True)
            logits, value = self.prediction(z, training=True)
            next_z, predicted_reward = self.dynamics(
                [z, tf.one_hot(a, self.actions)], training=True)
            target_z = tf.stop_gradient(
                self.representation(ns, training=False))
            _, next_value = self.prediction(target_z, training=False)
            target_value = tf.stop_gradient(
                r + config.GAMMA * (1 - t) * tf.squeeze(next_value, 1))
            policy_loss = - \
                tf.reduce_mean(tf.reduce_sum(
                    pi * tf.nn.log_softmax(logits), axis=1))
            value_loss = tf.reduce_mean(
                tf.square(target_value - tf.squeeze(value, 1)))
            reward_loss = tf.reduce_mean(
                tf.square(r - tf.squeeze(predicted_reward, 1)))
            consistency = tf.reduce_mean(tf.square(next_z - target_z))
            loss = policy_loss + value_loss + reward_loss + consistency
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, variables), variables))
        return loss

    def learn(self):
        if self.size < max(config.BATCH_SIZE, config.LEARNING_STARTS):
            return None
        i = self.rng.integers(self.size, size=config.BATCH_SIZE)
        return float(self._train(*[tf.convert_to_tensor(x) for x in [self.s[i],
                     self.a[i], self.r[i], self.ns[i], self.t[i], self.pi[i]]]))

    def save(self, path):
        self.representation.save_weights(path)
        self.dynamics.save_weights(
            path.replace(".weights.h5", ".dynamics.weights.h5")
        )
        self.prediction.save_weights(
            path.replace(".weights.h5", ".prediction.weights.h5")
        )

    def load(self, path):
        self.representation.load_weights(path)
        self.dynamics.load_weights(
            path.replace(".weights.h5", ".dynamics.weights.h5")
        )
        self.prediction.load_weights(
            path.replace(".weights.h5", ".prediction.weights.h5")
        )
