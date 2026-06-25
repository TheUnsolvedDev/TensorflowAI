import silence_tensorflow.auto
import tensorflow as tf
import numpy as np

from model import build_q_network as QNetwork
from buffer import ReplayBuffer


class DQN:
    def __init__(self, obs_dim, action_dim, lr, gamma, buffer_size):
        self.strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.NcclAllReduce())
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.lr = lr
        self.buffer = ReplayBuffer(buffer_size, obs_dim)

        with self.strategy.scope():
            self.q_net = QNetwork(obs_dim, action_dim)
            self.target_net = QNetwork(obs_dim, action_dim)

            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr,
                clipnorm=10.0
            )
            self.update_target()

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    @tf.function
    def _distributed_q(self, obs):
        def step_fn(obs):
            return self.q_net(obs, training=False)
        per_replica = self.strategy.run(step_fn, args=(obs,))
        return tf.concat(
            self.strategy.experimental_local_results(per_replica),
            axis=0
        )

    def sample_actions(self, obs, epsilon=0.1):
        obs_tf = tf.convert_to_tensor(obs, tf.float32)
        per_replica_batch = obs.shape[0] // self.strategy.num_replicas_in_sync
        dist_obs = self.strategy.experimental_distribute_values_from_function(
            lambda ctx: obs_tf[
                ctx.replica_id_in_sync_group * per_replica_batch:
                (ctx.replica_id_in_sync_group + 1) * per_replica_batch
            ]
        )
        q_values = self._distributed_q(dist_obs).numpy()
        batch_size, action_dim = q_values.shape
        greedy_actions = np.argmax(q_values, axis=-1)
        random_actions = np.random.randint(
            0, action_dim, size=batch_size, dtype=np.int32)
        explore = np.random.rand(batch_size) < epsilon
        actions = np.where(explore, random_actions, greedy_actions)
        return actions.astype(np.int32)

    @tf.function
    def distributed_train_step(self, obs, actions, rewards, next_obs, dones):

        def step_fn(obs, actions, rewards, next_obs, dones):
            with tf.GradientTape() as tape:
                q_values = self.q_net(obs, training=True)
                action_mask = tf.one_hot(
                    actions, self.action_dim, dtype=tf.float32)
                pred_q = tf.reduce_sum(q_values * action_mask, axis=-1)
                next_q = self.target_net(next_obs, training=False)
                max_next_q = tf.reduce_max(next_q, axis=-1)
                target_q = rewards + (1.0 - dones) * self.gamma * max_next_q
                target_q = tf.stop_gradient(target_q)
                loss = tf.reduce_mean(tf.keras.losses.Huber()(target_q, pred_q))
            grads = tape.gradient(loss, self.q_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.q_net.trainable_variables))
            return loss
        per_replica_loss = self.strategy.run(
            step_fn, args=(obs, actions, rewards, next_obs, dones))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)

    def train_step(self, experiences):
        obs, actions, rewards, next_obs, dones = experiences
        loss = self.distributed_train_step(
            obs.astype(np.float32),
            actions.astype(np.int32),
            rewards.astype(np.float32),
            next_obs.astype(np.float32),
            dones.astype(np.float32)
        )
        return loss
    
    def save(self, path):
        self.q_net.save_weights(f"{path}/q_net.weights.h5")
        self.target_net.save_weights(f"{path}/target_net.weights.h5")
        
    def load(self, path):
        dummy = np.zeros((1, self.obs_dim), dtype=np.float32)
        self.q_net(dummy)
        self.target_net(dummy)
        self.q_net.load_weights(f"{path}/q_net.weights.h5")
        self.target_net.load_weights(f"{path}/target_net.weights.h5")
