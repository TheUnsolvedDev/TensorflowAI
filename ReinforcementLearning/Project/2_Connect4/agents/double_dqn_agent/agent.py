import os

import tensorflow as tf

from .model import build_q_network as QNetwork
from .buffer import ReplayBuffer


class DoubleDQNAgent:
    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.99, buffer_size=100000):
        self.strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.NcclAllReduce())
        self.gamma = gamma
        self.action_dim = action_dim
        self.buffer = ReplayBuffer(buffer_size, obs_dim, action_dim)

        with self.strategy.scope():
            self.q_net = QNetwork(obs_dim, action_dim)
            self.target_net = QNetwork(obs_dim, action_dim)

            self.optimizer = tf.keras.optimizers.Adam(lr)
            self.update_target()

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    @tf.function
    def _distributed_act(self, obs, mask, eps):
        def step_fn(obs, mask):
            q = self.q_net(obs)
            neg_inf = tf.constant(-1e9, dtype=q.dtype)
            q_masked = tf.where(mask > 0, q, neg_inf)
            greedy = tf.argmax(q_masked, axis=-1, output_type=tf.int32)
            valid_probs = mask / tf.reduce_sum(mask, axis=-1, keepdims=True)
            valid_probs = tf.where(mask > 0, valid_probs, 0.0)
            random_actions = tf.random.categorical(
                tf.math.log(valid_probs), 1, dtype=tf.int32
            )
            random_actions = tf.squeeze(random_actions, axis=-1)
            explore = tf.random.uniform(tf.shape(greedy)) < eps
            actions = tf.where(explore, random_actions, greedy)
            batch_idx = tf.range(tf.shape(actions)[0], dtype=tf.int32)
            indices = tf.stack([batch_idx, actions], axis=1)
            valid = tf.gather_nd(mask, indices) > 0
            fallback = tf.argmax(mask, axis=-1, output_type=tf.int32)
            actions = tf.where(valid, actions, fallback)
            return actions

        per_replica = self.strategy.run(step_fn, args=(obs, mask))
        return tf.concat(self.strategy.experimental_local_results(per_replica), axis=0)

    def act(self, obs, mask, eps=0.1):
        obs = tf.convert_to_tensor(obs, tf.float32)
        mask = tf.convert_to_tensor(mask, tf.float32)

        per_replica_batch = obs.shape[0] // self.strategy.num_replicas_in_sync

        dist_obs = self.strategy.experimental_distribute_values_from_function(
            lambda ctx: obs[
                ctx.replica_id_in_sync_group * per_replica_batch:
                (ctx.replica_id_in_sync_group + 1) * per_replica_batch
            ]
        )

        dist_mask = self.strategy.experimental_distribute_values_from_function(
            lambda ctx: mask[
                ctx.replica_id_in_sync_group * per_replica_batch:
                (ctx.replica_id_in_sync_group + 1) * per_replica_batch
            ]
        )

        return self._distributed_act(dist_obs, dist_mask, eps)

    @tf.function
    def _train_step(self, obs, actions, rewards, next_obs, dones, next_mask):
        def step_fn(obs, actions, rewards, next_obs, dones, next_mask):
            with tf.GradientTape() as tape:
                q = self.q_net(obs)
                q_action = tf.reduce_sum(
                    q * tf.one_hot(actions, self.action_dim), axis=-1)
                next_q_online = self.q_net(next_obs)
                neg_inf = tf.constant(-1e9, dtype=next_q_online.dtype)
                next_q_online_masked = tf.where(
                    next_mask > 0, next_q_online, neg_inf)
                next_actions = tf.argmax(
                    next_q_online_masked, axis=-1, output_type=tf.int32)
                next_q_target = self.target_net(next_obs)
                next_q_target_action = tf.reduce_sum(
                    next_q_target * tf.one_hot(next_actions, self.action_dim), axis=-1)
                target = rewards + self.gamma * \
                    (1.0 - dones) * next_q_target_action
                target = tf.stop_gradient(target)
                loss = tf.reduce_mean(tf.keras.losses.huber(target, q_action))
            grads = tape.gradient(loss, self.q_net.trainable_variables)
            grads = [tf.clip_by_norm(
                g, 10.0) if g is not None else None for g in grads]
            self.optimizer.apply_gradients(
                zip(grads, self.q_net.trainable_variables))
            return loss
        per_replica_loss = self.strategy.run(step_fn, args=(
            obs, actions, rewards, next_obs, dones, next_mask))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)

    def train(self, batch):
        obs = tf.convert_to_tensor(batch["obs"], tf.float32)
        actions = tf.convert_to_tensor(batch["actions"], tf.int32)
        rewards = tf.convert_to_tensor(batch["rewards"], tf.float32)
        next_obs = tf.convert_to_tensor(batch["next_obs"], tf.float32)
        dones = tf.convert_to_tensor(batch["dones"], tf.float32)
        next_mask = tf.convert_to_tensor(batch["next_mask"], tf.float32)

        return self._train_step(obs, actions, rewards, next_obs, dones, next_mask)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.q_net.save_weights(f"{path}/q_net.weights.h5")
        self.target_net.save_weights(f"{path}/target_net.weights.h5")

        opt_weights = self.optimizer.variables

        ckpt = tf.train.Checkpoint(
            optimizer=self.optimizer,
            q_net=self.q_net,
            target_net=self.target_net
        )

        ckpt.write(f"{path}/checkpoint")

    def load(self, path):
        dummy = tf.zeros((1, self.q_net.input_shape[-1]), dtype=tf.float32)
        self.q_net(dummy)
        self.target_net(dummy)

        self.q_net.load_weights(f"{path}/q_net.weights.h5")
        self.target_net.load_weights(f"{path}/target_net.weights.h5")

        ckpt = tf.train.Checkpoint(
            optimizer=self.optimizer,
            q_net=self.q_net,
            target_net=self.target_net
        )
        ckpt.restore(f"{path}/checkpoint").expect_partial()
