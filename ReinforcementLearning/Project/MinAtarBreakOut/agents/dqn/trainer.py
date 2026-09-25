import tensorflow as tf


def restore_batch_order(parts):
    """Interleave results from replica-strided input shards."""
    parts = tuple(parts)
    if len(parts) == 1:
        return parts[0]
    total = tf.add_n([tf.shape(part)[0] for part in parts])
    indices = [tf.range(replica, total, len(parts)) for replica in range(len(parts))]
    return tf.dynamic_stitch(indices, parts)


class DistributedDQNTrainer:
    """DQN batch updates and action inference across MirroredStrategy replicas."""

    def __init__(self, model, strategy, gamma=0.99, learning_rate=0.001, double=False):
        self.model = model
        self.strategy = strategy
        self.double = double
        with strategy.scope():
            self.target_model = tf.keras.models.clone_model(model)
            self.target_model.set_weights(model.get_weights())
            self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.gamma = gamma

    @tf.function
    def _train_replica(
        self, states, actions, rewards, next_states, dones, weights, global_batch_size
    ):
        actions = tf.cast(actions, tf.int32)
        rewards = tf.cast(rewards, tf.float32)
        dones = tf.cast(dones, tf.float32)
        weights = tf.cast(weights, tf.float32)
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            chosen_q = tf.gather(q_values, actions, axis=1, batch_dims=1)
            next_values = self.model(next_states, training=False)
            if self.double:
                next_actions = tf.argmax(next_values, axis=1, output_type=tf.int32)
                target_values = self.target_model(next_states, training=False)
                next_q = tf.gather(target_values, next_actions, axis=1, batch_dims=1)
            else:
                next_q = tf.reduce_max(self.target_model(next_states, training=False), axis=1)
            targets = rewards + self.gamma * next_q * (1.0 - dones)
            per_example_loss = weights * tf.keras.losses.huber(targets, chosen_q)
            loss = tf.reduce_sum(per_example_loss) / tf.cast(
                global_batch_size, tf.float32
            )

        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, tf.abs(targets - chosen_q)

    def train_batch(self, states, actions, rewards, next_states, dones, weights=None):
        """Run one replay batch; MirroredStrategy performs NCCL gradient reduction."""
        if weights is None:
            weights = tf.ones(tf.shape(rewards), dtype=tf.float32)
        values = tuple(
            tf.convert_to_tensor(value)
            for value in (states, actions, rewards, next_states, dones, weights)
        )
        distributed_values = tuple(
            self.strategy.experimental_distribute_values_from_function(
                lambda context, value=value: value[context.replica_id_in_sync_group :: self.strategy.num_replicas_in_sync]
            )
            for value in values
        )
        global_batch_size = tf.shape(values[0])[0]
        per_replica_loss, per_replica_errors = self.strategy.run(
            self._train_replica, args=(*distributed_values, global_batch_size)
        )
        loss = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None
        )
        errors = restore_batch_order(
            self.strategy.experimental_local_results(per_replica_errors)
        )
        return loss, errors

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    @tf.function
    def _act_replica(self, states):
        return tf.argmax(self.model(states, training=False), axis=-1, output_type=tf.int32)

    def act(self, states, epsilon=0.0):
        """Run batched action inference on strategy replicas and combine actions."""
        states = tf.convert_to_tensor(states)
        single = states.shape.rank == 3
        if single:
            states = states[None, ...]
        parts = self.strategy.experimental_distribute_values_from_function(
            lambda context: states[
                context.replica_id_in_sync_group :: self.strategy.num_replicas_in_sync
            ]
        )
        actions = self.strategy.run(self._act_replica, args=(parts,))
        actions = restore_batch_order(self.strategy.experimental_local_results(actions))
        if epsilon:
            random_actions = tf.random.uniform(
                tf.shape(actions), maxval=self.model.output_shape[-1], dtype=tf.int32
            )
            choose_random = tf.random.uniform(tf.shape(actions)) < epsilon
            actions = tf.where(choose_random, random_actions, actions)
        return actions[0] if single else actions
