from unittest.mock import Mock

import numpy as np
import tensorflow as tf

from agents.dqn.replay_buffer import PrioritizedReplayBuffer
from agents.dqn.trainer import DistributedDQNTrainer, restore_batch_order
from env.minatar_env import MinAtarEnv
from models.dqn import build_model


def main():
    restored = restore_batch_order(
        (tf.constant([0, 2, 4]), tf.constant([1, 3]))
    ).numpy()
    assert restored.tolist() == [0, 1, 2, 3, 4]

    buffer = PrioritizedReplayBuffer(7, seed=0)
    for value in range(10):
        buffer.add(value, 0, 0.0, value + 1, False)

    batch = buffer.sample(7)
    assert len(batch[0]) == 7
    assert np.all(buffer.last_indices < len(buffer))

    buffer.update_priorities(np.arange(1, 8, dtype=np.float64))
    expected_total = buffer.scaled_priorities[: len(buffer)].sum()
    tree_total = 0.0
    index = len(buffer)
    while index:
        tree_total += buffer.tree[index]
        index -= index & -index
    assert np.isclose(buffer.total_priority, expected_total)
    assert np.isclose(tree_total, expected_total)

    strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
    with strategy.scope():
        model = build_model(6)
        trainer = DistributedDQNTrainer(model, strategy)
    states = np.zeros((8, 10, 10, 4), dtype=np.bool_)
    loss, errors = trainer.train_batch(
        states,
        np.arange(8) % 6,
        np.ones(8),
        states,
        np.zeros(8),
    )
    assert np.isfinite(float(loss))
    assert errors.shape == (8,)

    env = MinAtarEnv(render_mode="human")
    env.env.display_state = Mock()
    env.env.close_display = Mock()
    env.reset(seed=0)
    env.step(0)
    env.close()
    assert env.env.display_state.call_count == 2
    env.env.display_state.assert_called_with(50)
    env.env.close_display.assert_called_once()
    print("training core check passed")


if __name__ == "__main__":
    main()
