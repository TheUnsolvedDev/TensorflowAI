import numpy as np
import tensorflow as tf

from agents.dqn.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from agents.dqn.trainer import DistributedDQNTrainer, restore_batch_order
from env.wrappers import training_env
from models.dqn import build_model


def main():
    restored = restore_batch_order(
        (tf.constant([0, 2, 4]), tf.constant([1, 3]))
    ).numpy()
    assert restored.tolist() == [0, 1, 2, 3, 4]

    replay = ReplayBuffer(3)
    for value in range(5):
        replay.add(value, 0, 0.0, value + 1, False)
    assert len(replay) == 3
    assert set(replay.sample(3)[0]) == {2, 3, 4}

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
        model = build_model(4)
        trainer = DistributedDQNTrainer(model, strategy)
    states = np.zeros((8, 84, 84, 4), dtype=np.uint8)
    loss, errors = trainer.train_batch(
        states,
        np.arange(8) % 4,
        np.ones(8),
        states,
        np.zeros(8),
    )
    assert np.isfinite(float(loss))
    assert errors.shape == (8,)

    env = training_env()
    state, _ = env.reset(seed=0)
    next_state, _, terminated, truncated, _ = env.step(env.action_space.sample())
    assert state.shape == next_state.shape == (84, 84, 4)
    assert state.dtype == next_state.dtype == np.uint8
    assert env.action_space.n == 4
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    env.close()
    print("training core check passed")


if __name__ == "__main__":
    main()
