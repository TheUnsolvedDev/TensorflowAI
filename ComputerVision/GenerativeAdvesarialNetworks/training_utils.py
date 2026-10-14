from contextlib import contextmanager

import tensorflow as tf


class GeneratorEMA(tf.Module):
    """Track and temporarily apply moving-average generator weights."""

    def __init__(self, models, decay=0.999):
        super().__init__(name="generator_ema")
        self.models = tuple(models)
        self.decay = tf.constant(decay, tf.float32)
        self.shadow = [
            tf.Variable(variable, trainable=False, name=f"shadow_{index}")
            for index, variable in enumerate(self.variables)
        ]

    @property
    def variables(self):
        return [variable for model in self.models for variable in model.weights]

    def update(self):
        for shadow, variable in zip(self.shadow, self.variables):
            value = tf.cast(variable, shadow.dtype)
            shadow.assign(self.decay * shadow + (1.0 - self.decay) * value)

    @contextmanager
    def average_weights(self):
        original = [tf.identity(variable) for variable in self.variables]
        try:
            for variable, shadow in zip(self.variables, self.shadow):
                variable.assign(tf.cast(shadow, variable.dtype))
            yield
        finally:
            for variable, value in zip(self.variables, original):
                variable.assign(value)


def build_full_checkpoint(model, log_dir):
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        f"{log_dir}/checkpoints",
        max_to_keep=2,
    )
    return checkpoint, manager


class FullCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, manager, frequency):
        super().__init__()
        self.manager = manager
        self.frequency = frequency

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1
        total_epochs = self.params.get("epochs", current_epoch)
        if current_epoch % self.frequency == 0 or current_epoch == total_epochs:
            self.manager.save(checkpoint_number=current_epoch)
