import tensorflow as tf


def configure_gpus():
    """Allow TensorFlow to grow GPU memory usage as training needs it."""
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def nccl_strategy():
    """Create mirrored training with NCCL gradient all-reduce."""
    return tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce()
    )


configure_gpus()
