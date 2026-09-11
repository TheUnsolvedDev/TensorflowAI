"""Dynamic-routing CapsNet with supervised reconstruction (Sabour et al., 2017)."""
import tensorflow as tf
from config import INPUT_SIZE


def squash(x):
    x = tf.cast(x, tf.float32)
    norm = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
    return (norm / (1. + norm)) * x * tf.math.rsqrt(norm + 1e-7)


def margin_loss(y_true, y_pred):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    return tf.reduce_sum(y_true * tf.square(tf.nn.relu(.9 - y_pred)) +
                        .5 * (1. - y_true) * tf.square(tf.nn.relu(y_pred - .1)), axis=-1)


class RoutingCapsules(tf.keras.layers.Layer):
    def __init__(self, classes, iterations=3, **kwargs):
        super().__init__(dtype='float32', **kwargs)
        self.classes, self.iterations = classes, iterations
    def build(self, shape):
        self.transforms = self.add_weight(name='transforms', shape=(shape[1], self.classes, shape[2], 16),
                                           initializer=tf.keras.initializers.RandomNormal(stddev=.1))
    def call(self, x):
        votes = tf.einsum('bid,ijdo->bijo', tf.cast(x, tf.float32), self.transforms)
        logits = tf.zeros(tf.shape(votes)[:3], dtype=tf.float32)
        for iteration in range(self.iterations):
            routing_votes = votes if iteration == self.iterations - 1 else tf.stop_gradient(votes)
            coupling = tf.nn.softmax(logits, axis=2)
            capsules = squash(tf.reduce_sum(coupling[..., None] * routing_votes, axis=1))
            if iteration < self.iterations - 1:
                logits += tf.reduce_sum(routing_votes * capsules[:, None], axis=-1)
        return capsules


class CapsuleNetwork(tf.keras.Model):
    def __init__(self, image_shape, num_classes, routing_iterations):
        super().__init__(name='capsnet_model', dtype='float32')
        self.image_shape, self.num_classes = tuple(image_shape), num_classes
        self.conv = tf.keras.layers.Conv2D(256, 9, activation='relu', dtype='float32')
        self.primary = tf.keras.layers.Conv2D(32 * 8, 9, strides=2, dtype='float32')
        self.routing = RoutingCapsules(num_classes, routing_iterations)
        pixels = image_shape[0] * image_shape[1] * image_shape[2]
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', dtype='float32'),
            tf.keras.layers.Dense(1024, activation='relu', dtype='float32'),
            tf.keras.layers.Dense(pixels, activation='sigmoid', dtype='float32')])
    def call(self, inputs, training=None):
        if isinstance(inputs, (tuple, list)):
            images, labels = inputs
        else:
            images, labels = inputs, None
        images = tf.cast(images, tf.float32) / 255.
        x = self.primary(self.conv(images))
        capsule_count = int(x.shape[1]) * int(x.shape[2]) * 32
        x = squash(tf.reshape(x, [-1, capsule_count, 8]))
        capsules = self.routing(x)
        lengths = tf.sqrt(tf.reduce_sum(tf.square(capsules), -1) + 1e-7)
        if labels is None:
            labels = tf.one_hot(tf.argmax(lengths, axis=-1), self.num_classes)
        masked = capsules * tf.cast(labels[..., None], tf.float32)
        reconstructed = self.decoder(tf.reshape(masked, [-1, self.num_classes * 16]))
        target = tf.reshape(images, [tf.shape(images)[0], -1])
        self.add_loss(.0005 * tf.reduce_mean(tf.reduce_sum(tf.square(target - reconstructed), axis=-1)))
        return lengths


def capsnet_model(input_shape=tuple(INPUT_SIZE), num_classes=10, routing_iterations=3):
    if routing_iterations < 1:
        raise ValueError('routing_iterations must be positive')
    model = CapsuleNetwork(input_shape, num_classes, routing_iterations)
    model(tf.zeros((1, *input_shape)))
    return model
