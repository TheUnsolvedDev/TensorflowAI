import silence_tensorflow.auto
import tensorflow as tf
from config import *

# =========================
# 1. Binary Activation (STE)
# =========================
def binary_activation(x):
    return x + tf.stop_gradient(tf.sign(x) - x)


# =========================
# 2. Binary Conv Layer
# =========================
class BinaryConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding="same"):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(self.kernel_size, self.kernel_size, input_shape[-1], self.filters),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel"
        )

    def call(self, x):
        # Binary weights with STE
        w = self.kernel
        w_bin = tf.sign(w)
        w_bin = w + tf.stop_gradient(w_bin - w)

        # Scaling factor alpha
        alpha = tf.reduce_mean(tf.abs(w), axis=[0, 1, 2], keepdims=True)
        w_bin = w_bin * alpha

        # Binary activation
        x_bin = binary_activation(x)

        return tf.nn.conv2d(
            x_bin,
            w_bin,
            strides=[1, self.strides, self.strides, 1],
            padding=self.padding
        )


def XNOR_net_model(input_shape=[INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]], num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Normalize
    x = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = BinaryConv2D(64, 3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = BinaryConv2D(128, 3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = BinaryConv2D(256, 3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(num_classes)(x)
    outputs = tf.keras.layers.Activation("softmax")(x)

    return tf.keras.Model(inputs, outputs)


# =========================
# 4. Run Check
# =========================
if __name__ == "__main__":
    model = XNOR_net_model()
    model.summary(expand_nested=True)