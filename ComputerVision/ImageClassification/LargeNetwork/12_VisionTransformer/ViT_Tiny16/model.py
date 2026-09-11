"""ViT_Tiny16; train from scratch. Patch projection and class-token pooling."""
import tensorflow as tf
from config import INPUT_SIZE


@tf.keras.utils.register_keras_serializable(package='LargeNetwork')
class PositionAndClassToken(tf.keras.layers.Layer):
    def build(self, shape):
        self.token = self.add_weight(name='class_token', shape=(1, 1, shape[-1]), initializer='zeros')
        self.position = self.add_weight(name='position', shape=(1, shape[1] + 1, shape[-1]),
                                        initializer=tf.keras.initializers.TruncatedNormal(stddev=.02))
    def call(self, x):
        token = tf.broadcast_to(self.token, [tf.shape(x)[0], 1, tf.shape(x)[-1]])
        return tf.concat([token, x], axis=1) + self.position


def vit_tiny16_model(input_shape=tuple(INPUT_SIZE), num_classes=10):
    patch, dim, depth, heads = 16, 192, 12, 3
    if input_shape[0] % patch or input_shape[1] % patch:
        raise ValueError('Input height and width must be divisible by patch size')
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)(inputs)
    x = tf.keras.layers.Conv2D(dim, patch, strides=patch, padding='valid')(x)
    x = tf.keras.layers.Reshape((-1, dim))(x)
    x = PositionAndClassToken()(x)
    x = tf.keras.layers.Dropout(.1)(x)
    for block in range(depth):
        y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        y = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads,
                                               dropout=.1, name=f'attention_{block}')(y, y)
        x = tf.keras.layers.Add()([x, y])
        y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        y = tf.keras.layers.Dense(dim * 4, activation='gelu')(y)
        y = tf.keras.layers.Dropout(.1)(y)
        y = tf.keras.layers.Dense(dim)(y)
        y = tf.keras.layers.Dropout(.1)(y)
        x = tf.keras.layers.Add()([x, y])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = x[:, 0]
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    return tf.keras.Model(inputs, outputs, name='vit_tiny16_model')
