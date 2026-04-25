import tensorflow as tf

# -------------------------
# Channel Shuffle
# -------------------------
class ChannelShuffle(tf.keras.layers.Layer):
    def __init__(self, groups=2):
        super().__init__()
        self.groups = groups

    def call(self, x):
        b, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [b, h, w, self.groups, c // self.groups])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        return tf.reshape(x, [b, h, w, c])


# -------------------------
# Basic Conv Block
# -------------------------
def conv_bn_relu(x, filters, k, s):
    x = tf.keras.layers.Conv2D(filters, k, strides=s, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)


# -------------------------
# Shuffle Unit (V2)
# -------------------------
class ShuffleUnit(tf.keras.layers.Layer):
    def __init__(self, out_channels, stride):
        super().__init__()
        self.out_channels = out_channels
        self.stride = stride
        self.mid_channels = out_channels // 2

    def build(self, input_shape):
        in_channels = input_shape[-1]

        if self.stride == 1:
            # branch2 only
            self.branch2 = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.mid_channels, 1, 1, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(self.mid_channels, 1, 1, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ])
        else:
            # stride = 2 → both branches active
            self.branch1 = tf.keras.Sequential([
                tf.keras.layers.DepthwiseConv2D(3, strides=2, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(self.mid_channels, 1, 1, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ])

            self.branch2 = tf.keras.Sequential([
                tf.keras.layers.Conv2D(self.mid_channels, 1, 1, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.DepthwiseConv2D(3, strides=2, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Conv2D(self.mid_channels, 1, 1, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ])

        self.shuffle = ChannelShuffle(2)

    def call(self, x):
        if self.stride == 1:
            x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
            out = tf.concat([x1, self.branch2(x2)], axis=-1)
        else:
            out = tf.concat([self.branch1(x), self.branch2(x)], axis=-1)

        return self.shuffle(out)


# -------------------------
# ShuffleNet V2 Model
# -------------------------
def ShuffleNetV2_model(input_shape=(224, 224, 3), num_classes=1000, scale=1.0):

    stage_repeats = [4, 8, 4]

    stage_out_channels = {
        0.5: [-1, 24, 48, 96, 192, 1024],
        1.0: [-1, 24, 116, 232, 464, 1024],
        1.5: [-1, 24, 176, 352, 704, 1024],
        2.0: [-1, 24, 244, 488, 976, 2048],
    }

    out_channels = stage_out_channels[scale]

    inputs = tf.keras.layers.Input(shape=input_shape)

    # Normalize
    x = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)

    # stem
    x = conv_bn_relu(x, out_channels[1], 3, 2)
    x = tf.keras.layers.MaxPool2D(3, strides=2, padding='same')(x)

    # stages
    input_channels = out_channels[1]
    for idx, repeats in enumerate(stage_repeats):
        output_channels = out_channels[idx + 2]

        # first block (downsample)
        x = ShuffleUnit(output_channels, stride=2)(x)

        # remaining blocks
        for _ in range(repeats - 1):
            x = ShuffleUnit(output_channels, stride=1)(x)

        input_channels = output_channels

    # final conv
    x = conv_bn_relu(x, out_channels[-1], 1, 1)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, x)


# -------------------------
# Test
# -------------------------
if __name__ == "__main__":
    model = ShuffleNetV2_model()
    model.summary()