import tensorflow as tf

# -------------------------------
# Grouped Convolution
# -------------------------------
class GroupConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, groups=1, use_bias=False):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.groups = groups
        self.use_bias = use_bias

    def build(self, input_shape):
        in_channels = input_shape[-1]

        assert in_channels % self.groups == 0, "Input channels must be divisible by groups"
        assert self.filters % self.groups == 0, "Filters must be divisible by groups"

        self.group_in = in_channels // self.groups
        self.group_out = self.filters // self.groups

        self.convs = [
            tf.keras.layers.Conv2D(
                self.group_out,
                self.kernel_size,
                strides=self.strides,
                padding='same',
                use_bias=self.use_bias
            )
            for _ in range(self.groups)
        ]

    def call(self, x):
        splits = tf.split(x, num_or_size_splits=self.groups, axis=-1)
        outputs = [conv(s) for conv, s in zip(self.convs, splits)]
        return tf.concat(outputs, axis=-1)


# -------------------------------
# Channel Shuffle
# -------------------------------
class ChannelShuffle(tf.keras.layers.Layer):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def call(self, x):
        if self.groups == 1:
            return x

        batch, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        assert_op = tf.debugging.assert_equal(c % self.groups, 0)

        with tf.control_dependencies([assert_op]):
            channels_per_group = c // self.groups
            x = tf.reshape(x, [batch, h, w, self.groups, channels_per_group])
            x = tf.transpose(x, [0, 1, 2, 4, 3])
            x = tf.reshape(x, [batch, h, w, c])

        return x


# -------------------------------
# ShuffleNet Unit
# -------------------------------
def shufflenet_unit(x, out_channels, groups, stride):
    in_channels = x.shape[-1]
    bottleneck_channels = out_channels // 4

    # For stride=2, output channels split after concat
    if stride == 2:
        out_channels = out_channels - in_channels

    # 1x1 Group Conv (bottleneck)
    x_res = GroupConv2D(bottleneck_channels, (1, 1), groups=groups)(x)
    x_res = tf.keras.layers.BatchNormalization()(x_res)
    x_res = tf.keras.layers.ReLU()(x_res)

    # Channel Shuffle
    x_res = ChannelShuffle(groups)(x_res)

    # Depthwise Conv
    x_res = tf.keras.layers.DepthwiseConv2D(
        (3, 3),
        strides=stride,
        padding='same',
        use_bias=False
    )(x_res)
    x_res = tf.keras.layers.BatchNormalization()(x_res)

    # 1x1 Group Conv (expand)
    x_res = GroupConv2D(out_channels, (1, 1), groups=groups)(x_res)
    x_res = tf.keras.layers.BatchNormalization()(x_res)

    # Shortcut connection
    if stride == 1:
        out = tf.keras.layers.Add()([x, x_res])
    else:
        x_proj = tf.keras.layers.AveragePooling2D(
            pool_size=3,
            strides=2,
            padding='same'
        )(x)
        out = tf.keras.layers.Concatenate()([x_res, x_proj])

    return tf.keras.layers.ReLU()(out)


# -------------------------------
# ShuffleNet v1 Model
# -------------------------------
def ShuffleNetV1_model(input_shape=(224, 224, 3), num_classes=1000, groups=3):
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)

    # Initial Conv
    x = tf.keras.layers.Conv2D(24, (3, 3), strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Stage 2
    x = shufflenet_unit(x, 144, groups, stride=2)
    for _ in range(3):
        x = shufflenet_unit(x, 144, groups, stride=1)

    # Stage 3
    x = shufflenet_unit(x, 288, groups, stride=2)
    for _ in range(7):
        x = shufflenet_unit(x, 288, groups, stride=1)

    # Stage 4
    x = shufflenet_unit(x, 576, groups, stride=2)
    for _ in range(3):
        x = shufflenet_unit(x, 576, groups, stride=1)

    # Global Pool + FC
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


# -------------------------------
# Test
# -------------------------------
if __name__ == "__main__":
    model = ShuffleNetV1(input_shape=(224, 224, 3), num_classes=1000, groups=3)
    model.summary()