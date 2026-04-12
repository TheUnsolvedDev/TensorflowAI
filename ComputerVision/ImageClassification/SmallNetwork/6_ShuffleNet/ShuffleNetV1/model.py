import tensorflow as tf

# -------------------------------
# Channel Shuffle
# -------------------------------
class ChannelShuffle(tf.keras.layers.Layer):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def call(self, x):
        batch_size, height, width, channels = x.shape
        groups = self.groups
        channels_per_group = channels // groups

        x = tf.reshape(x, [-1, height, width, groups, channels_per_group])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        x = tf.reshape(x, [-1, height, width, channels])
        return x


# -------------------------------
# ShuffleNet Unit
# -------------------------------
def shufflenet_unit(x, out_channels, groups, stride, stage):
    in_channels = int(x.shape[-1])

    # Bottleneck channels
    bottleneck_channels = out_channels // 4

    # For stride=2, reduce residual branch channels
    if stride == 2:
        out_channels = out_channels - in_channels

    # First 1x1 conv
    first_groups = 1 if stage == 2 else groups
    x_res = tf.keras.layers.Conv2D(
        bottleneck_channels,
        kernel_size=1,
        strides=1,
        padding='same',
        groups=first_groups,
        use_bias=False
    )(x)
    x_res = tf.keras.layers.BatchNormalization()(x_res)
    x_res = tf.keras.layers.ReLU()(x_res)

    # Channel shuffle
    x_res = ChannelShuffle(groups)(x_res)

    # Depthwise conv
    x_res = tf.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding='same',
        use_bias=False
    )(x_res)
    x_res = tf.keras.layers.BatchNormalization()(x_res)

    # Second 1x1 conv
    x_res = tf.keras.layers.Conv2D(
        out_channels,
        kernel_size=1,
        strides=1,
        padding='same',
        groups=groups,
        use_bias=False
    )(x_res)
    x_res = tf.keras.layers.BatchNormalization()(x_res)

    # Shortcut
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
# Channel Configurations (Paper)
# -------------------------------
def get_stage_channels(groups):
    if groups == 1:
        return [144, 288, 576]
    elif groups == 2:
        return [200, 400, 800]
    elif groups == 3:
        return [240, 480, 960]
    elif groups == 4:
        return [272, 544, 1088]
    elif groups == 8:
        return [384, 768, 1536]
    else:
        raise ValueError("Invalid groups value")


# -------------------------------
# ShuffleNet v1 Model
# -------------------------------
def ShuffleNetV1_model(input_shape=(224, 224, 3), num_classes=1000, groups=3):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Normalize
    x = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)

    # Initial conv
    x = tf.keras.layers.Conv2D(
        24, kernel_size=3, strides=2, padding='same', use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=3, strides=2, padding='same'
    )(x)

    # Stage channels
    stage_channels = get_stage_channels(groups)

    # Stage 2
    x = shufflenet_unit(x, stage_channels[0], groups, stride=2, stage=2)
    for _ in range(3):
        x = shufflenet_unit(x, stage_channels[0], groups, stride=1, stage=2)

    # Stage 3
    x = shufflenet_unit(x, stage_channels[1], groups, stride=2, stage=3)
    for _ in range(7):
        x = shufflenet_unit(x, stage_channels[1], groups, stride=1, stage=3)

    # Stage 4
    x = shufflenet_unit(x, stage_channels[2], groups, stride=2, stage=4)
    for _ in range(3):
        x = shufflenet_unit(x, stage_channels[2], groups, stride=1, stage=4)

    # Head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


# -------------------------------
# Test
# -------------------------------
if __name__ == "__main__":
    model = ShuffleNetV1_model(input_shape=(224, 224, 3), num_classes=1000, groups=3)
    model.summary()