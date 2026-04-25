import tensorflow as tf

def nin_model(input_shape=[32, 32, 3], num_classes=10, ksize=3):
    initializer = tf.initializers.glorot_normal()
    inputs = tf.keras.Input(shape=tuple(input_shape))

    def elu(x): return tf.nn.elu(x)

    def ninconv(x, k, channels, name):
        x = tf.keras.layers.Conv2D(filters=channels[1], kernel_size=k, padding='same',
                                   kernel_initializer=initializer, name=f"nin{name}_1")(x)
        x = tf.keras.layers.BatchNormalization(name=f"nin{name}_1_bn")(x)
        x = tf.keras.layers.Activation(elu)(x)
        x = tf.keras.layers.Conv2D(filters=channels[2], kernel_size=1, padding='same',
                                   kernel_initializer=initializer, name=f"nin{name}_2")(x)
        x = tf.keras.layers.BatchNormalization(name=f"nin{name}_2_bn")(x)
        x = tf.keras.layers.Activation(elu)(x)
        x = tf.keras.layers.Conv2D(filters=channels[3], kernel_size=1, padding='same',
                                   kernel_initializer=initializer, name=f"nin{name}_3")(x)
        x = tf.keras.layers.BatchNormalization(name=f"nin{name}_3_bn")(x)
        x = tf.keras.layers.Activation(elu)(x)
        return x

    def residual(x, ksize, inchannel, outchannel, name):
        channels = [inchannel, outchannel * 2, outchannel * 2, outchannel]
        y = ninconv(x, ksize, channels, name=f"{name}_1")
        channels = [outchannel, outchannel * 2, outchannel * 2, outchannel]
        y = ninconv(y, ksize, channels, name=f"{name}_2")
        if x.shape[-1] != y.shape[-1]:
            sc = tf.keras.layers.Conv2D(filters=outchannel, kernel_size=1, padding='same',
                                        kernel_initializer=initializer, name=f"{name}_sc")(x)
            sc = tf.keras.layers.BatchNormalization(name=f"{name}_sc_bn")(sc)
            sc = tf.keras.layers.Activation(elu)(sc)
            x = sc
        out = tf.keras.layers.Add()([x, y])
        return out

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same',
                               kernel_initializer=initializer, name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="conv1_bn")(x)
    x = tf.keras.layers.Activation(elu)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = residual(x, ksize, inchannel=16, outchannel=32, name="conv2_1")
    x = residual(x, ksize, inchannel=32, outchannel=32, name="conv2_2")
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)

    x = residual(x, ksize, inchannel=32, outchannel=64, name="conv3_1")
    x = residual(x, ksize, inchannel=64, outchannel=num_classes, name="conv3_2")

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model
