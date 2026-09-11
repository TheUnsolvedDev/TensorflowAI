import tensorflow as tf

def solo(x_inp, filters, depth):
    x_inp = tf.keras.layers.Conv2D(
        filters, 3, padding='same', activation='relu')(x_inp)
    x_inp = tf.keras.layers.BatchNormalization()(x_inp)

    x_inp = tf.keras.layers.MaxPool2D(4**(depth), 4**(depth))(x_inp)
    return x_inp


def fractal_block(input_tensor, filters, depth=2):
    if depth == 1:
        x = tf.keras.layers.Conv2D(
            filters, 3, padding='same', activation='relu')(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)

        x = tf.keras.layers.Conv2D(
            filters, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(2, 2)(x)

        y = solo(input_tensor, filters, depth)
        return tf.keras.layers.Add()([x, y])

    new1 = fractal_block(input_tensor, filters, depth - 1)
    new2 = fractal_block(new1, filters, depth - 1)
    new3 = solo(input_tensor, filters, depth)
    x = tf.keras.layers.Add()(
        [new2, new3])

    return x

# Define the fractal net model


def fractal_net(input_shape=(64, 64, 3), num_classes=10):
    data_augmentation = tf.keras.models.Sequential(
        [
            tf.keras.layers.Normalization(),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(factor=0.02),
            tf.keras.layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )
    inputs = tf.keras.layers.Input(shape=input_shape)
    x_inp = tf.keras.layers.Lambda(lambda i: i/255.0)(inputs)
    x_inp = data_augmentation(x_inp)
    x = x_inp

    # Create the first recursive block
    x = fractal_block(x, 64)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(num_classes)(x)
    outputs = tf.keras.layers.Activation('softmax', name='predictions')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model