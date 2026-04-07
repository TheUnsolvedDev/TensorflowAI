import silence_tensorflow.auto
import tensorflow as tf



def fire_module(inputs, s1x1, e1x1, e3x3, name="fire"):
    squeeze = tf.keras.layers.Conv2D(s1x1, (1,1), strides=1, padding="valid", activation="relu", name=f"{name}_squeeze")(inputs)
    
    expand_1x1 = tf.keras.layers.Conv2D(e1x1, (1,1), strides=1, padding="valid", activation="relu", name=f"{name}_expand1x1")(squeeze)
    expand_3x3 = tf.keras.layers.Conv2D(e3x3, (3,3), strides=1, padding="same", activation="relu", name=f"{name}_expand3x3")(squeeze)
    
    return tf.keras.layers.Concatenate(name=f"{name}_concat")([expand_1x1, expand_3x3])

def squeezenet_complexbypass_model(input_shape=(224,224,3), num_classes=1000):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)

    x = tf.keras.layers.Conv2D(96, (7,7), strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((3,3), strides=2)(x)

    f2 = fire_module(x, 16, 64, 64, name="fire2")
    f3 = fire_module(f2, 16, 64, 64, name="fire3")
    x = tf.keras.layers.Add()([f2, f3])

    f4 = fire_module(x, 32, 128, 128, name="fire4")
    x = tf.keras.layers.MaxPooling2D((3,3), strides=2)(f4)

    f5 = fire_module(x, 32, 128, 128, name="fire5")
    x = tf.keras.layers.Add()([x, f5])

    f6 = fire_module(x, 48, 192, 192, name="fire6")
    f7 = fire_module(f6, 48, 192, 192, name="fire7")
    f8 = fire_module(f7, 64, 256, 256, name="fire8")

    x = tf.keras.layers.MaxPooling2D((3,3), strides=2)(f8)

    f9 = fire_module(x, 64, 256, 256, name="fire9")
    x = tf.keras.layers.Add()([x, f9])

    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(num_classes, (1,1), activation="relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Softmax()(x)

    return tf.keras.Model(inputs, outputs)


if __name__ == '__main__':
    model1 = squeezenet_complexbypass_model()
    model1.summary(expand_nested=True)
