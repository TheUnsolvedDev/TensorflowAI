import sys

import tensorflow as tf

from config import *


class NoiseInjection(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.weight = self.add_weight(
            name="noise_weight",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs, training=None):
        batch = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        noise = tf.random.normal((batch, height, width, 1))
        weight = tf.reshape(self.weight, (1, 1, 1, -1))
        return inputs + noise * weight


class MinibatchStdDev(tf.keras.layers.Layer):
    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=0, keepdims=True)
        stddev = tf.sqrt(variance + 1e-8)
        stat = tf.reduce_mean(stddev, keepdims=True)
        shape = tf.concat([tf.shape(inputs)[:3], [1]], axis=0)
        stat_map = tf.broadcast_to(stat, shape)
        return tf.concat([inputs, stat_map], axis=-1)


class LearnedConstant(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def build(self, input_shape):
        self.constant = self.add_weight(
            name="learned_constant",
            shape=(1, 4, 4, self.channels),
            initializer="ones",
            trainable=True,
        )

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        return tf.tile(self.constant, [batch, 1, 1, 1])


class StyleGAN(tf.keras.Model):
    def __init__(self, strategy, input_shape, latent_dim, batch_size):
        super().__init__()
        self.strategy = strategy
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.generator_channels = STYLEGAN_CHANNELS_64 if self.input_shape[0] >= 64 else STYLEGAN_CHANNELS_32

        with self.strategy.scope():
            self.mapping_network = self.build_mapping_network()
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator()

            self.generator_optimizer = tf.keras.optimizers.Adam(
                GENERATOR_LEARNING_RATE, beta_1=0.0, beta_2=0.99
            )
            self.discriminator_optimizer = tf.keras.optimizers.Adam(
                DISCRIMINATOR_LEARNING_RATE, beta_1=0.0, beta_2=0.99
            )
            self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        tf.keras.utils.plot_model(
            self.mapping_network, to_file="mapping_network.png", show_shapes=True, expand_nested=True
        )
        tf.keras.utils.plot_model(
            self.generator, to_file="generator.png", show_shapes=True, expand_nested=True
        )
        tf.keras.utils.plot_model(
            self.discriminator, to_file="discriminator.png", show_shapes=True, expand_nested=True
        )

    def build_mapping_network(self):
        inputs = tf.keras.layers.Input(shape=(self.latent_dim[0],))
        x = inputs
        for _ in range(STYLEGAN_MAPPING_LAYERS):
            x = tf.keras.layers.Dense(MAPPING_DIM)(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)
        return tf.keras.Model(inputs, x, name="mapping_network")

    def adain(self, x, style, channels):
        scale = tf.keras.layers.Dense(channels)(style)
        bias = tf.keras.layers.Dense(channels)(style)
        scale = tf.keras.layers.Reshape((1, 1, channels))(scale)
        bias = tf.keras.layers.Reshape((1, 1, channels))(bias)
        x = tf.keras.layers.LayerNormalization(axis=[1, 2])(x)
        return x * (scale + 1.0) + bias

    def style_block(self, x, style, filters, upsample=False):
        if upsample:
            x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = NoiseInjection()(x)
        x = self.adain(x, style, filters)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = NoiseInjection()(x)
        x = self.adain(x, style, filters)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        return x

    def build_generator(self):
        inputs = tf.keras.layers.Input(shape=(self.latent_dim[0],))
        style = self.mapping_network(inputs)

        channels = self.generator_channels
        x = LearnedConstant(channels[0])(inputs)
        x = NoiseInjection()(x)
        x = self.adain(x, style, channels[0])
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        for index, filters in enumerate(channels):
            x = self.style_block(x, style, filters, upsample=index > 0)

        outputs = tf.keras.layers.Conv2D(self.input_shape[-1], 1, padding="same", activation="tanh")(x)
        return tf.keras.Model(inputs, outputs, name="generator")

    def disc_block(self, x, filters):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Conv2D(filters, 3, strides=2, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        return x

    def build_discriminator(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = inputs

        for filters in reversed(self.generator_channels):
            x = self.disc_block(x, filters)

        x = MinibatchStdDev()(x)
        x = tf.keras.layers.Conv2D(max(self.generator_channels[0], 128), 3, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(max(self.generator_channels[0], 128))(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inputs, outputs, name="discriminator")

    @tf.function
    def train_generator_step(self, noise):
        valid = tf.ones((tf.shape(noise)[0], 1))
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            loss = self.loss_fn(valid, fake_output)
        variables = self.generator.trainable_variables
        grads = tape.gradient(loss, variables)
        self.generator_optimizer.apply_gradients(zip(grads, variables))
        return loss

    @tf.function
    def train_discriminator_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim[0]])
        valid = tf.ones((batch_size, 1))
        fake = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(tf.stop_gradient(fake_images), training=True)
            loss = self.loss_fn(valid, real_output) + self.loss_fn(fake, fake_output)
        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return loss

    @tf.function
    def dist_generator_step(self, noise):
        per_replica_loss = self.strategy.run(self.train_generator_step, args=(noise,))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)

    @tf.function
    def dist_discriminator_step(self, real_images):
        per_replica_loss = self.strategy.run(self.train_discriminator_step, args=(real_images,))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)

    def fit(self, dataset, epochs, initial_epoch=0, callbacks=None):
        if callbacks is None:
            callbacks = []

        callback_params = {
            "epochs": epochs,
            "initial_epoch": initial_epoch,
            "steps": None,
            "verbose": 1,
        }
        for callback in callbacks:
            callback.set_model(self)
            callback.set_params(callback_params)
            callback.on_train_begin()

        for epoch in range(initial_epoch, epochs):
            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            for step, real_images in enumerate(dataset):
                disc_loss = self.dist_discriminator_step(real_images)
                noise = tf.random.normal([self.batch_size, self.latent_dim[0]])
                gen_loss = self.dist_generator_step(noise)

                print(
                    f"\rEpoch [{step}/{epoch + 1}], Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}",
                    end="",
                )
                sys.stdout.flush()

                logs = {"gen_loss": gen_loss, "disc_loss": disc_loss}
                for callback in callbacks:
                    callback.on_train_batch_end(step, logs)

            print()
            logs = {"gen_loss": gen_loss.numpy(), "disc_loss": disc_loss.numpy()}
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

        for callback in callbacks:
            callback.on_train_end()
