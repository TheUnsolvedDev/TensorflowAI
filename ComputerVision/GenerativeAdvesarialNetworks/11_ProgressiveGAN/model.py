import sys

import tensorflow as tf

from config import *


class PixelNormalization(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs / tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + 1e-8)


class MinibatchStdDev(tf.keras.layers.Layer):
    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=0, keepdims=True)
        stddev = tf.sqrt(variance + 1e-8)
        stat = tf.reduce_mean(stddev, keepdims=True)
        shape = tf.concat([tf.shape(inputs)[:3], [1]], axis=0)
        stat_map = tf.broadcast_to(stat, shape)
        return tf.concat([inputs, stat_map], axis=-1)


class ProgressiveGAN(tf.keras.Model):
    def __init__(self, strategy, input_shape, latent_dim, batch_size, alpha=1.0):
        super().__init__()
        self.strategy = strategy
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.alpha = alpha
        self.channels = PROGAN_CHANNELS_64 if self.input_shape[0] >= 64 else PROGAN_CHANNELS_32

        with self.strategy.scope():
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
            self.generator, to_file="generator.png", show_shapes=True, expand_nested=True
        )
        tf.keras.utils.plot_model(
            self.discriminator, to_file="discriminator.png", show_shapes=True, expand_nested=True
        )

    def conv_block(self, x, filters):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = PixelNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = PixelNormalization()(x)
        return x

    def to_rgb(self, x, channels):
        return tf.keras.layers.Conv2D(channels, 1, padding="same", activation="tanh")(x)

    def from_rgb(self, x, filters):
        x = tf.keras.layers.Conv2D(filters, 1, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        return x

    def build_generator(self):
        inputs = tf.keras.layers.Input(shape=(self.latent_dim[0],))
        x = PixelNormalization()(inputs)
        x = tf.keras.layers.Dense(4 * 4 * self.channels[0])(x)
        x = tf.keras.layers.Reshape((4, 4, self.channels[0]))(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = PixelNormalization()(x)
        x = self.conv_block(x, self.channels[0])

        if len(self.channels) == 1:
            outputs = self.to_rgb(x, self.input_shape[-1])
            return tf.keras.Model(inputs, outputs, name="generator")

        prev_x = x
        for idx in range(1, len(self.channels)):
            prev_rgb = self.to_rgb(prev_x, self.input_shape[-1])
            x = tf.keras.layers.UpSampling2D()(prev_x)
            x = self.conv_block(x, self.channels[idx])
            prev_x = x

        current_rgb = self.to_rgb(x, self.input_shape[-1])
        upsampled_prev_rgb = tf.keras.layers.UpSampling2D()(prev_rgb)
        outputs = self.alpha * current_rgb + (1.0 - self.alpha) * upsampled_prev_rgb
        return tf.keras.Model(inputs, outputs, name="generator")

    def disc_block(self, x, filters):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        return x

    def build_discriminator(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        current_filters = self.channels[-1]
        x = self.from_rgb(inputs, current_filters)

        if len(self.channels) > 1:
            downsampled_inputs = tf.keras.layers.AveragePooling2D(pool_size=2)(inputs)
            skip = self.from_rgb(downsampled_inputs, self.channels[-2])

            for idx in range(len(self.channels) - 1, 0, -1):
                x = self.disc_block(x, self.channels[idx - 1])
                if idx == len(self.channels) - 1:
                    x = self.alpha * x + (1.0 - self.alpha) * skip

        x = MinibatchStdDev()(x)
        x = tf.keras.layers.Conv2D(self.channels[0], 3, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.channels[0])(x)
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
        grads = tape.gradient(loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
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

        for callback in callbacks:
            callback.set_model(self)
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
