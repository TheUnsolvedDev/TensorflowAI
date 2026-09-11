import sys

import numpy as np
import tensorflow as tf

from config import *


class NN_GAN(tf.keras.Model):
    def __init__(self, strategy, input_shape, latent_dim, batch_size):
        super().__init__()
        self.strategy = strategy
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.global_batch_size = batch_size * self.strategy.num_replicas_in_sync

        with self.strategy.scope():
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator()

            self.generator_optimizer = tf.keras.optimizers.Adam(
                GENERATOR_LEARNING_RATE, beta_1=0.0, beta_2=0.9
            )
            self.discriminator_optimizer = tf.keras.optimizers.Adam(
                DISCRIMINATOR_LEARNING_RATE
            )

        self.generator.build(input_shape=(None, latent_dim[0]))
        self.discriminator.build(input_shape=(None, *input_shape))
        self.generator.summary()
        self.discriminator.summary()

    def build_generator(self):
        height, width, channels = self.input_shape
        flat_dim = height * width * channels

        inputs = tf.keras.layers.Input((self.latent_dim[0],))
        x = tf.keras.layers.Dense(256, use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Dense(512, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Dense(1024, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Dense(flat_dim, activation="tanh")(x)
        outputs = tf.keras.layers.Reshape((height, width, channels))(x)
        return tf.keras.Model(inputs, outputs, name="generator")

    def build_discriminator(self):
        inputs = tf.keras.layers.Input(self.input_shape)
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inputs, outputs, name="discriminator")

    @tf.function
    def train_generator_step(self, noise):
        noise = tf.random.normal([self.global_batch_size, self.latent_dim[0]])
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )
        return gen_loss

    @tf.function
    def train_discriminator_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim[0]])
        with tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_images += tf.random.normal(tf.shape(real_images), stddev=0.05)
            generated_images += tf.random.normal(tf.shape(generated_images), stddev=0.05)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            disc_loss = (
                self.cross_entropy(tf.ones_like(real_output), real_output)
                + self.cross_entropy(tf.zeros_like(fake_output), fake_output)
            )

        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        return disc_loss

    @tf.function
    def dist_generator_step(self, noise):
        per_replica_gen_loss = self.strategy.run(self.train_generator_step, args=(noise,))
        gen_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_gen_loss, axis=None
        )
        return gen_loss

    @tf.function
    def dist_discriminator_step(self, dataset_inputs):
        per_replica_disc_loss = self.strategy.run(
            self.train_discriminator_step, args=(dataset_inputs,)
        )
        disc_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_disc_loss, axis=None
        )
        return disc_loss

    def fit(self, dataset, epochs, initial_epoch=0, path="folder", callbacks=None):
        if callbacks is None:
            callbacks = []
        for callback in callbacks:
            callback.set_model(self)
            callback.on_train_begin()

        for epoch in range(initial_epoch, epochs):
            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            for step, image_batch in enumerate(dataset):
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim[0]))
                for _ in range(N_DISC_STEP):
                    disc_loss = self.dist_discriminator_step(image_batch)

                for _ in range(N_GEN_STEP):
                    gen_loss = self.dist_generator_step(noise)
                print(
                    f"\rEpoch [{step}/{epoch + 1}], Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}",
                    end="",
                )
                sys.stdout.flush()
                logs = {
                    "gen_loss": gen_loss,
                    "disc_loss": disc_loss,
                }
                for callback in callbacks:
                    callback.on_train_batch_end(step, logs)
            print()
            logs = {"gen_loss": gen_loss.numpy(), "disc_loss": disc_loss.numpy()}
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

        for callback in callbacks:
            callback.on_train_end()


if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce()
    )
    NN_GAN(
        strategy=strategy,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        latent_dim=(LATENT_DIM,),
        batch_size=BATCH_SIZE,
    )
