import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import tqdm
import numpy as np
from config import *

# Create images folder
os.makedirs("images", exist_ok=True)


class Cond_GAN(tf.keras.Model):
    def __init__(self, strategy, input_shape, latent_dim, batch_size, num_classes):
        super().__init__()
        self.strategy = strategy
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.factor = 2 if self.input_shape[0] >= 64 else 1
        self.lambda_gp = LAMBDA_GP
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.global_batch_size = batch_size * self.strategy.num_replicas_in_sync

        with self.strategy.scope():
            self.generator = self.build_generator(self.num_classes)
            self.discriminator = self.build_discriminator(self.num_classes)

            self.generator_optimizer = tf.keras.optimizers.Adam(
                GENERATOR_LEARNING_RATE, beta_1=0.5, beta_2=0.999)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(
                DISCRIMINATOR_LEARNING_RATE, beta_1=0.5, beta_2=0.999)

        tf.keras.utils.plot_model(
            self.generator, to_file="generator.png", show_shapes=True, expand_nested=True, dpi=96)
        tf.keras.utils.plot_model(
            self.discriminator, to_file="discriminator.png", show_shapes=True, expand_nested=True, dpi=96)
        self.generator.summary()
        self.discriminator.summary()

    def build_generator(self, num_classes):
        z = tf.keras.layers.Input(shape=(self.latent_dim[0],))
        label = tf.keras.layers.Input(
            shape=(num_classes,))  # one-hot or multi-label

        # project noise
        x = tf.keras.layers.Dense(4 * 4 * 128 * self.factor, use_bias=False)(z)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Reshape((4, 4, 128 * self.factor))(x)

        # project label to spatial map
        y = tf.keras.layers.Dense(4 * 4 * 16 * self.factor, use_bias=False)(label)
        y = tf.keras.layers.Reshape((4, 4, 16 * self.factor))(y)

        x = tf.keras.layers.Concatenate()([x, y])

        x = tf.keras.layers.Conv2DTranspose(
            256 * self.factor, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(
            128 * self.factor, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        if self.input_shape[0] >= 64:
            x = tf.keras.layers.Conv2DTranspose(
                64 * self.factor, 4, strides=2, padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        out = tf.keras.layers.Conv2DTranspose(
            self.input_shape[2], 4, strides=2, padding='same',
            use_bias=False, activation='tanh'
        )(x)

        return tf.keras.Model([z, label], out, name="generator")

    def build_discriminator(self, num_classes):
        inp = tf.keras.layers.Input(shape=self.input_shape)
        label = tf.keras.layers.Input(
            shape=(num_classes,))  # one-hot or multi-label

        # project label to spatial map
        y = tf.keras.layers.Dense(
            self.input_shape[0] * self.input_shape[1], use_bias=False)(label)
        y = tf.keras.layers.Reshape(
            (self.input_shape[0], self.input_shape[1], 1)
        )(y)

        x = tf.keras.layers.Concatenate()([inp, y])

        x = tf.keras.layers.Conv2D(32 * self.factor, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv2D(
            64 * self.factor, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv2D(
            128 * self.factor, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv2D(
            256 * self.factor, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Flatten()(x)
        out = tf.keras.layers.Dense(1)(x)

        return tf.keras.Model([inp, label], out, name="discriminator")

    @tf.function
    def train_generator_step(self, noise, labels):
        batch_size = tf.shape(labels)[0]
        noise = tf.random.normal([batch_size, self.latent_dim[0]])
        with tf.GradientTape() as tape:
            fake_images = self.generator((noise, labels), training=True)
            fake_out = self.discriminator((fake_images, labels), training=True)
            loss = -tf.reduce_mean(fake_out)
        grads = tape.gradient(loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables))
        return loss

    @tf.function
    def train_discriminator_step(self, real_images, labels):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim[0]])

        with tf.GradientTape() as tape:
            fake_images = self.generator((noise, labels), training=True)

            real_out = self.discriminator((real_images, labels), training=True)
            fake_out = self.discriminator((fake_images, labels), training=True)

            alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
            interpolated = real_images + alpha * (fake_images - real_images)

            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                pred = self.discriminator(
                    (interpolated, labels), training=True)

            grads = gp_tape.gradient(pred, interpolated)
            grads = tf.reshape(grads, [batch_size, -1])
            norm = tf.norm(grads, axis=1)
            gp = tf.reduce_mean((norm - 1.0) ** 2)

            loss = tf.reduce_mean(fake_out) - \
                tf.reduce_mean(real_out) + self.lambda_gp * gp

        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_variables)
        )

        return loss

    @tf.function
    def dist_generator_step(self, noise, labels):
        per_replica_gen_loss = self.strategy.run(
            self.train_generator_step, args=(noise, labels))
        gen_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_gen_loss, axis=None)
        return gen_loss

    @tf.function
    def dist_discriminator_step(self, dataset_inputs, labels):
        per_replica_disc_loss = self.strategy.run(
            self.train_discriminator_step, args=(dataset_inputs, labels))
        disc_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_disc_loss, axis=None)
        return disc_loss

    def fit(self, dataset, epochs, initial_epoch=0, path='folder', callbacks=None):
        if callbacks is None:
            callbacks = []
        for callback in callbacks:
            callback.set_model(self)
            callback.on_train_begin()

        for epoch in range(initial_epoch, epochs):
            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            for step, batch in enumerate(dataset):
                images, labels = batch
                noise = np.random.normal(
                    0, 1, (self.batch_size, self.latent_dim[0]))
                for _ in range(N_DISC_STEP):
                    disc_loss = self.dist_discriminator_step(images, labels)

                for _ in range(N_GEN_STEP):
                    gen_loss = self.dist_generator_step(noise, labels)
                print(
                    f'\rEpoch [{step}/{epoch+1}], Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}', end='')
                sys.stdout.flush()
                logs = {
                    "gen_loss": gen_loss,
                    "disc_loss": disc_loss
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
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = Cond_GAN(strategy, input_shape=(64, 64, 3), latent_dim=(
            128,), batch_size=BATCH_SIZE, num_classes=100)
    tf.keras.utils.plot_model(model.generator, to_file="generator.png",
                              show_shapes=True, expand_nested=True, dpi=96)
    tf.keras.utils.plot_model(model.discriminator, to_file="discriminator.png",
                              show_shapes=True, expand_nested=True, dpi=96)
