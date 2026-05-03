import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import tqdm
import numpy as np
from config import *


class LS_GAN(tf.keras.Model):
    def __init__(self, strategy, input_shape, latent_dim, batch_size):
        super().__init__()
        self.strategy = strategy
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.global_batch_size = batch_size * self.strategy.num_replicas_in_sync
        self.factor = 2 if self.input_shape[0] >= 64 else 1
        with self.strategy.scope():
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator()

            self.generator_optimizer = tf.keras.optimizers.Adam(
                GENERATOR_LEARNING_RATE, beta_1=0.5, beta_2=0.999)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(
                DISCRIMINATOR_LEARNING_RATE, beta_1=0.5, beta_2=0.999)

        self.generator.build(input_shape=(None, latent_dim[0]))
        self.discriminator.build(input_shape=(None, *input_shape))

        self.generator.summary()
        self.discriminator.summary()

    def build_generator(self, latent_dim=100, channels=3):
        _, _, channels = self.input_shape
        z = tf.keras.layers.Input(shape=(self.latent_dim[0],))

        x = tf.keras.layers.Dense(4 * 4 * 256 * self.factor, use_bias=False)(z)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Reshape((4, 4, 256 * self.factor))(x)

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
            channels, 4, strides=2, padding='same',
            use_bias=False, activation='tanh')(x)

        return tf.keras.Model(z, out, name="generator")

    def build_discriminator(self):
        inp = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(
            64 * self.factor, 4, strides=2, padding='same')(inp)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv2D(
            128 * self.factor, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv2D(
            256 * self.factor, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv2D(
            256 * self.factor, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Flatten()(x)
        out = tf.keras.layers.Dense(1)(x)  # logits

        return tf.keras.Model(inp, out, name="discriminator")

    @tf.function
    def train_generator_step(self, noise):
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = self.loss_fn(
                tf.ones_like(fake_output), fake_output)* 0.5

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        return gen_loss

    @tf.function
    def train_discriminator_step(self, noise, real_images):
        with tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_images += tf.random.normal(tf.shape(real_images), stddev=0.05)
            generated_images += tf.random.normal(
                tf.shape(generated_images), stddev=0.05)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            disc_loss = (self.loss_fn(tf.ones_like(real_output), real_output) +
                         self.loss_fn(tf.zeros_like(fake_output), fake_output))* 0.5

        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return disc_loss

    @tf.function
    def dist_generator_step(self, noise):
        per_replica_gen_loss = self.strategy.run(
            self.train_generator_step, args=(noise,))
        gen_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_gen_loss, axis=None)
        return gen_loss

    @tf.function
    def dist_discriminator_step(self, noise, real_images):
        per_replica_disc_loss = self.strategy.run(
            self.train_discriminator_step, args=(noise, real_images))
        disc_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_disc_loss, axis=None)
        return disc_loss

    def generate_and_save_images(self, epoch, num_examples=16, path='folder'):
        path = f'images/{path}'
        os.makedirs(path, exist_ok=True)
        noise = tf.random.normal([num_examples, self.latent_dim[0]])
        generated_images = self.generator(noise, training=False)
        generated_images = (generated_images + 1) / 2.0  # Rescale [0,1]

        fig, axs = plt.subplots(4, 4, figsize=(4, 4))
        idx = 0
        for i in range(4):
            for j in range(4):
                img = generated_images[idx]
                if self.input_shape[2] == 1:
                    img = img[:, :, 0]
                    axs[i, j].imshow(img, cmap='gray')
                else:
                    axs[i, j].imshow(img)
                axs[i, j].axis('off')
                idx += 1

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(f"{path}/image_at_epoch_{epoch:03d}.png")
        plt.close()

    def fit(self, dataset, epochs, initial_epoch=0, path='folder', callbacks=None):
        if callbacks is None:
            callbacks = []
        for callback in callbacks:
            callback.set_model(self)
            callback.on_train_begin()

        for epoch in range(initial_epoch, epochs):
            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            for step, image_batch in enumerate(dataset):
                noise = np.random.normal(
                    0, 1, (self.batch_size, self.latent_dim[0]))
                for _ in range(N_DISC_STEP):
                    disc_loss = self.dist_discriminator_step(noise, image_batch)

                for _ in range(N_GEN_STEP):
                    gen_loss = self.dist_generator_step(noise)
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


if __name__ == '__main__':
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce())
    gan = LS_GAN(strategy=strategy, input_shape=(
        IMAGE_SIZE[0]*2, IMAGE_SIZE[1]*2, 3), latent_dim=(LATENT_DIM*4,), batch_size=BATCH_SIZE)
    gan = LS_GAN(strategy=strategy, input_shape=(
        IMAGE_SIZE[0], IMAGE_SIZE[1], 3), latent_dim=(LATENT_DIM,), batch_size=BATCH_SIZE)
    # gan = GAN(strategy=strategy, input_shape=(
    #     IMAGE_SIZE[0], IMAGE_SIZE[1], 3), latent_dim=(LATENT_DIM,), batch_size=BATCH_SIZE)
    # gan = GAN(strategy=strategy, input_shape=(
    #     28, 28, 3), latent_dim=(LATENT_DIM,), batch_size=BATCH_SIZE)
