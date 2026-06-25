import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import tqdm
import numpy as np
from config import *

class MinibatchDiscrimination(tf.keras.layers.Layer):
    def __init__(self, num_kernels=100, kernel_dim=5):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim

    def build(self, input_shape):
        features = input_shape[-1]
        self.T = self.add_weight(shape=(features, self.num_kernels * self.kernel_dim),
                                 initializer="glorot_uniform",
                                 trainable=True,
                                 name="T")

    def call(self, x):
        M = tf.matmul(x, self.T)
        M = tf.reshape(M, (-1, self.num_kernels, self.kernel_dim))
        M1 = tf.expand_dims(M, 3)
        M2 = tf.expand_dims(tf.transpose(M, [1, 2, 0]), 0)
        abs_diff = tf.reduce_sum(tf.abs(M1 - M2), axis=2)
        c = tf.reduce_sum(tf.exp(-abs_diff), axis=2)
        return tf.concat([x, c], axis=1)

class GAN(tf.keras.Model):
    def __init__(self, strategy, input_shape, latent_dim, batch_size):
        super().__init__()
        self.strategy = strategy
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)
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
        tf.keras.utils.plot_model(self.generator, to_file='generator.png', show_shapes=True, expand_nested=True)
        tf.keras.utils.plot_model(self.discriminator, to_file='discriminator.png', show_shapes=True, expand_nested=True)
        self.generator.summary()
        self.discriminator.summary()
        
    def sn_conv(self, filters, kernel_size, strides=1, padding='same', use_bias=True):
        return (tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)) # tf.keras.layers.SpectralNormalization

    def sn_dense(self, units, use_bias=True):
        return (tf.keras.layers.Dense(units, use_bias=use_bias)) # tf.keras.layers.SpectralNormalization
        
    def res_block_up(self, x, filters):
        shortcut = tf.keras.layers.UpSampling2D()(x)
        shortcut = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)

        out = tf.keras.layers.UpSampling2D()(x)
        out = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(out)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.ReLU()(out)

        out = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(out)
        out = tf.keras.layers.BatchNormalization()(out)

        out = tf.keras.layers.Add()([shortcut, out])
        out = tf.keras.layers.ReLU()(out)
        return out

    def build_generator(self, latent_dim=100, channels=3):
        _, _, channels = self.input_shape
        z = tf.keras.layers.Input(shape=(self.latent_dim[0],))

        x = tf.keras.layers.Dense(4 * 4 * 256 * self.factor, use_bias=False)(z)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Reshape((4, 4, 256 * self.factor))(x)

        x = self.res_block_up(x, 128 * self.factor)
        x = self.res_block_up(x, 64 * self.factor)

        if self.input_shape[0] >= 64:
            x = self.res_block_up(x, 64 * self.factor)

        out = tf.keras.layers.Conv2DTranspose(
            channels, 4, strides=2, padding='same',
            use_bias=False, activation='tanh')(x)

        return tf.keras.Model(z, out, name="generator")
    
    def res_block_down(self, x, filters):
        shortcut = self.sn_conv(filters, 1, strides=2)(x)
        out = self.sn_conv(filters, 4, strides=2)(x)
        out = tf.keras.layers.LeakyReLU(0.2)(out)
        out = self.sn_conv(filters, 3)(out)
        out = tf.keras.layers.LeakyReLU(0.2)(out)
        out = tf.keras.layers.Add()([shortcut, out])
        return out

    def build_discriminator(self):
        inp = tf.keras.layers.Input(shape=self.input_shape)

        x = self.res_block_down(inp, 32 * self.factor)
        x = self.res_block_down(x, 64 * self.factor)
        x = self.res_block_down(x, 64 * self.factor)
        x = self.res_block_down(x, 128 * self.factor)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # x = MinibatchDiscrimination(num_kernels=32, kernel_dim=4)(x)

        out = self.sn_dense(1)(x)
        return tf.keras.Model(inp, out, name="discriminator")

    @tf.function
    def train_generator_step(self, noise):
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = self.loss_fn(
                tf.ones_like(fake_output), fake_output)

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
            disc_loss = (self.loss_fn(tf.ones_like(real_output) * 0.9, real_output) +
                         self.loss_fn(tf.zeros_like(fake_output), fake_output))

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
    gan = GAN(strategy=strategy, input_shape=(
        IMAGE_SIZE[0]*2, IMAGE_SIZE[1]*2, 3), latent_dim=(LATENT_DIM*4,), batch_size=BATCH_SIZE)
    # gan = GAN(strategy=strategy, input_shape=(
    #     IMAGE_SIZE[0], IMAGE_SIZE[1], 3), latent_dim=(LATENT_DIM,), batch_size=BATCH_SIZE)
    # gan = GAN(strategy=strategy, input_shape=(
    #     IMAGE_SIZE[0], IMAGE_SIZE[1], 3), latent_dim=(LATENT_DIM,), batch_size=BATCH_SIZE)
    # gan = GAN(strategy=strategy, input_shape=(
    #     28, 28, 3), latent_dim=(LATENT_DIM,), batch_size=BATCH_SIZE)
