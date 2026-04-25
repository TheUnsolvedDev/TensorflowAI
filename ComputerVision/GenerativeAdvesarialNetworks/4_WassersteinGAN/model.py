import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import tqdm
import numpy as np
from config import *

# Create images folder
os.makedirs("images", exist_ok=True)


class WGAN(tf.keras.Model):
    def __init__(self, strategy, input_shape, latent_dim, batch_size):
        super().__init__()
        self.strategy = strategy
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        # self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.global_batch_size = batch_size * self.strategy.num_replicas_in_sync

        with self.strategy.scope():
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator()

            self.generator_optimizer = tf.keras.optimizers.RMSprop(
                GENERATOR_LEARNING_RATE)
            self.discriminator_optimizer = tf.keras.optimizers.RMSprop(
                DISCRIMINATOR_LEARNING_RATE)

        self.generator.build(input_shape=(None, latent_dim[0]))
        self.discriminator.build(input_shape=(None, *input_shape))

        self.generator.summary()
        self.discriminator.summary()
    
    def build_generator(self, latent_dim=100, channels=3):
        _, _, channels = self.input_shape
        z = tf.keras.layers.Input(shape=(self.latent_dim[0],))

        x = tf.keras.layers.Dense(4 * 4 * 256, use_bias=False)(z)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Reshape((4, 4, 256))(x)

        x = tf.keras.layers.Conv2DTranspose(
            256, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(
            128, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        if self.input_shape[0] >= 64:
            x = tf.keras.layers.Conv2DTranspose(
                64, 4, strides=2, padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        out = tf.keras.layers.Conv2DTranspose(
            channels, 4, strides=2, padding='same',
            use_bias=False, activation='tanh')(x)

        return tf.keras.Model(z, out, name="generator")

    def build_discriminator(self):
        inp = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(
            64, 4, strides=2, padding='same')(inp)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv2D(
            128, 4, strides=2, padding='same', use_bias=False)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv2D(
            256, 4, strides=2, padding='same', use_bias=False)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv2D(
            256, 4, strides=2, padding='same', use_bias=False)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Flatten()(x)
        out = tf.keras.layers.Dense(1)(x)  # logits

        return tf.keras.Model(inp, out, name="discriminator")
    
    
    @tf.function
    def train_generator_step(self, noise):
        noise = tf.random.normal([self.batch_size, self.latent_dim[0]])
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            fake_out = self.discriminator(fake_images, training=True)
            loss = -tf.reduce_mean(fake_out)
        grads = tape.gradient(loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        return loss
    
    @tf.function
    def train_discriminator_step(self, real_images):
        noise = tf.random.normal([self.batch_size, self.latent_dim[0]])
        real_images+= tf.random.normal(tf.shape(real_images), stddev=0.05)

        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            fake_images += tf.random.normal(tf.shape(fake_images), stddev=0.05)
            real_out = self.discriminator(real_images, training=True)
            fake_out = self.discriminator(fake_images, training=True)

            loss = tf.reduce_mean(fake_out) - tf.reduce_mean(real_out)
        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        for var in self.discriminator.trainable_variables:
            var.assign(tf.clip_by_value(var, -0.01, 0.01))
        return loss
    
    @tf.function
    def dist_generator_step(self, noise):
        per_replica_gen_loss = self.strategy.run(
            self.train_generator_step, args=(noise,))
        gen_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_gen_loss, axis=None)
        return gen_loss

    @tf.function
    def dist_discriminator_step(self, dataset_inputs):
        per_replica_disc_loss = self.strategy.run(
            self.train_discriminator_step, args=(dataset_inputs,))
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
                    disc_loss = self.dist_discriminator_step(image_batch)

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
            self.generate_and_save_images(epoch+1, path=path)
            logs = {"gen_loss": gen_loss.numpy(), "disc_loss": disc_loss.numpy()}
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

        for callback in callbacks:
            callback.on_train_end()