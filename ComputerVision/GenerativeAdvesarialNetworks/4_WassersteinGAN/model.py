from numpy import var
import tensorflow as tf
import matplotlib.pyplot as plt
import os, sys
from config import *

os.makedirs("images", exist_ok=True)


class WGAN(tf.keras.Model):
    def __init__(self, strategy, input_shape, latent_dim, batch_size):
        super().__init__()
        self.strategy = strategy
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.n_critic = 5

        with self.strategy.scope():
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator()

            self.generator_optimizer = tf.keras.optimizers.Adam(
                GENERATOR_LEARNING_RATE, beta_1=0.0, beta_2=0.9
            )
            self.discriminator_optimizer = tf.keras.optimizers.Adam(
                DISCRIMINATOR_LEARNING_RATE, beta_1=0.0, beta_2=0.9
            )
        self.generator.build(input_shape=(None, latent_dim[0]))
        self.discriminator.build(input_shape=(None, *input_shape))

        self.generator.summary()
        self.discriminator.summary()
    
    def build_generator(self):
        inputs = tf.keras.layers.Input(shape=(self.latent_dim[0],))

        # Start from a small feature map size depending on input resolution
        init_height = self.input_shape[0] // 4
        init_width = self.input_shape[1] // 4
        init_channels = 32

        x = tf.keras.layers.Dense(
            init_height * init_width * init_channels, use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Reshape(
            (init_height, init_width, init_channels))(x)

        x = tf.keras.layers.Conv2DTranspose(
            128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

        # Final layer outputs the desired channels (1 or 3) and final size
        x = tf.keras.layers.Conv2DTranspose(self.input_shape[2], (5, 5), strides=(
            1, 1), padding='same', use_bias=False, activation='tanh')(x)

        outputs = x
        return tf.keras.Model(inputs, outputs, name="generator")

    def build_discriminator(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding='same')(inputs)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Conv2D(
            128, (5, 5), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Conv2D(
            128, (5, 5), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.LeakyReLU()(x)
        
        x = tf.keras.layers.Conv2D(
            128, (5, 5), strides=(2, 2), padding='same')(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dense(1)(x)

        outputs = x
        return tf.keras.Model(inputs, outputs, name="discriminator")
    
    
    @tf.function
    def train_generator_step(self):
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

        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            real_out = self.discriminator(real_images, training=True)
            fake_out = self.discriminator(fake_images, training=True)

            loss = tf.reduce_mean(fake_out) - tf.reduce_mean(real_out)
        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        for var in self.discriminator.trainable_variables:
            var.assign(tf.clip_by_value(var, -0.01, 0.01))
        return loss
    

    # ---------------- Distributed Wrappers ----------------
    @tf.function
    def dist_discriminator_step(self, images):
        per_replica = self.strategy.run(self.train_discriminator_step, args=(images,))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica, axis=None)

    @tf.function
    def dist_generator_step(self):
        per_replica = self.strategy.run(self.train_generator_step)
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica, axis=None)
    
    def fit(self, dataset, epochs, path='folder', callbacks=None):
        if callbacks is None:
            callbacks = []
        for callback in callbacks:
            callback.set_model(self)
            callback.on_train_begin()

        for epoch in range(epochs):
            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            for step, image_batch in enumerate(dataset):
                for _ in range(self.n_critic):
                    disc_loss = self.dist_discriminator_step(image_batch)

                    # Train generator once
                gen_loss = self.dist_generator_step()
                print(
                    f'\rEpoch [{step}/{epoch+1}], Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}',end='')
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



    # ---------------- Visualization ----------------
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