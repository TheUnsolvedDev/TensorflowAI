import tensorflow as tf
import matplotlib.pyplot as plt
import os,sys
import tqdm
import numpy as np
from config import *

# Create images folder
os.makedirs("images", exist_ok=True)


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
                DISCRIMINATOR_LEARNING_RATE)

        self.generator.build(input_shape=(None, latent_dim[0]))
        self.discriminator.build(input_shape=(None, *input_shape))
        self.generator.summary()
        self.discriminator.summary()

    def build_generator(self):
        init_channels = 64
        inputs = tf.keras.layers.Input(shape=(self.latent_dim[0],))
        x = tf.keras.layers.Dense(init_channels * 2, use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Dense(init_channels * 4, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Dense(init_channels * 8, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

        output_dim = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
        x = tf.keras.layers.Dense(output_dim, activation='tanh')(x)
        x = tf.keras.layers.Reshape(self.input_shape)(x)

        return tf.keras.Model(inputs, x, name="generator")


    def build_discriminator(self):
        init_channels = 16
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Flatten()(inputs)

        x = tf.keras.layers.Dense(init_channels * 4)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        x = tf.keras.layers.Dense(init_channels * 4)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        x = tf.keras.layers.Dense(1)(x)  # Single output neuron
        return tf.keras.Model(inputs, x, name="discriminator")
    
    
    @tf.function
    def train_generator_step(self, noise):
        noise = tf.random.normal([self.global_batch_size, self.latent_dim[0]])
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = self.cross_entropy(
                tf.ones_like(fake_output), fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
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
            disc_loss = (self.cross_entropy(tf.ones_like(real_output), real_output) +
                         self.cross_entropy(tf.zeros_like(fake_output), fake_output)) 

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
                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim[0]))
                for _ in range(N_DISC_STEP):
                    disc_loss = self.dist_discriminator_step(image_batch)

                for _ in range(N_GEN_STEP):
                    gen_loss = self.dist_generator_step(noise)
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


if __name__ == '__main__':
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce())
    gan = NN_GAN(strategy=strategy, input_shape=(
        IMAGE_SIZE[0]*4, IMAGE_SIZE[1]*4, 3), latent_dim=(LATENT_DIM,), batch_size=BATCH_SIZE)
