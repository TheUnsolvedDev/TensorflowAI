import sys

import tensorflow as tf

from config import *


class SRGAN(tf.keras.Model):
    def __init__(self, strategy, lr_input_shape, hr_input_shape, batch_size):
        super().__init__()
        self.strategy = strategy
        self.lr_input_shape = lr_input_shape
        self.hr_input_shape = hr_input_shape
        self.batch_size = batch_size

        with self.strategy.scope():
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator()

            self.generator_optimizer = tf.keras.optimizers.Adam(
                GENERATOR_LEARNING_RATE, beta_1=0.9, beta_2=0.999
            )
            self.discriminator_optimizer = tf.keras.optimizers.Adam(
                DISCRIMINATOR_LEARNING_RATE, beta_1=0.9, beta_2=0.999
            )

            self.content_loss_fn = tf.keras.losses.MeanSquaredError()
            self.adversarial_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        tf.keras.utils.plot_model(
            self.generator, to_file="generator.png", show_shapes=True, expand_nested=True
        )
        tf.keras.utils.plot_model(
            self.discriminator, to_file="discriminator.png", show_shapes=True, expand_nested=True
        )

    def residual_block(self, x, filters):
        shortcut = x

        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        return tf.keras.layers.Add()([shortcut, x])

    def upsample_block(self, x, filters):
        x = tf.keras.layers.Conv2D(filters * 4, 3, padding="same")(x)
        x = tf.keras.layers.Lambda(lambda tensor: tf.nn.depth_to_space(tensor, 2))(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        return x

    def build_generator(self):
        inputs = tf.keras.layers.Input(shape=self.lr_input_shape)

        x = tf.keras.layers.Conv2D(64, 9, padding="same")(inputs)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
        skip = x

        for _ in range(SRGAN_NUM_RES_BLOCKS):
            x = self.residual_block(x, 64)

        x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, skip])

        x = self.upsample_block(x, 64)
        x = self.upsample_block(x, 64)

        outputs = tf.keras.layers.Conv2D(3, 9, padding="same", activation="tanh")(x)
        return tf.keras.Model(inputs, outputs, name="generator")

    def disc_block(self, x, filters, strides, use_batch_norm=True):
        x = tf.keras.layers.Conv2D(filters, 3, strides=strides, padding="same")(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        return x

    def build_discriminator(self):
        inputs = tf.keras.layers.Input(shape=self.hr_input_shape)

        x = self.disc_block(inputs, 64, 1, use_batch_norm=False)
        x = self.disc_block(x, 64, 2)
        x = self.disc_block(x, 128, 1)
        x = self.disc_block(x, 128, 2)
        x = self.disc_block(x, 256, 1)
        x = self.disc_block(x, 256, 2)
        x = self.disc_block(x, 512, 1)
        x = self.disc_block(x, 512, 2)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        outputs = tf.keras.layers.Dense(1)(x)

        return tf.keras.Model(inputs, outputs, name="discriminator")

    @tf.function
    def train_generator_step(self, lr_images, hr_images):
        valid = tf.ones((tf.shape(hr_images)[0], 1))

        with tf.GradientTape() as tape:
            sr_images = self.generator(lr_images, training=True)
            fake_output = self.discriminator(sr_images, training=True)

            content_loss = self.content_loss_fn(hr_images, sr_images)
            adv_loss = self.adversarial_loss_fn(valid, fake_output)
            total_loss = CONTENT_LOSS_WEIGHT * content_loss + ADVERSARIAL_LOSS_WEIGHT * adv_loss

        grads = tape.gradient(total_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        return total_loss

    @tf.function
    def train_discriminator_step(self, lr_images, hr_images):
        valid = tf.ones((tf.shape(hr_images)[0], 1))
        fake = tf.zeros((tf.shape(hr_images)[0], 1))

        with tf.GradientTape() as tape:
            sr_images = self.generator(lr_images, training=True)
            real_output = self.discriminator(hr_images, training=True)
            fake_output = self.discriminator(tf.stop_gradient(sr_images), training=True)

            real_loss = self.adversarial_loss_fn(valid, real_output)
            fake_loss = self.adversarial_loss_fn(fake, fake_output)
            total_loss = 0.5 * (real_loss + fake_loss)

        grads = tape.gradient(total_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return total_loss

    @tf.function
    def dist_generator_step(self, lr_images, hr_images):
        per_replica_loss = self.strategy.run(
            self.train_generator_step, args=(lr_images, hr_images)
        )
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)

    @tf.function
    def dist_discriminator_step(self, lr_images, hr_images):
        per_replica_loss = self.strategy.run(
            self.train_discriminator_step, args=(lr_images, hr_images)
        )
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)

    def fit(self, dataset, epochs, initial_epoch=0, callbacks=None):
        if callbacks is None:
            callbacks = []

        callback_params = {
            "epochs": epochs,
            "steps": None,
            "verbose": 1,
            "do_validation": False,
            "metrics": ["gen_loss", "disc_loss"],
        }
        for callback in callbacks:
            callback.set_model(self)
            callback.set_params(callback_params)
            callback.on_train_begin()

        for epoch in range(initial_epoch, epochs):
            self._current_epoch = epoch
            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            for step, (lr_images, hr_images) in enumerate(dataset):
                disc_loss = self.dist_discriminator_step(lr_images, hr_images)
                gen_loss = self.dist_generator_step(lr_images, hr_images)

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
