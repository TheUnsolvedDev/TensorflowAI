import sys

import tensorflow as tf

from config import *


class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gamma = self.add_weight(
            name="gamma",
            shape=(channels,),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(channels,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        gamma = tf.reshape(self.gamma, (1, 1, 1, -1))
        beta = tf.reshape(self.beta, (1, 1, 1, -1))
        return normalized * gamma + beta


class CycleGAN(tf.keras.Model):
    def __init__(self, strategy, input_shape, batch_size):
        super().__init__()
        self.strategy = strategy
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.global_batch_size = batch_size * self.strategy.num_replicas_in_sync
        self.lambda_cycle = LAMBDA_CYCLE
        self.lambda_identity = LAMBDA_IDENTITY

        with self.strategy.scope():
            self.generator_a2b = self.build_generator(name="generator_a2b")
            self.generator_b2a = self.build_generator(name="generator_b2a")
            self.discriminator_a = self.build_discriminator(name="discriminator_a")
            self.discriminator_b = self.build_discriminator(name="discriminator_b")

            self.generator_optimizer = tf.keras.optimizers.Adam(
                GENERATOR_LEARNING_RATE, beta_1=0.5, beta_2=0.999
            )
            self.discriminator_optimizer = tf.keras.optimizers.Adam(
                DISCRIMINATOR_LEARNING_RATE, beta_1=0.5, beta_2=0.999
            )

            self.adv_loss_fn = tf.keras.losses.MeanSquaredError()
            self.cycle_loss_fn = tf.keras.losses.MeanAbsoluteError()

        tf.keras.utils.plot_model(
            self.generator_a2b, to_file="generator_a2b.png", show_shapes=True, expand_nested=True
        )
        tf.keras.utils.plot_model(
            self.generator_b2a, to_file="generator_b2a.png", show_shapes=True, expand_nested=True
        )
        tf.keras.utils.plot_model(
            self.discriminator_a, to_file="discriminator_a.png", show_shapes=True, expand_nested=True
        )
        tf.keras.utils.plot_model(
            self.discriminator_b, to_file="discriminator_b.png", show_shapes=True, expand_nested=True
        )

    def residual_block(self, x, filters):
        shortcut = x
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        return tf.keras.layers.Add()([shortcut, x])

    def build_generator(self, name):
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(64, 7, padding="same", use_bias=False)(inputs)
        x = InstanceNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(256, 3, strides=2, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        for _ in range(CYCLEGAN_RES_BLOCKS):
            x = self.residual_block(x, 256)

        x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        outputs = tf.keras.layers.Conv2D(self.input_shape[-1], 7, padding="same", activation="tanh")(x)
        return tf.keras.Model(inputs, outputs, name=name)

    def build_discriminator(self, name):
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(64, 4, strides=2, padding="same")(inputs)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv2D(128, 4, strides=2, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv2D(256, 4, strides=2, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv2D(512, 4, padding="same", use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        outputs = tf.keras.layers.Conv2D(1, 4, padding="same")(x)
        return tf.keras.Model(inputs, outputs, name=name)

    def adversarial_loss(self, pred, is_real):
        target = tf.ones_like(pred) if is_real else tf.zeros_like(pred)
        return self.adv_loss_fn(target, pred)

    @tf.function
    def train_generator_step(self, real_a, real_b):
        with tf.GradientTape() as tape:
            fake_b = self.generator_a2b(real_a, training=True)
            cycled_a = self.generator_b2a(fake_b, training=True)

            fake_a = self.generator_b2a(real_b, training=True)
            cycled_b = self.generator_a2b(fake_a, training=True)

            same_a = self.generator_b2a(real_a, training=True)
            same_b = self.generator_a2b(real_b, training=True)

            disc_fake_a = self.discriminator_a(fake_a, training=True)
            disc_fake_b = self.discriminator_b(fake_b, training=True)

            adv_loss = self.adversarial_loss(disc_fake_a, True) + self.adversarial_loss(disc_fake_b, True)
            cycle_loss = self.cycle_loss_fn(real_a, cycled_a) + self.cycle_loss_fn(real_b, cycled_b)
            identity_loss = self.cycle_loss_fn(real_a, same_a) + self.cycle_loss_fn(real_b, same_b)

            total_loss = adv_loss + self.lambda_cycle * cycle_loss + self.lambda_identity * identity_loss

        variables = self.generator_a2b.trainable_variables + self.generator_b2a.trainable_variables
        grads = tape.gradient(total_loss, variables)
        self.generator_optimizer.apply_gradients(zip(grads, variables))
        return total_loss

    @tf.function
    def train_discriminator_step(self, real_a, real_b):
        fake_b = self.generator_a2b(real_a, training=True)
        fake_a = self.generator_b2a(real_b, training=True)

        with tf.GradientTape() as tape:
            disc_real_a = self.discriminator_a(real_a, training=True)
            disc_real_b = self.discriminator_b(real_b, training=True)
            disc_fake_a = self.discriminator_a(tf.stop_gradient(fake_a), training=True)
            disc_fake_b = self.discriminator_b(tf.stop_gradient(fake_b), training=True)

            loss_a = self.adversarial_loss(disc_real_a, True) + self.adversarial_loss(disc_fake_a, False)
            loss_b = self.adversarial_loss(disc_real_b, True) + self.adversarial_loss(disc_fake_b, False)
            total_loss = 0.5 * (loss_a + loss_b)

        variables = self.discriminator_a.trainable_variables + self.discriminator_b.trainable_variables
        grads = tape.gradient(total_loss, variables)
        self.discriminator_optimizer.apply_gradients(zip(grads, variables))
        return total_loss

    @tf.function
    def dist_generator_step(self, real_a, real_b):
        per_replica_loss = self.strategy.run(self.train_generator_step, args=(real_a, real_b))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)

    @tf.function
    def dist_discriminator_step(self, real_a, real_b):
        per_replica_loss = self.strategy.run(self.train_discriminator_step, args=(real_a, real_b))
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

            for step, (real_a, real_b) in enumerate(dataset):
                disc_loss = self.dist_discriminator_step(real_a, real_b)
                gen_loss = self.dist_generator_step(real_a, real_b)

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
