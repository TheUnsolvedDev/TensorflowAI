import os
import sys

import numpy as np
import tensorflow as tf

from config import *
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from training_utils import GeneratorEMA

os.makedirs("images", exist_ok=True)


class Cond_GAN(tf.keras.Model):
    def __init__(self, strategy, input_shape, latent_dim, batch_size, num_classes):
        super().__init__()
        self.strategy = strategy
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.lambda_gp = LAMBDA_GP
        if batch_size % self.strategy.num_replicas_in_sync:
            raise ValueError("Global batch size must be divisible by the replica count")

        with self.strategy.scope():
            self.generator = self.build_generator(num_classes)
            self.discriminator = self.build_discriminator(num_classes)

            self.generator_optimizer = tf.keras.optimizers.Adam(
                GENERATOR_LEARNING_RATE, beta_1=0.5, beta_2=0.999
            )
            self.discriminator_optimizer = tf.keras.optimizers.Adam(
                DISCRIMINATOR_LEARNING_RATE, beta_1=0.5, beta_2=0.999
            )
            self.generator_optimizer.build(self.generator.trainable_variables)
            self.discriminator_optimizer.build(self.discriminator.trainable_variables)
            self.ema = GeneratorEMA([self.generator], EMA_DECAY)

        tf.keras.utils.plot_model(
            self.generator,
            to_file="generator.png",
            show_shapes=True,
            expand_nested=True,
            dpi=96,
        )
        tf.keras.utils.plot_model(
            self.discriminator,
            to_file="discriminator.png",
            show_shapes=True,
            expand_nested=True,
            dpi=96,
        )
        self.generator.summary()
        self.discriminator.summary()

    def build_generator(self, num_classes):
        start_size = 4
        upscale_steps = int(np.log2(self.input_shape[0] // start_size))
        base_filters = 512 if self.input_shape[0] >= 64 else 256

        noise_input = tf.keras.layers.Input(shape=(self.latent_dim[0],))
        label_input = tf.keras.layers.Input(shape=(num_classes,))

        x = tf.keras.layers.Concatenate()([noise_input, label_input])
        x = tf.keras.layers.Dense(start_size * start_size * base_filters, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Reshape((start_size, start_size, base_filters))(x)

        label_map = tf.keras.layers.Dense(start_size * start_size)(label_input)
        label_map = tf.keras.layers.Reshape((start_size, start_size, 1))(label_map)
        x = tf.keras.layers.Concatenate()([x, label_map])

        filters = base_filters
        for _ in range(max(0, upscale_steps - 1)):
            filters = max(filters // 2, 64)
            x = tf.keras.layers.Conv2DTranspose(
                filters, 4, strides=2, padding="same", use_bias=False
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        outputs = tf.keras.layers.Conv2DTranspose(
            self.input_shape[2], 4, strides=2, padding="same", use_bias=False, activation="tanh"
        )(x)
        return tf.keras.Model([noise_input, label_input], outputs, name="generator")

    def build_discriminator(self, num_classes):
        image_input = tf.keras.layers.Input(shape=self.input_shape)
        label_input = tf.keras.layers.Input(shape=(num_classes,))

        label_map = tf.keras.layers.Dense(self.input_shape[0] * self.input_shape[1])(label_input)
        label_map = tf.keras.layers.Reshape((self.input_shape[0], self.input_shape[1], 1))(label_map)

        x = tf.keras.layers.Concatenate()([image_input, label_map])
        filters = [64, 128, 256]
        if self.input_shape[0] >= 64:
            filters.append(512)

        for num_filters in filters:
            x = tf.keras.layers.Conv2D(num_filters, 4, strides=2, padding="same")(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model([image_input, label_input], outputs, name="discriminator")

    @tf.function
    def train_generator_step(self, labels):
        batch_size = tf.shape(labels)[0]
        noise = tf.random.normal([batch_size, self.latent_dim[0]])
        with tf.GradientTape() as tape:
            fake_images = self.generator((noise, labels), training=True)
            fake_out = self.discriminator((fake_images, labels), training=True)
            loss = -tf.reduce_mean(fake_out) / self.strategy.num_replicas_in_sync
        grads = tape.gradient(loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
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
                pred = self.discriminator((interpolated, labels), training=True)

            grads = gp_tape.gradient(pred, interpolated)
            grads = tf.reshape(grads, [batch_size, -1])
            norm = tf.norm(grads, axis=1)
            gp = tf.reduce_mean((norm - 1.0) ** 2)

            loss = (
                tf.reduce_mean(fake_out)
                - tf.reduce_mean(real_out)
                + self.lambda_gp * gp
            ) / self.strategy.num_replicas_in_sync

        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_variables)
        )

        return loss

    @tf.function
    def dist_generator_step(self, labels):
        per_replica_gen_loss = self.strategy.run(
            self.train_generator_step, args=(labels,)
        )
        gen_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_gen_loss, axis=None
        )
        return gen_loss

    @tf.function
    def dist_discriminator_step(self, dataset_inputs, labels):
        per_replica_disc_loss = self.strategy.run(
            self.train_discriminator_step, args=(dataset_inputs, labels)
        )
        disc_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_disc_loss, axis=None
        )
        return disc_loss

    def fit(self, dataset, epochs, initial_epoch=0, path="folder", callbacks=None):
        if callbacks is None:
            callbacks = []
        steps_per_epoch = int(dataset.cardinality)
        for callback in callbacks:
            callback.set_model(self)
            callback.set_params({"epochs": epochs, "steps": steps_per_epoch})
            callback.on_train_begin()

        for epoch in range(initial_epoch, epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            progress_bar = tf.keras.utils.Progbar(steps_per_epoch)
            gen_loss_metric = tf.keras.metrics.Mean()
            disc_loss_metric = tf.keras.metrics.Mean()
            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            for step, batch in enumerate(dataset):
                images, labels = batch
                for _ in range(N_DISC_STEP):
                    disc_loss = self.dist_discriminator_step(images, labels)

                for _ in range(N_GEN_STEP):
                    gen_loss = self.dist_generator_step(labels)
                self.ema.update()
                gen_loss_metric.update_state(gen_loss)
                disc_loss_metric.update_state(disc_loss)
                progress_bar.update(
                    step + 1,
                    values=[
                        ("gen_loss", float(gen_loss)),
                        ("disc_loss", float(disc_loss)),
                    ],
                )
                logs = {"gen_loss": gen_loss, "disc_loss": disc_loss}
                for callback in callbacks:
                    callback.on_train_batch_end(step, logs)

            logs = {
                "gen_loss": gen_loss_metric.result().numpy(),
                "disc_loss": disc_loss_metric.result().numpy(),
            }
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

        for callback in callbacks:
            callback.on_train_end()


if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        Cond_GAN(
            strategy,
            input_shape=(64, 64, 3),
            latent_dim=(128,),
            batch_size=BATCH_SIZE,
            num_classes=100,
        )
