import sys

import tensorflow as tf

from config import *


class StarGAN(tf.keras.Model):
    def __init__(self, strategy, input_shape, batch_size, label_dim, label_mode):
        super().__init__()
        self.strategy = strategy
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.label_dim = label_dim
        self.label_mode = label_mode
        self.lambda_recon = LAMBDA_RECON
        self.lambda_cls = LAMBDA_CLS

        with self.strategy.scope():
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator()

            self.generator_optimizer = tf.keras.optimizers.Adam(
                GENERATOR_LEARNING_RATE, beta_1=0.5, beta_2=0.999
            )
            self.discriminator_optimizer = tf.keras.optimizers.Adam(
                DISCRIMINATOR_LEARNING_RATE, beta_1=0.5, beta_2=0.999
            )

            self.adv_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            if self.label_mode == "multilabel":
                self.cls_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            else:
                self.cls_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            self.rec_loss_fn = tf.keras.losses.MeanAbsoluteError()

        tf.keras.utils.plot_model(
            self.generator, to_file="generator.png", show_shapes=True, expand_nested=True
        )
        tf.keras.utils.plot_model(
            self.discriminator, to_file="discriminator.png", show_shapes=True, expand_nested=True
        )

    def residual_block(self, x, filters):
        shortcut = x
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Add()([shortcut, x])

    def build_generator(self):
        image_input = tf.keras.layers.Input(shape=self.input_shape)
        label_input = tf.keras.layers.Input(shape=(self.label_dim,))

        label_map = tf.keras.layers.Dense(self.input_shape[0] * self.input_shape[1])(label_input)
        label_map = tf.keras.layers.Reshape((self.input_shape[0], self.input_shape[1], 1))(label_map)

        x = tf.keras.layers.Concatenate()([image_input, label_map])
        x = tf.keras.layers.Conv2D(64, 7, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(128, 4, strides=2, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(256, 4, strides=2, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        for _ in range(STARGAN_RES_BLOCKS):
            x = self.residual_block(x, 256)

        x = tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        outputs = tf.keras.layers.Conv2D(self.input_shape[-1], 7, padding="same", activation="tanh")(x)
        return tf.keras.Model([image_input, label_input], outputs, name="generator")

    def build_discriminator(self):
        image_input = tf.keras.layers.Input(shape=self.input_shape)

        x = image_input
        for num_filters in [64, 128, 256, 512]:
            x = tf.keras.layers.Conv2D(num_filters, 4, strides=2, padding="same")(x)
            x = tf.keras.layers.LeakyReLU(0.2)(x)

        adv_out = tf.keras.layers.Conv2D(1, 3, padding="same", name="adv_head")(x)
        cls_out = tf.keras.layers.Conv2D(self.label_dim, x.shape[1], padding="valid", name="cls_head")(x)
        cls_out = tf.keras.layers.Reshape((self.label_dim,))(cls_out)
        return tf.keras.Model(image_input, [adv_out, cls_out], name="discriminator")

    def classification_loss(self, target, pred):
        return self.cls_loss_fn(target, pred)

    @tf.function
    def train_generator_step(self, real_images, real_labels):
        target_labels = tf.random.shuffle(real_labels)
        valid = tf.ones_like(self.discriminator(real_images, training=True)[0])

        with tf.GradientTape() as tape:
            fake_images = self.generator((real_images, target_labels), training=True)
            reconstructed_images = self.generator((fake_images, real_labels), training=True)

            disc_fake, cls_fake = self.discriminator(fake_images, training=True)

            adv_loss = self.adv_loss_fn(valid, disc_fake)
            cls_loss = self.classification_loss(target_labels, cls_fake)
            recon_loss = self.rec_loss_fn(real_images, reconstructed_images)
            total_loss = adv_loss + self.lambda_cls * cls_loss + self.lambda_recon * recon_loss

        grads = tape.gradient(total_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        return total_loss

    @tf.function
    def train_discriminator_step(self, real_images, real_labels):
        target_labels = tf.random.shuffle(real_labels)

        with tf.GradientTape() as tape:
            fake_images = self.generator((real_images, target_labels), training=True)
            disc_real, cls_real = self.discriminator(real_images, training=True)
            disc_fake, _ = self.discriminator(tf.stop_gradient(fake_images), training=True)

            valid = tf.ones_like(disc_real)
            fake = tf.zeros_like(disc_fake)
            adv_loss = self.adv_loss_fn(valid, disc_real) + self.adv_loss_fn(fake, disc_fake)
            cls_loss = self.classification_loss(real_labels, cls_real)
            total_loss = adv_loss + self.lambda_cls * cls_loss

        grads = tape.gradient(total_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return total_loss

    @tf.function
    def dist_generator_step(self, real_images, real_labels):
        per_replica_loss = self.strategy.run(self.train_generator_step, args=(real_images, real_labels))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)

    @tf.function
    def dist_discriminator_step(self, real_images, real_labels):
        per_replica_loss = self.strategy.run(self.train_discriminator_step, args=(real_images, real_labels))
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

            for step, (images, labels) in enumerate(dataset):
                disc_loss = self.dist_discriminator_step(images, labels)
                gen_loss = self.dist_generator_step(images, labels)

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
