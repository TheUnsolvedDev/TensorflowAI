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

        self.lambda_gp = LAMBDA_GP
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.global_batch_size = batch_size * self.strategy.num_replicas_in_sync

        with self.strategy.scope():
            self.generator = self.build_generator(self.num_classes)
            self.discriminator = self.build_discriminator(self.num_classes)

            self.generator_optimizer = tf.keras.optimizers.Adam(GENERATOR_LEARNING_RATE)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(DISCRIMINATOR_LEARNING_RATE)

        # DO NOT manually call build() for multi-input models
        # Instead run one forward pass to initialize weights
        self._init_models()

        self.generator.summary()
        self.discriminator.summary()

    def _init_models(self):
        dummy_z = tf.random.normal([1, self.latent_dim[0]])
        dummy_label = tf.zeros([1, 1], dtype=tf.int32)
        dummy_img = tf.zeros([1, *self.input_shape])

        _ = self.generator([dummy_z, dummy_label])
        _ = self.discriminator([dummy_img, dummy_label])

    # ---------------- GENERATOR ----------------
    def build_generator(self, num_classes, embedding_dim=100):
        z = tf.keras.layers.Input(shape=(self.latent_dim[0],))
        label = tf.keras.layers.Input(shape=(1,), dtype=tf.int32)

        label_embedding = tf.keras.layers.Embedding(num_classes, embedding_dim)(label)
        label_embedding = tf.keras.layers.Flatten()(label_embedding)

        x = tf.keras.layers.Concatenate()([z, label_embedding])

        x = tf.keras.layers.Dense(4 * 4 * 256, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Reshape((4, 4, 256))(x)

        x = tf.keras.layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        if self.input_shape[0] >= 64:
            x = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        out = tf.keras.layers.Conv2DTranspose(
            self.input_shape[2], 4, strides=2, padding='same',
            use_bias=False, activation='tanh'
        )(x)

        return tf.keras.Model([z, label], out, name="generator")

    # ---------------- DISCRIMINATOR ----------------
    def build_discriminator(self, num_classes, embedding_dim=50):
        inp = tf.keras.layers.Input(shape=self.input_shape)
        label = tf.keras.layers.Input(shape=(1,), dtype=tf.int32)

        label_embedding = tf.keras.layers.Embedding(num_classes, embedding_dim)(label)
        label_embedding = tf.keras.layers.Flatten()(label_embedding)

        # project to spatial map
        label_embedding = tf.keras.layers.Dense(self.input_shape[0] * self.input_shape[1])(label_embedding)
        label_embedding = tf.keras.layers.Reshape(
            (self.input_shape[0], self.input_shape[1], 1)
        )(label_embedding)

        x = tf.keras.layers.Concatenate()([inp, label_embedding])

        x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = tf.keras.layers.Flatten()(x)
        out = tf.keras.layers.Dense(1)(x)

        return tf.keras.Model([inp, label], out, name="discriminator")
    
    
if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = Cond_GAN(strategy, input_shape=IMAGE_SIZE + (3,), latent_dim=(100,), batch_size=BATCH_SIZE)