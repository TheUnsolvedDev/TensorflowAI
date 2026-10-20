import math

import tensorflow as tf

from config import (
    BASE_CHANNELS,
    DENOISER_LEARNING_RATE,
    DIFFUSION_STEPS,
    DROPOUT_RATE,
    TIME_EMBED_DIM,
    WEIGHT_DECAY,
)


class TimeEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj1 = tf.keras.layers.Dense(dim * 2, activation="swish")
        self.proj2 = tf.keras.layers.Dense(dim)

    def call(self, timesteps):
        half = max(self.dim // 2, 1)
        freqs = tf.exp(-tf.math.log(10000.0) * tf.range(half, dtype=tf.float32) / max(float(half - 1), 1.0))
        angles = tf.cast(timesteps[:, None], tf.float32) * freqs[None, :]
        embedding = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)
        return self.proj2(self.proj1(embedding))


def add_time_bias(x, time_embedding, channels, name):
    time_bias = tf.keras.layers.Dense(channels, name=f"{name}_time_dense")(time_embedding)
    time_bias = tf.keras.layers.Reshape((1, 1, channels), name=f"{name}_time_reshape")(time_bias)
    return tf.keras.layers.Add(name=f"{name}_time_add")([x, time_bias])


def residual_block(x, time_embedding, channels, name, dropout_rate=0.0, dilation_rate=1):
    residual = x
    x = tf.keras.layers.GroupNormalization(groups=min(8, channels), epsilon=1e-5, name=f"{name}_gn1")(x)
    x = tf.keras.layers.Activation("swish", name=f"{name}_act1")(x)
    x = tf.keras.layers.Conv2D(channels, 3, padding="same", dilation_rate=dilation_rate, name=f"{name}_conv1")(x)
    x = add_time_bias(x, time_embedding, channels, name)
    x = tf.keras.layers.GroupNormalization(groups=min(8, channels), epsilon=1e-5, name=f"{name}_gn2")(x)
    x = tf.keras.layers.Activation("swish", name=f"{name}_act2")(x)
    if dropout_rate > 0.0:
        x = tf.keras.layers.Dropout(dropout_rate, name=f"{name}_dropout")(x)
    x = tf.keras.layers.Conv2D(
        channels, 3, padding="same", dilation_rate=dilation_rate,
        kernel_initializer="zeros", name=f"{name}_conv2"
    )(x)
    if residual.shape[-1] != channels:
        residual = tf.keras.layers.Conv2D(channels, 1, padding="same", name=f"{name}_skip")(residual)
    return tf.keras.layers.Add(name=f"{name}_out")([x, residual])


def build_denoiser(input_shape, channels, output_channels):
    noisy_inputs = tf.keras.layers.Input(shape=input_shape, name="noisy_inputs")
    timestep_inputs = tf.keras.layers.Input(shape=(), dtype=tf.int32, name="timesteps")

    time_embedding = TimeEmbedding(TIME_EMBED_DIM)
    time_emb = time_embedding(timestep_inputs)

    x = tf.keras.layers.Conv2D(channels, 3, padding="same", name="input_conv")(noisy_inputs)

    levels = max(3, min(5, int(math.log2(input_shape[0])) - 2))
    widths = [channels * multiplier for multiplier in (1, 2, 4, 4, 6)[:levels]]
    skips = []
    for level, width in enumerate(widths):
        x = residual_block(x, time_emb, width, f"down_block{level + 1}_1", DROPOUT_RATE)
        x = residual_block(x, time_emb, width, f"down_block{level + 1}_2", DROPOUT_RATE)
        skips.append(x)
        if level < levels - 1:
            x = tf.keras.layers.Conv2D(widths[level + 1], 3, strides=2, padding="same", name=f"downsample{level + 1}")(x)
    for index, dilation in enumerate((1, 2, 4, 1), start=1):
        x = residual_block(
            x, time_emb, widths[-1], f"mid_block{index}", DROPOUT_RATE,
            dilation_rate=dilation,
        )
    for level in range(levels - 2, -1, -1):
        x = tf.keras.layers.UpSampling2D(interpolation="bilinear", name=f"upsample{level + 1}")(x)
        x = tf.keras.layers.Conv2D(widths[level], 3, padding="same", name=f"upsample_conv{level + 1}")(x)
        x = tf.keras.layers.Concatenate(name=f"skip_concat{level + 1}")([x, skips[level]])
        x = residual_block(x, time_emb, widths[level], f"up_block{level + 1}_1", DROPOUT_RATE)
        x = residual_block(x, time_emb, widths[level], f"up_block{level + 1}_2", DROPOUT_RATE)

    x = tf.keras.layers.GroupNormalization(groups=8, epsilon=1e-5, name="output_gn")(x)
    x = tf.keras.layers.Activation("swish", name="output_act")(x)
    outputs = tf.keras.layers.Conv2D(output_channels, 3, padding="same", kernel_initializer="zeros", name="output_conv")(x)

    return tf.keras.Model(
        inputs=[noisy_inputs, timestep_inputs],
        outputs=outputs,
        name="denoiser",
    )


class DDPMTrainer:
    def __init__(self, strategy, input_shape):
        self.strategy = strategy
        self.input_shape_value = input_shape
        self._interrupt_requested = False
        # A linear 50-step schedule leaves x_T about 60% signal, while sampling
        # starts from pure noise.  The cosine schedule reaches noise at x_T and
        # is well behaved with the deliberately small number of steps used here.
        schedule_steps = tf.linspace(0.0, 1.0, DIFFUSION_STEPS + 1)
        alpha_bars = tf.cos((schedule_steps + 0.008) / 1.008 * math.pi / 2.0) ** 2
        alpha_bars = alpha_bars / alpha_bars[0]
        self.betas = tf.clip_by_value(1.0 - alpha_bars[1:] / alpha_bars[:-1], 1e-4, 0.999)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = tf.math.cumprod(self.alphas)

        with self.strategy.scope():
            self.denoiser = build_denoiser(input_shape, BASE_CHANNELS, input_shape[-1])
            self.denoiser_optimizer = tf.keras.optimizers.AdamW(
                DENOISER_LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                clipnorm=1.0,
            )
            self.mse = tf.keras.losses.MeanSquaredError()
            self.denoiser_optimizer.build(self.denoiser.trainable_variables)
        self.denoiser.summary(expand_nested=True, show_trainable=True)

    def sample_timesteps(self, batch_size):
        return tf.random.uniform((batch_size,), minval=0, maxval=DIFFUSION_STEPS, dtype=tf.int32)

    def q_sample(self, x_start, timesteps, noise):
        alpha_bar = tf.gather(self.alpha_bars, timesteps)
        while len(alpha_bar.shape) < len(x_start.shape):
            alpha_bar = alpha_bar[..., None]
        return tf.sqrt(alpha_bar) * x_start + tf.sqrt(1.0 - alpha_bar) * noise

    @tf.function
    def train_diffusion_step(self, images):
        batch_size = tf.shape(images)[0]
        noise = tf.random.normal(tf.shape(images))
        timesteps = self.sample_timesteps(batch_size)
        noisy_inputs = self.q_sample(images, timesteps, noise)
        with tf.GradientTape() as tape:
            pred = self.denoiser([noisy_inputs, timesteps], training=True)
            loss = self.mse(noise, pred)
        denoiser_grads = tape.gradient(loss, self.denoiser.trainable_variables)
        self.denoiser_optimizer.apply_gradients(zip(denoiser_grads, self.denoiser.trainable_variables))
        return loss

    @tf.function
    def distributed_step(self, images):
        per_replica = self.strategy.run(self.train_diffusion_step, args=(images,))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica, axis=None)

    def p_sample(self, sample, timesteps):
        """Draw one exact DDPM reverse transition, x_t -> x_(t-1)."""
        pred = self.denoiser([sample, timesteps], training=False)
        alpha = tf.gather(self.alphas, timesteps)
        alpha_bar = tf.gather(self.alpha_bars, timesteps)
        beta = tf.gather(self.betas, timesteps)
        previous_alpha_bar = tf.where(
            timesteps > 0,
            tf.gather(self.alpha_bars, tf.maximum(timesteps - 1, 0)),
            tf.ones_like(alpha_bar),
        )

        def expand(values):
            while len(values.shape) < len(sample.shape):
                values = values[..., None]
            return values

        alpha = expand(alpha)
        alpha_bar = expand(alpha_bar)
        previous_alpha_bar = expand(previous_alpha_bar)
        beta = expand(beta)

        predicted_x0 = (sample - tf.sqrt(1.0 - alpha_bar) * pred) / tf.sqrt(alpha_bar)
        predicted_x0 = tf.clip_by_value(predicted_x0, -1.0, 1.0)
        mean = (
            beta * tf.sqrt(previous_alpha_bar) / (1.0 - alpha_bar) * predicted_x0
            + tf.sqrt(alpha) * (1.0 - previous_alpha_bar) / (1.0 - alpha_bar) * sample
        )
        posterior_variance = beta * (1.0 - previous_alpha_bar) / (1.0 - alpha_bar)
        noise = tf.random.normal(tf.shape(sample))
        has_previous_step = expand(tf.cast(timesteps > 0, sample.dtype))
        return mean + has_previous_step * tf.sqrt(tf.maximum(posterior_variance, 0.0)) * noise

    def reverse_timesteps(self):
        return range(DIFFUSION_STEPS - 1, -1, -1)

    def generate_samples(self, sample_count=4):
        def generate_replica_samples():
            sample = tf.random.normal((samples_per_replica, *self.input_shape_value))
            for step in self.reverse_timesteps():
                timesteps = tf.fill((samples_per_replica,), step)
                sample = self.p_sample(sample, timesteps)
            return sample

        replica_count = self.strategy.num_replicas_in_sync
        samples_per_replica = math.ceil(sample_count / replica_count)
        if replica_count > 1:
            replica_samples = self.strategy.run(generate_replica_samples)
            sample = tf.concat(self.strategy.experimental_local_results(replica_samples), axis=0)[:sample_count]
        else:
            sample = generate_replica_samples()[:sample_count]
        return tf.clip_by_value(sample, -1.0, 1.0)

    def fit(self, dataset, epochs, initial_epoch=0, callbacks=None, steps_per_epoch=None):
        callbacks = callbacks or []
        self._interrupt_requested = False
        progress = tf.keras.callbacks.ProgbarLogger()
        progress.set_model(self)
        progress.set_params({"verbose": 1, "epochs": epochs, "steps": steps_per_epoch})
        progress.on_train_begin()
        for callback in callbacks:
            callback.set_model(self)
            callback.set_params({"epochs": epochs, "initial_epoch": initial_epoch, "steps": steps_per_epoch})
            callback.on_train_begin()
        for epoch in range(initial_epoch, epochs):
            epoch_loss = tf.keras.metrics.Mean()
            progress.on_epoch_begin(epoch)
            for callback in callbacks:
                callback.on_epoch_begin(epoch)
            for step, batch in enumerate(dataset):
                loss = self.distributed_step(batch)
                epoch_loss.update_state(loss)
                logs = {"loss": float(loss.numpy())}
                progress.on_train_batch_end(step, {"loss": logs["loss"]})
                for callback in callbacks:
                    callback.on_train_batch_end(step, logs)
                if self._interrupt_requested:
                    raise KeyboardInterrupt
            epoch_logs = {"loss": float(epoch_loss.result().numpy())}
            progress.on_epoch_end(epoch, {"loss": epoch_logs["loss"]})
            for callback in callbacks:
                callback.on_epoch_end(epoch, epoch_logs)
            if self._interrupt_requested:
                raise KeyboardInterrupt
        for callback in callbacks:
            callback.on_train_end()
        progress.on_train_end()
