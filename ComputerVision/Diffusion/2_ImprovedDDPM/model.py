import math

import tensorflow as tf

from config import (
    BASE_CHANNELS,
    DENOISER_LEARNING_RATE,
    DIFFUSION_STEPS,
    DROPOUT_RATE,
    HYBRID_VLB_WEIGHT,
    SAMPLER_EMA_DECAY,
    SAMPLING_STEPS,
    TIME_EMBED_DIM,
    WEIGHT_DECAY,
)

LOG_2 = math.log(2.0)


class TimeEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj1 = tf.keras.layers.Dense(dim * 2, activation="swish")
        self.proj2 = tf.keras.layers.Dense(dim)

    def call(self, timesteps):
        half = max(self.dim // 2, 1)
        frequencies = tf.exp(
            -tf.math.log(10000.0) * tf.range(half, dtype=tf.float32)
            / max(float(half - 1), 1.0)
        )
        angles = tf.cast(timesteps[:, None], tf.float32) * frequencies[None, :]
        embedding = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)
        return self.proj2(self.proj1(embedding))


def residual_block(x, time_embedding, channels, name, dilation_rate=1):
    residual = x
    x = tf.keras.layers.GroupNormalization(
        groups=min(8, channels), epsilon=1e-5, name=f"{name}_gn1"
    )(x)
    x = tf.keras.layers.Activation("swish", name=f"{name}_act1")(x)
    x = tf.keras.layers.Conv2D(
        channels, 3, padding="same", dilation_rate=dilation_rate,
        name=f"{name}_conv1",
    )(x)

    scale_shift = tf.keras.layers.Dense(
        channels * 2, name=f"{name}_time_scale_shift"
    )(time_embedding)
    scale, shift = tf.keras.layers.Lambda(
        lambda values: tf.split(values, 2, axis=-1), name=f"{name}_time_split"
    )(scale_shift)
    scale = tf.keras.layers.Reshape((1, 1, channels), name=f"{name}_scale")(scale)
    shift = tf.keras.layers.Reshape((1, 1, channels), name=f"{name}_shift")(shift)

    x = tf.keras.layers.GroupNormalization(
        groups=min(8, channels), epsilon=1e-5, name=f"{name}_gn2"
    )(x)
    x = tf.keras.layers.Add(name=f"{name}_shift_add")([
        tf.keras.layers.Multiply(name=f"{name}_scale_multiply")([x, scale + 1.0]),
        shift,
    ])
    x = tf.keras.layers.Activation("swish", name=f"{name}_act2")(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE, name=f"{name}_dropout")(x)
    x = tf.keras.layers.Conv2D(
        channels, 3, padding="same", dilation_rate=dilation_rate,
        kernel_initializer="zeros", name=f"{name}_conv2",
    )(x)
    if residual.shape[-1] != channels:
        residual = tf.keras.layers.Conv2D(
            channels, 1, padding="same", name=f"{name}_skip"
        )(residual)
    return tf.keras.layers.Add(name=f"{name}_out")([x, residual])


def build_denoiser(input_shape):
    noisy_inputs = tf.keras.layers.Input(shape=input_shape, name="noisy_inputs")
    timestep_inputs = tf.keras.layers.Input(shape=(), dtype=tf.int32, name="timesteps")
    time_embedding = TimeEmbedding(TIME_EMBED_DIM)(timestep_inputs)
    x = tf.keras.layers.Conv2D(
        BASE_CHANNELS, 3, padding="same", name="input_conv"
    )(noisy_inputs)

    levels = max(3, min(5, int(math.log2(input_shape[0])) - 2))
    widths = [BASE_CHANNELS * multiplier for multiplier in (1, 2, 4, 4, 6)[:levels]]
    skips = []
    for level, width in enumerate(widths):
        x = residual_block(x, time_embedding, width, f"down_block{level + 1}_1")
        x = residual_block(x, time_embedding, width, f"down_block{level + 1}_2")
        skips.append(x)
        if level < levels - 1:
            x = tf.keras.layers.Conv2D(
                widths[level + 1], 3, strides=2, padding="same",
                name=f"downsample{level + 1}",
            )(x)

    for index, dilation in enumerate((1, 2, 4, 1), start=1):
        x = residual_block(
            x, time_embedding, widths[-1], f"mid_block{index}", dilation
        )

    for level in range(levels - 2, -1, -1):
        x = tf.keras.layers.UpSampling2D(
            interpolation="bilinear", name=f"upsample{level + 1}"
        )(x)
        x = tf.keras.layers.Conv2D(
            widths[level], 3, padding="same", name=f"upsample_conv{level + 1}"
        )(x)
        x = tf.keras.layers.Concatenate(name=f"skip_concat{level + 1}")(
            [x, skips[level]]
        )
        x = residual_block(x, time_embedding, widths[level], f"up_block{level + 1}_1")
        x = residual_block(x, time_embedding, widths[level], f"up_block{level + 1}_2")

    x = tf.keras.layers.GroupNormalization(
        groups=8, epsilon=1e-5, name="output_gn"
    )(x)
    x = tf.keras.layers.Activation("swish", name="output_act")(x)
    outputs = tf.keras.layers.Conv2D(
        input_shape[-1] * 2, 3, padding="same", kernel_initializer="zeros",
        name="noise_and_variance",
    )(x)
    return tf.keras.Model(
        inputs=[noisy_inputs, timestep_inputs], outputs=outputs, name="improved_denoiser"
    )


def mean_flat(values):
    return tf.reduce_mean(values, axis=tf.range(1, tf.rank(values)))


def normal_kl(mean1, log_variance1, mean2, log_variance2):
    return 0.5 * (
        -1.0 + log_variance2 - log_variance1
        + tf.exp(log_variance1 - log_variance2)
        + tf.square(mean1 - mean2) * tf.exp(-log_variance2)
    )


def approximate_standard_normal_cdf(values):
    return 0.5 * (
        1.0 + tf.tanh(tf.sqrt(2.0 / math.pi) * (values + 0.044715 * values ** 3))
    )


def discretized_gaussian_log_likelihood(images, means, log_variances):
    centered = images - means
    inverse_std = tf.exp(-0.5 * log_variances)
    plus = inverse_std * (centered + 1.0 / 255.0)
    minus = inverse_std * (centered - 1.0 / 255.0)
    cdf_plus = approximate_standard_normal_cdf(plus)
    cdf_minus = approximate_standard_normal_cdf(minus)
    log_cdf_plus = tf.math.log(tf.maximum(cdf_plus, 1e-12))
    log_one_minus_cdf_minus = tf.math.log(tf.maximum(1.0 - cdf_minus, 1e-12))
    log_cdf_delta = tf.math.log(tf.maximum(cdf_plus - cdf_minus, 1e-12))
    return tf.where(
        images < -0.999,
        log_cdf_plus,
        tf.where(images > 0.999, log_one_minus_cdf_minus, log_cdf_delta),
    )


class ImprovedDDPMTrainer:
    def __init__(self, strategy, input_shape):
        self.strategy = strategy
        self.input_shape_value = input_shape
        self._interrupt_requested = False

        steps = tf.linspace(0.0, 1.0, DIFFUSION_STEPS + 1)
        alpha_bars = tf.cos((steps + 0.008) / 1.008 * math.pi / 2.0) ** 2
        alpha_bars /= alpha_bars[0]
        self.betas = tf.clip_by_value(
            1.0 - alpha_bars[1:] / alpha_bars[:-1], 1e-4, 0.999
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = tf.math.cumprod(self.alphas)
        self.alpha_bars_previous = tf.concat(
            [tf.ones((1,), tf.float32), self.alpha_bars[:-1]], axis=0
        )
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bars_previous) / (1.0 - self.alpha_bars)
        )
        self.posterior_log_variance = tf.math.log(
            tf.maximum(self.posterior_variance, 1e-20)
        )
        self.posterior_x0_coefficient = (
            self.betas * tf.sqrt(self.alpha_bars_previous) / (1.0 - self.alpha_bars)
        )
        self.posterior_xt_coefficient = (
            tf.sqrt(self.alphas) * (1.0 - self.alpha_bars_previous)
            / (1.0 - self.alpha_bars)
        )

        with strategy.scope():
            self.denoiser = build_denoiser(input_shape)
            self.optimizer = tf.keras.optimizers.AdamW(
                DENOISER_LEARNING_RATE, weight_decay=WEIGHT_DECAY, clipnorm=1.0
            )
            self.optimizer.build(self.denoiser.trainable_variables)
            self.loss_second_moment = tf.Variable(
                tf.ones((DIFFUSION_STEPS,), tf.float32), trainable=False,
                name="loss_second_moment",
            )
        self.denoiser.summary(expand_nested=True, show_trainable=True)

    def timestep_probabilities(self):
        weights = tf.sqrt(tf.maximum(self.loss_second_moment, 1e-12))
        return weights / tf.reduce_sum(weights)

    def sample_timesteps(self, batch_size):
        probabilities = self.timestep_probabilities()
        timesteps = tf.cast(
            tf.random.categorical(tf.math.log(probabilities[None, :]), batch_size)[0],
            tf.int32,
        )
        importance_weights = 1.0 / (
            tf.cast(DIFFUSION_STEPS, tf.float32) * tf.gather(probabilities, timesteps)
        )
        return timesteps, importance_weights

    def q_sample(self, images, timesteps, noise):
        alpha_bar = tf.gather(self.alpha_bars, timesteps)[:, None, None, None]
        return tf.sqrt(alpha_bar) * images + tf.sqrt(1.0 - alpha_bar) * noise

    def q_posterior(self, images, noisy_images, timesteps):
        x0_coefficient = tf.gather(
            self.posterior_x0_coefficient, timesteps
        )[:, None, None, None]
        xt_coefficient = tf.gather(
            self.posterior_xt_coefficient, timesteps
        )[:, None, None, None]
        mean = x0_coefficient * images + xt_coefficient * noisy_images
        log_variance = tf.gather(
            self.posterior_log_variance, timesteps
        )[:, None, None, None]
        return mean, log_variance

    def model_statistics(self, noisy_images, timesteps, prediction, stop_mean=False):
        predicted_noise, variance_values = tf.split(prediction, 2, axis=-1)
        if stop_mean:
            predicted_noise = tf.stop_gradient(predicted_noise)
        alpha_bar = tf.gather(self.alpha_bars, timesteps)[:, None, None, None]
        predicted_x0 = (
            noisy_images - tf.sqrt(1.0 - alpha_bar) * predicted_noise
        ) / tf.sqrt(alpha_bar)
        predicted_x0 = tf.clip_by_value(predicted_x0, -1.0, 1.0)
        model_mean, _ = self.q_posterior(predicted_x0, noisy_images, timesteps)

        minimum_log = tf.gather(
            self.posterior_log_variance, timesteps
        )[:, None, None, None]
        maximum_log = tf.math.log(tf.gather(self.betas, timesteps))[:, None, None, None]
        fraction = (tf.tanh(variance_values) + 1.0) * 0.5
        model_log_variance = fraction * maximum_log + (1.0 - fraction) * minimum_log
        return predicted_noise, model_mean, model_log_variance

    def variational_bound(self, images, noisy_images, timesteps, prediction):
        true_mean, true_log_variance = self.q_posterior(images, noisy_images, timesteps)
        _, model_mean, model_log_variance = self.model_statistics(
            noisy_images, timesteps, prediction, stop_mean=True
        )
        kl = mean_flat(normal_kl(
            true_mean, true_log_variance, model_mean, model_log_variance
        )) / LOG_2
        decoder_nll = -mean_flat(discretized_gaussian_log_likelihood(
            images, model_mean, model_log_variance
        )) / LOG_2
        return tf.where(tf.equal(timesteps, 0), decoder_nll, kl)

    @tf.function
    def train_step(self, images):
        timesteps, importance_weights = self.sample_timesteps(tf.shape(images)[0])
        noise = tf.random.normal(tf.shape(images))
        noisy_images = self.q_sample(images, timesteps, noise)
        with tf.GradientTape() as tape:
            prediction = self.denoiser([noisy_images, timesteps], training=True)
            predicted_noise, _, _ = self.model_statistics(
                noisy_images, timesteps, prediction
            )
            simple_loss = mean_flat(tf.square(noise - predicted_noise))
            vlb = self.variational_bound(images, noisy_images, timesteps, prediction)
            raw_loss = simple_loss + HYBRID_VLB_WEIGHT * DIFFUSION_STEPS * vlb
            loss = tf.reduce_mean(importance_weights * raw_loss)
        gradients = tape.gradient(loss, self.denoiser.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.denoiser.trainable_variables))
        return loss, timesteps, tf.stop_gradient(raw_loss)

    @tf.function
    def distributed_step(self, images):
        losses, timesteps, raw_losses = self.strategy.run(self.train_step, args=(images,))
        loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, losses, axis=None)
        timesteps = self.strategy.gather(timesteps, axis=0)
        raw_losses = self.strategy.gather(raw_losses, axis=0)
        squared_loss_sum = tf.math.unsorted_segment_sum(
            tf.square(raw_losses), timesteps, DIFFUSION_STEPS
        )
        counts = tf.math.unsorted_segment_sum(
            tf.ones_like(raw_losses), timesteps, DIFFUSION_STEPS
        )
        observed_moment = squared_loss_sum / tf.maximum(counts, 1.0)
        updated = (
            SAMPLER_EMA_DECAY * self.loss_second_moment
            + (1.0 - SAMPLER_EMA_DECAY) * observed_moment
        )
        self.loss_second_moment.assign(
            tf.where(counts > 0.0, updated, self.loss_second_moment)
        )
        return loss

    def p_sample(self, sample, timestep, previous_timestep):
        batch_size = tf.shape(sample)[0]
        timesteps = tf.fill((batch_size,), timestep)
        prediction = self.denoiser([sample, timesteps], training=False)
        predicted_noise, variance_values = tf.split(prediction, 2, axis=-1)
        alpha_bar = self.alpha_bars[timestep]
        previous_alpha_bar = (
            self.alpha_bars[previous_timestep]
            if previous_timestep >= 0 else tf.constant(1.0, tf.float32)
        )
        predicted_x0 = (
            sample - tf.sqrt(1.0 - alpha_bar) * predicted_noise
        ) / tf.sqrt(alpha_bar)
        predicted_x0 = tf.clip_by_value(predicted_x0, -1.0, 1.0)

        alpha = alpha_bar / previous_alpha_bar
        beta = 1.0 - alpha
        posterior_variance = beta * (1.0 - previous_alpha_bar) / (1.0 - alpha_bar)
        mean = (
            beta * tf.sqrt(previous_alpha_bar) / (1.0 - alpha_bar) * predicted_x0
            + tf.sqrt(alpha) * (1.0 - previous_alpha_bar) / (1.0 - alpha_bar) * sample
        )
        minimum_log = tf.math.log(tf.maximum(posterior_variance, 1e-20))
        maximum_log = tf.math.log(tf.maximum(beta, 1e-20))
        fraction = (tf.tanh(variance_values) + 1.0) * 0.5
        model_log_variance = fraction * maximum_log + (1.0 - fraction) * minimum_log
        if previous_timestep < 0:
            return mean
        return mean + tf.exp(0.5 * model_log_variance) * tf.random.normal(tf.shape(sample))

    def reverse_timestep_pairs(self):
        ascending = [
            round(index * (DIFFUSION_STEPS - 1) / (SAMPLING_STEPS - 1))
            for index in range(SAMPLING_STEPS)
        ]
        descending = list(reversed(ascending))
        return list(zip(descending, descending[1:] + [-1]))

    def generate_samples(self, sample_count=16):
        replica_count = self.strategy.num_replicas_in_sync
        samples_per_replica = math.ceil(sample_count / replica_count)

        def generate_on_replica():
            sample = tf.random.normal((samples_per_replica, *self.input_shape_value))
            for timestep, previous_timestep in self.reverse_timestep_pairs():
                sample = self.p_sample(sample, timestep, previous_timestep)
            return sample

        if replica_count > 1:
            replica_samples = self.strategy.run(generate_on_replica)
            sample = tf.concat(
                self.strategy.experimental_local_results(replica_samples), axis=0
            )
        else:
            sample = generate_on_replica()
        return tf.clip_by_value(sample[:sample_count], -1.0, 1.0)

    def fit(self, dataset, epochs, initial_epoch=0, callbacks=None, steps_per_epoch=None):
        callbacks = callbacks or []
        progress = tf.keras.callbacks.ProgbarLogger()
        progress.set_model(self)
        progress.set_params({"verbose": 1, "epochs": epochs, "steps": steps_per_epoch})
        progress.on_train_begin()
        for callback in callbacks:
            callback.set_model(self)
            callback.set_params({"epochs": epochs, "steps": steps_per_epoch})
            callback.on_train_begin()

        for epoch in range(initial_epoch, epochs):
            self._current_epoch = epoch
            mean_loss = tf.keras.metrics.Mean()
            progress.on_epoch_begin(epoch)
            for callback in callbacks:
                callback.on_epoch_begin(epoch)
            for step, images in enumerate(dataset):
                loss = self.distributed_step(images)
                mean_loss.update_state(loss)
                logs = {"loss": float(loss.numpy())}
                progress.on_train_batch_end(step, logs)
                for callback in callbacks:
                    callback.on_train_batch_end(step, logs)
                if self._interrupt_requested:
                    raise KeyboardInterrupt
            logs = {"loss": float(mean_loss.result().numpy())}
            progress.on_epoch_end(epoch, logs)
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

        for callback in callbacks:
            callback.on_train_end()
        progress.on_train_end()
