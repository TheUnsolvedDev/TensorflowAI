import sys

import tensorflow as tf

from config import *


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


class TextEncoder(tf.keras.Model):
    def __init__(self, vocab_size=TEXT_VOCAB_SIZE, seq_len=TEXT_SEQUENCE_LENGTH, embed_dim=EMBED_DIM):
        super().__init__()
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.position_embedding = tf.keras.layers.Embedding(seq_len, embed_dim)
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=max(embed_dim // 4, 1))
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(embed_dim * 2, activation="gelu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.pool = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, tokens, training=False):
        positions = tf.range(tf.shape(tokens)[1])
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        attn_out = self.attn(x, x, training=training)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x, training=training))
        return self.pool(x)


class SimpleAutoencoder(tf.keras.Model):
    def __init__(self, latent_channels=4):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer((IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
            tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="swish"),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="swish"),
            tf.keras.layers.Conv2D(latent_channels, 3, padding="same"),
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer((LATENT_IMAGE_SIZE[0], LATENT_IMAGE_SIZE[1], latent_channels)),
            tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding="same", activation="swish"),
            tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="swish"),
            tf.keras.layers.Conv2D(3, 3, padding="same", activation="tanh"),
        ])

    def call(self, images, training=False):
        latents = self.encoder(images, training=training)
        recon = self.decoder(latents, training=training)
        return latents, recon


class CLIPLikeEncoder(tf.keras.Model):
    def __init__(self, embedding_dim=EMBED_DIM):
        super().__init__()
        self.image_proj = tf.keras.Sequential([
            tf.keras.layers.InputLayer((IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
            tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="swish"),
            tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="swish"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(embedding_dim),
        ])
        self.text_proj = TextEncoder(embed_dim=embedding_dim)
        self.image_head = tf.keras.layers.Dense(embedding_dim)
        self.text_head = tf.keras.layers.Dense(embedding_dim)

    def encode_image(self, images, training=False):
        return tf.math.l2_normalize(self.image_head(self.image_proj(images, training=training)), axis=-1)

    def encode_text(self, tokens, training=False):
        return tf.math.l2_normalize(self.text_head(self.text_proj(tokens, training=training)), axis=-1)


class DiffusionBackbone(tf.keras.Model):
    def __init__(self, channels, output_channels, conditioning_dim=0, use_control=False):
        super().__init__()
        self.channels = channels
        self.time_embedding = TimeEmbedding(TIME_EMBED_DIM)
        self.condition_projection = tf.keras.layers.Dense(channels) if conditioning_dim else None
        self.control_projection = tf.keras.layers.Conv2D(channels, 1, padding="same") if use_control else None
        self.input_projection = tf.keras.layers.Conv2D(channels, 3, padding="same")
        self.down1 = tf.keras.layers.Conv2D(channels, 3, padding="same", activation="swish")
        self.down2 = tf.keras.layers.Conv2D(channels * 2, 3, strides=2, padding="same", activation="swish")
        self.mid = tf.keras.layers.Conv2D(channels * 2, 3, padding="same", activation="swish")
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=max(channels // 2, 1))
        self.up = tf.keras.layers.Conv2DTranspose(channels, 4, strides=2, padding="same", activation="swish")
        self.out = tf.keras.layers.Conv2D(output_channels, 3, padding="same")

    def call(self, noisy_inputs, timesteps, cond_vector=None, control=None, training=False):
        time_emb = self.time_embedding(timesteps)
        time_map = tf.reshape(time_emb, (-1, 1, 1, TIME_EMBED_DIM))
        x = self.input_projection(noisy_inputs)
        if self.condition_projection is not None and cond_vector is not None:
            cond_map = tf.reshape(self.condition_projection(cond_vector), (-1, 1, 1, self.channels))
            x = x + cond_map
        if self.control_projection is not None and control is not None:
            control_resized = tf.image.resize(control, tf.shape(noisy_inputs)[1:3])
            x = x + self.control_projection(control_resized)
        x = x + tf.image.resize(time_map, tf.shape(x)[1:3])
        skip = self.down1(x)
        x = self.down2(skip)
        x = self.mid(x)
        shape = tf.shape(x)
        flat = tf.reshape(x, (shape[0], shape[1] * shape[2], shape[3]))
        flat = self.attn(flat, flat, training=training)
        x = tf.reshape(flat, shape)
        x = self.up(x)
        x = tf.concat([x, skip], axis=-1)
        return self.out(x)


class BaseDiffusionModel(tf.keras.Model):
    def __init__(self, strategy, input_shape, batch_size, use_text=False, use_latent=False, use_control=False, use_clip=False, use_consistency=False, model_label=MODEL_NAME):
        super().__init__()
        self.strategy = strategy
        self.input_shape_value = input_shape
        self.batch_size = batch_size
        self.use_text = use_text
        self.use_latent = use_latent
        self.use_control = use_control
        self.use_clip = use_clip
        self.use_consistency = use_consistency
        self.model_label = model_label
        self.global_batch_size = batch_size * self.strategy.num_replicas_in_sync
        self.betas = tf.cast(tf.linspace(BETA_START, BETA_END, DIFFUSION_STEPS), tf.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = tf.math.cumprod(self.alphas)

        with self.strategy.scope():
            self.text_encoder = TextEncoder() if self.use_text else None
            self.text_optimizer = tf.keras.optimizers.Adam(TEXT_ENCODER_LEARNING_RATE) if self.use_text else None
            self.autoencoder = SimpleAutoencoder() if (self.use_latent or self.model_label in {"StableDiffusion", "ControlNet"}) else None
            self.autoencoder_optimizer = tf.keras.optimizers.Adam(AUTOENCODER_LEARNING_RATE) if self.autoencoder is not None else None
            self.clip_encoder = CLIPLikeEncoder() if self.use_clip else None
            self.clip_optimizer = tf.keras.optimizers.Adam(TEXT_ENCODER_LEARNING_RATE) if self.use_clip else None
            self.prior_projection = None
            self.prior_optimizer = None
            if self.use_clip:
                self.prior_projection = tf.keras.Sequential([
                    tf.keras.layers.InputLayer((EMBED_DIM,)),
                    tf.keras.layers.Dense(EMBED_DIM, activation="swish"),
                    tf.keras.layers.Dense(EMBED_DIM),
                ])
                self.prior_optimizer = tf.keras.optimizers.Adam(PRIOR_LEARNING_RATE)
            output_channels = input_shape[-1] * (2 if USE_LEARNED_VARIANCE else 1)
            conditioning_dim = EMBED_DIM if (self.use_text or self.use_clip) else 0
            self.denoiser = DiffusionBackbone(BASE_CHANNELS, output_channels, conditioning_dim=conditioning_dim, use_control=self.use_control)
            self.denoiser_optimizer = tf.keras.optimizers.Adam(DENOISER_LEARNING_RATE)
            self.consistency_model = DiffusionBackbone(BASE_CHANNELS, input_shape[-1]) if self.use_consistency else None
            self.consistency_optimizer = tf.keras.optimizers.Adam(CONSISTENCY_LEARNING_RATE) if self.use_consistency else None
            self.mse = tf.keras.losses.MeanSquaredError()
            self.mae = tf.keras.losses.MeanAbsoluteError()
            self._initialize_components()
            self._print_model_summaries()

    def _initialize_components(self):
        image_batch = tf.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32)
        base_batch = tf.zeros((1, *self.input_shape_value), dtype=tf.float32)
        timesteps = tf.zeros((1,), dtype=tf.int32)
        cond_vector = None

        if self.text_encoder is not None:
            token_batch = tf.zeros((1, TEXT_SEQUENCE_LENGTH), dtype=tf.int32)
            cond_vector = self.text_encoder(token_batch, training=False)
            self.text_optimizer.build(self.text_encoder.trainable_variables)

        if self.autoencoder is not None:
            latent_batch, _ = self.autoencoder(image_batch, training=False)
            self.autoencoder_optimizer.build(self.autoencoder.trainable_variables)
        else:
            latent_batch = None

        if self.clip_encoder is not None:
            token_batch = tf.zeros((1, TEXT_SEQUENCE_LENGTH), dtype=tf.int32)
            self.clip_encoder.encode_image(image_batch, training=False)
            self.clip_encoder.encode_text(token_batch, training=False)
            self.clip_optimizer.build(self.clip_encoder.trainable_variables)
            self.prior_projection(tf.zeros((1, EMBED_DIM), dtype=tf.float32), training=False)
            self.prior_optimizer.build(self.prior_projection.trainable_variables)
            cond_vector = self.clip_encoder.encode_text(token_batch, training=False)

        denoiser_inputs = latent_batch if latent_batch is not None else base_batch
        self.denoiser(denoiser_inputs, timesteps, cond_vector=cond_vector, training=False)
        self.denoiser_optimizer.build(self.denoiser.trainable_variables)

        if self.consistency_model is not None:
            self.consistency_model(base_batch, timesteps, training=False)
            self.consistency_optimizer.build(self.consistency_model.trainable_variables)

    def _print_model_summaries(self):
        if self.text_encoder is not None:
            self.text_encoder.summary(expand_nested=True)
        if self.autoencoder is not None:
            self.autoencoder.summary(expand_nested=True)
        if self.clip_encoder is not None:
            self.clip_encoder.summary(expand_nested=True)
        if self.prior_projection is not None:
            self.prior_projection.summary(expand_nested=True)
        self.denoiser.summary(expand_nested=True)
        if self.consistency_model is not None:
            self.consistency_model.summary(expand_nested=True)

    def sample_timesteps(self, batch_size):
        return tf.random.uniform((batch_size,), minval=0, maxval=DIFFUSION_STEPS, dtype=tf.int32)

    def q_sample(self, x_start, timesteps, noise):
        alpha_bar = tf.gather(self.alpha_bars, timesteps)
        while len(alpha_bar.shape) < len(x_start.shape):
            alpha_bar = alpha_bar[..., None]
        return tf.sqrt(alpha_bar) * x_start + tf.sqrt(1.0 - alpha_bar) * noise

    def prepare_condition(self, batch, training=False):
        cond_vector = None
        control = batch.get("control") if isinstance(batch, dict) else None
        if self.use_text and isinstance(batch, dict):
            cond_vector = self.text_encoder(batch["tokens"], training=training)
            if USE_CLASSIFIER_FREE_GUIDANCE and training:
                mask = tf.cast(tf.random.uniform((tf.shape(cond_vector)[0], 1)) > GUIDANCE_DROPOUT, tf.float32)
                cond_vector = cond_vector * mask
        if self.use_clip and isinstance(batch, dict):
            cond_vector = self.clip_encoder.encode_text(batch["tokens"], training=training)
        return cond_vector, control

    def get_training_images(self, batch, training=False):
        if isinstance(batch, dict):
            images = batch.get("image")
        else:
            images = batch
        recon = None
        if self.autoencoder is not None and images is not None and training:
            latents, recon = self.autoencoder(images, training=True)
            return latents, recon
        if self.autoencoder is not None and images is not None and not training:
            latents, recon = self.autoencoder(images, training=False)
            return latents, recon
        return images, recon

    @tf.function
    def train_autoencoder_step(self, images):
        with tf.GradientTape() as tape:
            latents, recon = self.autoencoder(images, training=True)
            loss = self.mse(images, recon) + 1e-3 * tf.reduce_mean(tf.square(latents))
        grads = tape.gradient(loss, self.autoencoder.trainable_variables)
        self.autoencoder_optimizer.apply_gradients(zip(grads, self.autoencoder.trainable_variables))
        return loss

    @tf.function
    def train_alignment_step(self, batch):
        images = batch["image"]
        tokens = batch["tokens"]
        with tf.GradientTape() as tape:
            image_embed = self.clip_encoder.encode_image(images, training=True)
            text_embed = self.clip_encoder.encode_text(tokens, training=True)
            logits = tf.matmul(image_embed, text_embed, transpose_b=True)
            labels = tf.range(tf.shape(logits)[0])
            loss_i = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
            loss_t = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(logits), from_logits=True))
            loss = 0.5 * (loss_i + loss_t)
        grads = tape.gradient(loss, self.clip_encoder.trainable_variables)
        self.clip_optimizer.apply_gradients(zip(grads, self.clip_encoder.trainable_variables))
        return loss

    @tf.function
    def train_prior_step(self, batch):
        images = batch["image"]
        tokens = batch["tokens"]
        with tf.GradientTape() as tape:
            image_embed = tf.stop_gradient(self.clip_encoder.encode_image(images, training=False))
            text_embed = self.clip_encoder.encode_text(tokens, training=False)
            pred_embed = self.prior_projection(text_embed, training=True)
            loss = self.mse(image_embed, pred_embed)
        grads = tape.gradient(loss, self.prior_projection.trainable_variables)
        self.prior_optimizer.apply_gradients(zip(grads, self.prior_projection.trainable_variables))
        return loss

    @tf.function
    def train_diffusion_step(self, batch, stage="diffusion"):
        cond_vector = None
        control = None
        if stage == "sr":
            target_images = batch["high_res"]
            cond_vector = tf.reshape(tf.reduce_mean(batch["low_res"], axis=[1, 2]), (tf.shape(batch["low_res"])[0], -1))
        elif self.autoencoder is not None and stage in {"diffusion", "control"}:
            target_images, _ = self.get_training_images(batch, training=True)
            cond_vector, control = self.prepare_condition(batch, training=True)
        else:
            target_images = batch["image"] if isinstance(batch, dict) else batch
            cond_vector, control = self.prepare_condition(batch, training=True)
        batch_size = tf.shape(target_images)[0]
        noise = tf.random.normal(tf.shape(target_images))
        timesteps = self.sample_timesteps(batch_size)
        noisy_inputs = self.q_sample(target_images, timesteps, noise)
        with tf.GradientTape(persistent=True) as tape:
            pred = self.denoiser(noisy_inputs, timesteps, cond_vector=cond_vector, control=control, training=True)
            if USE_LEARNED_VARIANCE:
                pred_noise, pred_var = tf.split(pred, 2, axis=-1)
                loss = self.mse(noise, pred_noise) + 1e-3 * tf.reduce_mean(tf.square(pred_var))
            else:
                loss = self.mse(noise, pred)
            if self.text_encoder is not None and cond_vector is not None:
                loss += 1e-3 * tf.reduce_mean(tf.square(cond_vector))
        denoiser_grads = tape.gradient(loss, self.denoiser.trainable_variables)
        self.denoiser_optimizer.apply_gradients(zip(denoiser_grads, self.denoiser.trainable_variables))
        if self.text_encoder is not None and cond_vector is not None:
            text_grads = tape.gradient(loss, self.text_encoder.trainable_variables)
            self.text_optimizer.apply_gradients(zip(text_grads, self.text_encoder.trainable_variables))
        del tape
        return loss

    @tf.function
    def train_consistency_step(self, images):
        batch_size = tf.shape(images)[0]
        noise = tf.random.normal(tf.shape(images))
        timesteps = self.sample_timesteps(batch_size)
        noisy_inputs = self.q_sample(images, timesteps, noise)
        teacher_pred = tf.stop_gradient(self.denoiser(noisy_inputs, timesteps, training=False))
        with tf.GradientTape() as tape:
            student_pred = self.consistency_model(noisy_inputs, timesteps, training=True)
            loss = self.mae(teacher_pred, student_pred)
        grads = tape.gradient(loss, self.consistency_model.trainable_variables)
        self.consistency_optimizer.apply_gradients(zip(grads, self.consistency_model.trainable_variables))
        return loss

    @tf.function
    def distributed_step(self, batch, stage="diffusion"):
        per_replica = self.strategy.run(self.train_diffusion_step, args=(batch, stage))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica, axis=None)

    @tf.function
    def distributed_autoencoder_step(self, images):
        per_replica = self.strategy.run(self.train_autoencoder_step, args=(images,))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica, axis=None)

    @tf.function
    def distributed_alignment_step(self, batch):
        per_replica = self.strategy.run(self.train_alignment_step, args=(batch,))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica, axis=None)

    @tf.function
    def distributed_prior_step(self, batch):
        per_replica = self.strategy.run(self.train_prior_step, args=(batch,))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica, axis=None)

    @tf.function
    def distributed_consistency_step(self, images):
        per_replica = self.strategy.run(self.train_consistency_step, args=(images,))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica, axis=None)

    def p_sample(self, sample, timesteps, cond_vector=None, control=None):
        pred = self.denoiser(sample, timesteps, cond_vector=cond_vector, control=control, training=False)
        if USE_LEARNED_VARIANCE:
            pred, _ = tf.split(pred, 2, axis=-1)
        alpha = tf.gather(self.alphas, timesteps)
        alpha_bar = tf.gather(self.alpha_bars, timesteps)
        beta = tf.gather(self.betas, timesteps)
        while len(alpha.shape) < len(sample.shape):
            alpha = alpha[..., None]
            alpha_bar = alpha_bar[..., None]
            beta = beta[..., None]
        mean = (sample - (beta / tf.sqrt(1.0 - alpha_bar)) * pred) / tf.sqrt(alpha)
        if USE_DDIM_SAMPLER:
            return mean
        noise = tf.random.normal(tf.shape(sample))
        return mean + tf.sqrt(beta) * noise

    def generate_samples(self, sample_count=4, cond_tokens=None, control=None, stage="diffusion"):
        if stage == "sample_consistency" and self.consistency_model is not None:
            sample = tf.random.normal((sample_count, *self.input_shape_value))
            timesteps = tf.fill((sample_count,), DIFFUSION_STEPS - 1)
            outputs = self.consistency_model(sample, timesteps, training=False)
            return tf.clip_by_value(outputs, -1.0, 1.0)
        cond_vector = None
        if self.text_encoder is not None and cond_tokens is not None:
            cond_vector = self.text_encoder(cond_tokens, training=False)
        if self.clip_encoder is not None and cond_tokens is not None and cond_vector is None:
            text_embed = self.clip_encoder.encode_text(cond_tokens, training=False)
            cond_vector = self.prior_projection(text_embed, training=False)
        sample = tf.random.normal((sample_count, *self.input_shape_value))
        for step in tf.linspace(float(DIFFUSION_STEPS - 1), 0.0, SAMPLING_STEPS):
            timesteps = tf.fill((sample_count,), tf.cast(step, tf.int32))
            sample = self.p_sample(sample, timesteps, cond_vector=cond_vector, control=control)
        if self.autoencoder is not None and stage in {"diffusion", "control"}:
            sample = self.autoencoder.decoder(sample, training=False)
        return tf.clip_by_value(sample, -1.0, 1.0)

    def fit(self, dataset, epochs, initial_epoch=0, callbacks=None, stage="diffusion"):
        callbacks = callbacks or []
        try:
            steps_per_epoch = len(dataset)
        except TypeError:
            steps_per_epoch = None
        for callback in callbacks:
            callback.set_model(self)
            callback.set_params({"epochs": epochs, "initial_epoch": initial_epoch, "steps": steps_per_epoch})
            callback.on_train_begin()
        for epoch in range(initial_epoch, epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            progress = tf.keras.utils.Progbar(target=steps_per_epoch, verbose=1)
            for callback in callbacks:
                callback.on_epoch_begin(epoch)
            for step, batch in enumerate(dataset):
                if stage == "vae":
                    source = batch["image"] if isinstance(batch, dict) else batch
                    loss = self.distributed_autoencoder_step(source)
                elif stage == "align":
                    loss = self.distributed_alignment_step(batch)
                elif stage == "prior":
                    loss = self.distributed_prior_step(batch)
                elif stage == "consistency":
                    source = batch["image"] if isinstance(batch, dict) else batch
                    loss = self.distributed_consistency_step(source)
                else:
                    loss = self.distributed_step(batch, stage=stage)
                logs = {"loss": float(loss.numpy()), "stage": stage}
                progress.update(step + 1, values=[("loss", logs["loss"])])
                for callback in callbacks:
                    callback.on_train_batch_end(step, logs)
            epoch_logs = {"loss": float(loss.numpy()), "stage": stage}
            for callback in callbacks:
                callback.on_epoch_end(epoch, epoch_logs)
        for callback in callbacks:
            callback.on_train_end()
