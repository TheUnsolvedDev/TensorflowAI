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


def add_time_bias(x, time_embedding, channels, name):
    time_bias = tf.keras.layers.Dense(channels, name=f"{name}_time_dense")(time_embedding)
    time_bias = tf.keras.layers.Reshape((1, 1, channels), name=f"{name}_time_reshape")(time_bias)
    return tf.keras.layers.Add(name=f"{name}_time_add")([x, time_bias])


def residual_block(x, time_embedding, channels, name, dropout_rate=0.0):
    residual = x
    x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"{name}_ln1")(x)
    x = tf.keras.layers.Activation("swish", name=f"{name}_act1")(x)
    x = tf.keras.layers.Conv2D(channels, 3, padding="same", name=f"{name}_conv1")(x)
    x = add_time_bias(x, time_embedding, channels, name)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"{name}_ln2")(x)
    x = tf.keras.layers.Activation("swish", name=f"{name}_act2")(x)
    if dropout_rate > 0.0:
        x = tf.keras.layers.Dropout(dropout_rate, name=f"{name}_dropout")(x)
    x = tf.keras.layers.Conv2D(channels, 3, padding="same", name=f"{name}_conv2")(x)
    if residual.shape[-1] != channels:
        residual = tf.keras.layers.Conv2D(channels, 1, padding="same", name=f"{name}_skip")(residual)
    return tf.keras.layers.Add(name=f"{name}_out")([x, residual])


def attention_block(x, channels, name):
    residual = x
    x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"{name}_ln")(x)
    x = tf.keras.layers.Reshape((-1, channels), name=f"{name}_flatten")(x)
    x = tf.keras.layers.MultiHeadAttention(
        num_heads=4,
        key_dim=max(channels // 4, 1),
        output_shape=channels,
        name=f"{name}_mha",
    )(x, x)
    spatial = residual.shape[1] * residual.shape[2]
    x = tf.keras.layers.Reshape((residual.shape[1], residual.shape[2], channels), name=f"{name}_unflatten")(x)
    del spatial
    return tf.keras.layers.Add(name=f"{name}_out")([x, residual])


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


def build_denoiser(input_shape, channels, output_channels):
    noisy_inputs = tf.keras.layers.Input(shape=input_shape, name="noisy_inputs")
    timestep_inputs = tf.keras.layers.Input(shape=(), dtype=tf.int32, name="timesteps")

    time_embedding = TimeEmbedding(TIME_EMBED_DIM)
    time_emb = time_embedding(timestep_inputs)

    x = tf.keras.layers.Conv2D(channels, 3, padding="same", name="input_conv")(noisy_inputs)

    skip1 = residual_block(x, time_emb, channels, "down_block1", dropout_rate=0.0)
    x = tf.keras.layers.Conv2D(channels * 2, 3, strides=2, padding="same", name="downsample1")(skip1)
    skip2 = residual_block(x, time_emb, channels * 2, "down_block2", dropout_rate=0.0)
    x = tf.keras.layers.Conv2D(channels * 4, 3, strides=2, padding="same", name="downsample2")(skip2)

    x = residual_block(x, time_emb, channels * 4, "mid_block1", dropout_rate=0.05)
    x = attention_block(x, channels * 4, "mid_attention")
    x = residual_block(x, time_emb, channels * 4, "mid_block2", dropout_rate=0.05)

    x = tf.keras.layers.Conv2DTranspose(channels * 2, 4, strides=2, padding="same", name="upsample1")(x)
    x = tf.keras.layers.Concatenate(name="skip_concat1")([x, skip2])
    x = residual_block(x, time_emb, channels * 2, "up_block1", dropout_rate=0.0)
    x = tf.keras.layers.Conv2DTranspose(channels, 4, strides=2, padding="same", name="upsample2")(x)
    x = tf.keras.layers.Concatenate(name="skip_concat2")([x, skip1])
    x = residual_block(x, time_emb, channels, "up_block2", dropout_rate=0.0)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="output_ln")(x)
    x = tf.keras.layers.Activation("swish", name="output_act")(x)
    outputs = tf.keras.layers.Conv2D(output_channels, 3, padding="same", name="output_conv")(x)

    return tf.keras.Model(
        inputs=[noisy_inputs, timestep_inputs],
        outputs=outputs,
        name="denoiser",
    )


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
        self._interrupt_requested = False
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
            del conditioning_dim
            self.denoiser = build_denoiser(input_shape, BASE_CHANNELS, output_channels)
            self.denoiser_optimizer = tf.keras.optimizers.Adam(DENOISER_LEARNING_RATE)
            self.consistency_model = build_denoiser(input_shape, BASE_CHANNELS, input_shape[-1]) if self.use_consistency else None
            self.consistency_optimizer = tf.keras.optimizers.Adam(CONSISTENCY_LEARNING_RATE) if self.use_consistency else None
            self.mse = tf.keras.losses.MeanSquaredError()
            self.huber = tf.keras.losses.Huber(delta=0.5)
            self.mae = tf.keras.losses.MeanAbsoluteError()
            self._initialize_components()
            self._print_model_summaries()

    def _initialize_components(self):
        image_batch = tf.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32)
        base_batch = tf.zeros((1, *self.input_shape_value), dtype=tf.float32)
        timesteps = tf.zeros((1,), dtype=tf.int32)
        if self.text_encoder is not None:
            token_batch = tf.zeros((1, TEXT_SEQUENCE_LENGTH), dtype=tf.int32)
            self.text_encoder(token_batch, training=False)
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

        denoiser_inputs = latent_batch if latent_batch is not None else base_batch
        self.denoiser([denoiser_inputs, timesteps], training=False)
        self.denoiser_optimizer.build(self.denoiser.trainable_variables)

        if self.consistency_model is not None:
            self.consistency_model([base_batch, timesteps], training=False)
            self.consistency_optimizer.build(self.consistency_model.trainable_variables)

    def _print_model_summaries(self):
        if self.text_encoder is not None:
            self.text_encoder.summary(expand_nested=True, show_trainable=True)
        if self.autoencoder is not None:
            self.autoencoder.summary(expand_nested=True, show_trainable=True)
        if self.clip_encoder is not None:
            self.clip_encoder.summary(expand_nested=True, show_trainable=True)
        if self.prior_projection is not None:
            self.prior_projection.summary(expand_nested=True, show_trainable=True)
        self.denoiser.summary(expand_nested=True, show_trainable=True)
        try:
            tf.keras.utils.plot_model(
                self.denoiser,
                to_file="denoiser.png",
                show_shapes=True,
                expand_nested=True,
                show_layer_names=True,
            )
        except Exception as exc:
            print(f"Skipping denoiser plot export: {exc}")
        if self.consistency_model is not None:
            self.consistency_model.summary(expand_nested=True, show_trainable=True)

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
        if stage == "sr":
            target_images = batch["high_res"]
        elif self.autoencoder is not None and stage in {"diffusion", "control"}:
            target_images, _ = self.get_training_images(batch, training=True)
        else:
            target_images = batch["image"] if isinstance(batch, dict) else batch
        batch_size = tf.shape(target_images)[0]
        noise = tf.random.normal(tf.shape(target_images))
        timesteps = self.sample_timesteps(batch_size)
        noisy_inputs = self.q_sample(target_images, timesteps, noise)
        with tf.GradientTape() as tape:
            pred = self.denoiser([noisy_inputs, timesteps], training=True)
            if USE_LEARNED_VARIANCE:
                pred_noise, pred_var = tf.split(pred, 2, axis=-1)
                loss = self.huber(noise, pred_noise) + 1e-3 * tf.reduce_mean(tf.square(pred_var))
            else:
                loss = self.huber(noise, pred)
        denoiser_grads = tape.gradient(loss, self.denoiser.trainable_variables)
        denoiser_grads = [
            tf.clip_by_norm(grad, 1.0) if grad is not None else None
            for grad in denoiser_grads
        ]
        self.denoiser_optimizer.apply_gradients(zip(denoiser_grads, self.denoiser.trainable_variables))
        return loss

    @tf.function
    def train_consistency_step(self, images):
        batch_size = tf.shape(images)[0]
        noise = tf.random.normal(tf.shape(images))
        timesteps = self.sample_timesteps(batch_size)
        noisy_inputs = self.q_sample(images, timesteps, noise)
        teacher_pred = tf.stop_gradient(self.denoiser([noisy_inputs, timesteps], training=False))
        with tf.GradientTape() as tape:
            student_pred = self.consistency_model([noisy_inputs, timesteps], training=True)
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

    def p_sample(self, sample, timesteps):
        """Draw one exact DDPM reverse transition, x_t -> x_(t-1)."""
        pred = self.denoiser([sample, timesteps], training=False)
        if USE_LEARNED_VARIANCE:
            pred, _ = tf.split(pred, 2, axis=-1)
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

    def generate_samples(self, sample_count=4, cond_tokens=None, control=None, stage="diffusion"):
        if stage == "sample_consistency" and self.consistency_model is not None:
            sample = tf.random.normal((sample_count, *self.input_shape_value))
            timesteps = tf.fill((sample_count,), DIFFUSION_STEPS - 1)
            outputs = self.consistency_model(sample, timesteps, training=False)
            return tf.clip_by_value(outputs, -1.0, 1.0)
        sample = tf.random.normal((sample_count, *self.input_shape_value))
        for step in self.reverse_timesteps():
            timesteps = tf.fill((sample_count,), step)
            sample = self.p_sample(sample, timesteps)
        if self.autoencoder is not None and stage in {"diffusion", "control"}:
            sample = self.autoencoder.decoder(sample, training=False)
        return tf.clip_by_value(sample, -1.0, 1.0)

    def fit(self, dataset, epochs, initial_epoch=0, callbacks=None, stage="diffusion"):
        callbacks = callbacks or []
        self._interrupt_requested = False
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
            epoch_loss = tf.keras.metrics.Mean()
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
                epoch_loss.update_state(loss)
                logs = {"loss": float(loss.numpy()), "stage": stage}
                progress.update(step + 1, values=[("loss", logs["loss"])])
                for callback in callbacks:
                    callback.on_train_batch_end(step, logs)
                if self._interrupt_requested:
                    raise KeyboardInterrupt
            epoch_logs = {"loss": float(epoch_loss.result().numpy()), "stage": stage}
            for callback in callbacks:
                callback.on_epoch_end(epoch, epoch_logs)
            if self._interrupt_requested:
                raise KeyboardInterrupt
        for callback in callbacks:
            callback.on_train_end()
