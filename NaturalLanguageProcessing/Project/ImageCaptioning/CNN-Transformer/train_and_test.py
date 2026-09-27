import argparse
import csv
import json
import math
import os
import textwrap

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from config import *
from dataset import Flickr30kDataset
from model import build_cnn, build_model, verify_captioner

AUTOTUNE = tf.data.AUTOTUNE
console = Console()


def stage(title, detail=""):
    text = Text(title.upper(), justify="center", style="bold cyan")
    if detail:
        text.append(f"\n{detail}", style="bold green")
    console.print(Panel(text, border_style="cyan", padding=(1, 4)))


def strategy_for():
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) < MIN_GPUS:
        raise RuntimeError(
            f"NCCL multi-GPU training requires at least {MIN_GPUS} GPUs; found {len(gpus)}")
    for device in gpus:
        tf.config.experimental.set_memory_growth(device, True)
    return tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce())


def load_image(path):
    image = tf.io.decode_image(
        tf.io.read_file(path), channels=IMAGE_CHANNELS, expand_animations=False)
    return tf.cast(tf.image.resize(image, IMAGE_SIZE), tf.float32)[None]


def caption_image(model, image_path, vocabulary):
    word_to_id = {word: index for index, word in enumerate(vocabulary)}
    ids = [word_to_id["start"]]
    image = load_image(image_path)
    for _ in range(MAX_LENGTH - 2):
        tokens = tf.constant([ids], tf.int32)
        logits = model({"image": image, "tokens": tokens}, training=False)
        next_id = int(tf.argmax(logits[0, -1]))
        if next_id <= 1 or vocabulary[next_id] == "end":
            break
        ids.append(next_id)
    return " ".join(vocabulary[index] for index in ids[1:])


def load_captioner(decoder, weights=None):
    if not os.path.isfile(VOCABULARY_PATH):
        raise FileNotFoundError(f"Run training first to create {VOCABULARY_PATH}")
    with open(VOCABULARY_PATH, encoding="utf-8") as handle:
        vocabulary = json.load(handle)
    weights = weights or os.path.join(CHECKPOINT_DIR, decoder, "best.weights.h5")
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"No trained weights found at {weights}")
    model = build_model(len(vocabulary), decoder)
    model.load_weights(weights)
    verify_captioner(model, len(vocabulary))
    return model, vocabulary


def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, PRETRAIN_BRIGHTNESS)
    image = tf.image.random_contrast(image, *PRETRAIN_CONTRAST)
    image = tf.image.random_saturation(image, *PRETRAIN_SATURATION)
    return tf.clip_by_value(image, 0.0, PIXEL_MAX_VALUE)


def load_views(path):
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=IMAGE_CHANNELS)
    image = tf.image.resize(image, PRETRAIN_RESIZE)
    crop_shape = (*IMAGE_SIZE, IMAGE_CHANNELS)
    return augment(tf.image.random_crop(image, crop_shape)), augment(
        tf.image.random_crop(image, crop_shape))


def image_dataset(batch_size):
    paths = list(dict.fromkeys(path for path, _ in Flickr30kDataset().rows()))
    dataset = tf.data.Dataset.from_tensor_slices(paths).shuffle(len(paths), seed=SEED)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.map(load_views, num_parallel_calls=AUTOTUNE).with_options(options)
    return dataset.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE), len(paths)


class DistributedSimCLRTrainer:
    def __init__(self, encoder, strategy, optimizer):
        self.encoder = encoder
        self.strategy = strategy
        self.optimizer = optimizer
        with strategy.scope():
            self.projector = tf.keras.Sequential([
                tf.keras.layers.Dense(PROJECTION_HIDDEN_DIM, activation="relu"),
                tf.keras.layers.Dense(PROJECTION_DIM),
            ], name="projection_head")
            self.projector(tf.zeros((1, CNN_FILTERS[-1])))

    @tf.function
    def _train_replica(self, view1, view2, global_batch_size):
        with tf.GradientTape() as tape:
            z1 = tf.math.l2_normalize(
                self.projector(self.encoder(view1, training=True), training=True), axis=1)
            z2 = tf.math.l2_normalize(
                self.projector(self.encoder(view2, training=True), training=True), axis=1)
            labels = tf.range(tf.shape(z1)[0])
            loss12 = tf.keras.losses.sparse_categorical_crossentropy(
                labels, tf.matmul(z1, z2, transpose_b=True) / PRETRAIN_TEMPERATURE,
                from_logits=True)
            loss21 = tf.keras.losses.sparse_categorical_crossentropy(
                labels, tf.matmul(z2, z1, transpose_b=True) / PRETRAIN_TEMPERATURE,
                from_logits=True)
            loss = tf.reduce_sum((loss12 + loss21) / 2) / tf.cast(
                global_batch_size, tf.float32)
        variables = self.encoder.trainable_variables + self.projector.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def train_batch(self, view1, view2):
        values = (tf.convert_to_tensor(view1), tf.convert_to_tensor(view2))
        distributed = tuple(
            self.strategy.experimental_distribute_values_from_function(
                lambda context, value=value: value[
                    context.replica_id_in_sync_group::self.strategy.num_replicas_in_sync])
            for value in values
        )
        losses = self.strategy.run(
            self._train_replica, args=(*distributed, tf.shape(view1)[0]))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)


def run_pretraining(args, strategy):
    if PRETRAIN_BATCH_SIZE % strategy.num_replicas_in_sync:
        raise ValueError("PRETRAIN_BATCH_SIZE must be divisible by the number of replicas")
    stage("Pretrain data", "Creating two augmented Flickr30k views")
    dataset, image_count = image_dataset(PRETRAIN_BATCH_SIZE)
    os.makedirs(os.path.dirname(PRETRAINED_CNN_PATH), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "pretrain"), exist_ok=True)
    history_path = os.path.join(LOG_DIR, "pretrain", "distributed_history.csv")
    previous_losses = []
    if os.path.isfile(history_path) and args.resume:
        with open(history_path, newline="", encoding="utf-8") as handle:
            previous_losses = [float(row["loss"]) for row in csv.DictReader(handle)
                               if row.get("loss") not in (None, "", "nan")]
    initial_epoch = len(previous_losses)
    with strategy.scope():
        encoder = build_cnn()
        stage("Pretrain encoder", "Scratch CNN model summary")
        encoder.summary(expand_nested=True, show_trainable=True)
        if os.path.isfile(PRETRAINED_CNN_PATH) and not args.pretrain_scratch:
            encoder.load_weights(PRETRAINED_CNN_PATH)
            encoder(tf.zeros((1, *IMAGE_SIZE, IMAGE_CHANNELS)), training=False)
            stage("Pretrained encoder loaded", PRETRAINED_CNN_PATH)
        steps = max(image_count // PRETRAIN_BATCH_SIZE, 1)
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            PRETRAIN_LEARNING_RATE, steps * args.pretrain_epochs, alpha=COSINE_ALPHA)
        optimizer = tf.keras.optimizers.AdamW(
            schedule, weight_decay=PRETRAIN_WEIGHT_DECAY,
            global_clipnorm=PRETRAIN_CLIP_NORM)
        encoder.optimizer = optimizer
        trainer = DistributedSimCLRTrainer(encoder, strategy, optimizer)
        checkpoint = tf.train.Checkpoint(
            encoder=encoder, projector=trainer.projector, optimizer=optimizer)
        manager = tf.train.CheckpointManager(
            checkpoint, os.path.join(CHECKPOINT_DIR, "pretrain", "training_state"),
            max_to_keep=PRETRAIN_CHECKPOINTS)
        if args.resume and not manager.latest_checkpoint:
            raise FileNotFoundError("No SimCLR training state is available to resume")
        if args.resume:
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
            initial_epoch = int(manager.latest_checkpoint.rsplit("-", 1)[-1])
            stage("SimCLR state restored", manager.latest_checkpoint)
    if initial_epoch >= args.pretrain_epochs:
        if not os.path.isfile(PRETRAINED_CNN_PATH):
            encoder.save_weights(PRETRAINED_CNN_PATH)
        open(PRETRAIN_COMPLETE_PATH, "a").close()
        stage("Pretraining already complete", f"Epoch {initial_epoch}")
        return
    if os.path.isfile(PRETRAIN_COMPLETE_PATH):
        os.remove(PRETRAIN_COMPLETE_PATH)
    patience = (PRETRAIN_EARLY_STOPPING_PATIENCE if args.pretrain_patience is None
                else args.pretrain_patience)
    min_delta = (PRETRAIN_EARLY_STOPPING_MIN_DELTA if args.pretrain_min_delta is None
                 else args.pretrain_min_delta)
    callbacks = tf.keras.callbacks.CallbackList([
        tf.keras.callbacks.ModelCheckpoint(
            PRETRAINED_CNN_PATH, monitor="loss", mode="min", save_best_only=True,
            save_weights_only=True, initial_value_threshold=min(previous_losses, default=None),
            verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", mode="min", patience=patience, min_delta=min_delta,
            baseline=min(previous_losses, default=None), restore_best_weights=True, verbose=1),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.TensorBoard(os.path.join(LOG_DIR, "pretrain")),
        tf.keras.callbacks.CSVLogger(history_path, append=args.resume),
    ])
    callbacks.set_model(encoder)
    callbacks.set_params({"epochs": args.pretrain_epochs, "initial_epoch": initial_epoch,
                          "steps": steps, "verbose": 1, "metrics": ["loss"]})
    encoder.stop_training = False
    callbacks.on_train_begin()
    for epoch in range(initial_epoch, args.pretrain_epochs):
        callbacks.on_epoch_begin(epoch)
        print(f"Epoch {epoch + 1}/{args.pretrain_epochs}")
        progress = tf.keras.utils.Progbar(steps, stateful_metrics=["loss"])
        total_loss = 0.0
        for step, (view1, view2) in enumerate(dataset.take(steps)):
            callbacks.on_train_batch_begin(step)
            loss = float(trainer.train_batch(view1, view2))
            if not math.isfinite(loss):
                raise FloatingPointError("SimCLR loss became non-finite")
            total_loss += loss
            running_loss = total_loss / (step + 1)
            progress.update(step + 1, values=[("loss", running_loss)])
            callbacks.on_train_batch_end(step, {"loss": running_loss})
        callbacks.on_epoch_end(epoch, {"loss": total_loss / steps})
        manager.save(checkpoint_number=epoch + 1)
        if encoder.stop_training:
            break
    callbacks.on_train_end()
    if not os.path.isfile(PRETRAINED_CNN_PATH):
        encoder.save_weights(PRETRAINED_CNN_PATH)
    open(PRETRAIN_COMPLETE_PATH, "a").close()
    stage("SimCLR pretraining complete", PRETRAINED_CNN_PATH)


class CaptionGrid(tf.keras.callbacks.Callback):
    def __init__(self, data, output_dir):
        super().__init__()
        self.data = data
        self.output_dir = output_dir
        self.paths = []
        for path, _ in data.rows(True):
            if path not in self.paths:
                self.paths.append(path)
            if len(self.paths) == GRID_ROWS * GRID_COLUMNS:
                break

    def on_epoch_end(self, epoch, logs=None):
        import matplotlib.pyplot as plt

        os.makedirs(self.output_dir, exist_ok=True)
        figure, axes = plt.subplots(GRID_ROWS, GRID_COLUMNS, figsize=GRID_FIGSIZE)
        for axis, path in zip(axes.flat, self.paths):
            image = tf.io.decode_image(
                tf.io.read_file(path), channels=IMAGE_CHANNELS, expand_animations=False)
            axis.imshow(image.numpy())
            axis.axis("off")
            caption = caption_image(self.model, path, self.data.vocabulary) or "<empty caption>"
            axis.text(0.5, GRID_CAPTION_Y, textwrap.fill(caption, GRID_CAPTION_WIDTH),
                      transform=axis.transAxes, ha="center", va="top",
                      fontsize=GRID_FONT_SIZE)
        figure.suptitle(f"Validation captions - epoch {epoch + 1}")
        figure.tight_layout()
        figure.savefig(
            os.path.join(self.output_dir, f"epoch_{epoch + 1:03d}.png"), dpi=GRID_DPI)
        plt.close(figure)


def bleu_scores(model, data, limit=BLEU_EXAMPLES):
    """Dependency-free corpus BLEU-1..4 over a bounded validation sample."""
    matches = [0] * BLEU_MAX_ORDER
    totals = [0] * BLEU_MAX_ORDER
    predicted_length = reference_length = 0
    for number, (path, caption) in enumerate(data.rows(True)):
        if number >= limit:
            break
        reference = caption.split()[1:-1]
        predicted = caption_image(model, path, data.vocabulary).split()
        predicted_length += len(predicted); reference_length += len(reference)
        for n in range(1, BLEU_MAX_ORDER + 1):
            ref = {}
            for i in range(len(reference) - n + 1):
                gram = tuple(reference[i:i + n]); ref[gram] = ref.get(gram, 0) + 1
            seen = {}
            for i in range(len(predicted) - n + 1):
                gram = tuple(predicted[i:i + n]); seen[gram] = seen.get(gram, 0) + 1
            matches[n - 1] += sum(min(count, ref.get(gram, 0)) for gram, count in seen.items())
            totals[n - 1] += max(len(predicted) - n + 1, 0)
    brevity = math.exp(min(0.0, 1.0 - reference_length / max(predicted_length, 1)))
    precisions = [matches[i] / max(totals[i], 1) for i in range(BLEU_MAX_ORDER)]
    return {f"bleu_{n}": brevity * math.exp(sum(math.log(max(p, 1e-12)) for p in precisions[:n]) / n)
            for n in range(1, BLEU_MAX_ORDER + 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", "--pretain", action="store_true",
                        help="run/resume SimCLR encoder pretraining")
    parser.add_argument("--model", action="store_true", help="train the caption model")
    parser.add_argument("--decoder", choices=("transformer",), default=DECODER)
    parser.add_argument("--epochs", "--model-epochs", type=int, default=EPOCHS)
    parser.add_argument("--pretrain-epochs", type=int, default=PRETRAIN_EPOCHS)
    parser.add_argument("--pretrain-patience", type=int, default=None)
    parser.add_argument("--pretrain-min-delta", type=float, default=None)
    parser.add_argument("--pretrain-scratch", action="store_true",
                        help="discard SimCLR state and pretrain a new encoder")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--image", help="generate a caption instead of training")
    parser.add_argument("--weights", help="caption weights used with --image")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--scratch", action="store_true", help="ignore checkpoints and train from random weights")
    start.add_argument("--from-pretrained", action="store_true", help="start a new caption model from SimCLR CNN weights")
    start.add_argument("--resume", action="store_true",
                       help="continue caption training from the last recorded epoch")
    args = parser.parse_args()

    if ((args.pretrain_patience is not None and args.pretrain_patience < 0) or
            (args.pretrain_min_delta is not None and args.pretrain_min_delta < 0)):
        parser.error("pretraining callback values must be non-negative")

    if args.image:
        if not os.path.isfile(args.image):
            parser.error(f"Image not found: {args.image}")
        stage("Inference", f"Loading {args.decoder.upper()} caption model")
        model, vocabulary = load_captioner(args.decoder, args.weights)
        print(caption_image(model, args.image, vocabulary))
        stage("Inference complete")
        return

    if args.resume and (args.scratch or args.pretrain_scratch):
        parser.error("--resume cannot be combined with a scratch option")
    if args.scratch and args.pretrain:
        parser.error("--scratch cannot be combined with --pretrain; use --pretrain-scratch")

    explicitly_selected = args.pretrain or args.model or args.from_pretrained or args.test_only
    run_pretrain = args.pretrain or args.pretrain_scratch or not explicitly_selected
    run_model = args.model or args.from_pretrained or args.test_only or not explicitly_selected
    if args.scratch:
        run_pretrain = False
        run_model = True

    stage("Multi-GPU setup", "Creating NCCL MirroredStrategy")
    strategy = strategy_for()
    stage("Multi-GPU ready", f"{strategy.num_replicas_in_sync} replicas")

    if run_pretrain:
        if args.pretrain_scratch:
            detail = "Starting from random weights at epoch 1"
        elif args.resume:
            detail = "Restoring weights, optimizer and last epoch"
        else:
            detail = "Loading saved encoder weights and starting at epoch 1"
        stage("Pipeline stage 1 - SimCLR pretraining", detail)
        run_pretraining(args, strategy)
        stage("Pipeline stage 1 complete", "Pretrained CNN weights are ready")
        if not args.resume:
            args.from_pretrained = True

    if not run_model:
        return
    stage("Pipeline stage 2 - Caption model", f"CNN + {args.decoder.upper()}")

    if BATCH_SIZE % strategy.num_replicas_in_sync:
        raise ValueError("BATCH_SIZE must be divisible by the number of replicas")

    stage("Stage 2 - Vocabulary", "Loading or calculating Flickr30k vocabulary")
    data = Flickr30kDataset()
    data.prepare_vocabulary()
    stage("Stage 2 complete", f"Vocabulary size: {len(data.vocabulary)}")
    checkpoint = os.path.join(CHECKPOINT_DIR, args.decoder, "best.weights.h5")
    last_checkpoint = os.path.join(CHECKPOINT_DIR, args.decoder, "last.weights.h5")
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, args.decoder), exist_ok=True)

    with strategy.scope():
        model = build_model(len(data.vocabulary), args.decoder)

    stage("Stage 3 - Model architecture", "Full nested summary and diagram")
    model.summary(expand_nested=True, show_trainable=True)
    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(LOG_DIR, args.decoder, "model.png"),
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        show_layer_activations=True,
        show_trainable=True,
        expand_nested=True,
    )
    stage("Stage 3 complete", "Model summary displayed and model.png saved")

    stage("Stage 4 - Weights", "Checking caption and pretrained CNN checkpoints")
    resume_checkpoint = last_checkpoint if os.path.isfile(last_checkpoint) else checkpoint
    if args.from_pretrained and not os.path.isfile(PRETRAINED_CNN_PATH):
        raise FileNotFoundError(f"Pretrained CNN not found: {PRETRAINED_CNN_PATH}")
    if args.resume and not os.path.isfile(resume_checkpoint):
        raise FileNotFoundError("No caption checkpoint is available to resume")
    resumed = args.resume
    pretrained = not args.scratch and not resumed and os.path.isfile(PRETRAINED_CNN_PATH)
    if resumed:
        model.load_weights(resume_checkpoint)
        print(f"Resuming from {resume_checkpoint}")
    elif pretrained:
        model.get_layer("cnn_encoder").load_weights(PRETRAINED_CNN_PATH)
        model.get_layer("cnn_encoder").trainable = False
        print(f"Loaded pretrained CNN from {PRETRAINED_CNN_PATH}")
    elif args.scratch:
        print("Training from scratch")

    verify_captioner(model, len(data.vocabulary))
    if resumed:
        weight_status = f"Caption checkpoint loaded and verified: {resume_checkpoint}"
    elif pretrained:
        weight_status = f"Pretrained CNN loaded and verified: {PRETRAINED_CNN_PATH}"
    else:
        weight_status = "Random scratch weights verified"
    stage("Stage 4 complete", weight_status)

    stage("Stage 5 - Data pipeline", "Building train and validation batches")
    train_data, validation_data = data.build(True), data.build(False)
    stage("Stage 5 complete", "Flickr30k pipelines ready")

    history_path = os.path.join(LOG_DIR, args.decoder, "history.csv")
    initial_epoch = 0
    if resumed and os.path.isfile(history_path):
        with open(history_path, encoding="utf-8") as handle:
            initial_epoch = max(sum(1 for _ in handle) - 1, 0)
        stage("Resume epoch restored", f"Continuing from epoch {initial_epoch + 1}")

    def compile_model(learning_rate):
        with strategy.scope():
            model.compile(
                optimizer=tf.keras.optimizers.AdamW(learning_rate, weight_decay=WEIGHT_DECAY, global_clipnorm=CLIP_NORM),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="token_accuracy")],
            )

    compile_model(LEARNING_RATE)
    if not args.test_only:
        if pretrained and FREEZE_EPOCHS:
            warmup_epochs = min(FREEZE_EPOCHS, args.epochs)
            stage("Stage 6A - Frozen CNN warmup", f"Training for {warmup_epochs} epochs")
            model.fit(train_data, validation_data=validation_data, epochs=warmup_epochs, callbacks=[
                tf.keras.callbacks.ModelCheckpoint(last_checkpoint, save_weights_only=True),
                tf.keras.callbacks.TerminateOnNaN(),
                tf.keras.callbacks.CSVLogger(os.path.join(LOG_DIR, args.decoder, "history.csv")),
                CaptionGrid(data, os.path.join(LOG_DIR, args.decoder, "grids")),
            ])
            model.get_layer("cnn_encoder").trainable = True
            compile_model(FINETUNE_LEARNING_RATE)
            stage("Stage 6A complete", "CNN warmup finished and encoder unfrozen")
        else:
            warmup_epochs = initial_epoch
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(last_checkpoint, save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=LR_PATIENCE, factor=LR_FACTOR,
                min_lr=MIN_LEARNING_RATE, verbose=1),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.TensorBoard(os.path.join(LOG_DIR, args.decoder)),
            tf.keras.callbacks.CSVLogger(history_path, append=resumed or warmup_epochs > 0),
            CaptionGrid(data, os.path.join(LOG_DIR, args.decoder, "grids")),
        ]
        if resumed:
            callbacks.insert(0, tf.keras.callbacks.BackupAndRestore(
                os.path.join(CHECKPOINT_DIR, args.decoder, "backup")))
        if warmup_epochs < args.epochs:
            stage("Stage 6B - Caption training",
                  f"Training epochs {warmup_epochs + 1} to {args.epochs}")
            model.fit(train_data, validation_data=validation_data, initial_epoch=warmup_epochs,
                      epochs=args.epochs, callbacks=callbacks)
            stage("Stage 6 complete", "Caption training finished")
    stage("Stage 7 - Evaluation", "Validation loss, accuracy, perplexity and BLEU")
    results = model.evaluate(validation_data, return_dict=True)
    results["perplexity"] = math.exp(min(results["loss"], PERPLEXITY_MAX_LOSS))
    results.update(bleu_scores(model, data))
    print({name: round(value, 4) for name, value in results.items()})
    path, _ = next(data.rows(True))
    print("Sample:", os.path.basename(path), "->", caption_image(model, path, data.vocabulary))
    stage("All stages complete", "Model trained, evaluated and ready for inference")


if __name__ == "__main__":
    main()
