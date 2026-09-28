import argparse
import csv
import json
import math
import os
import textwrap

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

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
    features = model.get_layer("cnn_encoder")(load_image(image_path), training=False)
    memory = model.get_layer("image_projection")(features)
    memory = model.get_layer("image_position_embedding")(memory)

    def next_logits(ids):
        tokens = tf.constant([ids], tf.int32)
        x = model.get_layer("token_position_embedding")(tokens)
        x = model.get_layer("embedding_dropout")(x, training=False)
        for number in range(1, TRANSFORMER_LAYERS + 1):
            x = model.get_layer(f"transformer_decoder_{number}")(
                [x, memory, tokens], training=False)
        return model.get_layer("logits")(x)[0, -1]

    return beam_search(next_logits, vocabulary, word_to_id)


def beam_search(next_logits, vocabulary, word_to_id):
    start_id, end_id = word_to_id["start"], word_to_id["end"]
    beams = [([start_id], 0.0, False)]
    for _ in range(MAX_LENGTH - 2):
        candidates = []
        for ids, score, finished in beams:
            if finished:
                candidates.append((ids, score, True))
                continue
            log_probs = tf.nn.log_softmax(next_logits(ids))
            blocked = tf.tensor_scatter_nd_update(
                log_probs, [[0], [1], [start_id]], [-1e9, -1e9, -1e9])
            values, indices = tf.math.top_k(blocked, BEAM_WIDTH)
            for value, index in zip(values.numpy(), indices.numpy()):
                token_id = int(index)
                candidates.append((ids + [token_id], score + float(value),
                                   token_id == end_id))
        def normalized(item):
            length = max(len(item[0]) - 1, 1)
            return item[1] / (((5.0 + length) / 6.0) ** BEAM_LENGTH_ALPHA)
        beams = sorted(candidates, key=normalized, reverse=True)[:BEAM_WIDTH]
        if all(finished for _, _, finished in beams):
            break
    best = max((beam for beam in beams if beam[2]),
               key=normalized, default=max(beams, key=normalized))
    return " ".join(vocabulary[index] for index in best[0][1:]
                    if index != end_id)


def load_captioner(decoder, weights=None):
    if not os.path.isfile(VOCABULARY_PATH):
        raise FileNotFoundError(f"Run training first to create {VOCABULARY_PATH}")
    with open(VOCABULARY_PATH, encoding="utf-8") as handle:
        vocabulary = json.load(handle)
    weights = weights or os.path.join(
        CHECKPOINT_DIR, decoder, "best_resnet18_scratch.weights.h5")
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


def coco_records(split):
    annotation_path = os.path.join(COCO_DIR, "annotations", f"captions_{split}2017.json")
    images_dir = os.path.join(COCO_DIR, f"{split}2017")
    if not os.path.isfile(annotation_path) or not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Missing COCO {split}2017 images or captions")
    with open(annotation_path, encoding="utf-8") as handle:
        content = json.load(handle)
    names = {image["id"]: image["file_name"] for image in content["images"]}
    captions = {}
    for annotation in content["annotations"]:
        captions.setdefault(annotation["image_id"], []).append(annotation["caption"])
    return [(os.path.join(images_dir, names[image_id]), captions[image_id])
            for image_id in names if image_id in captions], content["annotations"]


def contrastive_vectorizer(annotations):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=CONTRASTIVE_VOCAB_SIZE, output_mode="int",
        output_sequence_length=MAX_LENGTH)
    if os.path.isfile(CONTRASTIVE_VOCABULARY_PATH):
        with open(CONTRASTIVE_VOCABULARY_PATH, encoding="utf-8") as handle:
            vectorizer.set_vocabulary(json.load(handle))
        return vectorizer
    captions = tf.data.Dataset.from_generator(
        lambda: (item["caption"] for item in annotations),
        output_signature=tf.TensorSpec((), tf.string)).batch(VOCAB_ADAPT_BATCH_SIZE)
    vectorizer.adapt(captions)
    os.makedirs(os.path.dirname(CONTRASTIVE_VOCABULARY_PATH), exist_ok=True)
    with open(CONTRASTIVE_VOCABULARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(vectorizer.get_vocabulary(), handle)
    return vectorizer


def contrastive_dataset(records, vectorizer, epoch, training):
    def rows():
        for index, (path, captions) in enumerate(records):
            yield path, captions[(epoch + index) % len(captions)]

    def decode(path, caption):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=IMAGE_CHANNELS)
        image = tf.image.resize(image, PRETRAIN_RESIZE if training else IMAGE_SIZE)
        if training:
            image = augment(tf.image.random_crop(
                image, (*IMAGE_SIZE, IMAGE_CHANNELS)))
        return tf.cast(image, tf.float32), vectorizer(caption)

    dataset = tf.data.Dataset.from_generator(
        rows, output_signature=(tf.TensorSpec((), tf.string), tf.TensorSpec((), tf.string)))
    if training:
        dataset = dataset.shuffle(SHUFFLE_BUFFER, seed=SEED + epoch,
                                  reshuffle_each_iteration=False)
    dataset = dataset.map(decode, num_parallel_calls=AUTOTUNE,
                          deterministic=not training)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    return dataset.with_options(options).batch(
        PRETRAIN_BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)


class DistributedContrastiveTrainer:
    def __init__(self, encoder, text_encoder, image_projector, text_projector,
                 strategy, optimizer):
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.image_projector = image_projector
        self.text_projector = text_projector
        self.strategy = strategy
        self.optimizer = optimizer
        optimizer.build(encoder.trainable_variables + text_encoder.trainable_variables +
                        image_projector.trainable_variables + text_projector.trainable_variables)

    def _loss(self, images, tokens, training):
        image_vectors = tf.math.l2_normalize(self.image_projector(
            self.encoder(images, training=training), training=training), axis=1)
        text_vectors = tf.math.l2_normalize(self.text_projector(
            self.text_encoder(tokens, training=training), training=training), axis=1)
        context = tf.distribute.get_replica_context()
        all_images = context.all_gather(image_vectors, axis=0)
        all_text = context.all_gather(text_vectors, axis=0)
        local_size = tf.shape(image_vectors)[0]
        labels = tf.range(local_size) + context.replica_id_in_sync_group * local_size
        image_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, tf.matmul(image_vectors, all_text, transpose_b=True) / PRETRAIN_TEMPERATURE,
            from_logits=True)
        text_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, tf.matmul(text_vectors, all_images, transpose_b=True) / PRETRAIN_TEMPERATURE,
            from_logits=True)
        return tf.nn.compute_average_loss(
            (image_loss + text_loss) / 2, global_batch_size=PRETRAIN_BATCH_SIZE)

    @tf.function
    def _train_replica(self, images, tokens):
        with tf.GradientTape() as tape:
            loss = self._loss(images, tokens, True)
        variables = (self.encoder.trainable_variables + self.text_encoder.trainable_variables +
                     self.image_projector.trainable_variables + self.text_projector.trainable_variables)
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    @tf.function
    def _eval_replica(self, images, tokens):
        return self._loss(images, tokens, False)

    def batch(self, values, training):
        function = self._train_replica if training else self._eval_replica
        losses = self.strategy.run(function, args=values)
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)


def run_pretraining(args, strategy):
    if PRETRAIN_BATCH_SIZE % strategy.num_replicas_in_sync:
        raise ValueError("PRETRAIN_BATCH_SIZE must be divisible by the number of replicas")
    pretrain_dir = os.path.join(CHECKPOINT_DIR, "pretrain")
    state_dir = os.path.join(pretrain_dir, "resnet18_training_state")
    history_path = os.path.join(LOG_DIR, "pretrain", "distributed_history.csv")
    if args.pretrain_scratch:
        for path in (PRETRAINED_CNN_PATH, PRETRAIN_COMPLETE_PATH,
                     CONTRASTIVE_VOCABULARY_PATH, history_path):
            if os.path.isfile(path):
                os.remove(path)
        for path in (state_dir, os.path.join(CHECKPOINT_DIR, args.decoder, "backup"),
                     os.path.join(LOG_DIR, args.decoder, "grids")):
            if os.path.isdir(path):
                tf.io.gfile.rmtree(path)
        for name in ("best_resnet18_scratch.weights.h5", "last_resnet18_scratch.weights.h5"):
            path = os.path.join(CHECKPOINT_DIR, args.decoder, name)
            if os.path.isfile(path):
                os.remove(path)
        caption_history = os.path.join(LOG_DIR, args.decoder, "history.csv")
        if os.path.isfile(caption_history):
            os.remove(caption_history)

    stage("Pretrain data", "Loading local COCO image-caption pairs")
    train_records, annotations = coco_records("train")
    validation_records, _ = coco_records("val")
    vectorizer = contrastive_vectorizer(annotations)
    steps = len(train_records) // PRETRAIN_BATCH_SIZE
    validation_steps = len(validation_records) // PRETRAIN_BATCH_SIZE
    os.makedirs(pretrain_dir, exist_ok=True)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    with strategy.scope():
        encoder = build_cnn()
        text_encoder = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(vectorizer.get_vocabulary()),
                                      CONTRASTIVE_EMBEDDING_DIM, mask_zero=True),
            tf.keras.layers.GRU(CONTRASTIVE_TEXT_DIM),
        ], name="contrastive_text_encoder")
        image_projector = tf.keras.Sequential([
            tf.keras.layers.Dense(PROJECTION_HIDDEN_DIM, activation="relu"),
            tf.keras.layers.Dense(PROJECTION_DIM),
        ], name="image_projection_head")
        text_projector = tf.keras.Sequential([
            tf.keras.layers.Dense(PROJECTION_HIDDEN_DIM, activation="relu"),
            tf.keras.layers.Dense(PROJECTION_DIM),
        ], name="text_projection_head")
        text_encoder(tf.zeros((1, MAX_LENGTH), tf.int32))
        image_projector(tf.zeros((1, ENCODER_DIM)))
        text_projector(tf.zeros((1, CONTRASTIVE_TEXT_DIM)))
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            PRETRAIN_LEARNING_RATE, steps * args.pretrain_epochs, alpha=COSINE_ALPHA)
        optimizer = tf.keras.optimizers.AdamW(
            schedule, weight_decay=PRETRAIN_WEIGHT_DECAY,
            global_clipnorm=PRETRAIN_CLIP_NORM)
        trainer = DistributedContrastiveTrainer(
            encoder, text_encoder, image_projector, text_projector, strategy, optimizer)
        checkpoint = tf.train.Checkpoint(
            encoder=encoder, text_encoder=text_encoder,
            image_projector=image_projector, text_projector=text_projector,
            optimizer=optimizer)
        manager = tf.train.CheckpointManager(
            checkpoint, state_dir, max_to_keep=PRETRAIN_CHECKPOINTS)
        initial_epoch = 0
        if args.resume_pretrain:
            if not manager.latest_checkpoint:
                raise FileNotFoundError("No contrastive pretraining state is available")
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
            initial_epoch = int(manager.latest_checkpoint.rsplit("-", 1)[-1])
            stage("Contrastive state restored", manager.latest_checkpoint)

    callbacks = tf.keras.callbacks.CallbackList([
        tf.keras.callbacks.ModelCheckpoint(
            PRETRAINED_CNN_PATH, monitor="val_loss", mode="min", save_best_only=True,
            save_weights_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min",
            patience=(PRETRAIN_EARLY_STOPPING_PATIENCE if args.pretrain_patience is None
                      else args.pretrain_patience),
            min_delta=(PRETRAIN_EARLY_STOPPING_MIN_DELTA if args.pretrain_min_delta is None
                       else args.pretrain_min_delta),
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger(history_path, append=args.resume_pretrain),
    ])
    callbacks.set_model(encoder)
    callbacks.set_params({"epochs": args.pretrain_epochs, "initial_epoch": initial_epoch,
                          "steps": steps, "verbose": 1,
                          "metrics": ["loss", "val_loss"]})
    encoder.stop_training = False
    callbacks.on_train_begin()
    validation = strategy.experimental_distribute_dataset(
        contrastive_dataset(validation_records, vectorizer, 0, False))
    for epoch in range(initial_epoch, args.pretrain_epochs):
        callbacks.on_epoch_begin(epoch)
        training = strategy.experimental_distribute_dataset(
            contrastive_dataset(train_records, vectorizer, epoch, True))
        progress = tf.keras.utils.Progbar(steps, stateful_metrics=["loss"])
        total_loss = 0.0
        for step, values in enumerate(training):
            loss = float(trainer.batch(values, True))
            if not math.isfinite(loss):
                raise FloatingPointError("Contrastive training loss became non-finite")
            total_loss += loss
            progress.update(step + 1, values=[("loss", total_loss / (step + 1))])
        validation_loss = sum(float(trainer.batch(values, False))
                              for values in validation) / validation_steps
        if not math.isfinite(validation_loss):
            raise FloatingPointError("Contrastive validation loss became non-finite")
        logs = {"loss": total_loss / steps, "val_loss": validation_loss}
        callbacks.on_epoch_end(epoch, logs)
        manager.save(checkpoint_number=epoch + 1)
        if encoder.stop_training:
            break
    callbacks.on_train_end()
    if not os.path.isfile(PRETRAINED_CNN_PATH):
        encoder.save_weights(PRETRAINED_CNN_PATH)
    open(PRETRAIN_COMPLETE_PATH, "a").close()
    stage("COCO contrastive pretraining complete", PRETRAINED_CNN_PATH)


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


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs["learning_rate"] = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate))


def smoothed_sparse_crossentropy(labels, logits):
    hard_loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True)
    smooth_loss = -tf.reduce_mean(tf.nn.log_softmax(logits), axis=-1)
    return (1.0 - LABEL_SMOOTHING) * hard_loss + LABEL_SMOOTHING * smooth_loss


def caption_scores(model, data, limit=BLEU_EXAMPLES):
    """Multi-reference corpus BLEU-1..4 and CIDEr over validation images."""
    groups = {}
    vocabulary = data.vocabulary
    for path, caption in data.rows(True):
        if path not in groups and len(groups) >= limit:
            continue
        ids = data.vectorizer(caption).numpy()
        groups.setdefault(path, []).append(
            [vocabulary[index] for index in ids if index > 1 and vocabulary[index] not in {"start", "end"}])

    def ngrams(words, order):
        counts = {}
        for index in range(len(words) - order + 1):
            gram = tuple(words[index:index + order])
            counts[gram] = counts.get(gram, 0) + 1
        return counts

    matches = [0] * BLEU_MAX_ORDER
    totals = [0] * BLEU_MAX_ORDER
    predicted_length = reference_length = 0
    predictions = {}
    for path, references in groups.items():
        predicted = caption_image(model, path, data.vocabulary).split()
        predictions[path] = predicted
        predicted_length += len(predicted)
        reference_length += min((len(ref) for ref in references),
                                key=lambda length: (abs(length - len(predicted)), length))
        for order in range(1, BLEU_MAX_ORDER + 1):
            candidate = ngrams(predicted, order)
            maximum = {}
            for reference in references:
                for gram, count in ngrams(reference, order).items():
                    maximum[gram] = max(maximum.get(gram, 0), count)
            matches[order - 1] += sum(min(count, maximum.get(gram, 0))
                                      for gram, count in candidate.items())
            totals[order - 1] += sum(candidate.values())
    brevity = math.exp(min(0.0, 1.0 - reference_length / max(predicted_length, 1)))
    precisions = [matches[i] / max(totals[i], 1) for i in range(BLEU_MAX_ORDER)]
    scores = {f"bleu_{n}": brevity * math.exp(
        sum(math.log(max(p, 1e-12)) for p in precisions[:n]) / n)
        for n in range(1, BLEU_MAX_ORDER + 1)}

    document_frequency = [{} for _ in range(BLEU_MAX_ORDER)]
    for references in groups.values():
        for order in range(1, BLEU_MAX_ORDER + 1):
            for gram in set().union(*(ngrams(ref, order) for ref in references)):
                document_frequency[order - 1][gram] = document_frequency[order - 1].get(gram, 0) + 1

    def vector(words, order):
        counts = ngrams(words, order)
        total = max(sum(counts.values()), 1)
        return {gram: count / total * math.log(len(groups) / frequency)
                for gram, count in counts.items()
                if (frequency := document_frequency[order - 1].get(gram))}

    cider = 0.0
    for path, references in groups.items():
        image_score = 0.0
        for order in range(1, BLEU_MAX_ORDER + 1):
            candidate = vector(predictions[path], order)
            candidate_norm = math.sqrt(sum(value * value for value in candidate.values()))
            similarities = []
            for reference in references:
                reference_vector = vector(reference, order)
                reference_norm = math.sqrt(sum(value * value for value in reference_vector.values()))
                dot = sum(value * reference_vector.get(gram, 0.0)
                          for gram, value in candidate.items())
                similarities.append(dot / max(candidate_norm * reference_norm, 1e-12))
            image_score += sum(similarities) / len(similarities)
        cider += 10.0 * image_score / BLEU_MAX_ORDER
    scores["cider"] = cider / max(len(groups), 1)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", "--pretain", action="store_true",
                        help="run COCO image-caption contrastive pretraining")
    parser.add_argument("--resume-pretrain", action="store_true",
                        help="resume COCO contrastive pretraining state")
    parser.add_argument("--model", action="store_true", help="train the caption model")
    parser.add_argument("--decoder", choices=("transformer",), default=DECODER)
    parser.add_argument("--epochs", "--model-epochs", type=int, default=EPOCHS)
    parser.add_argument("--pretrain-epochs", type=int, default=PRETRAIN_EPOCHS)
    parser.add_argument("--pretrain-patience", type=int, default=None)
    parser.add_argument("--pretrain-min-delta", type=float, default=None)
    parser.add_argument("--pretrain-scratch", action="store_true",
                        help="replace old runs and contrastively pretrain a new encoder")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--image", help="generate a caption instead of training")
    parser.add_argument("--weights", help="caption weights used with --image")
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--scratch", action="store_true", help="ignore checkpoints and train from random weights")
    start.add_argument("--from-pretrained", action="store_true",
                       help="start a new caption model from COCO contrastive CNN weights")
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
    if args.resume_pretrain and args.pretrain_scratch:
        parser.error("--resume-pretrain cannot be combined with --pretrain-scratch")
    if args.scratch and args.pretrain:
        parser.error("--scratch cannot be combined with --pretrain; use --pretrain-scratch")

    explicitly_selected = any((
        args.pretrain, args.pretrain_scratch, args.resume_pretrain,
        args.model, args.from_pretrained,
        args.test_only, args.resume, args.scratch,
    ))
    run_pretrain = args.pretrain or args.pretrain_scratch or args.resume_pretrain
    run_model = any((
        args.model, args.from_pretrained, args.test_only, args.resume,
        args.scratch,
    )) or not explicitly_selected
    if args.scratch:
        run_pretrain = False
        run_model = True

    stage("Multi-GPU setup", "Creating NCCL MirroredStrategy")
    strategy = strategy_for()
    stage("Multi-GPU ready", f"{strategy.num_replicas_in_sync} replicas")

    if run_pretrain:
        if args.pretrain_scratch:
            detail = "Starting from random weights at epoch 1"
        elif args.resume_pretrain:
            detail = "Restoring weights, optimizer and last epoch"
        else:
            detail = "Loading saved encoder weights and starting at epoch 1"
        stage("Pipeline stage 1 - COCO contrastive pretraining", detail)
        run_pretraining(args, strategy)
        stage("Pipeline stage 1 complete", "Pretrained CNN weights are ready")
        if not args.resume_pretrain:
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
    checkpoint = os.path.join(
        CHECKPOINT_DIR, args.decoder, "best_resnet18_scratch.weights.h5")
    last_checkpoint = os.path.join(
        CHECKPOINT_DIR, args.decoder, "last_resnet18_scratch.weights.h5")
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
    if args.test_only:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"No scratch caption checkpoint found at {checkpoint}")
        model.load_weights(checkpoint)
        print(f"Testing best checkpoint {checkpoint}")
    elif resumed:
        model.load_weights(resume_checkpoint)
        print(f"Resuming from {resume_checkpoint}")
    elif args.from_pretrained:
        model.get_layer("cnn_encoder").load_weights(PRETRAINED_CNN_PATH)
        model.get_layer("cnn_encoder").trainable = False
        print(f"Loaded COCO contrastive CNN from {PRETRAINED_CNN_PATH}")
    elif pretrained:
        model.get_layer("cnn_encoder").load_weights(PRETRAINED_CNN_PATH)
        model.get_layer("cnn_encoder").trainable = False
        print(f"Loaded COCO contrastive CNN from {PRETRAINED_CNN_PATH}")
    elif args.scratch:
        print("Training from scratch")

    verify_captioner(model, len(data.vocabulary))
    if args.test_only:
        weight_status = f"Best caption checkpoint loaded: {checkpoint}"
    elif resumed:
        weight_status = f"Caption checkpoint loaded and verified: {resume_checkpoint}"
    elif pretrained:
        weight_status = "COCO contrastive CNN loaded and verified"
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
            epochs = [int(line.split(",", 1)[0]) for line in handle
                      if line.split(",", 1)[0].isdigit()]
        initial_epoch = max(epochs, default=-1) + 1
        stage("Resume epoch restored", f"Continuing from epoch {initial_epoch + 1}")

    def compile_model(learning_rate):
        with strategy.scope():
            model.compile(
                optimizer=tf.keras.optimizers.AdamW(learning_rate, weight_decay=WEIGHT_DECAY, global_clipnorm=CLIP_NORM),
                loss=smoothed_sparse_crossentropy,
                weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="token_accuracy")],
            )

    compile_model(FINETUNE_LEARNING_RATE if resumed else LEARNING_RATE)
    if not args.test_only:
        if pretrained and FREEZE_EPOCHS:
            warmup_epochs = min(FREEZE_EPOCHS, args.epochs)
            stage("Stage 6A - Frozen CNN warmup", f"Training for {warmup_epochs} epochs")
            model.fit(train_data, validation_data=validation_data, epochs=warmup_epochs, callbacks=[
                tf.keras.callbacks.ModelCheckpoint(last_checkpoint, save_weights_only=True),
                tf.keras.callbacks.TerminateOnNaN(),
                LearningRateLogger(),
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
            LearningRateLogger(),
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
    if os.path.isfile(checkpoint) and not args.test_only:
        model.load_weights(checkpoint)
        print(f"Evaluating best checkpoint {checkpoint}")
    stage("Stage 7 - Evaluation", "Validation loss, token accuracy, perplexity, BLEU and CIDEr")
    results = model.evaluate(validation_data, return_dict=True)
    results["perplexity"] = math.exp(min(results["loss"], PERPLEXITY_MAX_LOSS))
    results.update(caption_scores(model, data))
    print({name: round(value, 4) for name, value in results.items()})
    path, _ = next(data.rows(True))
    print("Sample:", os.path.basename(path), "->", caption_image(model, path, data.vocabulary))
    stage("All stages complete", "Model trained, evaluated and ready for inference")


if __name__ == "__main__":
    main()
