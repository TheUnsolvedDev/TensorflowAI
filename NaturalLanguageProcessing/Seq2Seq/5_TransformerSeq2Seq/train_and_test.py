# train_and_test.py

from config import *
from dataset import *
from model import *

import os
import sys
import json
import signal
import argparse
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def setup_strategy(gpu_id):
    gpus = tf.config.list_physical_devices("GPU")
    if gpu_id < -1:
        raise ValueError("--gpu must be -1 for all GPUs or a non-negative GPU index")
    if gpu_id >= 0:
        if gpu_id >= len(gpus):
            raise ValueError(
                f"GPU {gpu_id} is unavailable; found {len(gpus)} physical GPUs")
        selected_gpus = [gpus[gpu_id]]
        tf.config.set_visible_devices(selected_gpus, "GPU")
    else:
        selected_gpus = gpus

    for gpu in selected_gpus:
        try: tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError: pass
    if not selected_gpus:
        print("Using CPU")
        return tf.distribute.OneDeviceStrategy("/CPU:0")

    logical_gpus = tf.config.list_logical_devices("GPU")
    devices = [gpu.name for gpu in logical_gpus]
    if len(devices) == 1:
        print(f"Using One GPU: {devices[0]}")
        return tf.distribute.OneDeviceStrategy(devices[0])

    print(f"Using {len(devices)} GPUs with NCCL: {', '.join(devices)}")
    return tf.distribute.MirroredStrategy(
        devices=devices, cross_device_ops=tf.distribute.NcclAllReduce())


def get_log_dir(dataset_name):
    timestamp = "./"  # datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(LOG_DIR, dataset_name, "Transformer", timestamp)


def get_weight_path(log_dir):
    return os.path.join(log_dir, "model.weights.h5")


def get_best_weight_path(log_dir):
    return os.path.join(log_dir, "best_model.weights.h5")


def get_history_path(log_dir):
    return os.path.join(log_dir, "history.json")


def get_state_path(log_dir):
    return os.path.join(log_dir, "training_state.json")


def get_epoch_log_path(log_dir):
    return os.path.join(log_dir, "epoch_logs.txt")


def get_evaluation_path(log_dir):
    return os.path.join(log_dir, "evaluation.txt")


def save_training_state(log_dir, epoch):
    with open(get_state_path(log_dir), "w") as f:
        json.dump({"epoch": epoch}, f)


def load_training_state(log_dir):
    path = get_state_path(log_dir)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f).get("epoch", 0)
    return 0


def load_weights_if_needed(model, log_dir, resume):
    weight_path = get_weight_path(log_dir)
    if not resume:
        return
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"Cannot resume because {weight_path} does not exist")
    model.load_weights(weight_path)
    print("Loaded Existing Weights")


def setup_interrupt_handler(model, log_dir):
    def handler(sig, frame):
        print("\nSaving Before Exit...")
        model.save_weights(get_weight_path(log_dir))
        current_epoch = getattr(model, "_current_epoch", 0)
        save_training_state(log_dir, current_epoch)
        print(f"Saved Epoch {current_epoch}")
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)


class EpochTracker(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super().__init__()
        self.model_ref = model

    def on_epoch_begin(self, epoch, logs=None):
        self.model_ref._current_epoch = epoch


class SaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, log_dir):
        super().__init__()
        self.model_ref = model
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model_ref.save_weights(get_weight_path(self.log_dir))
        save_training_state(self.log_dir, epoch + 1)
        print(f"Saved Epoch {epoch+1}")


class HistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, resume=False):
        super().__init__()
        self.log_dir = log_dir
        path = get_history_path(log_dir)
        if resume and os.path.exists(path):
            with open(path) as f:
                self.history = json.load(f)
        else:
            self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(float(value))
        with open(get_history_path(self.log_dir), "w") as f:
            json.dump(self.history, f, indent=4)


class EpochLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_path = get_epoch_log_path(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        with open(self.log_path, "a") as f:
            f.write("\n" + "=" * 140 + "\n")
            f.write(f"EPOCH : {epoch + 1}\n")
            f.write("=" * 140 + "\n")
            for key, value in logs.items():
                f.write(f"{key:<30}: {value:.6f}\n")


class PredictionLogger(tf.keras.callbacks.Callback):

    def __init__(self, val_ds, dataset, log_dir, max_length=TARGET_MAX_LENGTH,
                 num_samples=4, every_n_epochs=5):
        super().__init__()
        self.val_ds = val_ds
        self.dataset = dataset
        self.log_dir = log_dir
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.max_length = max_length

    def decode(self, encoder_input, max_length):
        target_vectorizer = self.dataset.dataset.target_vectorizer
        start_token = target_vectorizer(["[start]"]).numpy()[0][0]
        end_token = target_vectorizer(["[end]"]).numpy()[0][0]
        decoder_input = [start_token]

        for _ in range(max_length-1):
            padded_decoder_input = decoder_input + \
                [0]*((self.max_length-1)-len(decoder_input))
            padded_decoder_input = tf.constant(
                [padded_decoder_input], dtype=tf.int32)
            predictions = self.model(
                [encoder_input, padded_decoder_input], training=False)
            next_token_logits = predictions[:, len(decoder_input)-1, :]
            next_token = int(tf.argmax(next_token_logits[0]).numpy())
            decoder_input.append(next_token)
            if next_token == end_token:
                break

        vocab = target_vectorizer.get_vocabulary()
        decoded_text = " ".join(
            [vocab[token] for token in decoder_input[1:] if token not in [0, 1]])
        return decoded_text.replace("[end]", "").strip()

    def on_epoch_end(self, epoch, logs=None):
        final_epoch = epoch + 1 == self.params.get("epochs")
        if (epoch + 1) % self.every_n_epochs and not final_epoch:
            return
        prediction_dir = os.path.join(self.log_dir, "predictions")
        os.makedirs(prediction_dir, exist_ok=True)
        path = os.path.join(
            prediction_dir, f"epoch_{epoch+1}_predictions.txt")
        with open(path, "w", encoding="utf-8") as f:
            count = 0
            for (encoder_input, decoder_input), decoder_target in self.val_ds.shuffle(1000).take(20):
                for i in range(encoder_input.shape[0]):
                    source_text = self.dataset.decode_source(
                        encoder_input[i].numpy())
                    target_text = self.dataset.decode_target(
                        decoder_target[i].numpy())
                    predicted_text = self.decode(tf.expand_dims(
                        encoder_input[i], axis=0), self.max_length)

                    f.write("\n"+"="*140+"\n")
                    f.write("SOURCE:\n"+source_text+"\n\n")
                    f.write("TARGET:\n"+target_text+"\n\n")
                    f.write("PREDICTION:\n"+predicted_text+"\n")

                    count += 1
                    if count >= self.num_samples:
                        return


def evaluate_model(model, val_ds, log_dir):
    results = model.evaluate(val_ds, return_dict=True)
    with open(get_evaluation_path(log_dir), "w") as f:
        f.write("\n" + "=" * 140 + "\n")
        f.write("FINAL EVALUATION\n")
        f.write("=" * 140 + "\n")
        for key, value in results.items():
            f.write(f"{key:<30}: {value:.6f}\n")

    print("\nFinal Evaluation")

    for key, value in results.items():
        print(f"{key:<30}: {value:.6f}")


class TransformerLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = float(d_model)
        self.warmup_steps = int(warmup_steps)

    def __call__(self, step):
        step = tf.cast(tf.maximum(step, 1), tf.float32)
        scale = tf.math.rsqrt(tf.cast(self.d_model, tf.float32))
        return scale * tf.minimum(
            tf.math.rsqrt(step),
            step * tf.cast(self.warmup_steps, tf.float32) ** -1.5)

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none")


def masked_loss(y_true, y_pred):
    loss = loss_object(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss *= mask
    return tf.reduce_sum(loss, axis=-1) / tf.maximum(
        tf.reduce_sum(mask, axis=-1), 1.0)


def masked_accuracy(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(y_true, tf.int64)
    matches = tf.cast(tf.equal(y_true, y_pred), tf.float32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    matches *= mask
    return tf.reduce_sum(matches, axis=-1) / tf.maximum(
        tf.reduce_sum(mask, axis=-1), 1.0)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--dataset", type=str, default="english_french")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    strategy = setup_strategy(args.gpu)
    print(f"Number Of Devices : {strategy.num_replicas_in_sync}")

    dataset_paths = {
        "english_french": f"{DATASET_ROOT}/english-french",
        "english_german": f"{DATASET_ROOT}/english-german",
        "cornell_movie_dialogs": f"{DATASET_ROOT}/cornell_movie_dialogs_corpus",
        "cnn_dailymail": f"{DATASET_ROOT}/cnn_dailymail",
        "manythings_english_french": f"{DATASET_ROOT}/fra-eng",
        "wikilarge": f"{DATASET_ROOT}/wikilarge-text-simplification"
    }

    train_config = DATASET_CONFIGS[args.dataset]
    dataset_path = dataset_paths[args.dataset]
    training_epochs = args.epochs if args.epochs is not None else train_config.get("epochs", EPOCHS)
    if training_epochs < 1:
        raise ValueError("--epochs must be at least 1")
    replicas = strategy.num_replicas_in_sync
    if train_config["batch_size"] % replicas:
        raise ValueError(
            f"Global batch size {train_config['batch_size']} must be divisible "
            f"by {replicas} replicas")
    print(f"Global Batch Size : {train_config['batch_size']}")
    print(f"Per-Replica Batch Size : {train_config['batch_size'] // replicas}")
    print(f"Training Epochs : {training_epochs}")

    dataset = Dataset(
        dataset_name=args.dataset,
        dataset_path=dataset_path,
        batch_size=train_config["batch_size"],
        source_max_length=train_config["source_max_length"],
        target_max_length=train_config["target_max_length"],
        vocab_size=train_config["vocab_size"],
        validation_split=VALIDATION_SPLIT,
        seed=SEED,
        lowercase=LOWERCASE
    )
    print(train_config)
    train_ds, val_ds = dataset.load_data()
    default_train_steps = dataset.get_steps_per_epoch(training=True)
    default_validation_steps = dataset.get_steps_per_epoch(training=False)

    with strategy.scope():

        model = build_transformer_seq2seq_model(
            source_vocab_size=dataset.get_source_vocab_size(),
            target_vocab_size=dataset.get_target_vocab_size(),
            source_max_length=train_config["source_max_length"],
            target_max_length=train_config["target_max_length"],
            d_model=train_config["d_model"],
            num_heads=train_config["num_heads"],
            dff=train_config["dff"],
            num_encoder_layers=train_config["num_encoder_layers"],
            num_decoder_layers=train_config["num_decoder_layers"],
            dropout_rate=train_config["dropout_rate"]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=TransformerLearningRate(train_config["d_model"]),
                beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipnorm=1.0),
            loss=masked_loss,
            metrics=[masked_accuracy],
        )

    model.summary(expand_nested=True, show_trainable=True)

    tf.keras.utils.plot_model(
        model,
        to_file=f"transformer_seq2seq_model_{args.dataset}.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        expand_nested=True,
        show_layer_activations=True
    )

    log_dir = args.log_dir if args.log_dir else get_log_dir(args.dataset)

    os.makedirs(log_dir, exist_ok=True)
    start_epoch = 0
    if args.resume:
        load_weights_if_needed(model, log_dir, True)
        start_epoch = load_training_state(log_dir)
        print(f"Resuming From Epoch {start_epoch}")

    setup_interrupt_handler(model, log_dir)
    callbacks = [
        EpochTracker(model),
        SaveCallback(model, log_dir),
        HistoryCallback(log_dir, resume=args.resume),
        EpochLoggerCallback(log_dir),
        PredictionLogger(val_ds, dataset, log_dir,
                         max_length=train_config["target_max_length"]),
        tf.keras.callbacks.ModelCheckpoint(filepath=get_best_weight_path(
            log_dir), monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True,
            verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(
            log_dir, "tensorboard"), histogram_freq=1, write_graph=True, update_freq="epoch"),
        tf.keras.callbacks.CSVLogger(filename=os.path.join(
            log_dir, "training_log.csv"), append=args.resume)
    ]

    train_steps = 20 if args.smoke else args.steps_per_epoch or default_train_steps
    validation_steps = 5 if args.smoke else args.validation_steps or default_validation_steps
    print(f"Steps per epoch: {train_steps}; validation steps: {validation_steps}")
    # Repeating avoids distributed generator exhaustion; explicit counts retain
    # one finite data pass per epoch unless the user overrides them.
    fit_train_ds = train_ds.repeat()
    fit_val_ds = val_ds.repeat()

    model.fit(
        fit_train_ds,
        validation_data=fit_val_ds,
        epochs=1 if args.smoke else training_epochs, initial_epoch=start_epoch, callbacks=callbacks,
        steps_per_epoch=train_steps,
        validation_steps=validation_steps)
    print("\nTraining Complete")
    best_weight_path = get_best_weight_path(log_dir)
    if os.path.exists(best_weight_path):
        model.load_weights(best_weight_path)
        print("Loaded Best Validation Weights For Evaluation")
    evaluate_model(model, val_ds, log_dir)


if __name__ == "__main__":
    main()
