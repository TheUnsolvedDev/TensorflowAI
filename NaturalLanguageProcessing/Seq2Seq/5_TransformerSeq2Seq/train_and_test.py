# train_and_test.py

from config import *
from dataset import *
from model import *

import os
import sys
import json
import signal
import argparse
import datetime

import numpy as np
import tensorflow as tf
import silence_tensorflow.auto
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


AUTOTUNE = tf.data.AUTOTUNE


def setup_gpu(gpu_id):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpu_id == -1:
        print("Using All GPUs")
    elif 0 <= gpu_id < len(gpus):
        tf.config.set_visible_devices(gpus[gpu_id], "GPU")
        print(f"Using GPU {gpu_id}")
    else:
        print("Using CPU")


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
    if resume and os.path.exists(weight_path):
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
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
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

    def __init__(self, val_ds, dataset, log_dir, max_length=TARGET_MAX_LENGTH, num_samples=10, temperature=0.8, top_k=5):
        super().__init__()
        self.val_ds = val_ds
        self.dataset = dataset
        self.log_dir = log_dir
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k
        self.max_length = max_length

    def sample_next_token(self, logits, used_tokens):
        logits = logits/self.temperature
        values, indices = tf.math.top_k(logits, k=self.top_k)
        probs = tf.nn.softmax(values).numpy().flatten()
        candidate_tokens = indices.numpy().flatten()
        filtered = [(t, p) for t, p in zip(
            candidate_tokens, probs) if t not in used_tokens]
        if len(filtered) == 0:
            return candidate_tokens[0]
        filtered_tokens, filtered_probs = zip(*filtered)
        filtered_probs = np.array(filtered_probs)/np.sum(filtered_probs)
        return np.random.choice(filtered_tokens, p=filtered_probs)

    def decode(self, encoder_input, max_length):
        target_vectorizer = self.dataset.dataset.target_vectorizer
        start_token = target_vectorizer(["[start]"]).numpy()[0][0]
        end_token = target_vectorizer(["[end]"]).numpy()[0][0]
        decoder_input = [start_token]
        used_tokens = set()

        for _ in range(max_length-1):
            padded_decoder_input = decoder_input + \
                [0]*((self.max_length-1)-len(decoder_input))
            padded_decoder_input = tf.constant(
                [padded_decoder_input], dtype=tf.int32)
            predictions, _ = self.model(
                [encoder_input, padded_decoder_input], training=False)
            next_token_logits = predictions[:, len(decoder_input)-1, :]
            next_token = self.sample_next_token(next_token_logits, used_tokens)
            decoder_input.append(next_token)
            used_tokens.add(next_token)
            if next_token == end_token:
                break

        vocab = target_vectorizer.get_vocabulary()
        decoded_text = " ".join(
            [vocab[token] for token in decoder_input[1:] if token not in [0, 1]])
        return decoded_text.replace("[end]", "").strip()

    def on_epoch_end(self, epoch, logs=None):
        beam_dir = os.path.join(self.log_dir, "beam")
        os.makedirs(beam_dir, exist_ok=True)
        path = os.path.join(beam_dir, f"epoch_beam_{epoch+1}_predictions.txt")
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


class PredictionLoggerGreedy(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, dataset, log_dir, max_length=TARGET_MAX_LENGTH, num_samples=10):
        super().__init__()
        self.val_ds = val_ds
        self.dataset = dataset
        self.log_dir = log_dir
        self.num_samples = num_samples
        self.max_length = max_length

    def sample_next_token(self, logits):
        logits = logits.numpy()[0]
        next_token = np.argmax(logits)
        return int(next_token)

    def decode(self, encoder_input, max_length):
        target_vectorizer = self.dataset.dataset.target_vectorizer
        start_token = target_vectorizer(["[start]"]).numpy()[0][0]
        end_token = target_vectorizer(["[end]"]).numpy()[0][0]
        decoder_input = [start_token]
        for _ in range(max_length - 1):
            padded_decoder_input = decoder_input + \
                [0] * ((self.max_length - 1) - len(decoder_input))
            padded_decoder_input = tf.constant(
                [padded_decoder_input], dtype=tf.int32)
            predictions, _ = self.model(
                [encoder_input, padded_decoder_input], training=False)
            next_token_logits = predictions[:, len(decoder_input) - 1, :]
            next_token = self.sample_next_token(next_token_logits)
            decoder_input.append(next_token)
            if next_token == end_token:
                break

        vocab = target_vectorizer.get_vocabulary()
        decoded_text = " ".join(
            [
                vocab[token]
                for token in decoder_input[1:]
                if token not in [0, 1]
            ]
        )
        decoded_text = decoded_text.replace("[end]", "").strip()
        return decoded_text

    def on_epoch_end(self, epoch, logs=None):
        greed_dir = os.path.join(self.log_dir, "greed")
        os.makedirs(greed_dir, exist_ok=True)
        path = os.path.join(greed_dir, f"epoch_{epoch+1}_predictions.txt")
        with open(path, "w", encoding="utf-8") as f:
            count = 0
            for (encoder_input, decoder_input), decoder_target in self.val_ds.shuffle(1000).take(20):
                for i in range(encoder_input.shape[0]):
                    source_text = self.dataset.decode_source(
                        encoder_input[i].numpy()
                    )
                    target_text = self.dataset.decode_target(
                        decoder_target[i].numpy()
                    )
                    predicted_text = self.decode(
                        tf.expand_dims(encoder_input[i], axis=0),
                        self.max_length
                    )
                    f.write("\n" + "=" * 140 + "\n")
                    f.write("SOURCE:\n")
                    f.write(source_text + "\n\n")
                    f.write("TARGET:\n")
                    f.write(target_text + "\n\n")
                    f.write("PREDICTION:\n")
                    f.write(predicted_text + "\n")
                    count += 1
                    if count >= self.num_samples:
                        return


class AttentionLogger(tf.keras.callbacks.Callback):

    def __init__(self, val_ds, dataset, log_dir, max_length=TARGET_MAX_LENGTH):
        super().__init__()
        self.val_ds = val_ds
        self.dataset = dataset
        self.log_dir = log_dir
        self.max_length = max_length

    def greedy_decode_with_attention(self, encoder_input):
        target_vectorizer = self.dataset.dataset.target_vectorizer
        start_token = target_vectorizer(["[start]"]).numpy()[0][0]
        end_token = target_vectorizer(["[end]"]).numpy()[0][0]
        decoder_input = [start_token]
        collected_attention = []
        for _ in range(self.max_length - 1):
            padded_decoder_input = decoder_input + \
                [0] * ((self.max_length - 1) - len(decoder_input))
            padded_decoder_input = tf.constant(
                [padded_decoder_input], dtype=tf.int32)
            predictions, attention_scores = self.model(
                [encoder_input, padded_decoder_input], training=False)
            next_token_logits = predictions[:, len(decoder_input) - 1, :]
            next_token = int(tf.argmax(next_token_logits[0]).numpy())
            attention_step = attention_step = tf.reduce_mean(
                attention_scores[0, :, len(decoder_input)-1, :],
                axis=0).numpy()
            collected_attention.append(attention_step)
            decoder_input.append(next_token)
            if next_token == end_token:
                break
        return decoder_input, np.array(collected_attention)

    def on_epoch_end(self, epoch, logs=None):
        attention_dir = os.path.join(self.log_dir, "attention")
        os.makedirs(attention_dir, exist_ok=True)
        vocab_source = self.dataset.dataset.source_vectorizer.get_vocabulary()
        vocab_target = self.dataset.dataset.target_vectorizer.get_vocabulary()
        fig, axes = plt.subplots(2, 2, figsize=(24, 18))
        axes = axes.flatten()
        count = 0
        for (encoder_input, decoder_input), decoder_target in self.val_ds.take(1):
            for i in range(encoder_input.shape[0]):
                sample_encoder_input = tf.expand_dims(encoder_input[i], axis=0)
                predicted_tokens, attention_matrix = self.greedy_decode_with_attention(
                    sample_encoder_input)
                source_ids = encoder_input[i].numpy()
                source_tokens = [vocab_source[token]
                                 for token in source_ids if token > 1]
                target_tokens = [vocab_target[token]
                                 for token in predicted_tokens[1:] if token > 1]
                attention_matrix = attention_matrix[:len(
                    target_tokens), :len(source_tokens)]
                ax = axes[count]
                im = ax.imshow(attention_matrix, aspect="auto")
                ax.set_xticks(np.arange(len(source_tokens)))
                ax.set_xticklabels(source_tokens, rotation=45,
                                   ha="right", fontsize=8)
                ax.set_yticks(np.arange(len(target_tokens)))
                ax.set_yticklabels(target_tokens, fontsize=8)
                ax.set_xlabel("Source Tokens")
                ax.set_ylabel("Target Tokens")
                ax.set_title(f"Sample {count+1}")
                fig.colorbar(im, ax=ax)
                count += 1
                if count >= 4:
                    save_path = os.path.join(
                        attention_dir, f"epoch_{epoch+1}.png")
                    plt.tight_layout()
                    plt.savefig(save_path)
                    plt.close()
                    return


def evaluate_model(model, val_ds, dataset, log_dir):
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


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction="none")


def masked_loss(y_true, y_pred):
    loss = loss_object(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_accuracy(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(y_true, tf.int64)
    matches = tf.cast(tf.equal(y_true, y_pred), tf.float32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    matches *= mask
    return tf.reduce_sum(matches) / tf.reduce_sum(mask)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--dataset", type=str, default="english_french")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_dir", type=str, default=None)
    args = parser.parse_args()

    setup_gpu(args.gpu)
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce())
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

    with strategy.scope():

        model = build_transformer_seq2seq_model(
            source_vocab_size=train_config["vocab_size"],
            target_vocab_size=train_config["vocab_size"],
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
            loss=[masked_loss, None],
            metrics=[[masked_accuracy], None],
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
        HistoryCallback(log_dir),
        EpochLoggerCallback(log_dir),
        PredictionLogger(val_ds, dataset, log_dir,
                         max_length=train_config["target_max_length"]),
        PredictionLoggerGreedy(val_ds, dataset, log_dir,
                               max_length=train_config["target_max_length"]),
        AttentionLogger(val_ds, dataset, log_dir,
                        max_length=train_config["target_max_length"]),
        tf.keras.callbacks.ModelCheckpoint(filepath=get_best_weight_path(
            log_dir), monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(
            log_dir, "tensorboard"), histogram_freq=1, write_graph=True, update_freq="epoch"),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.9, patience=5, verbose=1),
        tf.keras.callbacks.CSVLogger(filename=os.path.join(
            log_dir, "training_log.csv"), append=True)
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS, initial_epoch=start_epoch, callbacks=callbacks)
    model.save_weights(get_weight_path(log_dir))
    print("\nTraining Complete")
    evaluate_model(model, val_ds, dataset, log_dir)


if __name__ == "__main__":
    main()
