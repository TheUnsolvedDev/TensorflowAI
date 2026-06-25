
from config import *
from dataset import *
from model import *
import silence_tensorflow.auto
import tensorflow as tf
import argparse
import signal
import json
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def setup_gpu(gpu_id):
    gpus = tf.config.list_physical_devices("GPU")
    [tf.config.experimental.set_memory_growth(g, True) for g in gpus]
    if gpu_id == -1:
        print("Using All GPUs")
    elif 0 <= gpu_id < len(gpus):
        tf.config.set_visible_devices(gpus[gpu_id], "GPU")
        print(f"Using GPU {gpu_id}")
    else:
        print("Using CPU")


def get_log_dir(dataset_name):
    return f"logs/{dataset_name}/VanillaRNN"


def get_weight_path(log_dir):
    return os.path.join(log_dir, "model.weights.h5")


def get_state_path(log_dir):
    return os.path.join(log_dir, "training_state.json")


def get_history_path(log_dir):
    return os.path.join(log_dir, "history.json")


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
        save_training_state(self.log_dir, epoch+1)
        print(f"Saved Epoch {epoch+1}")


class HistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(float(v))
        with open(get_history_path(self.log_dir), "w") as f:
            json.dump(self.history, f, indent=4)


class PredictionLogger(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, dataset, log_dir, num_samples=20):
        super().__init__()
        self.val_ds = val_ds
        self.dataset = dataset
        self.log_dir = log_dir
        self.num_samples = num_samples

    def on_epoch_end(self, epoch, logs=None):
        path = os.path.join(self.log_dir, f"epoch_{epoch+1}_predictions.txt")
        with open(path, "w", encoding="utf-8") as f:
            count = 0
            for x, y in self.val_ds:
                preds = self.model(x, training=False)
                preds = tf.argmax(preds, axis=-1)
                for i in range(len(x)):
                    text = self.dataset.decode_tokens(x[i].numpy())
                    true_label = int(y[i].numpy())
                    pred_label = int(preds[i].numpy())
                    f.write(f"TEXT:\n{text}\n")
                    f.write(f"TRUE LABEL : {true_label}\n")
                    f.write(f"PRED LABEL : {pred_label}\n")
                    f.write(f"MATCH      : {true_label == pred_label}\n")
                    f.write("="*120+"\n")
                    count += 1
                    if count >= self.num_samples:
                        return


def evaluate_model(model, val_ds, dataset):
    loss, acc = model.evaluate(val_ds)
    print(f"\nValidation Loss     : {loss:.4f}")
    print(f"Validation Accuracy : {acc:.4f}")
    for x, y in val_ds.take(1):
        preds = model(x, training=False)
        preds = tf.argmax(preds, axis=-1)
        for i in range(5):
            text = dataset.decode_tokens(x[i].numpy())
            true_label = int(y[i].numpy())
            pred_label = int(preds[i].numpy())
            print("\n"+"="*120)
            print("TEXT:")
            print(text)
            print(f"\nTRUE  : {true_label}")
            print(f"PRED  : {pred_label}")
            print(f"MATCH : {true_label == pred_label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--dataset", type=str, default="ag_news")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    setup_gpu(args.gpu)
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.NcclAllReduce())
    print(f"Number Of Devices: {strategy.num_replicas_in_sync}")

    dataset_path = f"{DATASET_ROOT}/{args.dataset}"
    dataset = Dataset(dataset_name=args.dataset, dataset_path=dataset_path, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
                      vocab_size=VOCAB_SIZE, validation_split=VALIDATION_SPLIT, seed=SEED, lowercase=LOWERCASE)
    train_ds, val_ds = dataset.load_data()
    # train_ds = strategy.experimental_distribute_dataset(train_ds)
    # val_ds = strategy.experimental_distribute_dataset(val_ds)

    with strategy.scope():
        model = build_bidirectional_rnn(vocab_size=dataset.get_vocab_size(
        ), num_classes=dataset.get_num_classes(), embedding_dim=256, hidden_dim=256, dropout=0.3)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

    log_dir = get_log_dir(args.dataset)
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
        PredictionLogger(val_ds, dataset, log_dir)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
              initial_epoch=start_epoch, callbacks=callbacks)
    model.save_weights(get_weight_path(log_dir))
    print("\nTraining Complete")
    evaluate_model(model, val_ds, dataset)


if __name__ == "__main__":
    main()
