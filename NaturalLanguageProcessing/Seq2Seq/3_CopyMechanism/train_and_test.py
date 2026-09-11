import argparse
import os
import tensorflow as tf
from config import *
from dataset import Dataset
from model import build_copy_seq2seq_model

PATHS = {"english_french": "english-french", "english_german": "english-german", "manythings_english_french": "fra-eng", "cornell_movie_dialogs": "cornell_movie_dialogs_corpus", "cnn_dailymail": "cnn_dailymail", "wikilarge": "wikilarge-text-simplification"}

def masked_loss(y_true, y_pred):
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), loss.dtype)
    return tf.math.divide_no_nan(tf.reduce_sum(loss * mask), tf.reduce_sum(mask))

def masked_accuracy(y_true, y_pred):
    correct = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)), tf.float32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    return tf.math.divide_no_nan(tf.reduce_sum(correct * mask), tf.reduce_sum(mask))

def strategy_for_runtime(gpu_id=-1):
    gpus = tf.config.list_physical_devices("GPU")
    if 0 <= gpu_id < len(gpus):
        tf.config.set_visible_devices(gpus[gpu_id], "GPU")
        gpus = [gpus[gpu_id]]
    for gpu in gpus:
        try: tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError: pass
    try: return tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())
    except (RuntimeError, ValueError): return tf.distribute.MirroredStrategy()

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", choices=PATHS, default="manythings_english_french")
    parser.add_argument("--epochs", type=int, default=EPOCHS); parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=None); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args(); strategy = strategy_for_runtime(args.gpu); print(f"Replicas: {strategy.num_replicas_in_sync}")
    data = Dataset(args.dataset, os.path.join(DATASET_ROOT, PATHS[args.dataset]), batch_size=BATCH_SIZE, source_max_length=SOURCE_MAX_LENGTH, target_max_length=TARGET_MAX_LENGTH, vocab_size=VOCAB_SIZE)
    train_ds, val_ds = data.load_data(); default_train_steps = data.get_steps_per_epoch(True); default_validation_steps = data.get_steps_per_epoch(False); log_dir = os.path.join(LOG_DIR, args.dataset, "CopyMechanism"); os.makedirs(log_dir, exist_ok=True)
    with strategy.scope():
        model = build_copy_seq2seq_model(data.get_source_vocab_size(), data.get_target_vocab_size(), SOURCE_MAX_LENGTH, TARGET_MAX_LENGTH, EMBEDDING_DIM, ENCODER_UNITS)
        model.compile(tf.keras.optimizers.Adam(LEARNING_RATE), loss=masked_loss, metrics=[masked_accuracy])
    path = os.path.join(log_dir, "model.weights.h5")
    if args.resume and os.path.exists(path): model.load_weights(path)
    train_steps = 20 if args.smoke else args.steps_per_epoch or default_train_steps; validation_steps = 5 if args.smoke else args.validation_steps or default_validation_steps
    print(f"Steps per epoch: {train_steps}; validation steps: {validation_steps}")
    fit_train_ds = train_ds.repeat(); fit_val_ds = val_ds.repeat()
    model.fit(fit_train_ds, validation_data=fit_val_ds, epochs=1 if args.smoke else args.epochs,
              steps_per_epoch=train_steps, validation_steps=validation_steps,
              callbacks=[tf.keras.callbacks.ModelCheckpoint(path, save_weights_only=True, save_best_only=True, monitor="val_loss"), tf.keras.callbacks.CSVLogger(os.path.join(log_dir, "history.csv"), append=args.resume)])

if __name__ == "__main__": main()
