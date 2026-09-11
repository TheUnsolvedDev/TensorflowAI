import argparse
import os
import tensorflow as tf
from config import *
from dataset import pair_dataset
from model import build_model

def strategy_for_runtime():
    for gpu in tf.config.list_physical_devices("GPU"):
        try: tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError: pass
    if MIXED_PRECISION: tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try: return tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())
    except (RuntimeError, ValueError): return tf.distribute.MirroredStrategy()

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--corpus", default=CORPUS); parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=None); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); data, vocab = pair_dataset(args.corpus); strategy = strategy_for_runtime(); print(f"Replicas: {strategy.num_replicas_in_sync}")
    os.makedirs(LOG_DIR, exist_ok=True); weights = os.path.join(LOG_DIR, "embeddings.weights.h5")
    with strategy.scope():
        model = build_model(len(vocab), EMBEDDING_DIM); model.compile(tf.keras.optimizers.Adam(LEARNING_RATE), loss="mse")
    if args.resume and os.path.exists(weights): model.load_weights(weights)
    model.fit(data, epochs=1 if args.smoke else args.epochs, steps_per_epoch=20 if args.smoke else args.steps_per_epoch,
              callbacks=[tf.keras.callbacks.ModelCheckpoint(weights, save_weights_only=True)])
    with open(os.path.join(LOG_DIR, "vocabulary.txt"), "w", encoding="utf-8") as handle: handle.write("\n".join(vocab))
if __name__ == "__main__": main()
