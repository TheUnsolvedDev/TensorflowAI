import argparse
import os
import tensorflow as tf
from config import *
from dataset import pair_dataset
from model import build_model


def configure_strategy(gpu_id):
    devices = tf.config.list_physical_devices("GPU")
    if 0 <= gpu_id < len(devices):
        tf.config.set_visible_devices(devices[gpu_id], "GPU"); devices = [devices[gpu_id]]
    for device in devices:
        try: tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError: pass
    if MIXED_PRECISION: tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try: return tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())
    except (RuntimeError, ValueError): return tf.distribute.MirroredStrategy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=CORPUS); parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=EPOCHS); parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); strategy = configure_strategy(args.gpu); print(f"Replicas: {strategy.num_replicas_in_sync}")
    dataset, vocabulary = pair_dataset(args.corpus); os.makedirs(LOG_DIR, exist_ok=True)
    with strategy.scope():
        model = build_model(len(vocabulary), EMBEDDING_DIM)
        model.compile(tf.keras.optimizers.Adam(LEARNING_RATE), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    weights = os.path.join(LOG_DIR, "embeddings.weights.h5")
    if args.resume and os.path.exists(weights): model.load_weights(weights)
    model.fit(dataset, epochs=1 if args.smoke else args.epochs, steps_per_epoch=20 if args.smoke else args.steps_per_epoch,
              callbacks=[tf.keras.callbacks.ModelCheckpoint(weights, save_weights_only=True, save_best_only=False)])
    with open(os.path.join(LOG_DIR, "vocabulary.txt"), "w", encoding="utf-8") as handle: handle.write("\n".join(vocabulary))


if __name__ == "__main__": main()
