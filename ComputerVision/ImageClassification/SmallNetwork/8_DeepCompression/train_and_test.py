import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import argparse
import os
import json
import matplotlib.pyplot as plt
import tqdm
from glob import glob

from config import *
from dataset import *
from model import *


def build_model(input_shape, num_classes):
    return alexnet_model(input_shape=input_shape, num_classes=num_classes)


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks(model_name, dataset_type):
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'./logs_{dataset_type}_{model_name}',
            histogram_freq=1,
            write_graph=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=4,
            min_delta=1e-4
        )
    ]


def train_model(model, train_ds, val_ds, callbacks):
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )


def evaluate_model(model, test_ds):
    return model.evaluate(test_ds)


def compute_pruning_mask(weights, sparsity):
    flat = tf.reshape(tf.abs(weights), [-1])
    total = tf.cast(tf.size(flat), tf.float32)
    k = tf.cast(total * (1.0 - sparsity), tf.int32)
    k = tf.clip_by_value(k, 1, tf.size(flat) - 1)
    topk = tf.math.top_k(flat, k=k)
    threshold = tf.reduce_min(topk.values)
    return tf.cast(tf.abs(weights) >= threshold, weights.dtype)


def is_prunable_layer(layer, idx, total):
    if not hasattr(layer, "kernel"):
        return False
    if idx == 0:
        return False
    if idx == total - 1:
        return False
    return isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense))


def apply_pruning_to_model(model, sparsity=0.8):
    masks = []
    total = len(model.layers)

    for i, layer in enumerate(model.layers):
        if is_prunable_layer(layer, i, total):
            w = layer.kernel
            mask = compute_pruning_mask(w, sparsity)
            layer.kernel.assign(w * mask)
            masks.append((layer.name, mask))

    return masks


def get_layer_by_name(model, name):
    for l in model.layers:
        if l.name == name:
            return l
    return None


def make_distributed_train_step(model, optimizer, strategy, masks):

    mask_dict = {name: mask for (name, mask) in masks}

    @tf.function
    def distributed_train_step(dist_inputs):
        def replica_step(inputs):
            x, y = inputs
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = tf.keras.losses.categorical_crossentropy(y, logits)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)

            grads_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
            optimizer.apply_gradients(grads_vars)

            for layer in model.layers:
                if layer.name in mask_dict and hasattr(layer, "kernel"):
                    layer.kernel.assign(layer.kernel * mask_dict[layer.name])

            return loss

        per_replica_loss = strategy.run(replica_step, args=(dist_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)

    return distributed_train_step


def fine_tune_distributed(model, train_ds, val_ds, strategy, masks, epochs):
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    dist_train_ds = strategy.experimental_distribute_dataset(train_ds)

    train_step = make_distributed_train_step(model, optimizer, strategy, masks)

    for epoch in tqdm.tqdm(range(epochs)):
        losses = []

        for batch in tqdm.tqdm(dist_train_ds):
            loss = train_step(batch)
            losses.append(loss.numpy())

        val_metrics = model.evaluate(val_ds, verbose=0)
        print(f"[FineTune] Epoch {epoch+1}: loss={np.mean(losses):.4f}, val_loss={val_metrics[0]:.4f}")


def export_tflite(model, name):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(f"{name}.tflite", "wb") as f:
        f.write(tflite_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    quant_model = converter.convert()

    with open(f"{name}_quant.tflite", "wb") as f:
        f.write(quant_model)


def save_history(history, log_dir):
    os.makedirs(log_dir, exist_ok=True)

    hist = {
        "accuracy": history.history.get("accuracy", []),
        "val_accuracy": history.history.get("val_accuracy", []),
        "loss": history.history.get("loss", []),
        "val_loss": history.history.get("val_loss", [])
    }

    with open(os.path.join(log_dir, "history.json"), "w") as f:
        json.dump(hist, f)


def visualize_predictions(model, test_ds, log_dir):
    x_batch, y_batch = next(iter(test_ds))
    preds = model.predict(x_batch)

    true_labels = tf.argmax(y_batch, axis=1).numpy()
    pred_labels = tf.argmax(preds, axis=1).numpy()

    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i >= len(x_batch):
            break

        img = x_batch[i].numpy()
        img = (img - img.min()) / (img.max() + 1e-8)

        if img.shape[-1] == 1:
            ax.imshow(img.squeeze(), cmap="gray")
        else:
            ax.imshow(img)

        color = "green" if true_labels[i] == pred_labels[i] else "red"
        ax.set_title(f"T={true_labels[i]} P={pred_labels[i]}", color=color)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "predictions.png"))
    plt.close()


def plot_all_logs(model_name):
    all_logs = sorted(glob(f'logs_*_{model_name}'))
    plt.figure(figsize=(10, 6))

    for log_dir in all_logs:
        hist_path = os.path.join(log_dir, "history.json")
        if not os.path.exists(hist_path):
            continue

        with open(hist_path, "r") as f:
            hist = json.load(f)

        acc = hist.get("val_accuracy") or hist.get("accuracy")
        if not acc:
            continue

        epochs = np.arange(1, len(acc) + 1)
        label = log_dir.replace("logs_", "").replace(f"_{model_name}", "")

        plt.plot(epochs, acc, marker='o', label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"all_datasets_accuracy_{model_name}.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--type', type=str, default='cifar10',
                        choices=['cifar10', 'fashion_mnist', 'mnist', 'cifar100',
                                 'skin_cancer', 'cassava_leaf_disease',
                                 'chest_xray', 'crop_disease'])
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

    if args.gpu != -1 and len(gpus) > 0:
        tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')

    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce())
    print(f"Devices: {strategy.num_replicas_in_sync}")

    dataset = Dataset()
    train_ds, val_ds, test_ds, num_classes, channels = dataset.load_data(args.type)

    with strategy.scope():
        model = build_model(
            input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], channels),
            num_classes=num_classes
        )
        model = compile_model(model)

    model.summary()

    callbacks = get_callbacks("alexnet_model", args.type)

    history = train_model(model, train_ds, val_ds, callbacks)

    evaluate_model(model, test_ds)

    print("\nApplying pruning...\n")
    masks = apply_pruning_to_model(model, sparsity=0.8)

    print("\nFine-tuning pruned model...\n")
    fine_tune_distributed(model, train_ds, val_ds, strategy, masks, epochs=EPOCHS)

    evaluate_model(model, test_ds)

    export_tflite(model, f"alexnet_{args.type}_pruned")

    log_dir = f'logs_{args.type}_alexnet_model'
    save_history(history, log_dir)
    visualize_predictions(model, test_ds, log_dir)
    plot_all_logs("alexnet_model")


if __name__ == "__main__":
    main()