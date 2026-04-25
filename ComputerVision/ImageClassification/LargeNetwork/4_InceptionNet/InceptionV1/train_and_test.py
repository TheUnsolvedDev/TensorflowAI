import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import argparse
from config import *
from dataset import *
from model import *


def main():
    model_fn = inception1_model
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--type', type=str, default='cifar10',
                        choices=['cifar10', 'fashion_mnist', 'mnist', 'cifar100', 'skin_cancer', 'cassava_leaf_disease', 'chest_xray', 'crop_disease'])
    args = parser.parse_args()

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for d in physical_devices:
        tf.config.experimental.set_memory_growth(d, True)
    if int(args.gpu) != -1:
        tf.config.experimental.set_visible_devices(
            physical_devices[args.gpu], 'GPU')

    dataset = Dataset()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'./logs_{args.type}_{model_fn.__name__}', histogram_freq=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=4)
    ]

    train_ds, validation_ds, test_ds, num_classes, channels = dataset.load_data(
        args.type)
    strategy = tf.distribute.MirroredStrategy()
    print(f'{args.type} | devices={strategy.num_replicas_in_sync}')

    with strategy.scope():
        model = model_fn(input_shape=(
            INPUT_SIZE[0], INPUT_SIZE[1], channels), num_classes=num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=['categorical_crossentropy']*3,
            metrics=['accuracy']*3
        )

    history = model.fit(train_ds, validation_data=validation_ds,
                        epochs=EPOCHS, callbacks=callbacks)
    model.evaluate(test_ds)

    import os
    import json
    import matplotlib.pyplot as plt
    from glob import glob

    log_dir = f'logs_{args.type}_{model_fn.__name__}'
    os.makedirs(log_dir, exist_ok=True)

    h = history.history

    # === aggregate metrics ===
    def avg_metric(keys):
        vals = [h[k] for k in keys if k in h]
        return np.mean(np.array(vals), axis=0).tolist() if vals else []

    acc_keys = [k for k in h if 'accuracy' in k and not k.startswith('val')]
    val_acc_keys = [k for k in h if 'val_' in k and 'accuracy' in k]
    loss_keys = [k for k in h if 'loss' in k and not k.startswith(
        'val') and k != 'loss']
    val_loss_keys = [
        k for k in h if 'val_' in k and 'loss' in k and k != 'val_loss']

    hist_save = {
        "accuracy": avg_metric(acc_keys),
        "val_accuracy": avg_metric(val_acc_keys),
        "loss": avg_metric(loss_keys),
        "val_loss": avg_metric(val_loss_keys)
    }

    with open(os.path.join(log_dir, "history.json"), "w") as f:
        json.dump(hist_save, f)

    # === prediction ===
    x_batch, y_batch = next(iter(test_ds))
    preds = model.predict(x_batch)

    true_labels = tf.argmax(y_batch[0], axis=1).numpy()
    pred_labels = tf.argmax(preds[0], axis=1).numpy()

    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i >= len(x_batch):
            ax.axis("off")
            continue
        img = x_batch[i].numpy()
        img = (img-img.min())/(img.max()+1e-8)*255
        img = img.astype("uint8")

        if img.shape[-1] == 1:
            ax.imshow(img.squeeze(), cmap="gray")
        else:
            ax.imshow(img)

        correct = true_labels[i] == pred_labels[i]
        ax.set_title(f"T={true_labels[i]} P={pred_labels[i]}",
                     color="green" if correct else "red", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "predictions.png"))
    plt.close()

    # === aggregate plot ===
    all_log_dirs = sorted(glob(f'logs_*_{model_fn.__name__}'))
    plt.figure(figsize=(10, 6))
    found_any = False
    for ld in all_log_dirs:
        hist_path = os.path.join(ld, "history.json")
        if not os.path.exists(hist_path):
            continue
        with open(hist_path, "r") as fh:
            h = json.load(fh)
        # prefer val_accuracy if available, else accuracy
        acc = h.get("val_accuracy") or h.get("accuracy") or []
        if not acc:
            continue
        epochs = np.arange(1, len(acc) + 1)
        # dataset name extraction: logs_{type}_{model}
        label = ld.replace(f'logs_', '').replace(f'_{model_fn.__name__}', '')
        plt.plot(epochs, acc, marker='o', label=label)
        found_any = True

    if found_any:
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (validation preferred)")
        plt.title(f"Accuracy vs Epochs for {model_fn.__name__}")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(".", f"all_datasets_accuracy_{model_fn.__name__}.png"))
        plt.close()
    else:
        # no histories found -> do nothing (or optionally warn)
        pass


if __name__ == '__main__':
    main()
