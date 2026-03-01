import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import argparse

from config import *
from dataset import *
from model import *

# tf.debugging.enable_check_numerics()

def main():
    model_fn = squeezenet_model
    parser = argparse.ArgumentParser(description='Select GPU[0-3]:')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU number')
    parser.add_argument('--type', type=str, default='cifar10',
                        help='Dataset type', choices=['cifar10', 'fashion_mnist',
                           'mnist',  'cifar100', 'skin_cancer', 'cassava_leaf_disease', 'chest_xray', 'crop_disease'])
    args = parser.parse_args()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    if int(args.gpu) != -1:
        tf.config.experimental.set_visible_devices(
            physical_devices[args.gpu], 'GPU')

    dataset = Dataset()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
        # tf.keras.callbacks.ModelCheckpoint(
        #     filepath=f'{model_fn.__name__}_' + args.type + '.weights.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'./logs_{args.type}_{model_fn.__name__}', histogram_freq=1, write_graph=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    ]

    train_ds, validation_ds, test_ds, num_classes, channels = dataset.load_data(
        args.type)
    strategy = tf.distribute.MirroredStrategy()
    print(f'Training on dataset {args.type} with {strategy.num_replicas_in_sync} devices')

    with strategy.scope():
        model = model_fn(input_shape=(
            INPUT_SIZE[0], INPUT_SIZE[1], channels), num_classes=num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    model.summary(expand_nested=True)
    tf.keras.utils.plot_model(
        model, to_file=model_fn.__name__+'.png',show_shapes=True, show_layer_names=True)
    
    # training (capture history)
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # evaluation
    model.evaluate(test_ds)

    # === prepare log dir and save history ===
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from glob import glob

    log_dir = f'logs_{args.type}_{model_fn.__name__}'
    os.makedirs(log_dir, exist_ok=True)

    hist_save = {
        "accuracy": history.history.get("accuracy", []),
        "val_accuracy": history.history.get("val_accuracy", []),
        "loss": history.history.get("loss", []),
        "val_loss": history.history.get("val_loss", [])
    }
    with open(os.path.join(log_dir, "history.json"), "w") as fh:
        json.dump(hist_save, fh)

    # === Prediction visualization (5x5 grid) ===
    x_batch, y_batch = next(iter(test_ds))
    preds = model.predict(x_batch)
    true_labels = tf.argmax(y_batch, axis=1).numpy()
    pred_labels = tf.argmax(preds, axis=1).numpy()

    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i >= len(x_batch):
            ax.axis("off")
            continue
        img = x_batch[i].numpy()

        # normalize image to [0,255]
        img = img.astype("float32")
        img_min = img.min()
        img = img - img_min
        img_max = img.max()
        if img_max > 0:
            img = img / img_max
        img = (img * 255.0).clip(0, 255).astype("uint8")

        # handle channels
        if img.ndim == 3 and img.shape[-1] == 1:
            ax.imshow(img.squeeze(-1), cmap="gray", vmin=0, vmax=255)
        elif img.ndim == 3 and img.shape[-1] == 3:
            ax.imshow(img)
        elif img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img[..., 0], cmap="gray", vmin=0, vmax=255)

        correct = (true_labels[i] == pred_labels[i])
        color = "green" if correct else "red"
        ax.set_title(f"true={true_labels[i]} pred={pred_labels[i]}",
                     fontsize=9, color=color)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "predictions.png"))
    plt.close()

    # === Aggregate plot: accuracy vs epochs across all saved logs ===
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
