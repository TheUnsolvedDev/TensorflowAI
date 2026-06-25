import silence_tensorflow.auto
import tensorflow as tf
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from config import *
from dataset import Dataset
from model import lenet5_model, alexnet_model, Distiller


def setup_gpu(gpu_id):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) == 0:
        return
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    if gpu_id != -1:
        tf.config.experimental.set_visible_devices(physical_devices[gpu_id], 'GPU')


def get_model(name, input_shape, num_classes):
    if name == "lenet":
        return lenet5_model(input_shape, num_classes)
    elif name == "alexnet":
        return alexnet_model(input_shape, num_classes)
    else:
        raise ValueError(name)


def get_callbacks(log_dir, model_name, dataset_name):
    os.makedirs(log_dir, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(log_dir, f"{model_name}_{dataset_name}.weights.h5"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=4,
            verbose=1,
            mode='auto',
            min_delta=1e-4,
            cooldown=0,
            min_lr=0
        )
    ]


def save_history(history, log_dir):
    hist_save = {
        "accuracy": history.history.get("accuracy", []),
        "val_accuracy": history.history.get("val_accuracy", []),
        "loss": history.history.get("loss", []),
        "val_loss": history.history.get("val_loss", [])
    }
    with open(os.path.join(log_dir, "history.json"), "w") as fh:
        json.dump(hist_save, fh)


def visualize_predictions(model, test_ds, log_dir):
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
        img = img.astype("float32")
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        img = (img * 255.0).clip(0, 255).astype("uint8")

        if img.ndim == 3 and img.shape[-1] == 1:
            ax.imshow(img.squeeze(-1), cmap="gray", vmin=0, vmax=255)
        elif img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)

        correct = (true_labels[i] == pred_labels[i])
        color = "green" if correct else "red"
        ax.set_title(f"t={true_labels[i]} p={pred_labels[i]}", fontsize=9, color=color)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "predictions.png"))
    plt.close()


def aggregate_plots(prefix):
    all_log_dirs = sorted(glob(f'{prefix}_*'))
    plt.figure(figsize=(10, 6))
    found_any = False

    for ld in all_log_dirs:
        hist_path = os.path.join(ld, "history.json")
        if not os.path.exists(hist_path):
            continue

        with open(hist_path, "r") as fh:
            h = json.load(fh)

        acc = h.get("val_accuracy") or h.get("accuracy") or []
        if not acc:
            continue

        epochs = np.arange(1, len(acc) + 1)
        label = ld.replace(f'{prefix}_', '')
        plt.plot(epochs, acc, marker='o', label=label)
        found_any = True

    if found_any:
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{prefix} Accuracy vs Epochs")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"{prefix}_aggregate.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--type', type=str, default='cifar10')
    args = parser.parse_args()

    setup_gpu(args.gpu)

    dataset = Dataset()
    train_ds, val_ds, test_ds, num_classes, channels = dataset.load_data(args.type)
    input_shape = (INPUT_SIZE[0], INPUT_SIZE[1], channels)

    strategy = tf.distribute.MirroredStrategy()

    teacher_log_dir = f'logs_teacher_{args.type}'
    distill_log_dir = f'logs_distill_{args.type}'
    student_log_dir = f'logs_student_{args.type}'

    with strategy.scope():
        teacher = get_model("alexnet", input_shape, num_classes)
        teacher.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    history_teacher = teacher.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=get_callbacks(teacher_log_dir, "teacher", args.type)
    )
    teacher.evaluate(test_ds)
    save_history(history_teacher, teacher_log_dir)
    visualize_predictions(teacher, test_ds, teacher_log_dir)

    with strategy.scope():
        student = get_model("lenet", input_shape, num_classes)
        distiller = Distiller(student=student, teacher=teacher)

        distiller.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
            student_loss_fn=tf.keras.losses.CategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE
            ),
            distillation_loss_fn=tf.keras.losses.KLDivergence(
                reduction=tf.keras.losses.Reduction.NONE
            ),
            alpha=0.1,
            temperature=5,
            global_batch_size=BATCH_SIZE
        )

    history_distill = distiller.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=get_callbacks(distill_log_dir, "distill", args.type)
    )
    distiller.evaluate(test_ds)
    save_history(history_distill, distill_log_dir)
    visualize_predictions(distiller, test_ds, distill_log_dir)

    with strategy.scope():
        student_scratch = get_model("lenet", input_shape, num_classes)
        student_scratch.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    history_student = student_scratch.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=get_callbacks(student_log_dir, "student", args.type)
    )
    student_scratch.evaluate(test_ds)
    save_history(history_student, student_log_dir)
    visualize_predictions(student_scratch, test_ds, student_log_dir)

    aggregate_plots("logs_teacher")
    aggregate_plots("logs_distill")
    aggregate_plots("logs_student")


if __name__ == "__main__":
    main()