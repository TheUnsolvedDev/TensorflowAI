"""Minimal training entry point."""
import argparse
from pathlib import Path

import tensorflow as tf


class PredictionGrid(tf.keras.callbacks.Callback):
    """Save 16 validation predictions after training completes."""

    def __init__(self, dataset, labels, output_path):
        super().__init__()
        self.dataset = dataset
        self.labels = labels
        self.output_path = output_path

    def on_train_end(self, logs=None):
        import matplotlib.pyplot as plt
        import numpy as np

        images, actual = next(iter(self.dataset.take(1)))
        predictions = self.model.predict(images, verbose=0).argmax(axis=1)
        actual = actual.numpy().argmax(axis=1)

        figure, axes = plt.subplots(4, 4, figsize=(10, 10))
        for index, axis in enumerate(axes.flat):
            axis.axis("off")
            if index >= len(images):
                continue
            image = np.clip(images[index].numpy(), 0, 255).astype("uint8")
            axis.imshow(image.squeeze(-1) if image.shape[-1] == 1 else image, cmap="gray")
            axis.set_title(
                f"True: {self.labels[actual[index]]}\nPred: {self.labels[predictions[index]]}",
                fontsize=8,
            )
        figure.tight_layout()
        figure.savefig(self.output_path)
        plt.close(figure)


def configure_devices(gpu_index, batch_size):
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("No GPU is available.")
    if gpu_index == -1:
        selected = physical
    elif 0 <= gpu_index < len(physical):
        selected = [physical[gpu_index]]
    else:
        raise ValueError(f"--gpu must be -1 or an index from 0 to {len(physical) - 1}")

    tf.config.set_visible_devices(selected, "GPU")
    for gpu in selected:
        tf.config.experimental.set_memory_growth(gpu, True)

    if len(selected) > 1:
        if batch_size % len(selected):
            raise ValueError(
                f"Batch size {batch_size} must be divisible by {len(selected)} selected GPUs"
            )
        return tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.NcclAllReduce()
        )
    return tf.distribute.OneDeviceStrategy("/GPU:0")


def main():
    from config import BATCH_SIZE, DATASET_PATH, EPOCHS, FROM_LOGITS, INPUT_SIZE, LEARNING_RATE, MODEL_FN, MODEL_KIND
    from dataset import DATASETS, Dataset
    import model as architecture

    parser = argparse.ArgumentParser(description="Train a classifier")
    parser.add_argument("--type", choices=DATASETS, default="cifar10")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Physical GPU index; use -1 for all GPUs with NCCL all-reduce")
    args = parser.parse_args()

    strategy = configure_devices(args.gpu, BATCH_SIZE)
    loader = Dataset(DATASET_PATH, INPUT_SIZE[:2], BATCH_SIZE)
    train, validation, _, classes, channels = loader.load_data(args.type)
    preview = validation
    labels = loader.manifest["classes"]

    build_args = {"input_shape": (INPUT_SIZE[0], INPUT_SIZE[1], channels), "num_classes": classes}
    if MODEL_KIND == "capsule":
        build_args["routing_iterations"] = 3
    with strategy.scope():
        network = getattr(architecture, MODEL_FN)(**build_args)
        loss = architecture.margin_loss if MODEL_KIND == "capsule" else tf.keras.losses.CategoricalCrossentropy(from_logits=FROM_LOGITS)
        network.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
            loss=loss,
            metrics=["accuracy"],
        )

    if MODEL_KIND == "capsule":
        train = train.map(lambda x, y: ((x, y), y))
        validation = validation.map(lambda x, y: ((x, y), y))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        PredictionGrid(
            preview,
            labels,
            Path(__file__).resolve().parent / f"prediction_grid_{args.type}.png",
        ),
    ]
    network.fit(train, validation_data=validation, epochs=EPOCHS, callbacks=callbacks)


if __name__ == "__main__":
    main()
