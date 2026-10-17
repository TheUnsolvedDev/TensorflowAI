import os

import numpy as np
try:
    import silence_tensorflow.auto  # noqa: F401
except ImportError:
    pass
import tensorflow as tf

from config import *

AUTOTUNE = tf.data.AUTOTUNE
VALID_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
BUILTIN_DATASETS = {"mnist", "fashion_mnist", "cifar10", "cifar100"}
FOLDER_DATASETS = {"celeba", "bedroom"}


class Dataset:
    def __init__(self, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
        self.data_types = SUPPORTED_DATASETS
        self.image_size = image_size
        self.batch_size = batch_size
        self.options = tf.data.Options()
        self.options.experimental_deterministic = not ENABLE_DATASET_NONDETERMINISM
        self._builtin = {}

    def _builtin_images(self, name):
        if name not in self._builtin:
            loaders = {
                "mnist": tf.keras.datasets.mnist.load_data,
                "fashion_mnist": tf.keras.datasets.fashion_mnist.load_data,
                "cifar10": tf.keras.datasets.cifar10.load_data,
                "cifar100": tf.keras.datasets.cifar100.load_data,
            }
            self._builtin[name] = loaders[name]()
        (images, _), _ = self._builtin[name]
        channels = 1 if name in {"mnist", "fashion_mnist"} else 3
        if channels == 1:
            images = images.reshape(-1, 28, 28, 1)
        return images[:MAX_DATASET_ITEMS] if MAX_DATASET_ITEMS else images, channels

    def _folder(self, name):
        paths = {
            "celeba": os.path.join(DATASET_PATH, "celeba-dataset", "img_align_celeba"),
            "bedroom": os.path.join(DATASET_PATH, "bedroom"),
        }
        path = paths[name]
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Dataset directory not found: {path}")
        return path

    def _image_paths(self, directory, recursive):
        if recursive:
            paths = [os.path.join(root, filename)
                     for root, _, files in os.walk(directory)
                     for filename in sorted(files)
                     if filename.lower().endswith(VALID_SUFFIXES)]
        else:
            paths = [os.path.join(directory, filename)
                     for filename in sorted(os.listdir(directory))
                     if filename.lower().endswith(VALID_SUFFIXES)]
        if MAX_DATASET_ITEMS:
            paths = paths[:MAX_DATASET_ITEMS]
        if not paths:
            raise FileNotFoundError(f"No images found in: {directory}")
        return np.asarray(paths)

    def _decode(self, path, channels=3):
        image = tf.io.decode_image(tf.io.read_file(path), channels=channels, expand_animations=False)
        image.set_shape([None, None, channels])
        return image

    def _prepare(self, image):
        image = tf.cast(image, tf.float32)
        side = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
        image = tf.image.resize_with_crop_or_pad(image, side, side)
        return tf.image.resize(image, self.image_size, antialias=True) / 127.5 - 1.0

    def _pipeline(self, values, decode=False, channels=3):
        dataset = tf.data.Dataset.from_tensor_slices(values)
        if decode:
            dataset = dataset.map(lambda path: self._decode(path, channels), num_parallel_calls=AUTOTUNE)
        dataset = dataset.map(self._prepare, num_parallel_calls=AUTOTUNE)
        if decode:
            dataset = dataset.map(tf.image.random_flip_left_right, num_parallel_calls=AUTOTUNE)
        dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        dataset = dataset.with_options(self.options)
        return dataset.batch(self.batch_size).prefetch(AUTOTUNE)

    def get_preview_batch(self, name, count=4):
        if name in BUILTIN_DATASETS:
            images, channels = self._builtin_images(name)
            return self._prepare(tf.convert_to_tensor(images[:count])), channels
        paths = self._image_paths(self._folder(name), recursive=name == "bedroom")[:count]
        return tf.stack([self._prepare(self._decode(path)) for path in paths]), 3

    def load_data(self, name=DEFAULT_DATASET):
        if name in BUILTIN_DATASETS:
            images, channels = self._builtin_images(name)
            return self._pipeline(images, channels=channels), channels
        if name in FOLDER_DATASETS:
            paths = self._image_paths(self._folder(name), recursive=name == "bedroom")
            return self._pipeline(paths, decode=True), 3
        raise ValueError(f"Unknown dataset type: {name}")


if __name__ == "__main__":
    data = Dataset()
    for name in SUPPORTED_DATASETS:
        dataset, channels = data.load_data(name)
        print(name, next(iter(dataset)).shape, channels)
