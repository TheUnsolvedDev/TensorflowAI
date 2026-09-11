import os
import shutil
import tempfile

import numpy as np
import silence_tensorflow.auto
import tensorflow as tf

from config import *

AUTOTUNE = tf.data.AUTOTUNE
LARGE_IMAGE_DATASETS = {"celeba", "anime_faces", "bedroom"}
VALID_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class Dataset:
    def __init__(self):
        self.data_types = [
            "cifar10",
            "fashion_mnist",
            "mnist",
            "cifar100",
            "celeba",
            "anime_faces",
            "bedroom",
        ]

        self.batch_size = BATCH_SIZE
        self.img_shape = IMAGE_SIZE
        self.cache_root = os.path.join(
            tempfile.gettempdir(),
            "tensorflowai_gan_cache",
            os.path.basename(os.path.dirname(__file__)),
        )
        self.train_options = tf.data.Options()
        self.train_options.experimental_deterministic = not ENABLE_DATASET_NONDETERMINISM
        self._builtin_datasets = {}

    def _get_builtin_dataset(self, dataset_name):
        if dataset_name not in self._builtin_datasets:
            loaders = {
                "mnist": tf.keras.datasets.mnist.load_data,
                "cifar10": tf.keras.datasets.cifar10.load_data,
                "fashion_mnist": tf.keras.datasets.fashion_mnist.load_data,
                "cifar100": tf.keras.datasets.cifar100.load_data,
            }
            self._builtin_datasets[dataset_name] = loaders[dataset_name]()
        return self._builtin_datasets[dataset_name]

    def _list_image_files(self, image_dir, recursive=False):
        image_paths = []
        if recursive:
            for root, _, filenames in os.walk(image_dir):
                for filename in sorted(filenames):
                    if filename.lower().endswith(VALID_SUFFIXES):
                        image_paths.append(os.path.join(root, filename))
        else:
            image_paths = [
                os.path.join(image_dir, filename)
                for filename in sorted(os.listdir(image_dir))
                if filename.lower().endswith(VALID_SUFFIXES)
            ]
        return np.array(image_paths)

    def _target_shape(self, dataset_type):
        if dataset_type in LARGE_IMAGE_DATASETS:
            return (IMAGE_SIZE[0] * 2, IMAGE_SIZE[1] * 2)
        return self.img_shape

    def _decode_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        return image

    def process_images(self, image, decode=False, dataset_type="cifar10"):
        if decode:
            image = self._decode_image(image)

        image = tf.image.resize(image, self._target_shape(dataset_type))
        image = tf.cast(image, tf.float32)
        return (image - 127.5) / 127.5

    def _cache_path(self, dataset_type):
        target_h, target_w = self._target_shape(dataset_type)
        cache_dir = os.path.join(self.cache_root, dataset_type)
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"images_{target_h}x{target_w}")

    def cleanup_cache(self):
        shutil.rmtree(self.cache_root, ignore_errors=True)

    def build_dataset(self, images, decode=False, dataset_type="cifar10", shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices(images)
        ds = ds.map(
            lambda image: self.process_images(
                image,
                decode=decode,
                dataset_type=dataset_type,
            ),
            num_parallel_calls=AUTOTUNE,
        )

        cache_mode = DECODE_CACHE_MODE if decode else ARRAY_CACHE_MODE
        if cache_mode == "disk":
            ds = ds.cache(self._cache_path(dataset_type))
        elif cache_mode == "memory":
            ds = ds.cache()

        if shuffle:
            ds = ds.shuffle(
                SHUFFLE_BUFFER_SIZE,
                reshuffle_each_iteration=True,
            )
            ds = ds.with_options(self.train_options)

        ds = ds.batch(self.batch_size)
        return ds.prefetch(AUTOTUNE)

    def _prepare_builtin_images(self, dataset_type):
        if dataset_type == "mnist":
            channels = 1
            (train_images, _), (test_images, _) = self._get_builtin_dataset("mnist")
            images = np.concatenate([train_images, test_images]).reshape(-1, 28, 28, 1)
        elif dataset_type == "cifar10":
            channels = 3
            (train_images, _), (test_images, _) = self._get_builtin_dataset("cifar10")
            images = np.concatenate([train_images, test_images])
        elif dataset_type == "fashion_mnist":
            channels = 1
            (train_images, _), (test_images, _) = self._get_builtin_dataset("fashion_mnist")
            images = np.concatenate([train_images, test_images]).reshape(-1, 28, 28, 1)
        elif dataset_type == "cifar100":
            channels = 3
            (train_images, _), (test_images, _) = self._get_builtin_dataset("cifar100")
            images = np.concatenate([train_images, test_images])
        else:
            raise ValueError(f"Unknown builtin dataset type: {dataset_type}")
        return images, channels

    def load_data(self, dataset_type="mnist"):
        if dataset_type in {"mnist", "cifar10", "fashion_mnist", "cifar100"}:
            images, channels = self._prepare_builtin_images(dataset_type)
            ds = self.build_dataset(images, dataset_type=dataset_type)
        elif dataset_type == "celeba":
            channels = 3
            image_dir = os.path.join(DATASET_PATH, "celeba-dataset", "img_align_celeba")
            images = self._list_image_files(image_dir)
            ds = self.build_dataset(images, decode=True, dataset_type=dataset_type)
        elif dataset_type == "anime_faces":
            channels = 3
            image_dir = os.path.join(DATASET_PATH, "anime_face_images")
            images = self._list_image_files(image_dir)
            ds = self.build_dataset(images, decode=True, dataset_type=dataset_type)
        elif dataset_type == "bedroom":
            channels = 3
            image_dir = os.path.join(DATASET_PATH, "bedroom")
            images = self._list_image_files(image_dir, recursive=True)
            ds = self.build_dataset(images, decode=True, dataset_type=dataset_type)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        return ds, channels


if __name__ == "__main__":
    dataset = Dataset()
    for dataset_type in dataset.data_types:
        print(f"\nTesting: {dataset_type}")
        ds, channels = dataset.load_data(dataset_type)
        for image in ds.take(1):
            print("Shape:", image.shape)
            print("Channels:", channels)
            break
