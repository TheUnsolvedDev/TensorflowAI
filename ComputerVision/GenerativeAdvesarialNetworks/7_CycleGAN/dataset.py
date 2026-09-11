from dataclasses import dataclass
import os
import shutil
import tempfile

import numpy as np
import silence_tensorflow.auto
import tensorflow as tf

from config import *

AUTOTUNE = tf.data.AUTOTUNE
VALID_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class CycleSpec:
    domain_a_name: str
    domain_b_name: str
    fixed_a: tf.Tensor
    fixed_b: tf.Tensor


class Dataset:
    def __init__(self):
        self.data_types = [
            "mnist_fashion",
            "monet2photo",
            "cezanne2photo",
            "ukiyoe2photo",
            "vangogh2photo",
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

    def _target_shape(self, dataset_type):
        if dataset_type == "mnist_fashion":
            return self.img_shape
        return (IMAGE_SIZE[0] * 2, IMAGE_SIZE[1] * 2)

    def _get_builtin_dataset(self, dataset_name):
        if dataset_name not in self._builtin_datasets:
            loaders = {
                "mnist": tf.keras.datasets.mnist.load_data,
                "fashion_mnist": tf.keras.datasets.fashion_mnist.load_data,
            }
            self._builtin_datasets[dataset_name] = loaders[dataset_name]()
        return self._builtin_datasets[dataset_name]

    def _list_image_files(self, image_dir):
        return np.array([
            os.path.join(image_dir, filename)
            for filename in sorted(os.listdir(image_dir))
            if filename.lower().endswith(VALID_SUFFIXES)
        ])

    def _prepare_mnist_domain(self, source):
        (train_images, _), (test_images, _) = source
        return np.concatenate([train_images, test_images]).reshape(-1, 28, 28, 1)

    def _decode_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        return image

    def process_image(self, image, decode=False, dataset_type="mnist_fashion"):
        if decode:
            image = self._decode_image(image)

        image = tf.image.resize(image, self._target_shape(dataset_type))
        image = tf.cast(image, tf.float32)
        return (image - 127.5) / 127.5

    def _cache_path(self, dataset_type, domain_name):
        target_shape = self._target_shape(dataset_type)
        cache_dir = os.path.join(self.cache_root, dataset_type)
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{domain_name}_{target_shape[0]}x{target_shape[1]}")

    def cleanup_cache(self):
        shutil.rmtree(self.cache_root, ignore_errors=True)

    def _apply_cache(self, ds, dataset_type, domain_name, decode):
        cache_mode = DECODE_CACHE_MODE if decode else ARRAY_CACHE_MODE
        if cache_mode == "disk":
            return ds.cache(self._cache_path(dataset_type, domain_name))
        if cache_mode == "memory":
            return ds.cache()
        return ds

    def _apply_training_options(self, ds, shuffle):
        if not shuffle:
            return ds
        ds = ds.shuffle(
            SHUFFLE_BUFFER_SIZE,
            reshuffle_each_iteration=True,
        )
        return ds.with_options(self.train_options)

    def _build_domain_dataset(self, images, domain_name, decode=False, dataset_type="mnist_fashion", shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices(images)
        ds = ds.map(
            lambda image: self.process_image(
                image,
                decode=decode,
                dataset_type=dataset_type,
            ),
            num_parallel_calls=AUTOTUNE,
        )
        ds = self._apply_cache(ds, dataset_type, domain_name, decode)
        ds = self._apply_training_options(ds, shuffle)
        ds = ds.batch(self.batch_size)
        return ds.prefetch(AUTOTUNE)

    def _build_preview_tensor(self, images, decode, dataset_type, count=8):
        preview_count = min(count, len(images))
        processed = [
            self.process_image(image, decode=decode, dataset_type=dataset_type).numpy()
            for image in images[:preview_count]
        ]
        return tf.convert_to_tensor(np.stack(processed, axis=0), dtype=tf.float32)

    def load_data(self, dataset_type="mnist_fashion"):
        if dataset_type == "mnist_fashion":
            images_a = self._prepare_mnist_domain(self._get_builtin_dataset("mnist"))
            images_b = self._prepare_mnist_domain(self._get_builtin_dataset("fashion_mnist"))
            channels = 1
            decode = False
            domain_a_name = "mnist"
            domain_b_name = "fashion_mnist"
        elif dataset_type in {"monet2photo", "cezanne2photo", "ukiyoe2photo", "vangogh2photo"}:
            dataset_root = os.path.join(DATASET_PATH, dataset_type)
            images_a = self._list_image_files(os.path.join(dataset_root, "trainA"))
            images_b = self._list_image_files(os.path.join(dataset_root, "trainB"))
            channels = 3
            decode = True
            domain_a_name = "domainA"
            domain_b_name = "domainB"
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        ds_a = self._build_domain_dataset(
            images_a,
            domain_name="domain_a",
            decode=decode,
            dataset_type=dataset_type,
        )
        ds_b = self._build_domain_dataset(
            images_b,
            domain_name="domain_b",
            decode=decode,
            dataset_type=dataset_type,
        )
        train_ds = tf.data.Dataset.zip((ds_a, ds_b)).prefetch(AUTOTUNE)

        cycle_spec = CycleSpec(
            domain_a_name=domain_a_name,
            domain_b_name=domain_b_name,
            fixed_a=self._build_preview_tensor(images_a, decode, dataset_type, count=4),
            fixed_b=self._build_preview_tensor(images_b, decode, dataset_type, count=4),
        )

        return train_ds, channels, cycle_spec


if __name__ == "__main__":
    dataset = Dataset()

    for dataset_type in dataset.data_types:
        print(f"\nTesting: {dataset_type}")
        train_ds, channels, cycle_spec = dataset.load_data(dataset_type)
        for images_a, images_b in train_ds.take(1):
            print("A shape:", images_a.shape)
            print("B shape:", images_b.shape)
            print("Domains:", cycle_spec.domain_a_name, cycle_spec.domain_b_name)
            print("Channels:", channels)
            break
