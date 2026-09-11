from dataclasses import dataclass
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import silence_tensorflow.auto
import tensorflow as tf

from config import *

AUTOTUNE = tf.data.AUTOTUNE
VALID_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class ConditionSpec:
    mode: str
    label_dim: int
    display_names: list[str]
    sample_labels: np.ndarray | None = None

    def make_sample_labels(self, num_samples):
        if self.sample_labels is not None and len(self.sample_labels) > 0:
            indices = np.arange(num_samples) % len(self.sample_labels)
            return self.sample_labels[indices].astype(np.float32)

        if self.mode == "onehot":
            base = np.arange(num_samples) % self.label_dim
            return np.eye(self.label_dim, dtype=np.float32)[base]

        return (np.random.rand(num_samples, self.label_dim) > 0.5).astype(np.float32)

    def format_label(self, label_vector, max_items=3):
        label_vector = np.asarray(label_vector)

        if self.mode == "onehot":
            class_index = int(np.argmax(label_vector))
            if 0 <= class_index < len(self.display_names):
                return str(self.display_names[class_index])
            return str(class_index)

        active_indices = np.flatnonzero(label_vector > 0.5)
        if len(active_indices) == 0:
            return "none"

        names = [
            self.display_names[index]
            if index < len(self.display_names) else str(index)
            for index in active_indices[:max_items]
        ]
        suffix = "..." if len(active_indices) > max_items else ""
        return ",".join(names) + suffix


class Dataset:
    CIFAR10_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    FASHION_MNIST_CLASSES = [
        "t-shirt", "trouser", "pullover", "dress", "coat",
        "sandal", "shirt", "sneaker", "bag", "ankle boot",
    ]

    def __init__(self):
        self.data_types = [
            "cifar10",
            "fashion_mnist",
            "mnist",
            "cifar100",
            "celeba",
            "chest_x_ray",
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
        self.preview_images = {}
        self._builtin_datasets = {}

    def _target_shape(self, dataset_type):
        if dataset_type in {"celeba", "chest_x_ray"}:
            return (IMAGE_SIZE[0] * 2, IMAGE_SIZE[1] * 2)
        return self.img_shape

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

    def _list_class_directories(self, root_dir):
        return sorted([
            entry for entry in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, entry))
        ])

    def _load_directory_labeled_dataset(self, root_dir):
        class_names = self._list_class_directories(root_dir)
        image_paths = []
        labels = []
        for index, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)
            for filename in sorted(os.listdir(class_dir)):
                if filename.lower().endswith(VALID_SUFFIXES):
                    image_paths.append(os.path.join(class_dir, filename))
                    labels.append(index)
        return np.array(image_paths), np.array(labels), class_names

    def _decode_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        return image

    def process_images(self, image, labels, decode=False, dataset_type="cifar10"):
        if decode:
            image = self._decode_image(image)

        image = tf.image.resize(image, self._target_shape(dataset_type))
        image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 127.5
        return image, tf.cast(labels, tf.float32)

    def _cache_path(self, dataset_type):
        target_h, target_w = self._target_shape(dataset_type)
        cache_dir = os.path.join(self.cache_root, dataset_type)
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"images_labels_{target_h}x{target_w}")

    def cleanup_cache(self):
        shutil.rmtree(self.cache_root, ignore_errors=True)

    def _apply_cache(self, ds, dataset_type, decode):
        cache_mode = DECODE_CACHE_MODE if decode else ARRAY_CACHE_MODE
        if cache_mode == "disk":
            return ds.cache(self._cache_path(dataset_type))
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

    def build_dataset(self, images, labels, decode=False, dataset_type="cifar10", shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        ds = ds.map(
            lambda image, y: self.process_images(
                image,
                y,
                decode=decode,
                dataset_type=dataset_type,
            ),
            num_parallel_calls=AUTOTUNE,
        )
        ds = self._apply_cache(ds, dataset_type, decode)
        ds = self._apply_training_options(ds, shuffle)
        ds = ds.batch(self.batch_size)
        return ds.prefetch(AUTOTUNE)

    def _build_onehot_spec(self, label_dim, display_names=None):
        if display_names is None:
            display_names = [str(index) for index in range(label_dim)]
        return ConditionSpec(mode="onehot", label_dim=label_dim, display_names=display_names)

    def _prepare_builtin_data(self, dataset_type):
        if dataset_type == "mnist":
            channels = 1
            (train_images, train_labels), (test_images, test_labels) = self._get_builtin_dataset("mnist")
            images = np.concatenate([train_images, test_images]).reshape(-1, 28, 28, 1)
            labels = tf.one_hot(np.concatenate([train_labels, test_labels]), depth=10).numpy()
            spec = self._build_onehot_spec(10, [str(index) for index in range(10)])
        elif dataset_type == "cifar10":
            channels = 3
            (train_images, train_labels), (test_images, test_labels) = self._get_builtin_dataset("cifar10")
            images = np.concatenate([train_images, test_images])
            flat_labels = np.concatenate([train_labels, test_labels]).squeeze(-1)
            labels = tf.one_hot(flat_labels, depth=10).numpy()
            spec = self._build_onehot_spec(10, self.CIFAR10_CLASSES)
        elif dataset_type == "fashion_mnist":
            channels = 1
            (train_images, train_labels), (test_images, test_labels) = self._get_builtin_dataset("fashion_mnist")
            images = np.concatenate([train_images, test_images]).reshape(-1, 28, 28, 1)
            labels = tf.one_hot(np.concatenate([train_labels, test_labels]), depth=10).numpy()
            spec = self._build_onehot_spec(10, self.FASHION_MNIST_CLASSES)
        elif dataset_type == "cifar100":
            channels = 3
            (train_images, train_labels), (test_images, test_labels) = self._get_builtin_dataset("cifar100")
            images = np.concatenate([train_images, test_images])
            flat_labels = np.concatenate([train_labels, test_labels]).squeeze(-1)
            labels = tf.one_hot(flat_labels, depth=100).numpy()
            spec = self._build_onehot_spec(100)
        else:
            raise ValueError(f"Unknown builtin dataset type: {dataset_type}")
        return images, labels, channels, spec, False

    def _prepare_decode_data(self, dataset_type):
        if dataset_type == "celeba":
            channels = 3
            dataset_path = os.path.join(DATASET_PATH, "celeba-dataset")
            image_dir = os.path.join(dataset_path, "img_align_celeba")
            attr_path = os.path.join(dataset_path, "list_attr_celeba.csv")

            df = pd.read_csv(attr_path)
            image_names = df.iloc[:, 0].astype(str).tolist()
            labels = ((df.iloc[:, 1:].values + 1) // 2).astype(np.float32)
            images = np.array([
                os.path.join(image_dir, image_name)
                for image_name in image_names
                if image_name.lower().endswith(VALID_SUFFIXES)
            ])
            labels = labels[: len(images)]
            spec = ConditionSpec(
                mode="multilabel",
                label_dim=labels.shape[-1],
                display_names=df.columns[1:].tolist(),
                sample_labels=labels[:256],
            )
        elif dataset_type == "chest_x_ray":
            channels = 3
            root_dir = os.path.join(DATASET_PATH, "chest_x_ray", "train")
            images, class_labels, class_names = self._load_directory_labeled_dataset(root_dir)
            labels = tf.one_hot(class_labels, depth=len(class_names)).numpy()
            spec = self._build_onehot_spec(len(class_names), class_names)
        else:
            raise ValueError(f"Unknown decode dataset type: {dataset_type}")

        return images, labels, channels, spec, True

    def _build_preview_images(self, images, decode, dataset_type, count=8):
        preview_count = min(count, len(images))
        processed = [
            self.process_images(image, np.zeros((1,), dtype=np.float32), decode=decode, dataset_type=dataset_type)[0].numpy()
            for image in images[:preview_count]
        ]
        return tf.convert_to_tensor(np.stack(processed, axis=0), dtype=tf.float32)

    def get_preview_images(self, dataset_type, count=8):
        preview = self.preview_images[dataset_type]
        return preview[: min(count, preview.shape[0])]

    def load_data(self, dataset_type="mnist"):
        if dataset_type in {"mnist", "cifar10", "fashion_mnist", "cifar100"}:
            images, labels, channels, condition_spec, decode = self._prepare_builtin_data(dataset_type)
        elif dataset_type in {"celeba", "chest_x_ray"}:
            images, labels, channels, condition_spec, decode = self._prepare_decode_data(dataset_type)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        train_ds = self.build_dataset(
            images,
            labels,
            decode=decode,
            dataset_type=dataset_type,
        )
        self.preview_images[dataset_type] = self._build_preview_images(
            images,
            decode,
            dataset_type,
            count=8,
        )
        return train_ds, channels, condition_spec


if __name__ == "__main__":
    dataset = Dataset()
    for dataset_type in dataset.data_types:
        print(f"\nTesting: {dataset_type}")
        train_ds, channels, condition_spec = dataset.load_data(dataset_type)
        for image, label in train_ds.take(1):
            print("Shape:", image.shape)
            print("Label shape:", label.shape)
            print("Mode:", condition_spec.mode)
            print("Channels:", channels)
            break
