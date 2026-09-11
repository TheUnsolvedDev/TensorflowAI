import os
import shutil
import tempfile

import numpy as np
import silence_tensorflow.auto
import tensorflow as tf

from config import *

AUTOTUNE = tf.data.AUTOTUNE
VALID_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class Dataset:
    def __init__(self):
        self.data_types = [
            "celeba",
            "anime_faces",
            "bedroom",
            "chest_x_ray",
            "skin_cancer",
        ]

        self.batch_size = BATCH_SIZE
        self.hr_shape = HR_IMAGE_SIZE
        self.lr_shape = LR_IMAGE_SIZE
        self.cache_root = os.path.join(
            tempfile.gettempdir(),
            "tensorflowai_gan_cache",
            os.path.basename(os.path.dirname(__file__)),
        )
        self.fallback_cache_root = os.path.join(
            os.path.dirname(__file__),
            CACHE_DIR,
        )
        self.train_options = tf.data.Options()
        self.train_options.experimental_deterministic = not ENABLE_DATASET_NONDETERMINISM
        self.preview_batches = {}

    def _collect_files(self, root_dir, recursive=False):
        image_paths = []

        if recursive:
            for root, _, filenames in os.walk(root_dir):
                for filename in sorted(filenames):
                    if filename.lower().endswith(VALID_SUFFIXES):
                        image_paths.append(os.path.join(root, filename))
        else:
            for filename in sorted(os.listdir(root_dir)):
                if filename.lower().endswith(VALID_SUFFIXES):
                    image_paths.append(os.path.join(root_dir, filename))

        return np.array(image_paths)

    def _resolve_dataset_root(self, dataset_type):
        if dataset_type == "celeba":
            return os.path.join(DATASET_PATH, "celeba-dataset", "img_align_celeba"), False
        if dataset_type == "anime_faces":
            return os.path.join(DATASET_PATH, "anime_face_images"), False
        if dataset_type == "bedroom":
            return os.path.join(DATASET_PATH, "bedroom"), True
        if dataset_type == "chest_x_ray":
            return os.path.join(DATASET_PATH, "chest_x_ray", "train"), True
        if dataset_type == "skin_cancer":
            return os.path.join(DATASET_PATH, "skin_cancer", "skin_cancer_images"), False
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    def _decode_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        return image

    def process_image(self, image_path):
        image = self._decode_image(image_path)
        hr = tf.image.resize(image, self.hr_shape, method="bicubic")
        hr = tf.cast(hr, tf.float32)
        hr = (hr / 127.5) - 1.0

        lr = tf.image.resize(hr, self.lr_shape, method="area")
        return lr, hr

    def _sample_cache_bytes(self):
        lr_bytes = int(np.prod(self.lr_shape) * 3 * 4)
        hr_bytes = int(np.prod(self.hr_shape) * 3 * 4)
        return lr_bytes + hr_bytes

    def _resolve_disk_cache_path(self, image_paths, dataset_type):
        estimated_bytes = len(image_paths) * self._sample_cache_bytes()
        candidate_roots = [self.cache_root, self.fallback_cache_root]

        for root in candidate_roots:
            os.makedirs(root, exist_ok=True)
            free_bytes = shutil.disk_usage(root).free
            if free_bytes > estimated_bytes:
                cache_dir = os.path.join(root, dataset_type)
                os.makedirs(cache_dir, exist_ok=True)
                return os.path.join(
                    cache_dir,
                    f"sr_pairs_{self.lr_shape[0]}x{self.lr_shape[1]}_{self.hr_shape[0]}x{self.hr_shape[1]}",
                )

        return None

    def cleanup_cache(self):
        shutil.rmtree(self.cache_root, ignore_errors=True)
        shutil.rmtree(self.fallback_cache_root, ignore_errors=True)

    def _apply_decode_cache(self, ds, image_paths, dataset_type):
        if DECODE_CACHE_MODE == "disk":
            cache_path = self._resolve_disk_cache_path(image_paths, dataset_type)
            if cache_path is not None:
                print(f"Using decoded disk cache: {cache_path}")
                return ds.cache(cache_path)
            print(
                "Skipping decoded disk cache due to insufficient free space in "
                f"'{self.cache_root}' and '{self.fallback_cache_root}'"
            )
            return ds
        if DECODE_CACHE_MODE == "memory":
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

    def build_dataset(self, image_paths, dataset_type, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices(image_paths)
        ds = ds.map(self.process_image, num_parallel_calls=AUTOTUNE)
        ds = self._apply_decode_cache(ds, image_paths, dataset_type)
        ds = self._apply_training_options(ds, shuffle)
        ds = ds.batch(self.batch_size)
        return ds.prefetch(AUTOTUNE)

    def _build_preview_batch(self, image_paths, count=4):
        preview_pairs = [self.process_image(image_path) for image_path in image_paths[: min(count, len(image_paths))]]
        lr_images = tf.stack([pair[0] for pair in preview_pairs], axis=0)
        hr_images = tf.stack([pair[1] for pair in preview_pairs], axis=0)
        return lr_images, hr_images

    def get_preview_batch(self, dataset_type, count=4):
        lr_images, hr_images = self.preview_batches[dataset_type]
        preview_count = min(count, lr_images.shape[0])
        return lr_images[:preview_count], hr_images[:preview_count]

    def load_data(self, dataset_type="celeba"):
        root_dir, recursive = self._resolve_dataset_root(dataset_type)
        image_paths = self._collect_files(root_dir, recursive=recursive)
        train_ds = self.build_dataset(image_paths, dataset_type=dataset_type)
        self.preview_batches[dataset_type] = self._build_preview_batch(image_paths, count=4)
        return train_ds, 3


if __name__ == "__main__":
    dataset = Dataset()

    for dataset_type in dataset.data_types:
        print(f"\nTesting: {dataset_type}")
        train_ds, channels = dataset.load_data(dataset_type)

        for lr_images, hr_images in train_ds.take(1):
            print("LR shape:", lr_images.shape)
            print("HR shape:", hr_images.shape)
            print("Channels:", channels)
            break
