import json
import os
import shutil
import tempfile

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
TEXT_DATASETS = {"coco", "flickr30k"}
FOLDER_DATASETS = {"celeba", "anime_faces", "bedroom", "chest_x_ray", "skin_cancer"}


class Dataset:
    def __init__(self):
        self.data_types = SUPPORTED_DATASETS
        self.batch_size = BATCH_SIZE
        self.image_size = IMAGE_SIZE
        self.super_res_size = SUPER_RES_IMAGE_SIZE
        self.cache_root = os.path.join(
            tempfile.gettempdir(),
            "tensorflowai_diffusion_cache",
            os.path.basename(os.path.dirname(__file__)),
        )
        self.train_options = tf.data.Options()
        self.train_options.experimental_deterministic = not ENABLE_DATASET_NONDETERMINISM
        self._builtin_datasets = {}
        self._vocab = {"[PAD]": 0, "[UNK]": 1}

    def cleanup_cache(self):
        shutil.rmtree(self.cache_root, ignore_errors=True)

    def _cache_path(self, dataset_type, suffix="images"):
        cache_dir = os.path.join(self.cache_root, dataset_type)
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, suffix)

    def _get_builtin_dataset(self, dataset_name):
        if dataset_name not in self._builtin_datasets:
            loaders = {
                "mnist": tf.keras.datasets.mnist.load_data,
                "fashion_mnist": tf.keras.datasets.fashion_mnist.load_data,
                "cifar10": tf.keras.datasets.cifar10.load_data,
                "cifar100": tf.keras.datasets.cifar100.load_data,
            }
            self._builtin_datasets[dataset_name] = loaders[dataset_name]()
        return self._builtin_datasets[dataset_name]

    def _prepare_builtin_images(self, dataset_type):
        if dataset_type in {"mnist", "fashion_mnist"}:
            (train_images, _), (test_images, _) = self._get_builtin_dataset(dataset_type)
            images = np.concatenate([train_images, test_images]).reshape(-1, 28, 28, 1)
            channels = 1
        elif dataset_type == "cifar10":
            (train_images, _), (test_images, _) = self._get_builtin_dataset(dataset_type)
            images = np.concatenate([train_images, test_images])
            channels = 3
        elif dataset_type == "cifar100":
            (train_images, _), (test_images, _) = self._get_builtin_dataset(dataset_type)
            images = np.concatenate([train_images, test_images])
            channels = 3
        else:
            raise ValueError(f"Unknown builtin dataset: {dataset_type}")
        if MAX_DATASET_ITEMS:
            images = images[:MAX_DATASET_ITEMS]
        return images, channels

    def _list_image_files(self, image_dir, recursive=False):
        image_paths = []
        if recursive:
            for current_root, _, filenames in os.walk(image_dir):
                for filename in sorted(filenames):
                    if filename.lower().endswith(VALID_SUFFIXES):
                        image_paths.append(os.path.join(current_root, filename))
        else:
            image_paths = [
                os.path.join(image_dir, filename)
                for filename in sorted(os.listdir(image_dir))
                if filename.lower().endswith(VALID_SUFFIXES)
            ]
        if MAX_DATASET_ITEMS:
            image_paths = image_paths[:MAX_DATASET_ITEMS]
        return np.array(image_paths)

    def _resolve_folder_dataset_dir(self, dataset_type):
        mapping = {
            "celeba": os.path.join(DATASET_PATH, "celeba-dataset", "img_align_celeba"),
            "anime_faces": os.path.join(DATASET_PATH, "anime_face_images"),
            "bedroom": os.path.join(DATASET_PATH, "bedroom"),
            "chest_x_ray": os.path.join(DATASET_PATH, "chest_x_ray", "train"),
            "skin_cancer": os.path.join(DATASET_PATH, "skin_cancer", "skin_cancer_images"),
        }
        return mapping[dataset_type]

    def _decode_image(self, image_path, channels=3):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=channels, expand_animations=False)
        image.set_shape([None, None, channels])
        return image

    def _normalize_image(self, image, size=None):
        size = size or self.image_size
        image = tf.image.resize(image, size)
        image = tf.cast(image, tf.float32)
        return (image / 127.5) - 1.0

    def _build_unconditional_dataset(self, images, dataset_type, decode=False, channels=3, size=None):
        ds = tf.data.Dataset.from_tensor_slices(images)
        if decode:
            ds = ds.map(lambda path: self._decode_image(path, channels=channels), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda image: self._normalize_image(image, size=size), num_parallel_calls=AUTOTUNE)
        cache_mode = DECODE_CACHE_MODE if decode else ARRAY_CACHE_MODE
        if cache_mode == "disk":
            ds = ds.cache(self._cache_path(dataset_type))
        elif cache_mode == "memory":
            ds = ds.cache()
        ds = ds.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        ds = ds.with_options(self.train_options)
        ds = ds.batch(self.batch_size)
        return ds.prefetch(AUTOTUNE)

    def _load_caption_pairs(self, dataset_type):
        pairs = []
        if dataset_type == "coco":
            annotation_path = os.path.join(DATASET_PATH, "coco", "annotations", "captions_train2017.json")
            image_dir = os.path.join(DATASET_PATH, "coco", "train2017")
            with open(annotation_path, "r") as handle:
                payload = json.load(handle)
            image_lookup = {item["id"]: os.path.join(image_dir, item["file_name"]) for item in payload["images"]}
            for ann in payload["annotations"]:
                image_path = image_lookup.get(ann["image_id"])
                if image_path and os.path.exists(image_path):
                    pairs.append((image_path, ann["caption"].strip()))
        elif dataset_type == "flickr30k":
            caption_path = os.path.join(DATASET_PATH, "flickr30k", "captions.txt")
            image_dir = os.path.join(DATASET_PATH, "flickr30k", "Images")
            with open(caption_path, "r") as handle:
                next(handle)
                for line in handle:
                    filename, caption = line.split(",", 1)
                    image_path = os.path.join(image_dir, filename.strip())
                    if os.path.exists(image_path):
                        pairs.append((image_path, caption.strip()))
        else:
            raise ValueError(f"Unknown text dataset: {dataset_type}")
        if MAX_DATASET_ITEMS:
            pairs = pairs[:MAX_DATASET_ITEMS]
        return pairs

    def _build_vocab(self, texts):
        for text in texts:
            for token in text.lower().split():
                if token not in self._vocab and len(self._vocab) < TEXT_VOCAB_SIZE:
                    self._vocab[token] = len(self._vocab)
        return self._vocab

    def _tokenize_text(self, text):
        tokens = [self._vocab.get(token, 1) for token in text.lower().split()[:TEXT_SEQUENCE_LENGTH]]
        if len(tokens) < TEXT_SEQUENCE_LENGTH:
            tokens.extend([0] * (TEXT_SEQUENCE_LENGTH - len(tokens)))
        return np.array(tokens, dtype=np.int32)

    def _build_text_dataset(self, pairs, dataset_type, include_control=False, include_sr=False):
        texts = [caption for _, caption in pairs]
        self._build_vocab(texts)
        image_paths = np.array([image_path for image_path, _ in pairs])
        token_ids = np.stack([self._tokenize_text(text) for text in texts])
        ds = tf.data.Dataset.from_tensor_slices((image_paths, token_ids))

        def preprocess(image_path, tokens):
            image = self._decode_image(image_path, channels=3)
            image = self._normalize_image(image, size=self.image_size)
            outputs = {"image": image, "tokens": tokens}
            if include_sr:
                original = self._decode_image(image_path, channels=3)
                outputs = {
                    "low_res": self._normalize_image(original, size=self.image_size),
                    "high_res": self._normalize_image(original, size=self.super_res_size),
                    "tokens": tokens,
                }
            if include_control:
                edge_source = (image + 1.0) / 2.0
                edges = tf.image.sobel_edges(edge_source[None, ...])[0]
                edges = tf.sqrt(tf.reduce_sum(tf.square(edges), axis=-1))
                edge_map = tf.reduce_mean(edges, axis=-1, keepdims=True)
                edge_map = tf.clip_by_value(edge_map / (tf.reduce_max(edge_map) + 1e-6), 0.0, 1.0)
                outputs = {"image": image, "tokens": tokens, "control": edge_map * 2.0 - 1.0}
            return outputs

        ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
        if DECODE_CACHE_MODE == "disk":
            ds = ds.cache(self._cache_path(dataset_type, suffix="text"))
        else:
            ds = ds.cache()
        ds = ds.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
        ds = ds.with_options(self.train_options)
        ds = ds.batch(self.batch_size)
        return ds.prefetch(AUTOTUNE)

    def get_preview_batch(self, dataset_type, count=4, include_control=False):
        if dataset_type in BUILTIN_DATASETS:
            images, channels = self._prepare_builtin_images(dataset_type)
            sample = images[:count]
            sample = tf.convert_to_tensor(sample, dtype=tf.float32)
            sample = self._normalize_image(sample, size=self.image_size)
            return sample, channels
        if dataset_type in TEXT_DATASETS:
            pairs = self._load_caption_pairs(dataset_type)[:count]
            texts = [caption for _, caption in pairs]
            self._build_vocab(texts)
            images = []
            controls = []
            for image_path, _ in pairs:
                image = self._normalize_image(self._decode_image(image_path, channels=3), size=self.image_size)
                images.append(image)
                if include_control:
                    edges = tf.image.sobel_edges(((image + 1.0) / 2.0)[None, ...])[0]
                    edges = tf.sqrt(tf.reduce_sum(tf.square(edges), axis=-1))
                    edge_map = tf.reduce_mean(edges, axis=-1, keepdims=True)
                    controls.append(tf.clip_by_value(edge_map / (tf.reduce_max(edge_map) + 1e-6), 0.0, 1.0) * 2.0 - 1.0)
            images = tf.stack(images)
            if include_control:
                return images, tf.stack(controls), texts
            return images, texts
        image_dir = self._resolve_folder_dataset_dir(dataset_type)
        image_paths = self._list_image_files(image_dir, recursive=dataset_type in {"bedroom", "chest_x_ray"})[:count]
        decoded = [self._normalize_image(self._decode_image(path, channels=3), size=self.image_size) for path in image_paths]
        return tf.stack(decoded), 3

    def load_data(self, dataset_type=DEFAULT_DATASET):
        if dataset_type in BUILTIN_DATASETS:
            images, channels = self._prepare_builtin_images(dataset_type)
            return self._build_unconditional_dataset(images, dataset_type, decode=False, channels=channels), channels
        if dataset_type in FOLDER_DATASETS:
            image_dir = self._resolve_folder_dataset_dir(dataset_type)
            image_paths = self._list_image_files(image_dir, recursive=dataset_type in {"bedroom", "chest_x_ray"})
            return self._build_unconditional_dataset(image_paths, dataset_type, decode=True, channels=3), 3
        if dataset_type in TEXT_DATASETS:
            pairs = self._load_caption_pairs(dataset_type)
            return self._build_text_dataset(pairs, dataset_type), 3
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    def load_super_resolution_data(self, dataset_type=DEFAULT_DATASET):
        pairs = self._load_caption_pairs(dataset_type)
        return self._build_text_dataset(pairs, dataset_type, include_sr=True), 3

    def load_control_data(self, dataset_type=DEFAULT_DATASET):
        pairs = self._load_caption_pairs(dataset_type)
        return self._build_text_dataset(pairs, dataset_type, include_control=True), 3


if __name__ == "__main__":
    dataset = Dataset()
    for dataset_type in SUPPORTED_DATASETS:
        print(f"Testing {dataset_type}")
        ds, channels = dataset.load_data(dataset_type)
        for batch in ds.take(1):
            if isinstance(batch, dict):
                print(batch.keys())
            else:
                print(batch.shape)
            print(channels)
            break
