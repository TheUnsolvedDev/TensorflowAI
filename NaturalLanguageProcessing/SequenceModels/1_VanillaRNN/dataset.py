"""Streaming text-classification datasets with folder-local corpus adapters."""
import csv
import hashlib
import os
import re
import tensorflow as tf
from config import *

AUTOTUNE = tf.data.AUTOTUNE
_URL = re.compile(r"http\S+")
_SPACE = re.compile(r"\s+")


class BaseTextDataset:
    def __init__(self, dataset_path, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, vocab_size=VOCAB_SIZE,
                 validation_split=VALIDATION_SPLIT, seed=SEED, lowercase=LOWERCASE):
        self.dataset_path, self.batch_size, self.max_length = dataset_path, batch_size, max_length
        self.vocab_size, self.validation_split, self.seed, self.lowercase = vocab_size, validation_split, seed, lowercase
        self.vectorizer = None; self.label_to_id = {}

    def clean_text(self, value):
        value = _URL.sub(" ", str(value))
        if self.lowercase: value = value.lower()
        return _SPACE.sub(" ", value).strip()

    def iter_records(self):
        raise NotImplementedError

    def prepare_labels(self):
        labels = sorted({label for _, label in self.iter_records()}, key=str)
        if not labels: raise ValueError(f"No valid records in {self.dataset_path}")
        self.label_to_id = {label: index for index, label in enumerate(labels)}

    def _is_validation(self, text):
        digest = hashlib.blake2b(f"{self.seed}\0{text}".encode("utf-8", "ignore"), digest_size=8).digest()
        return int.from_bytes(digest, "big") / 2**64 < self.validation_split

    def _text_stream(self, validation):
        for text, _ in self.iter_records():
            if bool(self._is_validation(text)) == bool(validation): yield text

    def prepare_vectorizer(self, name):
        path = f"text_vectorizer_{name}.keras"
        if os.path.exists(path):
            self.vectorizer = tf.keras.models.load_model(path).layers[0]; print(f"Loaded vocabulary: {path}"); return
        self.vectorizer = tf.keras.layers.TextVectorization(max_tokens=self.vocab_size, output_mode="int", output_sequence_length=self.max_length, standardize=None)
        print("Adapting vocabulary from raw training records...")
        data = tf.data.Dataset.from_generator(lambda: self._text_stream(False), output_signature=tf.TensorSpec((), tf.string)).batch(1024)
        self.vectorizer.adapt(data)
        wrapper = tf.keras.Sequential([self.vectorizer]); wrapper(tf.constant(["warmup"])); wrapper.save(path)

    def _examples(self, validation):
        for text, label in self.iter_records():
            if bool(self._is_validation(text)) == bool(validation): yield text, self.label_to_id[label]

    def _encode(self, text, label):
        return tf.ensure_shape(self.vectorizer(text), [self.max_length]), label

    def build_tf_dataset(self, training):
        ds = tf.data.Dataset.from_generator(lambda: self._examples(not training), output_signature=(tf.TensorSpec((), tf.string), tf.TensorSpec((), tf.int32)))
        if training: ds = ds.shuffle(8192, seed=self.seed, reshuffle_each_iteration=True)
        options = tf.data.Options(); options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        return ds.map(self._encode, num_parallel_calls=AUTOTUNE, deterministic=not training).with_options(options).batch(
            self.batch_size, drop_remainder=training).prefetch(AUTOTUNE)

    def decode_tokens(self, tokens):
        vocab = self.vectorizer.get_vocabulary()
        return " ".join(vocab[int(token)] for token in tokens if int(token) and int(token) < len(vocab))
    def get_vocab_size(self): return len(self.vectorizer.get_vocabulary())
    def get_num_classes(self): return len(self.label_to_id)


class AGNewsDataset(BaseTextDataset):
    def iter_records(self):
        for filename in ("train.csv", "test.csv"):
            path = os.path.join(self.dataset_path, filename)
            with open(path, newline="", encoding="utf-8", errors="replace") as handle:
                for row in csv.DictReader(handle):
                    text = self.clean_text(f"{row.get('Title', '')} {row.get('Description', '')}")
                    if text: yield text, row["Class Index"]


class DBPediaDataset(BaseTextDataset):
    def iter_records(self):
        for filename in ("DBPEDIA_train.csv", "DBPEDIA_val.csv", "DBPEDIA_test.csv"):
            path = os.path.join(self.dataset_path, filename)
            if not os.path.exists(path): continue
            with open(path, newline="", encoding="utf-8", errors="replace") as handle:
                for row in csv.DictReader(handle):
                    text, label = self.clean_text(row.get("text", "")), row.get("l1", "")
                    if text and label: yield text, label


class IMDBDataset(BaseTextDataset):
    def iter_records(self):
        with open(os.path.join(self.dataset_path, "IMDB_Dataset.csv"), newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                text, label = self.clean_text(row.get("review", "")), row.get("sentiment", "").lower()
                if text and label in ("positive", "negative"): yield text, label


class Dataset:
    _types = {"ag_news": AGNewsDataset, "dbpedia": DBPediaDataset, "imdb": IMDBDataset}
    def __init__(self, dataset_name, dataset_path, **kwargs):
        try: implementation = self._types[dataset_name.lower()]
        except KeyError as error: raise ValueError(f"Unsupported dataset {dataset_name}; choose from {sorted(self._types)}") from error
        self.name = dataset_name.lower(); self.dataset = implementation(dataset_path, **kwargs)
    def load_data(self):
        self.dataset.prepare_labels(); self.dataset.prepare_vectorizer(self.name)
        return self.dataset.build_tf_dataset(True), self.dataset.build_tf_dataset(False)
    def __getattr__(self, name): return getattr(self.dataset, name)
