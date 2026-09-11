"""Streaming local seq2seq datasets; no examples are cached or materialized."""
import ast
import csv
import hashlib
import os
import re

import tensorflow as tf
from config import BATCH_SIZE, LOWERCASE, SEED, SOURCE_MAX_LENGTH, TARGET_MAX_LENGTH, VALIDATION_SPLIT, VOCAB_SIZE

AUTOTUNE = tf.data.AUTOTUNE
_URL = re.compile(r"http\S+")
_SPACE = re.compile(r"\s+")


class BaseSeq2SeqDataset:
    """Folder-local base class. Subclasses only describe their raw records."""

    def __init__(self, dataset_path, batch_size=BATCH_SIZE, source_max_length=SOURCE_MAX_LENGTH,
                 target_max_length=TARGET_MAX_LENGTH, vocab_size=VOCAB_SIZE,
                 validation_split=VALIDATION_SPLIT, seed=SEED, lowercase=LOWERCASE):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length
        self.vocab_size = vocab_size
        self.validation_split = validation_split
        self.seed = seed
        self.lowercase = lowercase
        self.source_vectorizer = None
        self.target_vectorizer = None

    def clean_text(self, value):
        value = _URL.sub(" ", str(value))
        if self.lowercase:
            value = value.lower()
        return _SPACE.sub(" ", value).strip()

    def iter_pairs(self):
        raise NotImplementedError

    def _is_validation(self, source, target):
        payload = f"{self.seed}\0{source}\0{target}".encode("utf-8", "ignore")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") / 2**64 < self.validation_split

    def _split_pairs(self, validation):
        for source, target in self.iter_pairs():
            if bool(self._is_validation(source, target)) == bool(validation):
                yield source, target

    def _text_stream(self, validation, target):
        for source, value in self._split_pairs(validation):
            yield f"[start] {value} [end]" if target else source

    def _text_dataset(self, validation, target):
        return tf.data.Dataset.from_generator(
            lambda: self._text_stream(validation, target),
            output_signature=tf.TensorSpec((), tf.string),
        ).batch(1024).prefetch(AUTOTUNE)

    def prepare_vectorizers(self, name):
        src_path = f"source_vectorizer_{name}.keras"
        tgt_path = f"target_vectorizer_{name}.keras"
        if os.path.exists(src_path) and os.path.exists(tgt_path):
            self.source_vectorizer = tf.keras.models.load_model(src_path).layers[0]
            self.target_vectorizer = tf.keras.models.load_model(tgt_path).layers[0]
            print(f"Loaded vectorizers for {name}")
            return
        self.source_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=self.vocab_size, output_mode="int", output_sequence_length=self.source_max_length, standardize=None)
        self.target_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=self.vocab_size, output_mode="int", output_sequence_length=self.target_max_length, standardize=None)
        print("Adapting source vocabulary from raw training records...")
        self.source_vectorizer.adapt(self._text_dataset(False, False))
        print("Adapting target vocabulary from raw training records...")
        self.target_vectorizer.adapt(self._text_dataset(False, True))
        for vectorizer, path in ((self.source_vectorizer, src_path), (self.target_vectorizer, tgt_path)):
            wrapper = tf.keras.Sequential([vectorizer]); wrapper(tf.constant(["warmup"])); wrapper.save(path)

    def _encode(self, source, target):
        source = self.source_vectorizer(source)
        target = self.target_vectorizer(target)
        return (tf.ensure_shape(source, [self.source_max_length]),
                tf.ensure_shape(target[:-1], [self.target_max_length - 1])), tf.ensure_shape(target[1:], [self.target_max_length - 1])

    def build_tf_dataset(self, training):
        def records():
            for source, target in self._split_pairs(not training):
                yield source, f"[start] {target} [end]"
        ds = tf.data.Dataset.from_generator(
            records,
            output_signature=(tf.TensorSpec((), tf.string), tf.TensorSpec((), tf.string)),
        )
        if training:
            ds = ds.shuffle(8192, seed=self.seed, reshuffle_each_iteration=True)
        options = tf.data.Options(); options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        return ds.map(self._encode, num_parallel_calls=AUTOTUNE, deterministic=not training).with_options(options).batch(
            self.batch_size, drop_remainder=training).prefetch(AUTOTUNE)

    def get_steps_per_epoch(self, training):
        """Return the finite dataset's batch count before it is repeated."""
        examples = sum(1 for _ in self._split_pairs(not training))
        steps = examples // self.batch_size if training else (examples + self.batch_size - 1) // self.batch_size
        if not steps:
            split = "training" if training else "validation"
            raise ValueError(f"No complete {split} batches were found in {self.dataset_path}")
        return steps

    def encode_source(self, text):
        return self.source_vectorizer(tf.constant([text])).numpy()[0]

    def _decode(self, vectorizer, tokens):
        vocab = vectorizer.get_vocabulary()
        return " ".join(vocab[int(token)] for token in tokens if int(token) > 1 and int(token) < len(vocab))

    def decode_source(self, tokens): return self._decode(self.source_vectorizer, tokens)
    def decode_target(self, tokens): return self._decode(self.target_vectorizer, tokens)
    def get_source_vocab_size(self): return len(self.source_vectorizer.get_vocabulary())
    def get_target_vocab_size(self): return len(self.target_vectorizer.get_vocabulary())


class DelimitedTranslationDataset(BaseSeq2SeqDataset):
    files = ()
    source_hints = ()
    target_hints = ()

    def _columns(self, fieldnames):
        normal = {name.lower(): name for name in fieldnames or []}
        source = next((normal[key] for key in normal if any(hint in key for hint in self.source_hints)), None)
        target = next((normal[key] for key in normal if any(hint in key for hint in self.target_hints)), None)
        if not source or not target:
            raise ValueError(f"Could not locate translation columns in {fieldnames}")
        return source, target

    def iter_pairs(self):
        for filename in self.files:
            path = os.path.join(self.dataset_path, filename)
            if not os.path.exists(path):
                continue
            with open(path, newline="", encoding="utf-8", errors="replace") as handle:
                reader = csv.DictReader(handle); source_key, target_key = self._columns(reader.fieldnames)
                for row in reader:
                    source, target = self.clean_text(row.get(source_key, "")), self.clean_text(row.get(target_key, ""))
                    if source and target: yield source, target


class EnglishFrenchDataset(DelimitedTranslationDataset):
    files = ("wmt14_translate_fr-en_train.csv", "wmt14_translate_fr-en_validation.csv", "wmt14_translate_fr-en_test.csv")
    source_hints, target_hints = ("en", "english"), ("fr", "french")


class EnglishGermanDataset(DelimitedTranslationDataset):
    files = ("wmt14_translate_de-en_train.csv", "wmt14_translate_de-en_validation.csv", "wmt14_translate_de-en_test.csv")
    source_hints, target_hints = ("en", "english"), ("de", "german")


class ManyThingsEnglishFrenchDataset(BaseSeq2SeqDataset):
    def iter_pairs(self):
        with open(os.path.join(self.dataset_path, "fra.txt"), encoding="utf-8", errors="replace") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 2:
                    source, target = self.clean_text(fields[0]), self.clean_text(fields[1])
                    if source and target: yield source, target


class WikiLargeDataset(BaseSeq2SeqDataset):
    def iter_pairs(self):
        for filename in ("wiki.full.aner.ori.train.95.tsv", "wiki.full.aner.ori.valid.95.tsv", "wiki.full.aner.ori.test.95.tsv"):
            path = os.path.join(self.dataset_path, filename)
            if not os.path.exists(path): continue
            with open(path, encoding="utf-8", errors="replace") as handle:
                for row in csv.reader(handle, delimiter="\t"):
                    if len(row) >= 2:
                        source, target = self.clean_text(row[0]), self.clean_text(row[1])
                        if source and target: yield source, target


class CNNDailyMailDataset(BaseSeq2SeqDataset):
    def iter_pairs(self):
        for filename in ("train.csv", "validation.csv", "test.csv"):
            path = os.path.join(self.dataset_path, filename)
            if not os.path.exists(path): continue
            with open(path, newline="", encoding="utf-8", errors="replace") as handle:
                reader = csv.DictReader(handle); names = {name.lower(): name for name in reader.fieldnames or []}
                source_key = next((names[key] for key in names if key in ("article", "text", "document")), None)
                target_key = next((names[key] for key in names if key in ("highlights", "summary", "target")), None)
                if not source_key or not target_key: raise ValueError(f"Could not locate article/summary columns in {path}")
                for row in reader:
                    source, target = self.clean_text(row.get(source_key, "")), self.clean_text(row.get(target_key, ""))
                    if source and target: yield source, target


class CornellMovieDataset(BaseSeq2SeqDataset):
    def iter_pairs(self):
        line_map = {}
        with open(os.path.join(self.dataset_path, "movie_lines.txt"), encoding="utf-8", errors="replace") as handle:
            for line in handle:
                fields = line.rstrip("\n").split(" +++$+++ ")
                if len(fields) == 5: line_map[fields[0]] = self.clean_text(fields[-1])
        with open(os.path.join(self.dataset_path, "movie_conversations.txt"), encoding="utf-8", errors="replace") as handle:
            for line in handle:
                fields = line.rstrip("\n").split(" +++$+++ ")
                if not fields: continue
                try: ids = ast.literal_eval(fields[-1])
                except (ValueError, SyntaxError): continue
                for first, second in zip(ids, ids[1:]):
                    source, target = line_map.get(first, ""), line_map.get(second, "")
                    if source and target: yield source, target


class Dataset:
    _types = {"english_french": EnglishFrenchDataset, "english_german": EnglishGermanDataset,
              "manythings_english_french": ManyThingsEnglishFrenchDataset, "cornell_movie_dialogs": CornellMovieDataset,
              "cnn_dailymail": CNNDailyMailDataset, "wikilarge": WikiLargeDataset}

    def __init__(self, dataset_name, dataset_path, **kwargs):
        try: implementation = self._types[dataset_name.lower()]
        except KeyError as error: raise ValueError(f"Unsupported dataset {dataset_name}; choose from {sorted(self._types)}") from error
        self.name = dataset_name.lower(); self.dataset = implementation(dataset_path, **kwargs)

    def load_data(self):
        self.dataset.prepare_vectorizers(self.name)
        return self.dataset.build_tf_dataset(True), self.dataset.build_tf_dataset(False)
    def __getattr__(self, name): return getattr(self.dataset, name)
