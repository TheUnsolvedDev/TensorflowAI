import os
import re
import random
import itertools

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm.auto import tqdm
from config import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

AUTOTUNE = tf.data.AUTOTUNE

tqdm.pandas()

URL_RE = re.compile(r"http\S+")
SPACE_RE = re.compile(r"\s+")


class BaseSeq2SeqDataset:

    def __init__(self, dataset_path, batch_size=64, source_max_length=128, target_max_length=128, vocab_size=30000, validation_split=0.1, seed=42, lowercase=True):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length
        self.vocab_size = vocab_size
        self.validation_split = validation_split
        self.seed = seed
        self.lowercase = lowercase

        self.sources = []
        self.targets = []

        self.source_vectorizer = None
        self.target_vectorizer = None

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def clean_text(self, text):
        if self.lowercase:
            text = text.lower()
        text = URL_RE.sub(" ", text)
        text = SPACE_RE.sub(" ", text)

        return text.strip()

    def build_vectorizers(self):
        self.source_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=self.vocab_size, output_mode="int", output_sequence_length=self.source_max_length, standardize=None)
        self.target_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=self.vocab_size, output_mode="int", output_sequence_length=self.target_max_length, standardize=None)
        print("Building source vectorizer...")
        self.source_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(
            self.sources).batch(4096).prefetch(AUTOTUNE))
        print("Building target vectorizer...")
        self.target_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(
            self.targets).batch(4096).prefetch(AUTOTUNE))

    def save_vectorizers(self, name):
        src_path = f"source_vectorizer_{name}.keras"
        tgt_path = f"target_vectorizer_{name}.keras"
        src_model = tf.keras.Sequential([self.source_vectorizer])
        tgt_model = tf.keras.Sequential([self.target_vectorizer])
        src_model(tf.constant(["test"]))
        tgt_model(tf.constant(["test"]))
        src_model.save(src_path)
        tgt_model.save(tgt_path)

    def load_vectorizers(self, name):
        src_path = f"source_vectorizer_{name}.keras"
        tgt_path = f"target_vectorizer_{name}.keras"
        self.source_vectorizer = tf.keras.models.load_model(src_path).layers[0]
        self.target_vectorizer = tf.keras.models.load_model(tgt_path).layers[0]

    def prepare_vectorizers(self, name):
        src_path = f"source_vectorizer_{name}.keras"
        tgt_path = f"target_vectorizer_{name}.keras"
        if os.path.exists(src_path) and os.path.exists(tgt_path):
            print("Loading Vectorizers")
            self.load_vectorizers(name)
        else:
            print("Building Vectorizers")
            self.build_vectorizers()
            self.save_vectorizers(name)

    def split_dataset(self):
        indices = np.arange(len(self.sources))
        np.random.shuffle(indices)
        sources = np.array(self.sources, dtype=object)[indices]
        targets = np.array(self.targets, dtype=object)[indices]
        val_size = int(len(sources) * self.validation_split)
        x_train = sources[val_size:]
        y_train = targets[val_size:]
        x_val = sources[:val_size]
        y_val = targets[:val_size]
        return (x_train, y_train), (x_val, y_val)

    def encode(self, source, target):
        source = self.source_vectorizer(tf.expand_dims(source, axis=0))
        target = self.target_vectorizer(tf.expand_dims(target, axis=0))
        source = tf.squeeze(source, axis=0)
        target = tf.squeeze(target, axis=0)
        decoder_input = target[:-1]
        decoder_target = target[1:]
        source = tf.ensure_shape(source, [self.source_max_length])
        decoder_input = tf.ensure_shape(
            decoder_input, [self.target_max_length - 1])
        decoder_target = tf.ensure_shape(
            decoder_target, [self.target_max_length - 1])
        return (source, decoder_input), decoder_target

    def build_tf_dataset(self, sources, targets, training=False):
        ds = tf.data.Dataset.from_tensor_slices((sources, targets))
        if training:
            ds = ds.shuffle(min(len(sources), 100000),
                            seed=self.seed, reshuffle_each_iteration=True)
        ds = ds.map(self.encode, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    def encode_source(self, text):
        return self.source_vectorizer(tf.constant([text])).numpy()[0]

    def decode_source(self, tokens):
        vocab = self.source_vectorizer.get_vocabulary()
        return " ".join([vocab[token] for token in tokens if token not in [0, 1]])

    def decode_target(self, tokens):
        vocab = self.target_vectorizer.get_vocabulary()
        return " ".join([vocab[token] for token in tokens if token not in [0, 1]])

    def get_source_vocab_size(self):
        return len(self.source_vectorizer.get_vocabulary())

    def get_target_vocab_size(self):
        return len(self.target_vectorizer.get_vocabulary())


class EnglishFrenchDataset(BaseSeq2SeqDataset):
    def _detect_columns(self, columns):
        source_col = next(
            (col for col in columns if "en" in col.lower() or "english" in col.lower()), None)
        target_col = next(
            (col for col in columns if "fr" in col.lower() or "french" in col.lower()), None)
        if source_col is None or target_col is None:
            raise ValueError("Could not detect English/French columns")
        return source_col, target_col

    # def load_data(self):
    #     files = ["wmt14_translate_fr-en_train.csv", "wmt14_translate_fr-en_validation.csv", "wmt14_translate_fr-en_test.csv"]
    #     self.sources = []
    #     self.targets = []
    #     for file_name in tqdm(files, desc="Loading CSV Files"):
    #         file_path = os.path.join(self.dataset_path, file_name)
    #         df = pd.read_csv(file_path, engine="pyarrow", on_bad_lines="skip", dtype=str)
    #         source_col, target_col = self._detect_columns(df.columns)
    #         print(f"Processing File : {len(df)} rows")
    #         sources = df[source_col].str.lower().str.replace(r"http\S+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip().tolist()
    #         targets = ("[start] " + df[target_col].str.lower().str.replace(r"http\S+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip() + " [end]").tolist()
    #         self.sources.extend(sources)
    #         self.targets.extend(targets)
    #     return self.sources, self.targets

    def load_data(self):
        files = ["wmt14_translate_fr-en_train.csv",
                 "wmt14_translate_fr-en_validation.csv", "wmt14_translate_fr-en_test.csv"]
        self.sources = []
        self.targets = []
        for file_name in tqdm(files, desc="Loading CSV Files"):
            file_path = os.path.join(self.dataset_path, file_name)
            chunk_iter = pd.read_csv(
                file_path, engine="python", on_bad_lines="skip", chunksize=1_000_000)
            first_chunk = next(chunk_iter)
            source_col, target_col = self._detect_columns(first_chunk.columns)
            count = 0
            for chunk in itertools.chain([first_chunk], chunk_iter):
                if not isinstance(chunk, pd.DataFrame):
                    continue
                print(
                    f"\rProcessing Chunk : {len(chunk)} rows {(count % 5)*"*"}", end='')
                count += 1
                sources = chunk[source_col].astype(str).str.lower().str.replace(
                    r"http\S+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip().tolist()
                targets = ("[start] " + chunk[target_col].astype(str).str.lower().str.replace(r"http\S+",
                           " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip() + " [end]").tolist()
                self.sources.extend(sources)
                self.targets.extend(targets)
        return self.sources, self.targets


class EnglishGermanDataset(BaseSeq2SeqDataset):
    def _detect_columns(self, columns):
        source_col = next(
            (col for col in columns if "en" in col.lower() or "english" in col.lower()), None)
        target_col = next(
            (col for col in columns if "de" in col.lower() or "german" in col.lower()), None)
        if source_col is None or target_col is None:
            raise ValueError("Could not detect English/German columns")
        return source_col, target_col

    def load_data(self):
        files = ["wmt14_translate_de-en_train.csv",
                 "wmt14_translate_de-en_validation.csv", "wmt14_translate_de-en_test.csv"]
        self.sources = []
        self.targets = []
        for file_name in tqdm(files, desc="Loading English-German CSV Files"):
            file_path = os.path.join(self.dataset_path, file_name)
            chunk_iter = pd.read_csv(
                file_path, engine="python", on_bad_lines="skip", chunksize=500_000)
            first_chunk = next(chunk_iter)
            source_col, target_col = self._detect_columns(first_chunk.columns)
            count = 0
            for chunk in itertools.chain([first_chunk], chunk_iter):
                if not isinstance(chunk, pd.DataFrame):
                    continue
                print(
                    f"\rProcessing Chunk : {len(chunk)} rows {(count % 5)*"*"}", end='')
                count += 1
                sources = chunk[source_col].astype(str).str.lower().str.replace(
                    r"http\S+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip().tolist()
                targets = ("[start] " + chunk[target_col].astype(str).str.lower().str.replace(r"http\S+",
                           " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip() + " [end]").tolist()
                self.sources.extend(sources)
                self.targets.extend(targets)
        return self.sources, self.targets
    
    # def load_data(self):
    #     files = ["wmt14_translate_de-en_train.csv", "wmt14_translate_de-en_validation.csv", "wmt14_translate_de-en_test.csv"]
    #     self.sources = []
    #     self.targets = []
    #     for file_name in tqdm(files, desc="Loading CSV Files"):
    #         file_path = os.path.join(self.dataset_path, file_name)
    #         df = pd.read_csv(file_path, engine="pyarrow", on_bad_lines="skip", dtype=str)
    #         source_col, target_col = self._detect_columns(df.columns)
    #         print(f"Processing File : {len(df)} rows")
    #         sources = df[source_col].str.lower().str.replace(r"http\S+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip().tolist()
    #         targets = ("[start] " + df[target_col].str.lower().str.replace(r"http\S+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip() + " [end]").tolist()
    #         self.sources.extend(sources)
    #         self.targets.extend(targets)
    #     return self.sources, self.targets


class CornellMovieDataset(BaseSeq2SeqDataset):

    def load_data(self):
        lines_path = os.path.join(self.dataset_path, "movie_lines.txt")
        conv_path = os.path.join(self.dataset_path, "movie_conversations.txt")
        id2line = {}
        with open(lines_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in tqdm(f, desc="Loading Movie Lines"):
                parts = line.split(" +++$+++ ")
                if len(parts) == 5:
                    id2line[parts[0]] = self.clean_text(parts[-1])
        with open(conv_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in tqdm(f, desc="Loading Conversations"):
                parts = line.split(" +++$+++ ")
                ids = eval(parts[-1])
                for i in range(len(ids) - 1):
                    source = id2line.get(ids[i], "")
                    target = id2line.get(ids[i + 1], "")
                    if source and target:
                        target = "[start] " + target + " [end]"
                        self.sources.append(source)
                        self.targets.append(target)
        return self.sources, self.targets


class CNNDailyMailDataset(BaseSeq2SeqDataset):
    def load_data(self):
        self.sources = []
        self.targets = []
        for file_name in ["train.csv", "test.csv", "validation.csv"]:
            file_path = os.path.join(self.dataset_path, file_name)
            if not os.path.exists(file_path):
                continue
            df = pd.read_csv(file_path)
            article_col = None
            summary_col = None
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ["article", "text", "document"]:
                    article_col = col
                if col_lower in ["highlights", "summary", "target"]:
                    summary_col = col
            if article_col is None or summary_col is None:
                raise ValueError("Could not find article/summary columns")
            texts = df[article_col].astype(str).str.lower().str.replace(
                r"http\S+", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip().tolist()
            summaries = ("[start] " + df[summary_col].astype(str).str.lower().str.replace(r"http\S+",
                         " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip() + " [end]").tolist()

            self.sources.extend(texts)
            self.targets.extend(summaries)

        return self.sources, self.targets


class ManyThingsEnglishFrenchDataset(BaseSeq2SeqDataset):
    def load_data(self):
        self.sources = []
        self.targets = []
        file_path = os.path.join(self.dataset_path, "fra.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading ManyThings English-French Dataset"):
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                source = self.clean_text(parts[0])
                target = self.clean_text(parts[1])
                if not source or not target:
                    continue
                target = f"[start] {target} [end]"
                self.sources.append(source)
                self.targets.append(target)
        return self.sources, self.targets


class WikiLargeDataset(BaseSeq2SeqDataset):

    def load_data(self):

        self.sources = []
        self.targets = []

        files = [
            "wiki.full.aner.ori.train.95.tsv",
            "wiki.full.aner.ori.valid.95.tsv",
            "wiki.full.aner.ori.test.95.tsv"
        ]

        for file_name in files:

            file_path = os.path.join(self.dataset_path, file_name)

            if not os.path.exists(file_path):
                continue

            df = pd.read_csv(file_path, sep="\t", header=None, names=[
                             "source", "target"], on_bad_lines="skip")

            sources = df["source"].astype(str).str.lower().str.replace(
                r"http\\S+", " ", regex=True).str.replace(r"\\s+", " ", regex=True).str.strip().tolist()

            targets = ("[start] " + df["target"].astype(str).str.lower().str.replace(r"http\\S+",
                       " ", regex=True).str.replace(r"\\s+", " ", regex=True).str.strip() + " [end]").tolist()

            self.sources.extend(sources)
            self.targets.extend(targets)

        return self.sources, self.targets


class Dataset:
    def __init__(self, dataset_name, dataset_path, batch_size=BATCH_SIZE, source_max_length=SOURCE_MAX_LENGTH, target_max_length=TARGET_MAX_LENGTH, vocab_size=VOCAB_SIZE, validation_split=VALIDATION_SPLIT, seed=42, lowercase=True):
        datasets = datasets = {
            "english_french": EnglishFrenchDataset,
            "manythings_english_french": ManyThingsEnglishFrenchDataset,
            "english_german": EnglishGermanDataset,
            "cornell_movie_dialogs": CornellMovieDataset,
            "cnn_dailymail": CNNDailyMailDataset,
            "wikilarge": WikiLargeDataset
        }
        if dataset_name.lower() not in datasets:
            raise ValueError(f"Unsupported Dataset: {dataset_name}")
        self.name = dataset_name.lower()
        self.dataset = datasets[self.name](dataset_path=dataset_path, batch_size=batch_size, source_max_length=source_max_length,
                                           target_max_length=target_max_length, vocab_size=vocab_size, validation_split=validation_split, seed=seed, lowercase=lowercase)

    def load_data(self):
        self.dataset.load_data()
        self.dataset.prepare_vectorizers(self.name)
        (x_train, y_train), (x_val, y_val) = self.dataset.split_dataset()
        train_ds = self.dataset.build_tf_dataset(
            x_train, y_train, training=True)
        val_ds = self.dataset.build_tf_dataset(x_val, y_val)
        return train_ds, val_ds

    def encode_source(self, text):
        return self.dataset.encode_source(text)

    def decode_source(self, tokens):
        return self.dataset.decode_source(tokens)

    def decode_target(self, tokens):
        return self.dataset.decode_target(tokens)

    def get_source_vocab_size(self):
        return self.dataset.get_source_vocab_size()

    def get_target_vocab_size(self):
        return self.dataset.get_target_vocab_size()


if __name__ == "__main__":

    datasets = [
        # ("english_french", "/home/shuvrajeet/Documents/Dataset/english-french"),
        ("english_german", "/home/shuvrajeet/Documents/Dataset/english-german"),
        # ("cornell_movie_dialogs", "/home/shuvrajeet/Documents/Dataset/cornell_movie_dialogs_corpus"),
        # ("cnn_dailymail", "/home/shuvrajeet/Documents/Dataset/cnn_dailymail"),
        # ("manythings_english_french","/home/shuvrajeet/Documents/Dataset/fra-eng/"),
        # ("wikilarge", "/home/shuvrajeet/Documents/Dataset/wikilarge-text-simplification")

    ]

    for dataset_name, dataset_path in datasets:

        print("\n" + "=" * 120)
        print(f"DATASET : {dataset_name}")
        print("=" * 120)
        train_config = DATASET_CONFIGS[dataset_name]

        dataset = Dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            batch_size=train_config["batch_size"],
            source_max_length=train_config["source_max_length"],
            target_max_length=train_config["target_max_length"],
            vocab_size=train_config["vocab_size"],
            validation_split=VALIDATION_SPLIT,
            seed=SEED,
            lowercase=LOWERCASE
        )
        train_ds, val_ds = dataset.load_data()
        print("\nSource Vocab :", dataset.get_source_vocab_size())
        print("Target Vocab :", dataset.get_target_vocab_size())

        sample = "how are you"
        encoded = dataset.encode_source(sample)
        decoded = dataset.decode_source(encoded)

        print("\nOriginal :", sample)
        print("\nEncoded :", encoded)
        print("\nDecoded :", decoded)
        count = 0
        for (encoder_input, decoder_input), decoder_target in train_ds:

            print("\nEncoder Input Shape  :", encoder_input.shape)
            print("Decoder Input Shape  :", decoder_input.shape)
            print("Decoder Target Shape :", decoder_target.shape)
            print(count)
            count += 1

            # print("\nSource Example :")
            # print(dataset.decode_source(encoder_input[0].numpy()))

            # print("\nTarget Input Example :")
            # print(dataset.decode_target(decoder_input[0].numpy()))

            # print("\nTarget Output Example :")
            # print(dataset.decode_target(decoder_target[0].numpy()))

        count = 0
        for (encoder_input, decoder_input), decoder_target in val_ds:

            print("\nEncoder Input Shape  :", encoder_input.shape)
            print("Decoder Input Shape  :", decoder_input.shape)
            print("Decoder Target Shape :", decoder_target.shape)
            print(count)
            count += 1
