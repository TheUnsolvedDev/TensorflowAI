# dataset.py

import tensorflow as tf
import pandas as pd
import numpy as np
import random
import re
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


AUTOTUNE = tf.data.AUTOTUNE


class BaseTextDataset:

    def __init__(self, dataset_path, batch_size=64, max_length=256, vocab_size=20000, validation_split=0.1, seed=42, lowercase=True):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.validation_split = validation_split
        self.seed = seed
        self.lowercase = lowercase

        self.texts = []
        self.labels = []

        self.vectorizer = None

        self.label2idx = {}
        self.idx2label = {}

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def clean_text(self, text):
        if self.lowercase:
            text = text.lower()
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def build_vectorizer(self):
        self.vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=self.vocab_size, output_mode="int", output_sequence_length=self.max_length, standardize=None)
        self.vectorizer.adapt(
            tf.data.Dataset.from_tensor_slices(self.texts).batch(1024))

    def save_vectorizer(self, name=None):
        save_path = os.path.join(f"text_vectorizer_{name}.keras")
        model = tf.keras.Sequential([self.vectorizer])
        model(tf.constant(["test"]))
        model.save(save_path)

    def load_vectorizer(self, name=None):
        save_path = os.path.join(f"text_vectorizer_{name}.keras")
        model = tf.keras.models.load_model(save_path)
        self.vectorizer = model.layers[0]

    def prepare_vectorizer(self, name=None):
        save_path = os.path.join(f"text_vectorizer_{name}.keras")
        if os.path.exists(save_path):
            print('loading')
            self.load_vectorizer(name)
        else:
            print('building')
            self.build_vectorizer()
            self.save_vectorizer(name)

    def split_dataset(self):
        indices = np.arange(len(self.texts))
        np.random.shuffle(indices)
        texts = np.array(self.texts, dtype=object)[indices]
        labels = np.array(self.labels, dtype=np.int32)[indices]
        val_size = int(len(texts)*self.validation_split)
        x_train = texts[val_size:]
        y_train = labels[val_size:]
        x_val = texts[:val_size]
        y_val = labels[:val_size]
        return (x_train, y_train), (x_val, y_val)

    def encode_text(self, text):
        return self.vectorizer(tf.constant([text])).numpy()[0]

    def decode_tokens(self, tokens):
        vocab = self.vectorizer.get_vocabulary()
        words = [vocab[token]
                 for token in tokens if token != 0 and token < len(vocab)]
        return " ".join(words)

    def encode(self, text, label):
        return self.vectorizer(text), label

    def build_tf_dataset(self, texts, labels, training=False):
        ds = tf.data.Dataset.from_tensor_slices((texts, labels))
        if training:
            ds = ds.shuffle(len(texts), seed=self.seed,
                            reshuffle_each_iteration=True)
        ds = ds.map(self.encode, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(self.batch_size).cache()
        ds = ds.prefetch(AUTOTUNE)
        return ds

    def get_vocab_size(self):
        return len(self.vectorizer.get_vocabulary())

    def get_num_classes(self):
        return len(self.label2idx)


class AGNewsDataset(BaseTextDataset):
    def load_data(self):
        texts = []
        labels = []
        for file_name in ["train.csv", "test.csv"]:
            df = pd.read_csv(os.path.join(self.dataset_path, file_name))
            texts.extend([self.clean_text(text) for text in (
                df["Title"]+" "+df["Description"]).tolist()])
            labels.extend((df["Class Index"]-1).tolist())
        self.texts = texts
        self.labels = labels
        unique_labels = sorted(set(labels))
        self.label2idx = {label: idx for idx,
                          label in enumerate(unique_labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        self.labels = [self.label2idx[label] for label in labels]


class DBPediaDataset(BaseTextDataset):
    def load_data(self):
        texts = []
        labels = []
        for file_name in ["DBPEDIA_train.csv", "DBPEDIA_val.csv"]:
            df=pd.read_csv(os.path.join(self.dataset_path,file_name))
            file_texts=df["text"].astype(str).tolist()
            file_labels=df["l1"].astype(str).tolist()
            texts.extend([self.clean_text(text) for text in file_texts])
            labels.extend(file_labels)
        unique_labels=sorted(set(labels))
        self.label2idx={label:idx for idx,label in enumerate(unique_labels)}
        self.idx2label={idx:label for label,idx in self.label2idx.items()}
        self.texts=texts
        self.labels=[self.label2idx[label] for label in labels]


class IMDBDataset(BaseTextDataset):
    def load_data(self):
        df = pd.read_csv(os.path.join(self.dataset_path, "IMDB_Dataset.csv"))
        self.texts = [self.clean_text(text) for text in df["review"].tolist()]
        self.labels = [1 if label ==
                       "positive" else 0 for label in df["sentiment"].tolist()]
        unique_labels = sorted(set(self.labels))
        self.label2idx = {label: idx for idx,
                          label in enumerate(unique_labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        self.labels = [self.label2idx[label] for label in self.labels]


class Dataset:
    def __init__(self, dataset_name, dataset_path, batch_size=64, max_length=256, vocab_size=20000, validation_split=0.1, seed=42, lowercase=True):
        datasets = {
            "ag_news": AGNewsDataset,
            "dbpedia": DBPediaDataset,
            "imdb": IMDBDataset
        }
        if dataset_name.lower() not in datasets:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.name = dataset_name.lower()
        self.dataset = datasets[dataset_name.lower()](dataset_path=dataset_path, batch_size=batch_size, max_length=max_length,
                                                      vocab_size=vocab_size, validation_split=validation_split, seed=seed, lowercase=lowercase)

    def load_data(self):
        self.dataset.load_data()
        self.dataset.prepare_vectorizer(self.name)
        (x_train, y_train), (x_val, y_val) = self.dataset.split_dataset()
        train_ds = self.dataset.build_tf_dataset(
            x_train, y_train, training=True)
        val_ds = self.dataset.build_tf_dataset(x_val, y_val)
        return train_ds, val_ds

    def encode_text(self, text):
        return self.dataset.encode_text(text)

    def decode_tokens(self, tokens):
        return self.dataset.decode_tokens(tokens)

    def get_vocab_size(self):
        return self.dataset.get_vocab_size()

    def get_num_classes(self):
        return self.dataset.get_num_classes()


if __name__ == "__main__":

    dataset = Dataset(dataset_name="ag_news",
                      dataset_path="/home/shuvrajeet/Documents/Dataset/ag_news")
    dataset2 = Dataset(dataset_name='dbpedia',
                       dataset_path='/home/shuvrajeet/Documents/Dataset/dbpedia/')
    dataset3 = Dataset(dataset_name='imdb',
                       dataset_path='/home/shuvrajeet/Documents/Dataset/imdb')

    train_ds, val_ds = dataset.load_data()
    train_ds, val_ds = dataset2.load_data()
    train_ds, val_ds = dataset3.load_data()
    print("Vocab Size:", dataset.get_vocab_size())
    print("Num Classes:", dataset.get_num_classes())
    
    print("Vocab Size:", dataset2.get_vocab_size())
    print("Num Classes:", dataset2.get_num_classes())
    
    print("Vocab Size:", dataset3.get_vocab_size())
    print("Num Classes:", dataset3.get_num_classes())

    sample_text = "Apple launches new AI model for mobile devices"
    encoded = dataset.encode_text(sample_text)
    decoded = dataset.decode_tokens(encoded)

    print("\nOriginal Text:\n", sample_text)
    print("\nEncoded Tokens:\n", encoded)
    print("\nDecoded Text:\n", decoded)

    for x, y in train_ds.take(1):
        print("\nBatch Input Shape:", x.shape)
        print("Batch Label Shape:", y.shape)
