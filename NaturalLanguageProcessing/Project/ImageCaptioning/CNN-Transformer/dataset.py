"""Flickr30k CSV reader and streaming tf.data pipeline."""
import csv
import hashlib
import json
import math
import os

import tensorflow as tf

from config import (BATCH_SIZE, DATASET_DIR, IMAGE_CHANNELS, IMAGE_SIZE, MAX_LENGTH,
                    SEED, SHUFFLE_BUFFER, VALIDATION_SPLIT, VOCAB_ADAPT_BATCH_SIZE,
                    VOCABULARY_PATH, VOCAB_SIZE)

AUTOTUNE = tf.data.AUTOTUNE


class Flickr30kDataset:
    def __init__(self, root=DATASET_DIR, batch_size=BATCH_SIZE):
        self.root = root
        self.batch_size = batch_size
        self.images_dir = os.path.join(root, "Images")
        self.captions_path = os.path.join(root, "captions.txt")
        self.vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=VOCAB_SIZE,
            output_mode="int",
            output_sequence_length=MAX_LENGTH,
        )

    def rows(self, validation=None):
        if not os.path.isdir(self.images_dir) or not os.path.isfile(self.captions_path):
            raise FileNotFoundError(f"Expected captions.txt and Images/ in {self.root}")
        with open(self.captions_path, newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                image, caption = row.get("image", "").strip(), row.get("caption", "").strip()
                if not image or not caption:
                    continue
                digest = hashlib.blake2b(f"{SEED}:{image}".encode(), digest_size=8).digest()
                is_validation = int.from_bytes(digest, "big") / 2**64 < VALIDATION_SPLIT
                if validation is None or is_validation == validation:
                    yield os.path.join(self.images_dir, image), f"start {caption} end"

    def prepare_vocabulary(self):
        if os.path.isfile(VOCABULARY_PATH):
            with open(VOCABULARY_PATH, encoding="utf-8") as handle:
                self.vectorizer.set_vocabulary(json.load(handle))
            print(f"Loaded vocabulary: {VOCABULARY_PATH}")
            return
        captions = tf.data.Dataset.from_generator(
            lambda: (caption for _, caption in self.rows(False)),
            output_signature=tf.TensorSpec((), tf.string),
        ).batch(VOCAB_ADAPT_BATCH_SIZE)
        self.vectorizer.adapt(captions)
        with open(VOCABULARY_PATH, "w", encoding="utf-8") as handle:
            json.dump(self.vectorizer.get_vocabulary(), handle)
        print(f"Saved vocabulary: {VOCABULARY_PATH}")

    def _decode(self, path, caption, training):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=IMAGE_CHANNELS)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32)
        if training:
            image = tf.image.random_flip_left_right(image, seed=SEED)
        tokens = self.vectorizer(caption)
        targets = tokens[1:]
        mask = tf.cast(targets != 0, tf.float32)
        weights = mask * tf.cast(MAX_LENGTH - 1, tf.float32) / tf.maximum(tf.reduce_sum(mask), 1.0)
        return {"image": image, "tokens": tokens[:-1]}, targets, weights

    def build(self, training):
        example_count = sum(1 for _ in self.rows(not training))
        batch_count = (example_count // self.batch_size if training else
                       math.ceil(example_count / self.batch_size))
        ds = tf.data.Dataset.from_generator(
            lambda: self.rows(not training),
            output_signature=(tf.TensorSpec((), tf.string), tf.TensorSpec((), tf.string)),
        )
        if training:
            ds = ds.shuffle(SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(lambda path, text: self._decode(path, text, training),
                    num_parallel_calls=AUTOTUNE, deterministic=not training)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds = ds.with_options(options).batch(self.batch_size, drop_remainder=training)
        return ds.apply(tf.data.experimental.assert_cardinality(batch_count)).prefetch(AUTOTUNE)

    def load_data(self):
        self.prepare_vocabulary()
        return self.build(True), self.build(False)

    @property
    def vocabulary(self):
        return self.vectorizer.get_vocabulary()


if __name__ == "__main__":
    data = Flickr30kDataset(batch_size=2)
    train, _ = data.load_data()
    inputs, targets, weights = next(iter(train))
    assert inputs["image"].shape == (2, *IMAGE_SIZE, IMAGE_CHANNELS)
    assert inputs["tokens"].shape == targets.shape == (2, MAX_LENGTH - 1)
    assert weights.shape == targets.shape
    print("Pipeline check passed")
