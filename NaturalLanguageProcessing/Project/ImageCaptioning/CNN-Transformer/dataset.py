"""Local COCO 2017 caption reader and streaming tf.data pipeline."""
import json
import math
import os
import string
from collections import Counter

import matplotlib.pyplot as plt
import tensorflow as tf

from config import (BATCH_SIZE, DATASET_DIR, IMAGE_CHANNELS, IMAGE_SIZE, MAX_LENGTH,
                    MIN_TOKEN_FREQUENCY, SEED, SHUFFLE_BUFFER, TOKEN_DROPOUT,
                    VOCAB_ADAPT_BATCH_SIZE,
                    VOCABULARY_HISTOGRAM_PATH, VOCABULARY_PATH,
                    VOCABULARY_STATS_PATH, VOCAB_SIZE)

AUTOTUNE = tf.data.AUTOTUNE


class CocoCaptionDataset:
    def __init__(self, root=DATASET_DIR, batch_size=BATCH_SIZE):
        self.root = root
        self.batch_size = batch_size
        self.vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=VOCAB_SIZE,
            output_mode="int",
            output_sequence_length=MAX_LENGTH,
        )

    def rows(self, validation=None):
        splits = ("train", "val") if validation is None else (("val",) if validation else ("train",))
        for split in splits:
            images_dir = os.path.join(self.root, f"{split}2017")
            captions_path = os.path.join(
                self.root, "annotations", f"captions_{split}2017.json")
            if not os.path.isdir(images_dir) or not os.path.isfile(captions_path):
                raise FileNotFoundError(
                    f"Expected {split}2017/ and annotations/captions_{split}2017.json in {self.root}")
            with open(captions_path, encoding="utf-8") as handle:
                annotations = json.load(handle)
            filenames = {image["id"]: image["file_name"] for image in annotations["images"]}
            for annotation in annotations["annotations"]:
                caption = annotation["caption"].strip()
                if caption:
                    yield (os.path.join(images_dir, filenames[annotation["image_id"]]),
                           f"start {caption} end")

    def prepare_vocabulary(self):
        self.save_vocabulary_statistics()
        if os.path.isfile(VOCABULARY_PATH):
            with open(VOCABULARY_PATH, encoding="utf-8") as handle:
                vocabulary = json.load(handle)
            if len(vocabulary) == VOCAB_SIZE:
                self.vectorizer.set_vocabulary(vocabulary)
                print(f"Loaded vocabulary: {VOCABULARY_PATH}")
                return
            print(f"Rebuilding vocabulary for {VOCAB_SIZE} tokens")
        captions = tf.data.Dataset.from_generator(
            lambda: (caption for _, caption in self.rows(False)),
            output_signature=tf.TensorSpec((), tf.string),
        ).batch(VOCAB_ADAPT_BATCH_SIZE)
        self.vectorizer.adapt(captions)
        with open(VOCABULARY_PATH, "w", encoding="utf-8") as handle:
            json.dump(self.vectorizer.get_vocabulary(), handle)
        print(f"Saved vocabulary: {VOCABULARY_PATH}")

    def save_vocabulary_statistics(self):
        counts = Counter()
        remove_punctuation = str.maketrans("", "", string.punctuation)
        for _, caption in self.rows(False):
            counts.update(caption.lower().translate(remove_punctuation).split())
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        frequencies = [count for _, count in ordered]
        total = sum(frequencies)
        kept = sum(count >= MIN_TOKEN_FREQUENCY for count in frequencies)
        recommended = kept + 2  # TextVectorization reserves padding and [UNK].
        covered = sum(frequencies[:kept])
        stats = {
            "training_token_count": total,
            "unique_training_tokens": len(counts),
            "minimum_frequency": MIN_TOKEN_FREQUENCY,
            "recommended_vocabulary_size": recommended,
            "configured_vocabulary_size": VOCAB_SIZE,
            "token_coverage": covered / total,
            "unknown_token_occurrences": total - covered,
            "token_counts": dict(ordered),
        }
        os.makedirs(os.path.dirname(VOCABULARY_STATS_PATH), exist_ok=True)
        with open(VOCABULARY_STATS_PATH, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)

        bins = [1]
        while bins[-1] <= frequencies[0]:
            bins.append(bins[-1] * 2)
        coverage = []
        running = 0
        for count in frequencies:
            running += count
            coverage.append(running / total)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(frequencies, bins=bins)
        axes[0].set_xscale("log", base=2)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Token frequency")
        axes[0].set_ylabel("Number of unique tokens")
        axes[0].set_title("Training vocabulary frequency")
        axes[1].plot(range(1, len(coverage) + 1), coverage)
        axes[1].axvline(kept, color="red", linestyle="--",
                        label=f"min frequency {MIN_TOKEN_FREQUENCY}: {recommended} entries")
        axes[1].set_xlabel("Most frequent tokens retained")
        axes[1].set_ylabel("Token coverage")
        axes[1].set_title("Vocabulary coverage")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(VOCABULARY_HISTOGRAM_PATH, dpi=150)
        plt.close(figure)
        print(f"Vocabulary recommendation: {recommended} entries, {covered / total:.2%} coverage")

    def _decode(self, path, caption, training):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=IMAGE_CHANNELS)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32)
        if training:
            image = tf.image.random_flip_left_right(image, seed=SEED)
        tokens = self.vectorizer(caption)
        inputs = tokens[:-1]
        targets = tokens[1:]
        if training and TOKEN_DROPOUT:
            eligible = (inputs > 1) & (inputs != self.vectorizer("start")[0])
            dropped = tf.random.uniform(tf.shape(inputs), seed=SEED) < TOKEN_DROPOUT
            inputs = tf.where(eligible & dropped, tf.ones_like(inputs), inputs)
        mask = tf.cast(targets != 0, tf.float32)
        weights = mask * tf.cast(MAX_LENGTH - 1, tf.float32) / tf.maximum(tf.reduce_sum(mask), 1.0)
        return {"image": image, "tokens": inputs}, targets, weights

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
    data = CocoCaptionDataset(batch_size=2)
    train, _ = data.load_data()
    inputs, targets, weights = next(iter(train))
    assert inputs["image"].shape == (2, *IMAGE_SIZE, IMAGE_CHANNELS)
    assert inputs["tokens"].shape == targets.shape == (2, MAX_LENGTH - 1)
    assert weights.shape == targets.shape
    print("Pipeline check passed")
