"""Small, runnable checks for the Transformer and official dataset splits."""
import csv
import os
import tempfile

import tensorflow as tf

from dataset import CNNDailyMailDataset
from model import build_transformer_seq2seq_model
from train_and_test import TransformerLearningRate, masked_accuracy, masked_loss


def check_model():
    model = build_transformer_seq2seq_model(
        101, 113, source_max_length=8, target_max_length=7,
        d_model=16, num_heads=4, dff=32,
        num_encoder_layers=2, num_decoder_layers=2, dropout_rate=0.0)
    source = tf.constant([[2, 3, 4, 0, 0, 0, 0, 0]], tf.int32)
    decoder = tf.constant([[2, 4, 3, 0, 0, 0]], tf.int32)
    target = tf.constant([[4, 3, 0, 0, 0, 0]], tf.int32)
    logits = model([source, decoder], training=False)
    assert logits.shape == (1, 6, 113)
    assert bool(tf.reduce_all(tf.math.is_finite(logits)))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            TransformerLearningRate(16), clipnorm=1.0),
        loss=masked_loss, metrics=[masked_accuracy])
    result = model.train_on_batch(
        [source, decoder], target, return_dict=True)
    assert all(bool(tf.math.is_finite(value)) for value in result.values())


def check_official_splits():
    with tempfile.TemporaryDirectory() as root:
        records = {
            "train.csv": ("train article", "train summary"),
            "validation.csv": ("validation article", "validation summary"),
            "test.csv": ("test article", "test summary"),
        }
        for filename, (article, highlights) in records.items():
            with open(os.path.join(root, filename), "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["article", "highlights"])
                writer.writeheader()
                writer.writerow({"article": article, "highlights": highlights})
        dataset = CNNDailyMailDataset(root)
        assert list(dataset._split_pairs(False)) == [records["train.csv"]]
        assert list(dataset._split_pairs(True)) == [records["validation.csv"]]


if __name__ == "__main__":
    check_model()
    check_official_splits()
    print("Transformer implementation checks passed")
