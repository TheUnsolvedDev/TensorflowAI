import os
import re
from collections import Counter
import tensorflow as tf
from config import *


def _words(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield from re.findall(r"[a-z0-9']+", line.lower())


def build_corpus(corpus=CORPUS, vocab_size=VOCAB_SIZE):
    path = corpus if os.path.isabs(
        corpus) else os.path.join(DATASET_ROOT, corpus)
    counts = Counter(_words(path))
    vocab = ["<pad>", "<unk>"] + \
        [w for w, _ in counts.most_common(vocab_size-2)]
    ids = {w: i for i, w in enumerate(vocab)}

    def ids_stream():
        for word in _words(path):
            yield ids.get(word, 1)
    return path, vocab, ids, ids_stream


def pair_dataset(corpus=CORPUS, vocab_size=VOCAB_SIZE, window=WINDOW_SIZE, batch_size=BATCH_SIZE):
    _, vocab, ids, stream = build_corpus(corpus, vocab_size)

    def gen():
        buf = []
        for token in stream():
            buf.append(token)
            if len(buf) >= 2*window+1:
                center = buf[window]
                for j, context in enumerate(buf):
                    if j != window:
                        yield center, context
                buf.pop(0)
    ds = tf.data.Dataset.from_generator(gen, output_signature=(
        tf.TensorSpec((), tf.int32), tf.TensorSpec((), tf.int32)))
    return ds.shuffle(8192, seed=SEED).batch(batch_size).prefetch(tf.data.AUTOTUNE), vocab
