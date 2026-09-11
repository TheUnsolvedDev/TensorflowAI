import os
import re
from collections import Counter, deque
import tensorflow as tf
from config import *

def _tokens(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle: yield from re.findall(r"[a-z0-9']+", line.lower())

def pair_dataset(corpus=CORPUS, vocab_size=VOCAB_SIZE, window=WINDOW_SIZE, batch_size=BATCH_SIZE):
    path = corpus if os.path.isabs(corpus) else os.path.join(DATASET_ROOT, corpus)
    vocab = ["<pad>", "<unk>"] + [word for word, _ in Counter(_tokens(path)).most_common(vocab_size - 2)]
    ids = {word: index for index, word in enumerate(vocab)}; width = 2 * window
    def records():
        buffer = deque(maxlen=width + 1)
        for word in _tokens(path):
            buffer.append(ids.get(word, 1))
            if len(buffer) == width + 1:
                values = list(buffer); yield values[:window] + values[window + 1:], values[window]
    ds = tf.data.Dataset.from_generator(records, output_signature=(tf.TensorSpec((width,), tf.int32), tf.TensorSpec((), tf.int32)))
    return ds.shuffle(SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE), vocab
