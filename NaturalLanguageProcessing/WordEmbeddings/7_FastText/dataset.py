import hashlib
import os
import re
from collections import Counter, deque
import tensorflow as tf
from config import *

def _tokens(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle: yield from re.findall(r"[a-z0-9']+", line.lower())

def _subwords(word):
    token = f"<{word}>"; values = []
    for size in range(MIN_NGRAM, MAX_NGRAM + 1):
        for start in range(max(0, len(token) - size + 1)):
            digest = hashlib.blake2b(token[start:start + size].encode(), digest_size=8).digest()
            values.append(int.from_bytes(digest, "little") % NGRAM_BUCKETS)
    return (values[:MAX_SUBWORDS] + [0] * MAX_SUBWORDS)[:MAX_SUBWORDS]

def pair_dataset(corpus=CORPUS, vocab_size=VOCAB_SIZE, window=WINDOW_SIZE, batch_size=BATCH_SIZE):
    path = corpus if os.path.isabs(corpus) else os.path.join(DATASET_ROOT, corpus)
    vocab = ["<pad>", "<unk>"] + [word for word, _ in Counter(_tokens(path)).most_common(vocab_size - 2)]
    ids = {word: index for index, word in enumerate(vocab)}
    def records():
        buffer = deque(maxlen=2 * window + 1)
        for word in _tokens(path):
            buffer.append((ids.get(word, 1), word))
            if len(buffer) == 2 * window + 1:
                center_id, center_word = buffer[window]
                for index, (context_id, _) in enumerate(buffer):
                    if index != window: yield {"word_id": center_id, "subword_ids": _subwords(center_word)}, context_id
    spec = ({"word_id": tf.TensorSpec((), tf.int32), "subword_ids": tf.TensorSpec((MAX_SUBWORDS,), tf.int32)}, tf.TensorSpec((), tf.int32))
    return tf.data.Dataset.from_generator(records, output_signature=spec).shuffle(SHUFFLE_BUFFER, seed=SEED).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE), vocab
