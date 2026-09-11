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
    ids = {word: index for index, word in enumerate(vocab)}; counts = Counter(); history = deque(maxlen=window)
    for word in _tokens(path):
        current = ids.get(word, 1)
        for distance, context in enumerate(reversed(history), start=1):
            if len(counts) < MAX_COOC_PAIRS or (current, context) in counts:
                counts[(current, context)] += 1.0 / distance; counts[(context, current)] += 1.0 / distance
        history.append(current)
    def records():
        for (center, context), value in counts.items():
            yield {"center": center, "context": context, "count": value}, 0.0
    signature = ({"center": tf.TensorSpec((), tf.int32), "context": tf.TensorSpec((), tf.int32), "count": tf.TensorSpec((), tf.float32)}, tf.TensorSpec((), tf.float32))
    return tf.data.Dataset.from_generator(records, output_signature=signature).shuffle(min(SHUFFLE_BUFFER, max(1, len(counts))), seed=SEED).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE), vocab
