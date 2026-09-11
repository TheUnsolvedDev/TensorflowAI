import os
import re
from collections import Counter, deque
import numpy as np
from config import *
def _tokens(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle: yield from re.findall(r"[a-z0-9']+", line.lower())
def cooccurrence_dense(corpus=CORPUS, vocab_size=VOCAB_SIZE, window=WINDOW_SIZE):
    path = corpus if os.path.isabs(corpus) else os.path.join(DATASET_ROOT, corpus)
    vocab = ["<pad>", "<unk>"] + [word for word, _ in Counter(_tokens(path)).most_common(vocab_size - 2)]
    ids, counts, history = {word: index for index, word in enumerate(vocab)}, Counter(), deque(maxlen=window)
    for word in _tokens(path):
        current = ids.get(word, 1)
        for distance, context in enumerate(reversed(history), start=1):
            if len(counts) < MAX_COOC_PAIRS or (current, context) in counts:
                counts[(current, context)] += 1.0 / distance; counts[(context, current)] += 1.0 / distance
        history.append(current)
    matrix = np.zeros((len(vocab), len(vocab)), dtype=np.float32)
    for (row, column), value in counts.items(): matrix[row, column] = value
    return vocab, matrix
