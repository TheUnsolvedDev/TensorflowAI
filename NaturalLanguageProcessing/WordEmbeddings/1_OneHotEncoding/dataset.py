# dataset.py

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def read_file(FILE_LOCATION):
    with open(FILE_LOCATION, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    return lines


def build_vocab(texts, vocab_size=None):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size,
        oov_token="<OOV>"
    )
    tokenizer.fit_on_texts(texts)

    word_index = tokenizer.word_index
    index_word = {v: k for k, v in word_index.items()}

    vocab_size = len(word_index) + 1
    return tokenizer, word_index, index_word, vocab_size


def texts_to_sequences(tokenizer, texts):
    return tokenizer.texts_to_sequences(texts)


def pad_sequences(sequences, max_len):
    return tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=max_len,
        padding="post"
    )


def sequences_to_onehot(sequences, vocab_size):
    return tf.one_hot(sequences, depth=vocab_size)


def onehot_to_sequences(onehot_tensor):
    return tf.argmax(onehot_tensor, axis=-1)


def sequences_to_texts(sequences, index_word):
    texts = []
    for seq in sequences:
        words = []
        for idx in seq:
            idx = int(idx)
            if idx == 0:
                continue
            words.append(index_word.get(idx, "<UNK>"))
        texts.append(" ".join(words))
    return texts