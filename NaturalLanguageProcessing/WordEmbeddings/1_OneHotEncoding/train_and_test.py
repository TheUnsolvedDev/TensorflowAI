import silence_tensorflow.auto
import tensorflow as tf
import tqdm
import numpy as np

from dataset import *
from config import FILE_LOCATION, BATCH_SIZE

def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            

            
if __name__ == "__main__":
    configure_gpu()

    texts = read_file(FILE_LOCATION)

    print("\nOriginal Sentences:")
    for t in texts[:10]:
        print(t)

    tokenizer, word_index, index_word, vocab_size = build_vocab(texts)

    print("\nVocabulary Size:", vocab_size)
    print("Word Index Sample:", dict(list(word_index.items())[:10]))

    sequences = texts_to_sequences(tokenizer, texts)

    print("\nSequences:")
    print(sequences)

    max_len = max(len(seq) for seq in sequences)
    padded = pad_sequences(sequences, max_len)

    print("\nPadded Sequences:")
    print(padded)

    onehot = sequences_to_onehot(padded, vocab_size)

    print("\nOne-hot shape:")
    print(onehot.shape)  # (num_samples, seq_len, vocab_size)

    recovered_sequences = onehot_to_sequences(onehot)

    print("\nRecovered Sequences:")
    print(recovered_sequences.numpy())

    recovered_texts = sequences_to_texts(recovered_sequences.numpy(), index_word)
    print("\nRecovered Sentences:")
    for t in recovered_texts:
        print(t)