import argparse
from config import *
from dataset import build_cooccurrence
from model import sparse_matrix

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--corpus", default=CORPUS); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); vocab, indices, values = build_cooccurrence(args.corpus, VOCAB_SIZE if not args.smoke else min(VOCAB_SIZE, 1000))
    matrix = sparse_matrix(len(vocab), indices, values)
    print(f"Vocabulary: {len(vocab)}; non-zero co-occurrences: {matrix.values.shape[0]}")
if __name__ == "__main__": main()
