import argparse
from config import *
from dataset import cooccurrence_dense
from model import ppmi_svd
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--corpus", default=CORPUS); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); vocab, matrix = cooccurrence_dense(args.corpus, min(VOCAB_SIZE, 500) if args.smoke else VOCAB_SIZE)
    embeddings = ppmi_svd(matrix, EMBEDDING_DIM); print(f"PPMI-SVD embeddings: {embeddings.shape}; vocabulary: {len(vocab)}")
if __name__ == "__main__": main()
