#!/usr/bin/env python3
"""Train the repo-local BPE tokenizer."""

from __future__ import annotations

from pathlib import Path

from tqdm.auto import tqdm

from storygen.core import train_bpe_tokenizer
from storygen.utils import ensure_dir, load_config_module, make_arg_parser, read_jsonl, resolve_cli_path, write_json


def main():
    parser = make_arg_parser("Train a minimal-dependency BPE tokenizer.")
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--input", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--sample-limit", type=int, default=0)
    args = parser.parse_args()

    config = load_config_module(args.config)
    input_path = resolve_cli_path(config, args.input, fallback=Path(resolve_cli_path(config, "", fallback_key="dataset_dir", fallback="artifacts/data_v1")) / "clean_train.jsonl")
    output_path = resolve_cli_path(config, args.out_dir, fallback_key="tokenizer_dir", fallback="artifacts/tokenizer_v1")

    def text_stream():
        for index, (_, row) in enumerate(tqdm(read_jsonl(input_path), desc="Loading tokenizer texts", unit="row")):
            yield row["clean_text"]
            if args.sample_limit and index + 1 >= args.sample_limit:
                return

    tokenizer = train_bpe_tokenizer(text_stream(), vocab_size=args.vocab_size, min_frequency=args.min_frequency)
    out_dir = ensure_dir(output_path)
    tokenizer.save(out_dir / "tokenizer.json")
    write_json(
        out_dir / "stats.json",
        {
            "vocab_size": tokenizer.vocab_size,
            "trained_texts": sum(1 for _ in read_jsonl(input_path)) if not args.sample_limit else args.sample_limit,
            "requested_vocab_size": args.vocab_size,
            "min_frequency": args.min_frequency,
        },
    )
    print(f"Tokenizer saved to {Path(out_dir).resolve()}")


if __name__ == "__main__":
    main()
