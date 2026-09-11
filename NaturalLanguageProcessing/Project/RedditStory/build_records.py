#!/usr/bin/env python3
"""Build TFRecord shards for language-model training."""

from __future__ import annotations

from pathlib import Path

from tqdm.auto import tqdm

from storygen.core import BPETokenizer, write_split_tfrecords
from storygen.utils import ensure_dir, load_config_module, make_arg_parser, resolve_cli_path, write_json


def main():
    parser = make_arg_parser("Convert cleaned JSONL splits into packed TFRecord shards.")
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--clean-dir", default="")
    parser.add_argument("--tokenizer-dir", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--shard-size", type=int, default=2048)
    args = parser.parse_args()

    config = load_config_module(args.config)
    clean_dir = resolve_cli_path(config, args.clean_dir, fallback_key="dataset_dir", fallback="artifacts/data_v1")
    tokenizer_dir = resolve_cli_path(config, args.tokenizer_dir, fallback_key="tokenizer_dir", fallback="artifacts/tokenizer_v1")
    output_path = resolve_cli_path(config, args.out_dir, fallback_key="records_dir", fallback="artifacts/records_v1")

    tokenizer = BPETokenizer.from_file(Path(tokenizer_dir) / "tokenizer.json")
    out_dir = ensure_dir(output_path)

    aggregate = {}
    for split in tqdm(("train", "val", "test"), desc="Building TFRecord splits", unit="split"):
        input_path = Path(clean_dir) / f"clean_{split}.jsonl"
        split_dir = ensure_dir(Path(out_dir) / split)
        aggregate[split] = write_split_tfrecords(
            input_path=input_path,
            output_dir=split_dir,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            split_name=split,
            shard_size=args.shard_size,
        )
    write_json(Path(out_dir) / "stats.json", aggregate)
    print(f"TFRecords saved to {Path(out_dir).resolve()}")


if __name__ == "__main__":
    main()
