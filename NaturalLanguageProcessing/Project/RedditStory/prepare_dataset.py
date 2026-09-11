#!/usr/bin/env python3
"""Prepare cleaned story dataset artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tqdm.auto import tqdm

from storygen.core import process_row, summarize_lengths
from storygen.utils import ensure_dir, load_config_module, make_arg_parser, read_jsonl, resolve_cli_path, split_from_id, write_json


def main():
    parser = make_arg_parser("Clean, score, deduplicate, and split the raw story dataset.")
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--input", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--min-chars", type=int, default=800)
    parser.add_argument("--min-quality-score", type=float, default=25.0)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    config = load_config_module(args.config)
    input_path = resolve_cli_path(config, args.input, fallback_key="raw_data_file", fallback="output/ghost_stories.jsonl")
    output_path = resolve_cli_path(config, args.out_dir, fallback_key="dataset_dir", fallback="artifacts/data_v1")

    out_dir = ensure_dir(output_path)
    subreddit_counts: Counter = Counter()
    for line_number, row in tqdm(read_jsonl(input_path), desc="Reading raw dataset", unit="row"):
        if not isinstance(row, dict):
            continue
        subreddit_counts[str(row.get("subreddit", ""))] += 1

    kept = []
    rejected = Counter()
    seen_ids = set()
    seen_signatures = set()
    length_by_split = {"train": [], "val": [], "test": []}

    input_rows = 0
    for line_number, row in tqdm(read_jsonl(input_path), desc="Cleaning and splitting", unit="row"):
        if not isinstance(row, dict):
            continue
        input_rows += 1
        row_id = str(row.get("id", "") or "")
        if not row_id or row_id in seen_ids:
            rejected["duplicate_id"] += 1
            continue
        seen_ids.add(row_id)

        result = process_row(row, args.min_chars, args.min_quality_score, subreddit_counts)
        if not result.accepted:
            for reason in result.reasons:
                rejected[reason] += 1
            continue

        signature = result.row["normalized_signature"]
        if signature in seen_signatures:
            rejected["normalized_duplicate"] += 1
            continue
        seen_signatures.add(signature)

        split = split_from_id(row_id, args.train_ratio, args.val_ratio)
        result.row["split"] = split
        kept.append(result.row)
        length_by_split[split].append(result.row["char_length"])

    kept.sort(key=lambda item: (item["split"], -item["quality_score"], item["id"]))

    split_handles = {
        "train": (out_dir / "clean_train.jsonl").open("w", encoding="utf-8"),
        "val": (out_dir / "clean_val.jsonl").open("w", encoding="utf-8"),
        "test": (out_dir / "clean_test.jsonl").open("w", encoding="utf-8"),
    }
    all_handle = (out_dir / "clean_all.jsonl").open("w", encoding="utf-8")
    try:
        for row in tqdm(kept, desc="Writing cleaned splits", unit="row"):
            payload = {key: value for key, value in row.items() if key != "normalized_signature"}
            line = json.dumps(payload, ensure_ascii=False) + "\n"
            split_handles[row["split"]].write(line)
            all_handle.write(line)
    finally:
        all_handle.close()
        for handle in split_handles.values():
            handle.close()

    stats = {
        "input_rows": input_rows,
        "kept_rows": len(kept),
        "rejected": dict(rejected),
        "split_counts": {
            "train": sum(1 for row in kept if row["split"] == "train"),
            "val": sum(1 for row in kept if row["split"] == "val"),
            "test": sum(1 for row in kept if row["split"] == "test"),
        },
        "length_summary": {split: summarize_lengths(values) for split, values in length_by_split.items()},
        "top_subreddits": Counter(row["subreddit"] for row in kept).most_common(20),
    }
    write_json(out_dir / "stats.json", stats)
    print(f"Wrote cleaned dataset to {Path(out_dir).resolve()}")
    print(stats)


if __name__ == "__main__":
    main()
