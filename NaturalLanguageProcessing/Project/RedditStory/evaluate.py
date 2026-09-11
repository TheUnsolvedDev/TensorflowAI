#!/usr/bin/env python3
"""Evaluate a trained story language model."""

from __future__ import annotations

from pathlib import Path

from tqdm.auto import tqdm

from storygen.utils import (
    add_runtime_args,
    configure_runtime,
    ensure_dir,
    load_config_module,
    make_arg_parser,
    resolve_cli_path,
    runtime_config_from_args,
    write_json,
)


def main():
    parser = make_arg_parser("Evaluate a trained language model on held-out records.")
    parser.add_argument("--records-dir", default="")
    parser.add_argument("--tokenizer-dir", default="")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--out-dir", default="")
    add_runtime_args(parser)
    args = parser.parse_args()
    runtime_info = configure_runtime(runtime_config_from_args(args))

    from storygen.core import BPETokenizer, build_model, evaluate_model, make_language_model_dataset, model_config_from_dict

    config = load_config_module(args.config)
    records_dir = resolve_cli_path(config, args.records_dir, fallback_key="records_dir", fallback="artifacts/records_v1")
    tokenizer_dir = resolve_cli_path(config, args.tokenizer_dir, fallback_key="tokenizer_dir", fallback="artifacts/tokenizer_v1")
    model_dir = resolve_cli_path(config, args.model_dir, fallback_key="model_dir", fallback="artifacts/model_v1")
    output_path = resolve_cli_path(config, args.out_dir, fallback_key="eval_dir", fallback="eval/run_001")
    tokenizer = BPETokenizer.from_file(Path(tokenizer_dir) / "tokenizer.json")
    model_config = model_config_from_dict(config, tokenizer.vocab_size)

    dataset = make_language_model_dataset(
        file_pattern=str(Path(records_dir) / args.split / "*.tfrecord"),
        batch_size=config["train"]["eval_batch_size"],
        seq_len=config["data"]["seq_len"],
        shuffle=False,
        shuffle_buffer=1,
        seed=config["train"]["seed"],
        for_fit=True,
        cache=bool(config["train"].get("cache_dataset", False)),
        prefetch_to_device=bool(config["train"].get("prefetch_to_device", False)),
    )
    model = build_model(model_config)

    weights_path = Path(model_dir) / "best" / "model.weights.h5"
    if not weights_path.exists():
        weights_path = Path(model_dir) / "last_model.weights.h5"
    model.load_weights(str(weights_path))

    metrics = evaluate_model(
        model,
        dataset,
        max_batches=args.max_batches or None,
        progress=tqdm(desc=f"Evaluating {args.split}", unit="batch"),
    )
    out_dir = ensure_dir(output_path)
    metrics["runtime"] = runtime_info
    write_json(Path(out_dir) / "metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
