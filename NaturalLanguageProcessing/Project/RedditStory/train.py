#!/usr/bin/env python3
"""Train the TensorFlow story language model."""

from __future__ import annotations

from pathlib import Path

from storygen.utils import add_runtime_args, configure_runtime, load_config_module, make_arg_parser, resolve_cli_path, runtime_config_from_args


def main():
    parser = make_arg_parser("Train the decoder-only language model.")
    parser.add_argument("--records-dir", default="")
    parser.add_argument("--tokenizer-dir", default="")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--reset-model", action="store_true", help="Ignore existing checkpoints and start from fresh weights.")
    parser.add_argument("--generate-every", type=int, default=None, help="Generate samples every N steps; 0 disables generation.")
    parser.add_argument("--generation-prompts", default="", help="Text file containing one generation prompt per line.")
    parser.add_argument("--generation-max-new-tokens", type=int, default=None)
    parser.add_argument("--generation-temperature", type=float, default=None)
    parser.add_argument("--generation-top-k", type=int, default=None)
    parser.add_argument("--generation-top-p", type=float, default=None)
    parser.add_argument("--generation-repetition-penalty", type=float, default=None)
    parser.add_argument("--generation-no-repeat-ngram-size", type=int, default=None)
    add_runtime_args(parser)
    args = parser.parse_args()
    runtime_info = configure_runtime(runtime_config_from_args(args))

    from storygen.core import BPETokenizer, make_language_model_dataset, model_config_from_dict, train_config_from_dict, train_model

    config = load_config_module(args.config)
    if args.generate_every is not None:
        config["train"]["generate_every"] = max(args.generate_every, 0)
    if args.generation_prompts:
        config["train"]["generation_prompts_file"] = args.generation_prompts
    if args.generation_max_new_tokens is not None:
        config["train"]["generation_max_new_tokens"] = max(args.generation_max_new_tokens, 1)
    if args.generation_temperature is not None:
        config["train"]["generation_temperature"] = args.generation_temperature
    if args.generation_top_k is not None:
        config["train"]["generation_top_k"] = max(args.generation_top_k, 0)
    if args.generation_top_p is not None:
        config["train"]["generation_top_p"] = args.generation_top_p
    if args.generation_repetition_penalty is not None:
        config["train"]["generation_repetition_penalty"] = max(args.generation_repetition_penalty, 1.0)
    if args.generation_no_repeat_ngram_size is not None:
        config["train"]["generation_no_repeat_ngram_size"] = max(args.generation_no_repeat_ngram_size, 0)
    records_dir = resolve_cli_path(config, args.records_dir, fallback_key="records_dir", fallback="artifacts/records_v1")
    tokenizer_dir = resolve_cli_path(config, args.tokenizer_dir, fallback_key="tokenizer_dir", fallback="artifacts/tokenizer_v1")
    model_dir = resolve_cli_path(config, args.model_dir, fallback_key="model_dir", fallback="artifacts/model_v1")
    tokenizer = BPETokenizer.from_file(Path(tokenizer_dir) / "tokenizer.json")

    model_config = model_config_from_dict(config, tokenizer.vocab_size)
    train_config = train_config_from_dict(config)

    train_dataset = make_language_model_dataset(
        file_pattern=str(Path(records_dir) / "train" / "*.tfrecord"),
        batch_size=train_config.train_batch_size,
        seq_len=train_config.seq_len,
        shuffle=True,
        shuffle_buffer=config["data"]["shuffle_buffer"],
        seed=train_config.seed,
        for_fit=True,
        cache=train_config.cache_dataset,
        prefetch_to_device=train_config.prefetch_to_device,
    )
    # Repeat before Keras creates the distributed iterator. This prevents
    # MirroredStrategy from treating the finite TFRecord pipeline as an
    # epoch-limited iterator when steps_per_epoch is larger than one pass.
    train_dataset = train_dataset.repeat()
    val_dataset = make_language_model_dataset(
        file_pattern=str(Path(records_dir) / "val" / "*.tfrecord"),
        batch_size=train_config.eval_batch_size,
        seq_len=train_config.seq_len,
        shuffle=False,
        shuffle_buffer=1,
        seed=train_config.seed,
        for_fit=True,
        cache=train_config.cache_dataset,
        prefetch_to_device=train_config.prefetch_to_device,
    )

    metrics = train_model(train_dataset, val_dataset, model_dir, model_config, train_config, tokenizer=tokenizer, reset_model=args.reset_model)
    print({"runtime": runtime_info})
    print(metrics)


if __name__ == "__main__":
    main()
