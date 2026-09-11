#!/usr/bin/env python3
"""Efficient batch generation CLI."""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
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
    set_global_seed,
    write_json,
)


def load_prompts(args) -> list[str]:
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts:
        with Path(args.prompts).open("r", encoding="utf-8") as handle:
            prompts.extend([line.rstrip("\n") for line in handle if line.strip()])
    if not prompts:
        raise ValueError("Provide --prompt or --prompts.")
    return prompts


def typewrite(text: str, words_per_minute: float) -> None:
    """Render text at a steady human typing pace without buffering a full story."""
    if words_per_minute <= 0:
        print(text)
        return

    characters_per_second = words_per_minute * 5 / 60
    started = time.monotonic()
    for index, character in enumerate(text, start=1):
        sys.stdout.write(character)
        sys.stdout.flush()
        target_time = started + index / characters_per_second
        time.sleep(max(0.0, target_time - time.monotonic()))
    print()


def run_interactive(model, tokenizer, args, strategy):
    """Run a persistent prompt-to-story session using one loaded model."""
    from storygen.core import generate_batch

    print("Interactive story generator. Enter a prompt; blank line or 'quit' exits.")
    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return
        if not prompt or prompt.lower() in {"quit", "exit", ":q"}:
            print("Exiting.")
            return
        # A single prompt cannot be split across replicas. Batched prompt-file
        # generation below uses MirroredStrategy when multiple GPUs exist.
        generated = generate_batch(
            model,
            tokenizer,
            [prompt],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            strategy=None,
        )[0]
        print("\nStory:")
        typewrite(generated, args.typing_wpm)


def main():
    parser = make_arg_parser("Generate story continuations from a trained model.")
    parser.add_argument("--tokenizer-dir", default="")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompts", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--interactive", action="store_true", help="Prompt repeatedly for stories in an interactive session.")
    parser.add_argument(
        "--typing-wpm",
        type=float,
        default=70.0,
        help="Interactive display speed in words per minute; use 0 to print instantly (default: 40).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    add_runtime_args(parser)
    args = parser.parse_args()
    if args.typing_wpm < 0:
        parser.error("--typing-wpm must be zero or greater.")

    runtime_info = configure_runtime(runtime_config_from_args(args))
    import tensorflow as tf
    from storygen.core import BPETokenizer, build_model, count_visible_gpus, generate_batch, model_config_from_dict

    set_global_seed(args.seed)
    config = load_config_module(args.config)
    tokenizer_dir = resolve_cli_path(config, args.tokenizer_dir, fallback_key="tokenizer_dir", fallback="artifacts/tokenizer_v1")
    model_dir = resolve_cli_path(config, args.model_dir, fallback_key="model_dir", fallback="artifacts/model_v1")
    output_path = resolve_cli_path(config, args.out_dir, fallback_key="generations_dir", fallback="generations/run_001")
    tokenizer = BPETokenizer.from_file(Path(tokenizer_dir) / "tokenizer.json")
    model_config = model_config_from_dict(config, tokenizer.vocab_size)
    strategy = tf.distribute.MirroredStrategy() if count_visible_gpus() > 1 else tf.distribute.get_strategy()
    with strategy.scope():
        model = build_model(model_config)

    weights_path = Path(model_dir) / "best" / "model.weights.h5"
    if not weights_path.exists():
        weights_path = Path(model_dir) / "last_model.weights.h5"
    model.load_weights(str(weights_path))

    if args.interactive:
        run_interactive(model, tokenizer, args, strategy)
        return

    prompts = load_prompts(args)

    groups = defaultdict(list)
    replica_count = strategy.num_replicas_in_sync
    batch_size = args.batch_size
    if replica_count > 1 and batch_size % replica_count:
        batch_size = max(replica_count, (batch_size // replica_count) * replica_count)
    for index, prompt in enumerate(prompts):
        length = len(tokenizer.encode(prompt, add_special_tokens=True))
        groups[length].append((index, prompt))

    outputs = [None] * len(prompts)
    total_batches = sum((len(group) + batch_size - 1) // batch_size for group in groups.values())
    batch_progress = tqdm(total=total_batches, desc="Generating batches", unit="batch")
    for length in sorted(groups):
        group = groups[length]
        for batch_start in range(0, len(group), batch_size):
            chunk = group[batch_start:batch_start + batch_size]
            chunk_indices = [index for index, _ in chunk]
            chunk_prompts = [prompt for _, prompt in chunk]
            generated = generate_batch(
                model,
                tokenizer,
                chunk_prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                strategy=strategy,
            )
            for target_index, text in zip(chunk_indices, generated):
                outputs[target_index] = text
            batch_progress.update(1)
    batch_progress.close()

    out_dir = ensure_dir(output_path)
    lines = []
    for prompt, output in zip(prompts, outputs):
        lines.append({"prompt": prompt, "generated_text": output})
    with (Path(out_dir) / "generations.jsonl").open("w", encoding="utf-8") as handle:
        for row in lines:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(
        Path(out_dir) / "run_config.json",
        {
            "batch_size": batch_size,
            "requested_batch_size": args.batch_size,
            "replicas": replica_count,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "runtime": runtime_info,
        },
    )
    print(f"Saved generations to {Path(out_dir).resolve()}")


if __name__ == "__main__":
    main()
