"""Core story generation pipeline components."""

from __future__ import annotations

import collections
import json
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

from storygen.utils import count_visible_gpus, ensure_dir, maybe_enable_memory_growth, read_json, read_jsonl, set_global_seed, stable_hash, write_json

PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
UNK_TOKEN = "[UNK]"
SEP_TOKEN = "[SEP]"
NL_TOKEN = "[NL]"
END_OF_WORD = "</w>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_TOKEN, NL_TOKEN]
GPT_INITIALIZER_STDDEV = 0.02

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
MULTISPACE_RE = re.compile(r"[ \t]+")
MULTINEWLINE_RE = re.compile(r"\n{3,}")
EDIT_LINE_RE = re.compile(r"^\s*(edit|update|eta)\b[: -]?", re.IGNORECASE)
PART_TITLE_RE = re.compile(r"\bpart\s+[0-9ivx]+\b", re.IGNORECASE)
BOT_RE = re.compile(r"\b(i am a bot|this post was removed|automoderator)\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
WORD_OR_PUNCT_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SPACE_MARKER = "^"


@dataclass
class CleanResult:
    accepted: bool
    row: dict
    reasons: list[str]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = CONTROL_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = MULTISPACE_RE.sub(" ", text)
    text = MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


def strip_low_value_sections(text: str) -> tuple[str, list[str]]:
    kept_lines = []
    removed_flags: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            kept_lines.append("")
            continue
        if EDIT_LINE_RE.match(line):
            removed_flags.append("edit_line_removed")
            continue
        kept_lines.append(raw_line)
    text = "\n".join(kept_lines).strip()
    text = PART_TITLE_RE.sub("", text).strip()
    if PART_TITLE_RE.search(text):
        removed_flags.append("part_marker_seen")
    return text, removed_flags


def lexical_diversity(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / float(len(tokens))


def paragraph_count(text: str) -> int:
    return sum(1 for chunk in text.split("\n\n") if chunk.strip())


def score_story(clean_text: str, score: int, num_comments: int, upvote_ratio: float) -> float:
    tokens = TOKEN_RE.findall(clean_text.lower())
    length = len(clean_text)
    diversity = lexical_diversity(tokens)
    paragraphs = paragraph_count(clean_text)
    punctuation_hits = sum(clean_text.count(mark) for mark in ".!?")
    length_score = min(length / 4000.0, 1.0)
    diversity_score = min(max(diversity, 0.05), 0.75) / 0.75
    paragraph_score = min(paragraphs / 6.0, 1.0)
    metadata_score = min(math.log1p(max(score, 0)) / 8.0, 1.0)
    comment_score = min(math.log1p(max(num_comments, 0)) / 6.0, 1.0)
    ratio_score = min(max(upvote_ratio or 0.0, 0.0), 1.0)
    punctuation_score = min(punctuation_hits / max(len(tokens), 1) * 12.0, 1.0)
    weighted = (
        0.30 * length_score
        + 0.18 * diversity_score
        + 0.12 * paragraph_score
        + 0.16 * metadata_score
        + 0.08 * comment_score
        + 0.10 * ratio_score
        + 0.06 * punctuation_score
    )
    return round(weighted * 100.0, 4)


def make_normalized_signature(text: str) -> str:
    return stable_hash(re.sub(r"\W+", "", text.lower()))


def process_row(raw_row: dict, min_chars: int, min_quality_score: float, subreddit_counts: Counter | None = None) -> CleanResult:
    reasons: list[str] = []
    title = normalize_text(str(raw_row.get("title", "")))
    story = normalize_text(str(raw_row.get("story", "")))
    if not title or not story:
        return CleanResult(False, raw_row, ["missing_title_or_story"])
    if story in {"[removed]", "[deleted]"}:
        return CleanResult(False, raw_row, ["deleted_story"])
    if BOT_RE.search(story):
        return CleanResult(False, raw_row, ["bot_or_removed_marker"])
    story, strip_flags = strip_low_value_sections(story)
    reasons.extend(strip_flags)
    text = f"{title}\n\n{story}".strip()
    if len(text) < min_chars:
        return CleanResult(False, raw_row, ["too_short"])
    quality_score = score_story(
        text,
        int(raw_row.get("score", 0) or 0),
        int(raw_row.get("num_comments", 0) or 0),
        float(raw_row.get("upvote_ratio", 0.0) or 0.0),
    )
    if quality_score < min_quality_score:
        return CleanResult(False, raw_row, ["quality_below_threshold"])
    subreddit = str(raw_row.get("subreddit", "") or "").strip()
    row = {
        "id": str(raw_row.get("id", "")),
        "subreddit": subreddit,
        "title": title,
        "story": story,
        "clean_text": text,
        "score": int(raw_row.get("score", 0) or 0),
        "upvote_ratio": float(raw_row.get("upvote_ratio", 0.0) or 0.0),
        "num_comments": int(raw_row.get("num_comments", 0) or 0),
        "created_utc": float(raw_row.get("created_utc", 0.0) or 0.0),
        "permalink": str(raw_row.get("permalink", "")),
        "quality_score": quality_score,
        "char_length": len(text),
        "word_count": len(TOKEN_RE.findall(text)),
        "paragraph_count": paragraph_count(text),
        "normalized_signature": make_normalized_signature(text),
        "subreddit_frequency": 0 if subreddit_counts is None else int(subreddit_counts.get(subreddit, 0)),
        "flags": reasons,
    }
    return CleanResult(True, row, reasons)


def summarize_lengths(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {"count": 0}
    ordered = sorted(lengths)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": round(statistics.mean(ordered), 4),
        "median": round(statistics.median(ordered), 4),
        "p90": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
        "max": ordered[-1],
    }


def pretokenize(text: str) -> list[str]:
    pieces: list[str] = []
    for line_index, line in enumerate(text.split("\n")):
        if line_index > 0:
            pieces.append(NL_TOKEN)
        for chunk_index, chunk in enumerate([chunk for chunk in line.split(" ") if chunk]):
            for token_index, token in enumerate(WORD_OR_PUNCT_RE.findall(chunk)):
                pieces.append(SPACE_MARKER + token if chunk_index > 0 and token_index == 0 else token)
    return pieces


def _get_stats(vocab: dict[tuple[str, ...], int]) -> collections.Counter:
    pairs: collections.Counter = collections.Counter()
    for symbols, freq in vocab.items():
        for index in range(len(symbols) - 1):
            pairs[(symbols[index], symbols[index + 1])] += freq
    return pairs


def _merge_vocab(pair: tuple[str, str], vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, ...], int]:
    merged_vocab: dict[tuple[str, ...], int] = {}
    replacement = "".join(pair)
    for symbols, freq in vocab.items():
        updated: list[str] = []
        index = 0
        while index < len(symbols):
            if index < len(symbols) - 1 and (symbols[index], symbols[index + 1]) == pair:
                updated.append(replacement)
                index += 2
            else:
                updated.append(symbols[index])
                index += 1
        merged_vocab[tuple(updated)] = freq
    return merged_vocab


class BPETokenizer:
    def __init__(self, token_to_id: dict[str, int], merges: list[tuple[str, str]]):
        self.token_to_id = token_to_id
        self.id_to_token = {value: key for key, value in token_to_id.items()}
        self.merges = merges
        self.merge_ranks = {pair: index for index, pair in enumerate(merges)}
        self.pad_id = token_to_id[PAD_TOKEN]
        self.bos_id = token_to_id[BOS_TOKEN]
        self.eos_id = token_to_id[EOS_TOKEN]
        self.unk_id = token_to_id[UNK_TOKEN]
        self.sep_id = token_to_id[SEP_TOKEN]
        self.nl_id = token_to_id[NL_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def to_dict(self) -> dict:
        return {"token_to_id": self.token_to_id, "merges": [list(pair) for pair in self.merges]}

    @classmethod
    def from_file(cls, path: str | Path) -> "BPETokenizer":
        payload = read_json(path)
        return cls(payload["token_to_id"], [tuple(pair) for pair in payload["merges"]])

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    def _apply_bpe(self, token: str) -> list[str]:
        if token in SPECIAL_TOKENS:
            return [token]
        symbols = list(token) + [END_OF_WORD]
        while True:
            pairs = [(symbols[index], symbols[index + 1]) for index in range(len(symbols) - 1)]
            ranked_pairs = [(self.merge_ranks[pair], pair) for pair in pairs if pair in self.merge_ranks]
            if not ranked_pairs:
                break
            _, best_pair = min(ranked_pairs, key=lambda item: item[0])
            merged: list[str] = []
            index = 0
            while index < len(symbols):
                if index < len(symbols) - 1 and (symbols[index], symbols[index + 1]) == best_pair:
                    merged.append(symbols[index] + symbols[index + 1])
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = merged
        if symbols and symbols[-1] == END_OF_WORD:
            symbols = [UNK_TOKEN] if len(symbols) == 1 else symbols[:-2] + [symbols[-2] + END_OF_WORD]
        return [piece if piece in self.token_to_id else UNK_TOKEN for piece in symbols]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        token_ids: list[int] = [self.bos_id] if add_special_tokens else []
        for token in pretokenize(text):
            if token == NL_TOKEN:
                token_ids.append(self.nl_id)
                continue
            token_ids.extend(self.token_to_id.get(piece, self.unk_id) for piece in self._apply_bpe(token))
        if add_special_tokens:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        output: list[str] = []
        current = ""
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), UNK_TOKEN)
            if token in {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN} and skip_special_tokens:
                continue
            if token == NL_TOKEN:
                if current:
                    output.append(_pretoken_to_text(current))
                    current = ""
                output.append("\n")
                continue
            if token == UNK_TOKEN:
                if current:
                    output.append(_pretoken_to_text(current))
                    current = ""
                output.append("<?>")
                continue
            current += token
            if current.endswith(END_OF_WORD):
                output.append(_pretoken_to_text(current[:-len(END_OF_WORD)]))
                current = ""
        if current:
            output.append(_pretoken_to_text(current.replace(END_OF_WORD, "")))
        return re.sub(r"\n{3,}", "\n\n", "".join(output)).strip()


def _pretoken_to_text(token: str) -> str:
    if not token:
        return ""
    return " " + token[1:] if token.startswith(SPACE_MARKER) else token


def train_bpe_tokenizer(texts, vocab_size: int, min_frequency: int = 2) -> BPETokenizer:
    vocab_counter: collections.Counter = collections.Counter()
    for text in texts:
        for token in pretokenize(text):
            if token != NL_TOKEN:
                vocab_counter[token] += 1
    bpe_vocab = {tuple(list(token) + [END_OF_WORD]): freq for token, freq in vocab_counter.items() if freq >= min_frequency}
    merges: list[tuple[str, str]] = []
    learned_tokens = set(SPECIAL_TOKENS)
    for token in bpe_vocab:
        learned_tokens.update(token)
    target_merges = max(vocab_size - len(learned_tokens), 0)
    merge_progress = tqdm(total=target_merges, desc="Learning BPE merges", unit="merge")
    try:
        for _ in range(target_merges):
            stats = _get_stats(bpe_vocab)
            if not stats:
                break
            best_pair, best_freq = stats.most_common(1)[0]
            if best_freq < min_frequency:
                break
            bpe_vocab = _merge_vocab(best_pair, bpe_vocab)
            merges.append(best_pair)
            merge_progress.update(1)
            merge_progress.set_postfix(freq=best_freq, vocab=len(learned_tokens) + len(merges))
    finally:
        merge_progress.close()
    token_to_id: dict[str, int] = {}
    for token in SPECIAL_TOKENS:
        token_to_id[token] = len(token_to_id)
    learned = set()
    for symbols in bpe_vocab:
        learned.update(symbols)
    for pair in merges:
        learned.add("".join(pair))
    for symbol in list(learned):
        if symbol != END_OF_WORD and not symbol.endswith(END_OF_WORD):
            learned.add(symbol + END_OF_WORD)
    for symbol in sorted(learned):
        if symbol not in token_to_id:
            token_to_id[symbol] = len(token_to_id)
    return BPETokenizer(token_to_id, merges)


def _int_feature(values: list[int]) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def write_split_tfrecords(input_path: str | Path, output_dir: str | Path, tokenizer: BPETokenizer, seq_len: int, split_name: str, shard_size: int = 2048) -> dict:
    output_dir = ensure_dir(output_dir)
    window_len = seq_len + 1
    shard_index = 0
    shard_count = 0
    sample_count = 0
    token_count = 0
    packed_windows = 0
    buffer: list[int] = []
    source_lengths: list[int] = []
    writer = None

    def open_writer(index: int):
        return tf.io.TFRecordWriter(str(output_dir / f"{split_name}-{index:05d}.tfrecord"))

    def flush_window(window: list[int], valid_length: int):
        nonlocal shard_count, shard_index, sample_count, packed_windows, writer, token_count
        if writer is None:
            writer = open_writer(shard_index)
            shard_count += 1
        writer.write(
            tf.train.Example(
                features=tf.train.Features(feature={"token_ids": _int_feature(window), "length": _int_feature([valid_length])})
            ).SerializeToString()
        )
        sample_count += 1
        packed_windows += 1
        token_count += valid_length
        if sample_count % shard_size == 0:
            writer.close()
            writer = None
            shard_index += 1

    progress = tqdm(desc=f"Packing {split_name} TFRecords", unit="row")
    try:
        for _, row in read_jsonl(input_path):
            token_ids = tokenizer.encode(row["clean_text"], add_special_tokens=True)
            source_lengths.append(len(token_ids))
            buffer.extend(token_ids)
            while len(buffer) >= window_len:
                flush_window(buffer[:window_len], window_len)
                buffer = buffer[window_len:]
            progress.update(1)
            progress.set_postfix(windows=packed_windows, shards=shard_count)
    finally:
        progress.close()
    if buffer:
        valid_length = len(buffer)
        flush_window(buffer + [tokenizer.pad_id] * (window_len - valid_length), valid_length)
    if writer is not None:
        writer.close()
    stats = {
        "split": split_name,
        "seq_len": seq_len,
        "window_len": window_len,
        "packed_windows": packed_windows,
        "source_texts": len(source_lengths),
        "source_token_mean": float(np.mean(source_lengths)) if source_lengths else 0.0,
        "source_token_p95": int(np.percentile(source_lengths, 95)) if source_lengths else 0,
        "written_token_count": token_count,
        "shards": shard_count,
    }
    write_json(output_dir / f"{split_name}_stats.json", stats)
    return stats


def make_language_model_dataset(
    file_pattern: str,
    batch_size: int,
    seq_len: int,
    shuffle: bool,
    shuffle_buffer: int,
    seed: int,
    for_fit: bool = False,
    cache: bool = False,
    prefetch_to_device: bool = False,
) -> tf.data.Dataset:
    window_len = seq_len + 1
    feature_spec = {
        "token_ids": tf.io.FixedLenFeature([window_len], tf.int64),
        "length": tf.io.FixedLenFeature([1], tf.int64),
    }

    def parse_record(example):
        parsed = tf.io.parse_single_example(example, feature_spec)
        token_ids = tf.cast(parsed["token_ids"], tf.int32)
        labels = token_ids[1:]
        inputs = token_ids[:-1]
        loss_mask = tf.cast(tf.not_equal(labels, 0), tf.float32)
        if for_fit:
            return inputs, labels, loss_mask
        return {"input_ids": inputs, "labels": labels, "loss_mask": loss_mask}

    files = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle, seed=seed)
    dataset = files.interleave(
        lambda file_path: tf.data.TFRecordDataset(file_path, num_parallel_reads=tf.data.AUTOTUNE),
        cycle_length=tf.data.AUTOTUNE,
        block_length=16,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle,
    )
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(parse_record, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=shuffle)
    if cache:
        dataset = dataset.cache()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    options.experimental_deterministic = not shuffle
    dataset = dataset.with_options(options)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if prefetch_to_device and tf.config.list_logical_devices("GPU"):
        dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=1))
    return dataset


@dataclass
class ModelConfig:
    vocab_size: int
    max_position_embeddings: int
    hidden_size: int
    num_layers: int
    num_heads: int
    ffn_hidden_size: int
    dropout_rate: float
    layer_norm_epsilon: float = 1e-5


class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float, **kwargs):
        super().__init__(**kwargs)
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        initializer = tf.keras.initializers.TruncatedNormal(stddev=GPT_INITIALIZER_STDDEV)
        self.q_proj = tf.keras.layers.Dense(hidden_size, use_bias=False, kernel_initializer=initializer)
        self.k_proj = tf.keras.layers.Dense(hidden_size, use_bias=False, kernel_initializer=initializer)
        self.v_proj = tf.keras.layers.Dense(hidden_size, use_bias=False, kernel_initializer=initializer)
        self.out_proj = tf.keras.layers.Dense(hidden_size, use_bias=False, kernel_initializer=initializer)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def _split_heads(self, tensor: tf.Tensor) -> tf.Tensor:
        batch = tf.shape(tensor)[0]
        length = tf.shape(tensor)[1]
        return tf.transpose(tf.reshape(tensor, [batch, length, self.num_heads, self.head_dim]), [0, 2, 1, 3])

    def _merge_heads(self, tensor: tf.Tensor) -> tf.Tensor:
        tensor = tf.transpose(tensor, [0, 2, 1, 3])
        return tf.reshape(tensor, [tf.shape(tensor)[0], tf.shape(tensor)[1], self.hidden_size])

    def call(self, x: tf.Tensor, training: bool | None = None, cache: tuple[tf.Tensor, tf.Tensor] | None = None, use_cache: bool = False):
        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))
        if cache is not None:
            key = tf.concat([cache[0], key], axis=2)
            value = tf.concat([cache[1], value], axis=2)
        scores = tf.matmul(query, key, transpose_b=True) / math.sqrt(float(self.head_dim))
        if cache is None:
            query_length = tf.shape(query)[2]
            key_length = tf.shape(key)[2]
            causal_mask = tf.reshape(tf.linalg.band_part(tf.ones((query_length, key_length), dtype=tf.bool), -1, 0), [1, 1, query_length, key_length])
            scores = tf.where(causal_mask, scores, tf.constant(-1e9, dtype=scores.dtype))
        attention = self.dropout(tf.nn.softmax(tf.cast(scores, tf.float32), axis=-1), training=training)
        output = self.out_proj(self._merge_heads(tf.matmul(attention, value)))
        return output, (key, value) if use_cache else None


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, hidden_size: int, ffn_hidden_size: int, dropout_rate: float, **kwargs):
        super().__init__(**kwargs)
        initializer = tf.keras.initializers.TruncatedNormal(stddev=GPT_INITIALIZER_STDDEV)
        self.dense_in = tf.keras.layers.Dense(ffn_hidden_size, activation="gelu", kernel_initializer=initializer)
        self.dense_out = tf.keras.layers.Dense(hidden_size, kernel_initializer=initializer)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        return self.dropout(self.dense_out(self.dense_in(x)), training=training)


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config.hidden_size, config.num_heads, config.dropout_rate)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon)
        self.ffn = FeedForward(config.hidden_size, config.ffn_hidden_size, config.dropout_rate)

    def call(self, x: tf.Tensor, training: bool | None = None, cache: tuple[tf.Tensor, tf.Tensor] | None = None, use_cache: bool = False):
        attn_out, new_cache = self.attn(self.norm1(x), training=training, cache=cache, use_cache=use_cache)
        x = x + attn_out
        return x + self.ffn(self.norm2(x), training=training), new_cache


@dataclass
class TrainConfig:
    seed: int
    seq_len: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    warmup_steps: int
    train_steps: int
    eval_every: int
    save_every: int
    log_every: int
    weight_decay: float
    grad_clip_norm: float
    mixed_precision: bool = False
    optimizer_epsilon: float = 1e-7
    fail_on_nonfinite: bool = True
    cache_dataset: bool = False
    prefetch_to_device: bool = False
    generate_every: int = 0
    generation_prompts_file: str = ""
    generation_max_new_tokens: int = 128
    generation_batch_size: int = 4
    generation_temperature: float = 0.8
    generation_top_k: int = 50
    generation_top_p: float = 0.9
    generation_repetition_penalty: float = 1.15
    generation_no_repeat_ngram_size: int = 3


def model_config_from_dict(config: dict, vocab_size: int) -> ModelConfig:
    values = config["model"]
    return ModelConfig(
        vocab_size=vocab_size,
        max_position_embeddings=values["max_position_embeddings"],
        hidden_size=values["hidden_size"],
        num_layers=values["num_layers"],
        num_heads=values["num_heads"],
        ffn_hidden_size=values["ffn_hidden_size"],
        dropout_rate=values["dropout_rate"],
        layer_norm_epsilon=values.get("layer_norm_epsilon", 1e-5),
    )


def train_config_from_dict(config: dict) -> TrainConfig:
    data, values = config["data"], config["train"]
    return TrainConfig(
        seed=values["seed"], seq_len=data["seq_len"],
        train_batch_size=values["train_batch_size"], eval_batch_size=values["eval_batch_size"],
        learning_rate=values["learning_rate"], warmup_steps=values["warmup_steps"],
        train_steps=values["train_steps"], eval_every=values["eval_every"],
        save_every=values["save_every"], log_every=values["log_every"],
        weight_decay=values["weight_decay"], grad_clip_norm=values["grad_clip_norm"],
        mixed_precision=False,
        optimizer_epsilon=float(values.get("optimizer_epsilon", 1e-7)),
        fail_on_nonfinite=bool(values.get("fail_on_nonfinite", True)),
        cache_dataset=bool(values.get("cache_dataset", False)),
        prefetch_to_device=bool(values.get("prefetch_to_device", False)),
        generate_every=max(int(values.get("generate_every", 0)), 0),
        generation_prompts_file=str(values.get("generation_prompts_file", "")),
        generation_max_new_tokens=max(int(values.get("generation_max_new_tokens", 128)), 1),
        generation_batch_size=max(int(values.get("generation_batch_size", 4)), 1),
        generation_temperature=float(values.get("generation_temperature", 0.8)),
        generation_top_k=max(int(values.get("generation_top_k", 50)), 0),
        generation_top_p=float(values.get("generation_top_p", 0.9)),
        generation_repetition_penalty=max(float(values.get("generation_repetition_penalty", 1.15)), 1.0),
        generation_no_repeat_ngram_size=max(int(values.get("generation_no_repeat_ngram_size", 3)), 0),
    )


class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(max(self.warmup_steps, 1), tf.float32)
        total_steps = tf.cast(max(self.total_steps, 1), tf.float32)
        warmup = self.learning_rate * (step / warmup_steps)
        progress = tf.clip_by_value((step - warmup_steps) / tf.maximum(total_steps - warmup_steps, 1.0), 0.0, 1.0)
        return tf.where(step < warmup_steps, warmup, self.learning_rate * 0.5 * (1.0 + tf.cos(math.pi * progress)))


def maybe_enable_mixed_precision(enabled: bool) -> None:
    if enabled:
        raise ValueError("Mixed precision is disabled for this training path; set mixed_precision=False.")
    tf.keras.mixed_precision.set_global_policy("float32")


def forward_story_model(
    layers,
    input_ids: tf.Tensor,
    training: bool | None = None,
    caches: list[tuple[tf.Tensor, tf.Tensor]] | None = None,
    use_cache: bool = False,
):
    cache_length = 0 if not caches else tf.shape(caches[0][0])[2]
    positions = tf.tile(tf.expand_dims(tf.range(cache_length, cache_length + tf.shape(input_ids)[1]), axis=0), [tf.shape(input_ids)[0], 1])
    x = layers.dropout(layers.token_embedding(input_ids) + layers.position_embedding(positions), training=training)
    new_caches = []
    for index, block in enumerate(layers.blocks):
        x, new_cache = block(x, training=training, cache=None if caches is None else caches[index], use_cache=use_cache)
        if use_cache:
            new_caches.append(new_cache)
    x = layers.final_norm(x)
    tied_embeddings = tf.cast(layers.token_embedding.embeddings, tf.float32)
    logits = tf.einsum("bsh,vh->bsv", tf.cast(x, tf.float32), tied_embeddings)
    return logits, new_caches if use_cache else None


class StoryBackbone(tf.keras.layers.Layer):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(name="story_backbone", **kwargs)
        initializer = tf.keras.initializers.TruncatedNormal(stddev=GPT_INITIALIZER_STDDEV)
        self.token_embedding = tf.keras.layers.Embedding(config.vocab_size, config.hidden_size, embeddings_initializer=initializer, name="token_embedding")
        self.position_embedding = tf.keras.layers.Embedding(config.max_position_embeddings, config.hidden_size, embeddings_initializer=initializer, name="position_embedding")
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
        self.blocks = [DecoderBlock(config, name=f"decoder_block_{index}") for index in range(config.num_layers)]
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name="final_norm")

    def call(
        self,
        input_ids: tf.Tensor,
        training: bool | None = None,
        caches: list[tuple[tf.Tensor, tf.Tensor]] | None = None,
        use_cache: bool = False,
    ):
        logits, new_caches = forward_story_model(self, input_ids, training=training, caches=caches, use_cache=use_cache)
        return (logits, new_caches) if use_cache else logits


def build_model(model_config: ModelConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    backbone = StoryBackbone(model_config)
    logits = backbone(inputs, training=None, caches=None, use_cache=False)
    model = tf.keras.Model(inputs=inputs, outputs=logits, name="story_language_model")
    model.story_backbone = backbone
    model.story_layers = backbone
    model.model_config = model_config
    model.forward_with_cache = lambda input_ids, training=False, caches=None, use_cache=False: model.story_backbone(
        input_ids,
        training=training,
        caches=caches,
        use_cache=use_cache,
    )
    return model


def forward_with_cache(
    model: tf.keras.Model,
    input_ids: tf.Tensor,
    training: bool = False,
    caches: list[tuple[tf.Tensor, tf.Tensor]] | None = None,
    use_cache: bool = False,
):
    return model.story_backbone(input_ids, training=training, caches=caches, use_cache=use_cache)


def sample_next_token(
    logits: tf.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    sequences: list[list[int]],
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> tf.Tensor:
    """Sample one token per row with bounded, repetition-aware decoding."""
    logits_np = np.asarray(logits, dtype=np.float64).copy()
    if repetition_penalty != 1.0:
        for row_index, sequence in enumerate(sequences):
            for token_id in set(sequence):
                if logits_np[row_index, token_id] < 0:
                    logits_np[row_index, token_id] *= repetition_penalty
                else:
                    logits_np[row_index, token_id] /= repetition_penalty

    if no_repeat_ngram_size >= 2:
        ngram_size = no_repeat_ngram_size
        for row_index, sequence in enumerate(sequences):
            if len(sequence) < ngram_size:
                continue
            prefix = tuple(sequence[-(ngram_size - 1):])
            blocked = set()
            for index in range(len(sequence) - ngram_size + 1):
                ngram = tuple(sequence[index:index + ngram_size])
                if ngram[:-1] == prefix:
                    blocked.add(ngram[-1])
            for token_id in blocked:
                logits_np[row_index, token_id] = -np.inf

    if temperature <= 0:
        return tf.convert_to_tensor(np.argmax(logits_np, axis=-1), dtype=tf.int32)

    logits_np /= temperature
    sampled_ids = []
    vocab_size = logits_np.shape[-1]
    for row in logits_np:
        if top_k > 0:
            k = min(top_k, vocab_size)
            kth = np.partition(row, -k)[-k]
            row[row < kth] = -np.inf
        if 0.0 < top_p < 1.0:
            order = np.argsort(row)[::-1]
            sorted_logits = row[order]
            sorted_probs = np.exp(sorted_logits - np.max(sorted_logits))
            sorted_probs /= np.sum(sorted_probs)
            cumulative = np.cumsum(sorted_probs)
            remove = cumulative > top_p
            if np.any(remove):
                remove[1:] = remove[:-1]
                remove[0] = False
                row[order[remove]] = -np.inf
        probs = np.exp(row - np.max(row))
        probs /= np.sum(probs)
        sampled_ids.append(int(np.random.choice(vocab_size, p=probs)))
    return tf.convert_to_tensor(sampled_ids, dtype=tf.int32)


def generate_batch(
    model: tf.keras.Model,
    tokenizer: BPETokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    strategy: tf.distribute.Strategy | None = None,
) -> list[str]:
    """Generate equal-length prompt batches using the shared decoder path."""
    encoded_prompts = [tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts]
    if len({len(item) for item in encoded_prompts}) != 1:
        raise ValueError("generate_batch requires prompts with equal token lengths")
    inputs = tf.convert_to_tensor(np.asarray(encoded_prompts, dtype=np.int32))

    def distributed_forward(input_tensor, cache_values=None):
        if strategy is None or strategy.num_replicas_in_sync == 1:
            return model.forward_with_cache(input_tensor, training=False, caches=cache_values, use_cache=True)
        replica_count = strategy.num_replicas_in_sync
        if input_tensor.shape[0] is not None and input_tensor.shape[0] % replica_count:
            raise ValueError("Distributed generation batch size must be divisible by the replica count")
        parts = tf.split(input_tensor, replica_count, axis=0)
        distributed_inputs = strategy.experimental_distribute_values_from_function(
            lambda context: parts[context.replica_id_in_sync_group]
        )

        def replica_forward(replica_input, replica_cache=None):
            return model.forward_with_cache(replica_input, training=False, caches=replica_cache, use_cache=True)

        per_replica_logits, per_replica_caches = strategy.run(
            replica_forward,
            args=(distributed_inputs, cache_values),
        )
        return strategy.gather(per_replica_logits, axis=0), per_replica_caches

    logits, caches = distributed_forward(inputs)
    sequences = [list(prompt_ids) for prompt_ids in encoded_prompts]
    finished = [False] * len(prompts)
    next_logits = logits[:, -1, :]

    for _ in range(max_new_tokens):
        next_ids = sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sequences=sequences,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        ).numpy().tolist()
        active_ids = list(next_ids)
        for index, token_id in enumerate(next_ids):
            if finished[index]:
                active_ids[index] = tokenizer.pad_id
                continue
            sequences[index].append(int(token_id))
            if token_id == tokenizer.eos_id:
                finished[index] = True
        if all(finished):
            break
        logits, caches = distributed_forward(
            tf.convert_to_tensor(np.asarray(active_ids, dtype=np.int32)[:, None]), caches
        )
        next_logits = logits[:, -1, :]
    return [tokenizer.decode(sequence) for sequence in sequences]


def masked_language_model_loss(labels: tf.Tensor, logits: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return tf.reduce_sum(tf.cast(loss, tf.float32) * tf.cast(mask, tf.float32)) / tf.maximum(tf.reduce_sum(mask), 1.0)


class FiniteTrainingGuard(tf.keras.callbacks.Callback):
    """Stop before a non-finite update can silently poison a long run."""

    def __init__(self, fail_on_nonfinite: bool = True):
        super().__init__()
        self.fail_on_nonfinite = fail_on_nonfinite
        self.batch_count = 0

    def on_train_batch_end(self, batch, logs=None):
        self.batch_count += 1
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None and not np.isfinite(float(loss)):
            self.model.stop_training = True
            raise FloatingPointError(f"Non-finite training loss detected at batch {batch}: {loss}")
        # Checking the scalar loss every batch is cheap. Inspecting every
        # parameter periodically catches silent optimizer corruption without
        # adding a full host/device synchronization to every step.
        if self.fail_on_nonfinite and self.batch_count % 100 == 0:
            for variable in self.model.trainable_variables:
                if not bool(tf.reduce_all(tf.math.is_finite(variable)).numpy()):
                    self.model.stop_training = True
                    raise FloatingPointError(f"Non-finite weights detected at batch {batch}: {variable.name}")


def masked_token_accuracy(labels: tf.Tensor, logits: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    matches = tf.cast(tf.equal(tf.cast(tf.argmax(logits, axis=-1), labels.dtype), labels), tf.float32) * tf.cast(mask, tf.float32)
    return tf.reduce_sum(matches) / tf.maximum(tf.reduce_sum(mask), 1.0)


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    max_batches: int | None = None,
    progress=None,
) -> dict[str, float]:
    if max_batches is not None:
        dataset = dataset.take(max_batches)
    verbose = 0
    results = model.evaluate(dataset, verbose=verbose, return_dict=True)
    loss = float(results.get("loss", 0.0))
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "token_accuracy": float(results.get("token_accuracy", 0.0)),
    }


class TrainingArtifactsCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        model_dir: Path,
        val_dataset: tf.data.Dataset,
        train_config: TrainConfig,
        checkpoint: tf.train.Checkpoint,
        manager: tf.train.CheckpointManager,
        tokenizer: BPETokenizer | None = None,
    ):
        super().__init__()
        self.model_dir = model_dir
        self.val_dataset = val_dataset
        self.train_config = train_config
        self.checkpoint = checkpoint
        self.manager = manager
        self.tokenizer = tokenizer
        self.logs_path = model_dir / "training_log.jsonl"
        self.best_path = ensure_dir(model_dir / "best")
        self.best_perplexity = float("inf")
        self.start_time = time.time()
        self.global_step = int(checkpoint.step.numpy())
        self.generation_dir = ensure_dir(model_dir / "generations")
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> list[str]:
        if self.train_config.generate_every <= 0 or self.tokenizer is None:
            return []
        path = Path(self.train_config.generation_prompts_file).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            print(f"Generation disabled: prompt file not found: {path}")
            return []
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _generate_samples(self, step: int) -> None:
        if not self.prompts or step % self.train_config.generate_every != 0:
            return
        started = time.time()
        outputs = []
        batch_size = self.train_config.generation_batch_size
        indexed_prompts = list(enumerate(self.prompts))
        grouped = collections.defaultdict(list)
        for index, prompt in indexed_prompts:
            length = len(self.tokenizer.encode(prompt, add_special_tokens=True))
            grouped[length].append((index, prompt))
        generated_by_index = {}
        for group in grouped.values():
            for start in range(0, len(group), batch_size):
                chunk = group[start:start + batch_size]
                chunk_indices = [index for index, _ in chunk]
                chunk_prompts = [prompt for _, prompt in chunk]
                generated = generate_batch(
                    self.model,
                    self.tokenizer,
                    chunk_prompts,
                    max_new_tokens=self.train_config.generation_max_new_tokens,
                    temperature=self.train_config.generation_temperature,
                    top_k=self.train_config.generation_top_k,
                    top_p=self.train_config.generation_top_p,
                    repetition_penalty=self.train_config.generation_repetition_penalty,
                    no_repeat_ngram_size=self.train_config.generation_no_repeat_ngram_size,
                )
                generated_by_index.update(zip(chunk_indices, generated))
        outputs.extend(
            {"prompt": prompt, "generated_text": generated_by_index[index]}
            for index, prompt in indexed_prompts
        )
        output_path = self.generation_dir / f"step_{step:08d}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for row in outputs:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.append_log({"step": step, "generation_seconds": round(time.time() - started, 4), "generation_file": str(output_path)})
        print({"step": step, "generated": len(outputs), "generation_seconds": round(time.time() - started, 4)})

    def current_learning_rate(self) -> float:
        value = self.model.optimizer.learning_rate
        return float(value(self.model.optimizer.iterations).numpy()) if callable(value) else float(tf.convert_to_tensor(value).numpy())

    def append_log(self, payload: dict[str, float]) -> None:
        with self.logs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.global_step += 1
        self.checkpoint.step.assign(self.global_step)

        if self.global_step % self.train_config.log_every == 0 or self.global_step == 1:
            record = {
                "step": self.global_step,
                "train_loss": float(logs.get("loss", 0.0)),
                "train_token_accuracy": float(logs.get("token_accuracy", 0.0)),
                "tokens_per_second": (self.global_step * self.train_config.train_batch_size * self.train_config.seq_len) / max(time.time() - self.start_time, 1e-6),
                "learning_rate": self.current_learning_rate(),
            }
            self.append_log(record)

        self._generate_samples(self.global_step)

        if self.global_step % self.train_config.eval_every == 0 or self.global_step == self.train_config.train_steps:
            metrics = self.model.evaluate(self.val_dataset, verbose=0, return_dict=True)
            record = {
                "step": self.global_step,
                "loss": float(metrics.get("loss", 0.0)),
                "perplexity": math.exp(min(float(metrics.get("loss", 0.0)), 20.0)),
                "token_accuracy": float(metrics.get("token_accuracy", 0.0)),
            }
            self.append_log(record)
            print(record)
            if record["perplexity"] < self.best_perplexity:
                self.best_perplexity = record["perplexity"]
                self.model.save_weights(str(self.best_path / "model.weights.h5"))
                write_json(self.best_path / "metrics.json", record)

        if self.global_step % self.train_config.save_every == 0 or self.global_step == self.train_config.train_steps:
            self.manager.save(checkpoint_number=self.global_step)


def train_model(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    model_dir: str | Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    tokenizer: BPETokenizer | None = None,
    reset_model: bool = False,
) -> dict[str, float]:
    from storygen.plot_utils import plot_model_graph, plot_training_log, write_history_json, write_model_summary

    model_dir = ensure_dir(model_dir)
    set_global_seed(train_config.seed)
    maybe_enable_memory_growth()
    maybe_enable_mixed_precision(train_config.mixed_precision)
    strategy = tf.distribute.MirroredStrategy() if count_visible_gpus() > 1 else tf.distribute.get_strategy()
    with strategy.scope():
        model = build_model(model_config)
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=WarmupCosineSchedule(train_config.learning_rate, train_config.warmup_steps, train_config.train_steps),
            weight_decay=train_config.weight_decay,
            global_clipnorm=train_config.grad_clip_norm,
            epsilon=train_config.optimizer_epsilon,
        )
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="token_accuracy")],
        )
        checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(checkpoint, str(model_dir / "checkpoints"), max_to_keep=3)
        optimizer.build(model.trainable_variables)
        if manager.latest_checkpoint and not reset_model:
            checkpoint.restore(manager.latest_checkpoint).expect_partial()

    write_model_summary(model, model_dir / "model_summary.txt")
    plot_model_graph(model, model_dir / "model_graph.png")

    start_step = int(checkpoint.step.numpy())
    remaining_steps = max(train_config.train_steps - start_step, 0)
    callbacks = [
        FiniteTrainingGuard(train_config.fail_on_nonfinite),
        TrainingArtifactsCallback(model_dir, val_dataset, train_config, checkpoint, manager, tokenizer=tokenizer),
    ]
    history = None
    if remaining_steps > 0:
        history = model.fit(
            train_dataset,
            epochs=1,
            steps_per_epoch=remaining_steps,
            callbacks=callbacks,
            verbose=1,
        )

    final_metrics = evaluate_model(model, val_dataset)
    model.save_weights(str(model_dir / "last_model.weights.h5"))
    write_json(model_dir / "final_metrics.json", final_metrics)
    if history is not None:
        write_history_json(history.history, model_dir / "fit_history.json")
    plot_training_log(model_dir / "training_log.jsonl", model_dir / "training_curves.png")
    return final_metrics
