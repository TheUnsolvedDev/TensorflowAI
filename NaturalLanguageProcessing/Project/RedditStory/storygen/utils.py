"""Common utility helpers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _get_tf():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import tensorflow as tf

    return tf


@dataclass
class RuntimeConfig:
    cpu_threads: int = 0
    inter_op_threads: int = 0
    gpu_memory_growth: bool = True
    gpu_memory_limit_mb: int = 0
    enable_xla: bool = False
    allocator: str = ""


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config_module(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".json":
        return read_json(path)
    if path.suffix != ".py":
        raise ValueError(f"Unsupported config format: {path}. Use .py or .json.")

    module_name = f"storygen_config_{stable_hash(str(path.resolve()))[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "CONFIG"):
        config = getattr(module, "CONFIG")
    elif hasattr(module, "get_config"):
        config = module.get_config()
    else:
        raise AttributeError(f"Config file {path} must define CONFIG or get_config().")

    if not isinstance(config, dict):
        raise TypeError(f"Config loaded from {path} must be a dict.")
    return config


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_number, json.loads(line)


def write_jsonl(path: str | Path, rows) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def path_from_config(config: dict[str, Any], key: str, fallback: str | Path) -> str:
    paths = config.get("paths", {}) if isinstance(config, dict) else {}
    value = paths.get(key, fallback)
    return str(value)


def resolve_cli_path(config: dict[str, Any], value: str | Path, fallback_key: str | None = None, fallback: str | Path | None = None) -> Path:
    raw = str(value or "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        if candidate.is_absolute() or candidate.exists():
            return candidate

        data_root = Path(path_from_config(config, "data_root", "."))
        first_part = candidate.parts[0] if candidate.parts else ""
        if first_part == "output":
            return data_root / candidate

        project_root = Path(path_from_config(config, "project_root", "."))
        return project_root / candidate

    if fallback_key is not None:
        return Path(path_from_config(config, fallback_key, fallback or ""))
    if fallback is not None:
        return Path(fallback)
    raise ValueError("A path value or fallback must be provided.")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_from_id(text: str, train_ratio: float, val_ratio: float) -> str:
    value = int(stable_hash(text)[:8], 16) / float(0xFFFFFFFF)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def set_global_seed(seed: int) -> None:
    tf = _get_tf()
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def count_visible_gpus() -> int:
    tf = _get_tf()
    return len(tf.config.list_physical_devices("GPU"))


def maybe_enable_memory_growth() -> None:
    tf = _get_tf()
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def add_runtime_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--cpu-threads", type=int, default=0, help="Set TensorFlow intra/inter op CPU thread count. 0 keeps TensorFlow defaults.")
    parser.add_argument("--inter-op-threads", type=int, default=0, help="Set TensorFlow inter-op thread count. 0 mirrors --cpu-threads or keeps defaults.")
    parser.add_argument("--gpu-memory-growth", action="store_true", default=True, help="Enable GPU memory growth when GPUs are visible.")
    parser.add_argument("--no-gpu-memory-growth", action="store_false", dest="gpu_memory_growth", help="Disable GPU memory growth.")
    parser.add_argument("--gpu-memory-limit-mb", type=int, default=0, help="Optional per-GPU memory cap in MB. 0 disables capping.")
    parser.add_argument("--enable-xla", action="store_true", help="Enable TensorFlow XLA JIT.")
    parser.add_argument("--allocator", default="", help="Optional TF_GPU_ALLOCATOR override, for example cuda_malloc_async.")
    return parser


def runtime_config_from_args(args) -> RuntimeConfig:
    return RuntimeConfig(
        cpu_threads=max(int(getattr(args, "cpu_threads", 0) or 0), 0),
        inter_op_threads=max(int(getattr(args, "inter_op_threads", 0) or 0), 0),
        gpu_memory_growth=bool(getattr(args, "gpu_memory_growth", True)),
        gpu_memory_limit_mb=max(int(getattr(args, "gpu_memory_limit_mb", 0) or 0), 0),
        enable_xla=bool(getattr(args, "enable_xla", False)),
        allocator=str(getattr(args, "allocator", "") or "").strip(),
    )


def configure_runtime(runtime: RuntimeConfig) -> dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    if runtime.allocator:
        os.environ["TF_GPU_ALLOCATOR"] = runtime.allocator

    tf = _get_tf()
    if runtime.cpu_threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(runtime.cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(runtime.inter_op_threads or runtime.cpu_threads)
    elif runtime.inter_op_threads > 0:
        tf.config.threading.set_inter_op_parallelism_threads(runtime.inter_op_threads)

    if runtime.enable_xla:
        tf.config.optimizer.set_jit(True)

    visible_gpus = tf.config.list_physical_devices("GPU")
    logical_gpu_count = 0
    memory_actions = []
    for gpu in visible_gpus:
        try:
            if runtime.gpu_memory_limit_mb > 0:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=runtime.gpu_memory_limit_mb)],
                )
                memory_actions.append(f"limited:{runtime.gpu_memory_limit_mb}MB")
            elif runtime.gpu_memory_growth:
                tf.config.experimental.set_memory_growth(gpu, True)
                memory_actions.append("growth")
            else:
                memory_actions.append("default")
        except RuntimeError as exc:
            memory_actions.append(f"skipped:{exc}")

    logical_gpu_count = len(tf.config.list_logical_devices("GPU"))
    return {
        "cpu_threads": runtime.cpu_threads,
        "inter_op_threads": runtime.inter_op_threads,
        "gpu_memory_growth": runtime.gpu_memory_growth,
        "gpu_memory_limit_mb": runtime.gpu_memory_limit_mb,
        "enable_xla": runtime.enable_xla,
        "allocator": runtime.allocator,
        "visible_gpu_count": len(visible_gpus),
        "logical_gpu_count": logical_gpu_count,
        "memory_actions": memory_actions,
    }


def make_arg_parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=description)


def format_float(value: float) -> float:
    if math.isfinite(value):
        return round(float(value), 6)
    return float("nan")
