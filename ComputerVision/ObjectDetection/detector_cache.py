"""Shared, CPU-only persistent record cache used by the standalone detectors.

The payload is deliberately Python/NumPy metadata and encoded targets, never
decoded images or TensorFlow tensors.  This keeps cache preparation and the
tf.data source on the host and leaves accelerators to receive only batches.
"""
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def default_cache_workers():
    return 4


def _stamp(path):
    try:
        info = os.stat(path)
    except OSError:
        return {"path": str(path), "missing": True}
    return {"path": os.path.abspath(path), "mtime_ns": info.st_mtime_ns, "size": info.st_size}


def _tree_stamp(path):
    summary = _stamp(path)
    if summary.get("missing"):
        return summary
    digest, count = hashlib.sha256(), 0
    for directory, _, names in os.walk(path):
        for name in sorted(names):
            if not name.endswith(".xml"):
                continue
            filename = os.path.join(directory, name)
            try:
                info = os.stat(filename)
            except OSError:
                continue
            digest.update(os.path.relpath(filename, path).encode())
            digest.update(f"{info.st_mtime_ns}:{info.st_size}".encode())
            count += 1
    summary.update(file_count=count, digest=digest.hexdigest())
    return summary


def _fingerprint(detector, dataset_name, split, root, settings, version):
    payload = {"version": version, "detector": detector, "dataset": dataset_name,
               "split": split, "root": _stamp(root), "settings": settings}
    if dataset_name == "coco":
        payload["annotations"] = _stamp(os.path.join(root, "annotations", f"instances_{split}.json"))
    else:
        payload["annotations"] = _tree_stamp(Path(root) / "Annotations" / "CLS-LOC" / split)
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _remove(path):
    if os.path.lexists(path):
        shutil.rmtree(path, ignore_errors=True)


def cleanup_cache(cache_dir):
    # Complete fingerprint generations are intentionally retained for reuse.
    # Only interrupted staging directories are disposable.
    if os.path.isdir(cache_dir):
        for directory, dirs, _ in os.walk(cache_dir):
            for name in list(dirs):
                if name.startswith(".build-"):
                    _remove(os.path.join(directory, name))
                    dirs.remove(name)


def cached_records(detector, dataset_name, split, root, settings, records, prepare=None,
                   *, cache_dir, enabled=True, rebuild=False, max_samples=2**22,
                   version=2, workers=None):
    """Reuse or atomically build one complete split-specific cache generation."""
    if not enabled or os.environ.get("DETECTOR_CACHE_ENABLED", "1") == "0":
        print("[cache] disabled; decoding and target preparation remain CPU-side")
        return records
    maximum = int(os.environ.get("DETECTOR_CACHE_MAX_SAMPLES", max_samples))
    if maximum <= 0:
        raise RuntimeError("--cache-max-samples must cover the complete split; use --no-cache to disable caching")
    if maximum < len(records):
        raise RuntimeError(f"--cache-max-samples={maximum} is smaller than split {dataset_name}/{split} ({len(records)}); use --no-cache or increase capacity")
    worker_count = max(1, int(os.environ.get("DETECTOR_CACHE_WORKERS", workers or default_cache_workers())))
    digest, manifest = _fingerprint(detector, dataset_name, split, root, settings, version)
    split_dir = os.path.join(cache_dir, detector, dataset_name, split)
    generation = os.path.join(split_dir, digest)
    payload_path = os.path.join(generation, "records.pkl")
    manifest_path = os.path.join(generation, "manifest.json")
    os.makedirs(split_dir, exist_ok=True)
    if not (rebuild or os.environ.get("DETECTOR_CACHE_REBUILD", "0") == "1"):
        try:
            with open(manifest_path, encoding="utf-8") as file:
                current = json.load(file)
            if current.get("complete") and current.get("fingerprint") == digest and current.get("record_count") == len(records):
                with open(payload_path, "rb") as file:
                    result = pickle.load(file)
                print(f"[cache] warm dataset={dataset_name} split={split} records={len(result)} workers={worker_count} id={digest[:12]}")
                return result
        except (OSError, ValueError, pickle.PickleError, EOFError):
            pass
    build = tempfile.mkdtemp(prefix=f".build-{digest[:12]}-", dir=split_dir)
    try:
        if prepare:
            # All target construction stays on CPU.  One-record calls keep the
            # API compatible with every detector and allow bounded parallelism.
            def build_one(record):
                prepared = prepare([record])
                return prepared[0] if prepared else record
            if worker_count == 1:
                prepared = [build_one(record) for record in records]
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as pool:
                    prepared = list(pool.map(build_one, records))
        else:
            prepared = records
        with open(os.path.join(build, "records.pkl"), "wb") as file:
            pickle.dump(prepared, file, protocol=pickle.HIGHEST_PROTOCOL)
        manifest.update(fingerprint=digest, record_count=len(prepared), cache_max_samples=maximum,
                        workers=worker_count, complete=True, payload="records.pkl")
        with open(os.path.join(build, "manifest.json"), "w", encoding="utf-8") as file:
            json.dump(manifest, file, sort_keys=True)
        with open(os.path.join(build, "manifest.json"), encoding="utf-8") as file:
            if not json.load(file).get("complete"):
                raise RuntimeError("cache manifest validation failed")
        if os.path.isdir(generation):
            _remove(generation)
        os.replace(build, generation)
        print(f"[cache] complete dataset={dataset_name} split={split} records={len(prepared)} workers={worker_count} id={digest[:12]}")
        return prepared
    finally:
        _remove(build)
