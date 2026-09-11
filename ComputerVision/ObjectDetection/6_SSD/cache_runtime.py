"""Bounded, transient metadata/target cache for this standalone detector."""
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

from config import CACHE_ACTIVE_DIR, CACHE_DIR, CACHE_ENABLED, CACHE_MAX_SAMPLES, CACHE_REBUILD, CACHE_TARGET_VERSION


def _stamp(path):
    try:
        info = os.stat(path)
    except OSError:
        return {"path": str(path), "missing": True}
    return {"path": os.path.abspath(path), "mtime_ns": info.st_mtime_ns, "size": info.st_size}




def _tree_stamp(path):
    """Compact modification summary for XML annotation trees."""
    summary = _stamp(path)
    if summary.get("missing"):
        return summary
    digest = hashlib.sha256()
    count = 0
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
            digest.update(("%d:%d" % (info.st_mtime_ns, info.st_size)).encode())
            count += 1
    summary.update(file_count=count, digest=digest.hexdigest())
    return summary


def fingerprint(detector, dataset_name, split, root, settings):
    """Stable identity for source annotations plus target-shaping settings."""
    payload = {"version": CACHE_TARGET_VERSION, "detector": detector, "dataset": dataset_name, "split": split, "root": _stamp(root), "settings": settings}
    if dataset_name == "coco":
        payload["annotations"] = _stamp(os.path.join(root, "annotations", "instances_%s.json" % split))
    else:
        payload["annotations"] = _tree_stamp(Path(root) / "Annotations" / "CLS-LOC" / split)
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _remove(path):
    if os.path.lexists(path):
        shutil.rmtree(path, ignore_errors=True)


def cleanup_cache():
    _remove(CACHE_ACTIVE_DIR)
    parent = os.path.dirname(CACHE_ACTIVE_DIR)
    if os.path.isdir(parent):
        for name in os.listdir(parent):
            if name.startswith(".active.tmp-"):
                _remove(os.path.join(parent, name))


def cached_records(detector, dataset_name, split, root, settings, records, prepare=None):
    """Return one run's cached CPU metadata/targets; never JPEG pixels."""
    if os.environ.get("DETECTOR_CACHE_ENABLED", "1") == "0":
        return records
    active, parent = CACHE_ACTIVE_DIR, os.path.dirname(CACHE_ACTIVE_DIR)
    os.makedirs(parent, exist_ok=True)
    for name in os.listdir(parent):
        if name.startswith(".active.tmp-"):
            _remove(os.path.join(parent, name))
    digest, manifest = fingerprint(detector, dataset_name, split, root, settings)
    manifest_path, payload_path = os.path.join(active, "manifest.json"), os.path.join(active, "records.pkl")
    try:
        with open(manifest_path, encoding="utf-8") as file:
            current = json.load(file)
        if os.environ.get("DETECTOR_CACHE_REBUILD", "0") != "1" and current.get("fingerprint") == digest and os.path.isfile(payload_path):
            with open(payload_path, "rb") as file:
                print("[cache] reusing active %s" % digest[:12])
                return pickle.load(file)
    except (OSError, ValueError, pickle.PickleError, EOFError):
        pass
    # The cap applies only to optional, derived target payloads.  Keep every
    # record in the generator so shuffling and epoch coverage remain complete.
    cap = max(0, int(os.environ.get("DETECTOR_CACHE_MAX_SAMPLES", CACHE_MAX_SAMPLES)))
    if prepare and cap:
        prepared = prepare(records[:cap]) + records[cap:]
    else:
        prepared = records
    temporary = tempfile.mkdtemp(prefix=".active.tmp-", dir=parent)
    try:
        with open(os.path.join(temporary, "records.pkl"), "wb") as file:
            pickle.dump(prepared, file, protocol=pickle.HIGHEST_PROTOCOL)
        manifest.update(fingerprint=digest, record_count=len(prepared), cache_max_samples=cap, cached_payload_records=min(cap, len(records)))
        with open(os.path.join(temporary, "manifest.json"), "w", encoding="utf-8") as file:
            json.dump(manifest, file, sort_keys=True)
        with open(os.path.join(temporary, "manifest.json"), encoding="utf-8") as file:
            if json.load(file).get("fingerprint") != digest:
                raise RuntimeError("cache manifest validation failed")
        previous = active + ".previous"
        _remove(previous)
        if os.path.lexists(active):
            os.replace(active, previous)
        os.replace(temporary, active)
        _remove(previous)
        print("[cache] prepared active %s (%d records)" % (digest[:12], len(prepared)))
    finally:
        _remove(temporary)
    return prepared


# Complete split-isolated cache generations shared by the detector folders.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector_cache import cached_records as _persistent_cached_records, cleanup_cache as _cleanup_persistent_cache, default_cache_workers


def cached_records(detector, dataset_name, split, root, settings, records, prepare=None):
    return _persistent_cached_records(
        detector, dataset_name, split, root, settings, records, prepare,
        cache_dir=CACHE_DIR, enabled=CACHE_ENABLED, rebuild=CACHE_REBUILD,
        max_samples=CACHE_MAX_SAMPLES, version=CACHE_TARGET_VERSION,
    )


def cleanup_cache():
    _cleanup_persistent_cache(CACHE_DIR)
