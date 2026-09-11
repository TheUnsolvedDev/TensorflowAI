"""Transient metadata cache plus a bounded, lazy proposal cache."""
import hashlib
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

from config import CACHE_DIR, CACHE_MAX_SAMPLES, CACHE_TARGET_VERSION


def _stamp(path):
    try:
        info = os.stat(path)
    except OSError:
        return {"path": str(path), "missing": True}
    return {"path": os.path.abspath(path), "mtime_ns": info.st_mtime_ns, "size": info.st_size}


def fingerprint(detector, dataset_name, split, root, settings):
    """Fast source identity; rebuild explicitly after nested ImageNet XML edits."""
    payload = {"version": CACHE_TARGET_VERSION, "detector": detector, "dataset": dataset_name,
               "split": split, "root": _stamp(root), "settings": settings}
    if dataset_name == "coco":
        payload["annotations"] = _stamp(os.path.join(root, "annotations", "instances_%s.json" % split))
    else:
        annotation_root = Path(root) / "Annotations" / "CLS-LOC"
        payload["annotations"] = {"root": _stamp(annotation_root), "split": split}
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _remove(path):
    if os.path.lexists(path):
        shutil.rmtree(path, ignore_errors=True)


def cleanup_cache():
    # Do not evict reusable completed generations at process exit.
    return None


def default_cache_workers():
    return 4


class ProposalCache:
    """Disk-backed proposal entries.  It never stores decoded image pixels."""
    def __init__(self, active, maximum):
        self.active = active
        self.maximum = max(0, int(maximum))
        self.entries_dir = os.path.join(active, "proposals")
        self.index_path = os.path.join(active, "proposal_index.json")
        self.index = self._read_index()

    def _read_index(self):
        try:
            with open(self.index_path, encoding="utf-8") as file:
                value = json.load(file)
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    @staticmethod
    def key(sample):
        identity = "%s\0%s" % (sample.get("image_id", ""), os.path.abspath(sample["image_path"]))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def load(self, sample):
        key = self.key(sample)
        name = self.index.get(key)
        if not name:
            return None
        try:
            with open(os.path.join(self.entries_dir, name), "rb") as file:
                return pickle.load(file)
        except (OSError, pickle.PickleError, EOFError):
            return None

    def store(self, sample, payload):
        if self.maximum == 0 or len(self.index) >= self.maximum:
            return payload
        key = self.key(sample)
        if key in self.index:
            return payload
        os.makedirs(self.entries_dir, exist_ok=True)
        name = key + ".pkl"
        fd, temporary = tempfile.mkstemp(prefix=".proposal-", suffix=".tmp", dir=self.entries_dir)
        try:
            with os.fdopen(fd, "wb") as file:
                pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, os.path.join(self.entries_dir, name))
            updated = dict(self.index)
            updated[key] = name
            index_temp = self.index_path + ".tmp"
            with open(index_temp, "w", encoding="utf-8") as file:
                json.dump(updated, file, sort_keys=True, separators=(",", ":"))
            os.replace(index_temp, self.index_path)
            self.index = updated
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return payload


def cached_records(detector, dataset_name, split, root, settings, records):
    """Create/reuse the active generation without decoding images or proposals."""
    if os.environ.get("DETECTOR_CACHE_ENABLED", "1") == "0":
        return records, None
    digest, manifest = fingerprint(detector, dataset_name, split, root, settings)
    # Generations are isolated by detector, dataset and split; completed
    # fingerprints are retained for reuse instead of one mutable "active" cache.
    parent = os.path.join(CACHE_DIR, detector, dataset_name, split)
    active = os.path.join(parent, digest)
    os.makedirs(parent, exist_ok=True)
    manifest_path = os.path.join(active, "manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as file:
            current = json.load(file)
        if os.environ.get("DETECTOR_CACHE_REBUILD", "0") != "1" and current.get("fingerprint") == digest:
            print("[cache] reusing active %s (%d lazy proposals)" % (digest[:12], len(current.get("proposal_index", {}))))
            return records, ProposalCache(active, os.environ.get("DETECTOR_CACHE_MAX_SAMPLES", CACHE_MAX_SAMPLES))
    except (OSError, ValueError):
        pass
    temporary = tempfile.mkdtemp(prefix=".active.tmp-", dir=parent)
    try:
        manifest.update(fingerprint=digest, record_count=len(records), proposal_index={}, cache_max_samples=int(os.environ.get("DETECTOR_CACHE_MAX_SAMPLES", CACHE_MAX_SAMPLES)))
        with open(os.path.join(temporary, "proposal_index.json"), "w", encoding="utf-8") as file:
            json.dump({}, file)
        with open(os.path.join(temporary, "manifest.json"), "w", encoding="utf-8") as file:
            json.dump(manifest, file, sort_keys=True)
        previous = active + ".previous"
        _remove(previous)
        if os.path.lexists(active):
            os.replace(active, previous)
        os.replace(temporary, active)
        _remove(previous)
        print("[cache] prepared dataset=%s split=%s id=%s (%d metadata records)" % (dataset_name, split, digest[:12], len(records)))
    finally:
        _remove(temporary)
    return records, ProposalCache(active, manifest["cache_max_samples"])
