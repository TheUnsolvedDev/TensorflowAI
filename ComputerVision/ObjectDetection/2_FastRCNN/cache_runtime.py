"""Persistent, split-scoped proposal caches for the Fast R-CNN input path."""
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import cv2
import numpy as np

from config import CACHE_DIR, CACHE_ENABLED, CACHE_MAX_SAMPLES, CACHE_REBUILD, CACHE_TARGET_VERSION
from utils import generate_region_proposals


def default_cache_workers():
    """Conservative default for CPU-heavy cold-cache preparation."""
    return 4


def _worker_init(proposal_builder, capacity):
    global _WORKER_PROPOSAL_BUILDER, _WORKER_PROPOSAL_CAPACITY
    _WORKER_PROPOSAL_BUILDER = proposal_builder
    _WORKER_PROPOSAL_CAPACITY = capacity


def _build_proposal_worker(index, sample):
    """CPU-only worker: decode one image and produce its bounded proposals."""
    image = cv2.imread(sample["image_path"])
    if image is None:
        raise RuntimeError(f"Cannot read cache image: {sample['image_path']}")
    value = np.asarray(_WORKER_PROPOSAL_BUILDER(
        sample, image), dtype=np.float32)
    value = value[:_WORKER_PROPOSAL_CAPACITY]
    if not len(value):
        raise RuntimeError(f"No proposals built for {sample['image_path']}")
    return index, value


def build_proposals(records, capacity, proposal_builder, workers, commit, label):
    """Compute proposals with at most two worker batches in flight.

    Only this function's workers touch image decoding/Selective Search.  The
    caller owns all arrays and metadata, so an interrupted build remains
    resumable and cannot promote partial worker output.
    """
    indexes = [index for index in range(
        len(records)) if not commit(index, None)]
    total = len(records)
    completed = total - len(indexes)
    started = time.monotonic()
    print(
        f"[cache] building {label} entries={total} resumed={completed} workers={workers}")

    def report(force=False):
        if force or completed % 100 == 0:
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = (completed - (total - len(indexes))) / elapsed
            remaining = total - completed
            eta = remaining / rate if rate > 0 else float("inf")
            eta_text = f"{eta:.0f}s" if np.isfinite(eta) else "?"
            print(
                f"\r[cache] {completed}/{total} {rate:.1f} images/s eta={eta_text}", end="", flush=True)

    def store(index, value):
        nonlocal completed
        commit(index, value)
        completed += 1
        report(completed == total)

    if workers == 1:
        for index in indexes:
            _worker_init(proposal_builder, capacity)
            _, value = _build_proposal_worker(index, records[index])
            store(index, value)
    elif indexes:
        iterator = iter(indexes)
        pending = {}
        executor = ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                                       initargs=(proposal_builder, capacity))
        try:
            def submit_next():
                try:
                    index = next(iterator)
                except StopIteration:
                    return False
                pending[executor.submit(
                    _build_proposal_worker, index, records[index])] = index
                return True

            for _ in range(min(2 * workers, len(indexes))):
                submit_next()
            while pending:
                ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    index = pending.pop(future)
                    try:
                        _, value = future.result()
                    except Exception as exc:
                        print(
                            f"\n[cache] worker failure index={index} image={records[index]['image_path']}: {exc}", flush=True)
                        for queued in pending:
                            queued.cancel()
                        raise RuntimeError(
                            f"Proposal-cache worker failed at sample {index}") from exc
                    store(index, value)
                    submit_next()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    if total:
        print()


def build_volatile_proposals(records, capacity, proposal_builder, workers):
    """Return fixed-shape proposal arrays for the explicit --no-cache path."""
    values = np.zeros((len(records), capacity, 4), np.float32)
    counts = np.zeros(len(records), np.int32)

    def commit(index, value):
        if value is None:
            return False
        values[index, :len(value)] = value
        counts[index] = len(value)
        return False

    build_proposals(records, capacity, proposal_builder,
                    workers, commit, "no-cache fallback")
    return values, counts


def _stamp(path):
    try:
        info = os.stat(path)
    except OSError:
        return {"path": str(path), "missing": True}
    return {"path": os.path.abspath(path), "mtime_ns": info.st_mtime_ns, "size": info.st_size}


def fingerprint(detector, dataset_name, split, root, settings):
    """Return a source/settings identity which deliberately includes the split."""
    payload = {"version": CACHE_TARGET_VERSION, "detector": detector, "dataset": dataset_name,
               "split": split, "root": _stamp(root), "settings": settings}
    annotation_sources = settings.get("annotation_sources")
    if annotation_sources:
        payload["annotations"] = [_stamp(path) for path in annotation_sources]
    elif dataset_name == "coco":
        payload["annotations"] = _stamp(os.path.join(
            root, "annotations", f"instances_{split}.json"))
    else:
        payload["annotations"] = {"root": _stamp(
            Path(root) / "Annotations" / "CLS-LOC"), "split": split}
    encoded = json.dumps(payload, sort_keys=True,
                         default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _remove(path):
    if os.path.lexists(path):
        shutil.rmtree(path, ignore_errors=True)


def cleanup_cache():
    """Caches are intentionally persistent; only abandoned completed-temp directories go away."""
    if not os.path.isdir(CACHE_DIR):
        return
    for root, dirs, _ in os.walk(CACHE_DIR):
        for name in dirs[:]:
            if name.startswith(".replace-"):
                _remove(os.path.join(root, name))
                dirs.remove(name)


class ProposalCache:
    """One immutable cache generation stored as fixed-shape NumPy arrays."""

    def __init__(self, path):
        self.path = path
        self.manifest_path = os.path.join(path, "manifest.json")
        with open(self.manifest_path, encoding="utf-8") as file:
            self.manifest = json.load(file)
        self.proposals = np.load(os.path.join(
            path, "proposals.npy"), mmap_mode="r")
        self.counts = np.load(os.path.join(path, "counts.npy"), mmap_mode="r")

    @property
    def ready(self):
        return bool(self.manifest.get("complete")) and len(self.proposals) == int(self.manifest["record_count"])

    @property
    def disk_bytes(self):
        return sum(os.path.getsize(os.path.join(self.path, name)) for name in ("proposals.npy", "counts.npy", "manifest.json") if os.path.exists(os.path.join(self.path, name)))


class PersistentProposalCache:
    def __init__(
        self, detector, dataset_name, split, root, settings, records,
        *, enabled=CACHE_ENABLED, rebuild=CACHE_REBUILD,
        max_samples=CACHE_MAX_SAMPLES, workers=None,
    ):
        self.enabled = bool(enabled)
        self.rebuild = bool(rebuild)
        self.maximum = max(0, int(max_samples))
        self.records = records
        self.digest, self.source_manifest = fingerprint(
            detector, dataset_name, split, root, settings)
        # Fingerprint directories are isolated by detector/dataset/split before the digest.
        self.split_dir = os.path.join(CACHE_DIR, detector, dataset_name, split)
        self.path = os.path.join(self.split_dir, self.digest)
        self.build_path = os.path.join(self.split_dir, f".build-{self.digest}")
        self.capacity = int(
            settings["max_proposals"] + settings.get("max_jittered_proposals", 0))
        self.workers = max(1, int(default_cache_workers()
                           if workers is None else workers))

    def _load_ready(self):
        try:
            cache = ProposalCache(self.path)
            return cache if cache.ready and cache.manifest.get("fingerprint") == self.digest else None
        except (OSError, ValueError, KeyError):
            return None

    def prepare(self, proposal_builder):
        """Build all records once. An incomplete .build directory is resumed safely."""
        if not self.enabled or self.maximum == 0:
            print(
                "[cache] disabled; proposals will be generated in the slower input fallback")
            return None
        if self.maximum < len(self.records):
            raise RuntimeError(
                f"--cache-max-samples={self.maximum} is smaller than this split ({len(self.records)}); use 0 for --no-cache or a full-split capacity.")
        if self.rebuild:
            _remove(self.path)
            _remove(self.build_path)
        cache = self._load_ready()
        if cache:
            print(
                f"[cache] warm split={self.source_manifest['split']} records={len(self.records)} limit={self.maximum} id={self.digest[:12]} disk={cache.disk_bytes / 2**30:.2f}GiB")
            return cache

        os.makedirs(self.build_path, exist_ok=True)
        manifest_path = os.path.join(self.build_path, "manifest.json")
        manifest = dict(self.source_manifest, fingerprint=self.digest, record_count=len(self.records),
                        cache_max_samples=self.maximum, proposal_capacity=self.capacity,
                        complete=False, built_entries=0)
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, encoding="utf-8") as file:
                    old = json.load(file)
                if old.get("fingerprint") == self.digest and old.get("proposal_capacity") == self.capacity:
                    manifest.update(old)
                else:
                    _remove(self.build_path)
                    os.makedirs(self.build_path)
            except (OSError, ValueError):
                _remove(self.build_path)
                os.makedirs(self.build_path)
        proposal_path, count_path, done_path = (os.path.join(
            self.build_path, n) for n in ("proposals.npy", "counts.npy", "done.npy"))
        shape = (len(self.records), self.capacity, 4)
        if not os.path.exists(proposal_path):
            np.lib.format.open_memmap(
                proposal_path, mode="w+", dtype=np.float32, shape=shape)[:] = 0
            np.lib.format.open_memmap(
                count_path, mode="w+", dtype=np.int32, shape=(len(self.records),))[:] = 0
            np.lib.format.open_memmap(
                done_path, mode="w+", dtype=np.bool_, shape=(len(self.records),))[:] = False
        proposals = np.lib.format.open_memmap(proposal_path, mode="r+")
        counts = np.lib.format.open_memmap(count_path, mode="r+")
        done = np.lib.format.open_memmap(done_path, mode="r+")
        built_entries = int(done.sum())

        def commit(index, value):
            nonlocal built_entries
            if value is None:
                return bool(done[index])
            proposals[index, :len(value)] = value
            counts[index] = len(value)
            done[index] = True
            built_entries += 1
            if built_entries % 100 == 0 or built_entries == len(self.records):
                proposals.flush()
                counts.flush()
                done.flush()
                manifest.update(built_entries=built_entries)
                with open(manifest_path, "w", encoding="utf-8") as file:
                    json.dump(manifest, file, sort_keys=True)
            return False

        build_proposals(self.records, self.capacity, proposal_builder, self.workers, commit,
                        f"split={self.source_manifest['split']} id={self.digest[:12]}")
        proposals.flush()
        counts.flush()
        done.flush()
        manifest.update(complete=True, built_entries=len(
            self.records), disk_bytes=os.path.getsize(proposal_path) + os.path.getsize(count_path))
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, sort_keys=True)
        del proposals, counts, done
        os.unlink(done_path)
        os.makedirs(self.split_dir, exist_ok=True)
        replacement = self.path + ".replace"
        _remove(replacement)
        if os.path.exists(self.path):
            os.replace(self.path, replacement)
        os.replace(self.build_path, self.path)
        _remove(replacement)
        cache = ProposalCache(self.path)
        print(
            f"[cache] ready split={self.source_manifest['split']} records={len(self.records)} limit={self.maximum} disk={cache.disk_bytes / 2**30:.2f}GiB")
        return cache
