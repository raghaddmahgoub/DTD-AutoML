"""
cache_manager.py
----------------
Content-addressed cache for the DTD pipeline.

Cache key = SHA-256(raw CSV bytes) + ":" + target_column
This makes the check byte-exact and filename-agnostic.

Performance: the file hash is computed ONCE per (path, mtime, size) tuple
and stored in an in-process LRU cache.  Every subsequent call within the
same server process — lookup, save_stage ×4, finalize — hits the in-memory
cache instead of re-reading the file.

Cache layout
------------
cache/
└── <first16_of_sha256>__<target_col>/
    ├── cache_meta.json          ← key, target, shape, timestamp, version
    ├── raw_analysis.json
    ├── preprocessing.json
    ├── clean_analysis.json
    ├── automl_training.json
    └── artifacts/               ← copies of every file the pipeline saved
        ├── full_preprocessed.csv
        ├── X_train.csv  ...
        └── <model>.pkl  ...
"""

import hashlib
import json
import os
import re
import shutil
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

# ── constants ─────────────────────────────────────────────────────────────────
CACHE_ROOT    = Path("cache")
META_FILENAME = "cache_meta.json"
CACHE_VERSION = "1.0"          # bump when cache schema changes

STAGE_KEYS = ["raw_analysis", "preprocessing", "clean_analysis", "automl_training"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_dir_name(target_column: str) -> str:
    return re.sub(r"[^\w\-]", "_", target_column)


def _hash_file(data_path: str) -> str:
    """SHA-256 of raw file bytes, read in 8 MB chunks."""
    h = hashlib.sha256()
    with open(data_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=64)
def _cached_file_hash(data_path: str, mtime: float, size: int) -> str:
    """
    Hash result keyed by (path, mtime, size).
    Same file re-uploaded → same mtime+size → same cache slot → no re-read.
    File actually changed → different mtime or size → cache miss → re-hashed.
    """
    return _hash_file(data_path)


def compute_file_hash(data_path: str) -> str:
    """
    Public entry point.  Returns SHA-256 hex digest.
    Uses the in-process LRU cache to avoid re-hashing the same file
    multiple times within a single server process lifetime.
    """
    stat = os.stat(data_path)
    return _cached_file_hash(data_path, stat.st_mtime, stat.st_size)


def _cache_key(file_hash: str, target_column: str) -> str:
    combined = f"{file_hash}:{target_column}"
    return hashlib.sha256(combined.encode()).hexdigest()


def _run_dir(cache_root: Path, cache_key_hex: str, target_column: str) -> Path:
    prefix   = cache_key_hex[:16]
    safe_col = _safe_dir_name(target_column)
    return cache_root / f"{prefix}__{safe_col}"


# ── public API ────────────────────────────────────────────────────────────────

class PipelineCacheManager:
    """
    Compute the file hash once, reuse it for every operation in a run.

    Internal _resolve() now accepts an optional pre-computed file_hash so
    lookup → save_stage → finalize never re-read the file.
    """

    def __init__(self, cache_root: str | Path = CACHE_ROOT):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # Per-request hash memo: (data_path, target_column) → (file_hash, ck, run_dir)
        # Populated on first lookup(); reused by save_stage() and finalize().
        self._resolved: dict[tuple, tuple] = {}

    # ── internals ─────────────────────────────────────────────────────────────

    def _resolve(self, data_path: str, target_column: str, file_hash: str | None = None):
        """
        Return (file_hash, cache_key_hex, run_dir).
        Hash is computed at most once per (path, target) key per process.
        """
        key = (data_path, target_column)
        if key not in self._resolved:
            fh  = file_hash or compute_file_hash(data_path)
            ck  = _cache_key(fh, target_column)
            rd  = _run_dir(self.cache_root, ck, target_column)
            self._resolved[key] = (fh, ck, rd)
        return self._resolved[key]

    def _meta_path(self, run_dir: Path) -> Path:
        return run_dir / META_FILENAME

    def _stage_path(self, run_dir: Path, stage: str) -> Path:
        return run_dir / f"{stage}.json"

    def _artifacts_dir(self, run_dir: Path) -> Path:
        return run_dir / "artifacts"

    # ── public methods ────────────────────────────────────────────────────────

    def lookup(self, data_path: str, target_column: str) -> tuple[bool, Optional[dict]]:
        """
        Strict content-based cache lookup.
        Hashes the file once; all subsequent calls reuse that hash.

        Returns
        -------
        (True,  cached_result_dict)   on a valid cache hit
        (False, None)                 on a miss or corrupt cache
        """
        try:
            t0 = time.monotonic()
            file_hash, ck, run_dir = self._resolve(data_path, target_column)
            t1 = time.monotonic()
            print(f"   Hash computed in {t1 - t0:.3f}s  ({file_hash[:12]}…)")

            meta_path = self._meta_path(run_dir)
            if not run_dir.exists() or not meta_path.exists():
                return False, None

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # ── strict validation ─────────────────────────────────────────────
            if meta.get("cache_key")     != ck:
                print("⚠️  Cache key mismatch — ignoring stale entry.")
                return False, None
            if meta.get("file_hash")     != file_hash:
                print("⚠️  File hash mismatch — ignoring stale entry.")
                return False, None
            if meta.get("target_column") != target_column:
                print("⚠️  Target column mismatch — ignoring stale entry.")
                return False, None
            if meta.get("cache_version") != CACHE_VERSION:
                print("⚠️  Cache schema version changed — ignoring stale entry.")
                return False, None

            # ── load each stage output ────────────────────────────────────────
            stages = {}
            for stage in STAGE_KEYS:
                sp = self._stage_path(run_dir, stage)
                if sp.exists():
                    with open(sp, "r", encoding="utf-8") as f:
                        stages[stage] = json.load(f)

            if not stages:
                print("⚠️  Cache entry exists but contains no stage data.")
                return False, None

            print(f"✅ Cache HIT  →  {run_dir}")
            print(f"   Stages cached : {list(stages.keys())}")
            print(f"   Cached on     : {meta.get('created_at', 'unknown')}")

            return True, {
                "meta":          meta,
                "stages":        stages,
                "artifacts_dir": str(self._artifacts_dir(run_dir)),
            }

        except Exception as exc:
            print(f"⚠️  Cache lookup error (treating as miss): {exc}")
            return False, None

    def save_stage(
        self,
        data_path:     str,
        target_column: str,
        stage_name:    str,
        agent_output:  dict,
    ) -> None:
        """
        Persist one stage's agent_output JSON to the cache folder.
        Reuses the hash computed during lookup() — no extra file I/O.
        """
        try:
            _, _, run_dir = self._resolve(data_path, target_column)
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(self._stage_path(run_dir, stage_name), "w", encoding="utf-8") as f:
                json.dump(agent_output, f, indent=2, default=str)
        except Exception as exc:
            print(f"⚠️  Could not save stage '{stage_name}' to cache: {exc}")

    def finalize(
        self,
        data_path:      str,
        target_column:  str,
        pipeline_state: dict,
    ) -> None:
        """
        Write cache_meta.json and copy every artifact file into artifacts/.
        Reuses the hash from lookup() — no file re-read.
        Dataset shape is read from the already-built clean_data_path split,
        not by re-reading the original raw file.
        """
        try:
            import pandas as pd

            file_hash, ck, run_dir = self._resolve(data_path, target_column)
            run_dir.mkdir(parents=True, exist_ok=True)
            artifacts_dir = self._artifacts_dir(run_dir)
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # ── dataset shape: read from the preprocessed file if available,
            #    otherwise a single fast header-only read of the raw file ──────
            shape_path = pipeline_state.get("clean_data_path") or data_path
            try:
                sample = pd.read_csv(shape_path, nrows=1)
                # count rows cheaply without loading the whole file
                with open(shape_path, "rb") as fh:
                    row_count = sum(1 for _ in fh) - 1   # subtract header
                shape = {"rows": row_count, "cols": len(sample.columns)}
            except Exception:
                shape = {"rows": None, "cols": None}

            # ── write meta ────────────────────────────────────────────────────
            meta = {
                "cache_version": CACHE_VERSION,
                "cache_key":     ck,
                "file_hash":     file_hash,
                "target_column": target_column,
                "original_path": str(data_path),
                "dataset_shape": shape,
                "created_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "task_type":     pipeline_state.get("task_type"),
            }
            with open(self._meta_path(run_dir), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            # ── copy artifact files ───────────────────────────────────────────
            artifact_keys = [
                "clean_data_path",
                "X_train_path", "X_test_path",
                "y_train_path", "y_test_path",
                "analysis_report_path",
            ]
            saved_files: dict = pipeline_state.get("saved_files") or {}
            copied = []

            for key in artifact_keys:
                src = pipeline_state.get(key)
                if src and os.path.isfile(src):
                    dst = artifacts_dir / Path(src).name
                    shutil.copy2(src, dst)
                    copied.append(str(dst))

            for _, src in saved_files.items():
                if src and os.path.isfile(str(src)):
                    dst = artifacts_dir / Path(src).name
                    shutil.copy2(str(src), dst)
                    copied.append(str(dst))

            print(f"✅ Cache written → {run_dir}  ({len(copied)} artifact(s))")

        except Exception as exc:
            print(f"⚠️  Cache finalize error (pipeline result NOT cached): {exc}")

    def invalidate(self, data_path: str, target_column: str) -> bool:
        """Delete the cache entry for this exact (file, target) pair."""
        try:
            _, _, run_dir = self._resolve(data_path, target_column)
            if run_dir.exists():
                shutil.rmtree(run_dir)
                # Also evict from in-process memo
                self._resolved.pop((data_path, target_column), None)
                print(f"🗑️  Cache invalidated: {run_dir}")
                return True
            return False
        except Exception as exc:
            print(f"⚠️  Cache invalidate error: {exc}")
            return False

    def list_entries(self) -> list[dict]:
        """Return a summary of every cached run (for debugging / UI)."""
        entries = []
        for run_dir in sorted(self.cache_root.iterdir()):
            meta_path = run_dir / META_FILENAME
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        entries.append(json.load(f))
                except Exception:
                    pass
        return entries