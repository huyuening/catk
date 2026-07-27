"""Small helpers for deterministic run and output artifact fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def stable_fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_identity(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def input_fingerprint(paths: Iterable[Path]) -> dict:
    files = [file_identity(path) for path in paths]
    return {
        "files": files,
        "fingerprint": stable_fingerprint(files),
    }


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path, *, logical_path: Path | None = None) -> dict:
    return {
        "path": str((logical_path or path).expanduser().resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def artifact_identity_record(
    path: Path,
    *,
    logical_path: Path | None = None,
) -> dict:
    stat = path.stat()
    return {
        "path": str((logical_path or path).expanduser().resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def artifact_identity_matches(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    try:
        path = Path(record["path"])
        expected_size = int(record["size_bytes"])
        expected_mtime_ns = int(record["mtime_ns"])
    except (KeyError, TypeError, ValueError):
        return False
    try:
        stat = path.stat()
        return (
            path.is_file()
            and stat.st_size == expected_size
            and stat.st_mtime_ns == expected_mtime_ns
        )
    except OSError:
        return False


def artifact_matches(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    try:
        path = Path(record["path"])
        expected_size = int(record["size_bytes"])
        expected_sha256 = str(record["sha256"])
    except (KeyError, TypeError, ValueError):
        return False
    try:
        return (
            path.is_file()
            and path.stat().st_size == expected_size
            and sha256_file(path) == expected_sha256
        )
    except OSError:
        return False
