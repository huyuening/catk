"""Deterministic, TensorFlow-free WOMD TFRecord streaming helpers."""

from __future__ import annotations

import glob
from pathlib import Path
import struct
from typing import Iterable, Iterator


def resolve_tfrecord_paths(
    entries: Iterable[str | Path],
) -> list[Path]:
    """Resolve files, directories, and glob expressions in stable order."""
    resolved: list[Path] = []
    for raw_entry in entries:
        entry = Path(raw_entry).expanduser()
        if entry.is_dir():
            matches = sorted(
                candidate
                for candidate in entry.iterdir()
                if candidate.is_file() and "tfrecord" in candidate.name
            )
        else:
            matches = [
                Path(match)
                for match in sorted(glob.glob(str(entry)))
                if Path(match).is_file()
            ]
            if not matches and entry.is_file():
                matches = [entry]
        resolved.extend(matches)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        absolute = path.resolve()
        if absolute not in seen:
            seen.add(absolute)
            unique.append(absolute)
    if not unique:
        raise FileNotFoundError("No TFRecord shards matched the input paths")
    return unique


def iter_tfrecord(path: Path) -> Iterator[tuple[int, bytes]]:
    """Yield `(record_index, payload)` pairs from an uncompressed TFRecord."""
    with Path(path).open("rb") as stream:
        record_index = 0
        while True:
            length_bytes = stream.read(8)
            if not length_bytes:
                return
            if len(length_bytes) != 8:
                raise ValueError(f"Truncated TFRecord length in {path}")
            length = struct.unpack("<Q", length_bytes)[0]
            if len(stream.read(4)) != 4:
                raise ValueError(f"Truncated length CRC in {path}")
            payload = stream.read(length)
            if len(payload) != length:
                raise ValueError(
                    f"Truncated record {record_index} payload in {path}"
                )
            if len(stream.read(4)) != 4:
                raise ValueError(
                    f"Truncated record {record_index} data CRC in {path}"
                )
            yield record_index, payload
            record_index += 1


def count_tfrecord_records(path: Path) -> int:
    """Count records while validating TFRecord framing without parsing payloads."""
    count = 0
    with Path(path).open("rb") as stream:
        while True:
            length_bytes = stream.read(8)
            if not length_bytes:
                return count
            if len(length_bytes) != 8:
                raise ValueError(f"Truncated TFRecord length in {path}")
            length = struct.unpack("<Q", length_bytes)[0]
            if len(stream.read(4)) != 4:
                raise ValueError(f"Truncated length CRC in {path}")
            stream.seek(length, 1)
            if len(stream.read(4)) != 4:
                raise ValueError(f"Truncated data CRC in {path}")
            count += 1
