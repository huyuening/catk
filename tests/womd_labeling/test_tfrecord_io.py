from pathlib import Path

import pytest

from src.womd_labeling.tfrecord_io import (
    count_tfrecord_records,
    iter_tfrecord,
    resolve_tfrecord_paths,
)

from .helpers import make_scenario, write_tfrecord


def test_resolves_directories_deterministically_and_deduplicates(tmp_path):
    second = tmp_path / "training.tfrecord-00001-of-00002"
    first = tmp_path / "training.tfrecord-00000-of-00002"
    ignored = tmp_path / "notes.txt"
    first.write_bytes(b"")
    second.write_bytes(b"")
    ignored.write_text("not a record", encoding="utf-8")

    resolved = resolve_tfrecord_paths([tmp_path, str(first)])

    assert resolved == [first.resolve(), second.resolve()]


def test_streams_and_counts_tfrecord_records(tmp_path):
    path = tmp_path / "training.tfrecord-00000-of-00001"
    payloads = [
        make_scenario("scenario-a").SerializeToString(),
        make_scenario("scenario-b").SerializeToString(),
    ]
    write_tfrecord(path, payloads)

    assert count_tfrecord_records(path) == 2
    assert list(iter_tfrecord(path)) == list(enumerate(payloads))


def test_rejects_truncated_tfrecord_payload(tmp_path):
    path = tmp_path / "training.tfrecord-00000-of-00001"
    write_tfrecord(path, [b"complete"])
    path.write_bytes(path.read_bytes()[:-3])

    with pytest.raises(ValueError, match="Truncated"):
        list(iter_tfrecord(path))


def test_rejects_empty_input_resolution(tmp_path):
    with pytest.raises(FileNotFoundError, match="No TFRecord"):
        resolve_tfrecord_paths([tmp_path])
