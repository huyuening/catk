from __future__ import annotations

import gzip
import json
import os

import pytest

from src.womd_labeling.annotate import (
    annotate_paths,
    output_path_for,
    parse_args,
    validate_completed_annotation,
)

from .helpers import make_scenario, write_tfrecord


def _args(input_path, output_dir, *extra):
    return parse_args(
        [
            "--input-path",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--workers",
            "1",
            *extra,
        ]
    )


def test_writes_ordered_atomic_annotation_shard_and_resumes(tmp_path):
    input_path = tmp_path / "training.tfrecord-00000-of-00001"
    write_tfrecord(
        input_path,
        [
            make_scenario("scenario-a").SerializeToString(),
            make_scenario("scenario-b").SerializeToString(),
        ],
    )
    output_dir = tmp_path / "labels"
    args = _args(input_path, output_dir)

    summary = annotate_paths(args)

    output_path = output_path_for(input_path, output_dir, "gzip")
    assert output_path.is_file()
    assert not output_path.with_name(output_path.name + ".partial").exists()
    with gzip.open(output_path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert [row["scenario_id"] for row in rows] == [
        "scenario-a",
        "scenario-b",
    ]
    assert summary["scenarios_written"] == 2
    assert summary["shards_written"] == 1
    assert summary["shards_skipped"] == 0

    original_mtime = output_path.stat().st_mtime_ns
    resumed = annotate_paths(args)

    assert resumed["scenarios_written"] == 0
    assert resumed["shards_written"] == 0
    assert resumed["shards_skipped"] == 1
    assert resumed["scenarios_skipped"] == 2
    assert output_path.stat().st_mtime_ns == original_mtime


def test_resume_validation_rejects_corrupt_gzip(tmp_path):
    input_path = tmp_path / "validation.tfrecord-00000-of-00001"
    write_tfrecord(
        input_path,
        [make_scenario("scenario-a").SerializeToString()],
    )
    output_dir = tmp_path / "labels"
    annotate_paths(_args(input_path, output_dir))
    output_path = output_path_for(input_path, output_dir, "gzip")
    output_path.write_bytes(output_path.read_bytes()[:-8])

    with pytest.raises(ValueError, match="Invalid completed annotation"):
        validate_completed_annotation(
            output_path,
            source_file=input_path.name,
            expected_records=1,
        )

    with pytest.raises(ValueError, match="cannot be resumed"):
        annotate_paths(_args(input_path, output_dir))


def test_overwrite_replaces_existing_annotation(tmp_path):
    input_path = tmp_path / "testing.tfrecord-00000-of-00001"
    write_tfrecord(
        input_path,
        [make_scenario("scenario-a").SerializeToString()],
    )
    output_dir = tmp_path / "labels"
    annotate_paths(_args(input_path, output_dir))
    output_path = output_path_for(input_path, output_dir, "gzip")
    first_mtime = output_path.stat().st_mtime_ns

    summary = annotate_paths(_args(input_path, output_dir, "--overwrite"))

    assert summary["shards_written"] == 1
    assert summary["shards_skipped"] == 0
    assert output_path.stat().st_mtime_ns >= first_mtime


def test_resume_rejects_changed_annotation_config(tmp_path):
    input_path = tmp_path / "training.tfrecord-00000-of-00001"
    write_tfrecord(
        input_path,
        [make_scenario("scenario-a").SerializeToString()],
    )
    output_dir = tmp_path / "labels"
    annotate_paths(_args(input_path, output_dir))

    with pytest.raises(ValueError, match="cannot be resumed"):
        annotate_paths(
            _args(
                input_path,
                output_dir,
                "--near-distance-m",
                "41",
            )
        )


def test_resume_rejects_changed_source_identity(tmp_path):
    input_path = tmp_path / "training.tfrecord-00000-of-00001"
    write_tfrecord(
        input_path,
        [make_scenario("scenario-a").SerializeToString()],
    )
    output_dir = tmp_path / "labels"
    annotate_paths(_args(input_path, output_dir))
    stat = input_path.stat()
    os.utime(
        input_path,
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
    )

    with pytest.raises(ValueError, match="cannot be resumed"):
        annotate_paths(_args(input_path, output_dir))
