from __future__ import annotations

import csv
import gzip
import json

import pytest

from src.womd_labeling.statistics import parse_args, run_statistics

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
            "--frame-number",
            "11",
            *extra,
        ]
    )


def test_writes_consistent_road_size_and_action_statistics(tmp_path):
    input_path = tmp_path / "training.tfrecord-00000-of-00001"
    write_tfrecord(
        input_path,
        [make_scenario("scenario-a", frame_count=11).SerializeToString()],
    )
    output_dir = tmp_path / "statistics"

    summary = run_statistics(_args(input_path, output_dir))

    aggregate = summary["aggregate"]
    assert aggregate["scenarios"] == 1
    assert aggregate["errors"] == 0
    assert sum(aggregate["road_counts"].values()) == 1
    assert sum(aggregate["agent_size_counts"].values()) == 1
    assert aggregate["action_diagnostics"]["valid_state_frames"] == 11
    assert summary["table_row_counts"] == {
        "road_counts": 22,
        "agent_counts": 18,
        "action_counts": 120,
        "action_counts_by_frame": 1320,
    }
    for redundant_key in (
        "per_source",
        "road_count_rows",
        "agent_count_rows",
        "agent_action_count_rows",
        "agent_action_count_rows_by_frame",
    ):
        assert redundant_key not in summary
    for output_path in summary["output_files"].values():
        assert output_path.endswith(
            (
                ".csv",
                ".csv.gz",
                ".json",
                ".jsonl",
            )
        )
        assert not (output_dir / (output_path + ".partial")).exists()

    with gzip.open(
        output_dir / "agent_actions_by_frame.csv.gz",
        "rt",
        encoding="utf-8",
        newline="",
    ) as stream:
        action_rows = list(csv.DictReader(stream))
    assert len(action_rows) == 11
    assert {row["scenario_id"] for row in action_rows} == {"scenario-a"}

    saved = json.loads((output_dir / "summary.json").read_text())
    assert saved == summary
    assert (output_dir / "summary.json").stat().st_size < 100_000


def test_requires_overwrite_for_complete_statistics_directory(tmp_path):
    input_path = tmp_path / "validation.tfrecord-00000-of-00001"
    write_tfrecord(
        input_path,
        [make_scenario("scenario-a", frame_count=11).SerializeToString()],
    )
    output_dir = tmp_path / "statistics"
    run_statistics(_args(input_path, output_dir))

    with pytest.raises(FileExistsError, match="--overwrite"):
        run_statistics(_args(input_path, output_dir))

    summary = run_statistics(_args(input_path, output_dir, "--overwrite"))
    assert summary["aggregate"]["scenarios"] == 1
