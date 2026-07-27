from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import time

import pytest

from src.womd_labeling import statistics as statistics_module
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

    action_detail_path = next(
        (output_dir / "shards").glob("*.agent-actions-by-frame.csv.gz")
    )
    with gzip.open(
        action_detail_path,
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

    resumed = run_statistics(_args(input_path, output_dir))
    assert resumed["resumed"] is True

    with pytest.raises(FileExistsError, match="--overwrite"):
        run_statistics(_args(input_path, output_dir, "--no-resume"))

    summary = run_statistics(_args(input_path, output_dir, "--overwrite"))
    assert summary["aggregate"]["scenarios"] == 1


def test_statistics_keeps_only_one_source_accumulator(tmp_path, monkeypatch):
    input_dir = tmp_path / "training"
    input_dir.mkdir()
    for index in range(3):
        write_tfrecord(
            input_dir / f"training.tfrecord-{index:05d}-of-00003",
            [
                make_scenario(
                    f"scenario-{index}",
                    frame_count=11,
                ).SerializeToString()
            ],
        )

    original_new_accumulator = statistics_module.new_accumulator

    class TrackedAccumulator(dict):
        live = 0
        max_live = 0

        def __init__(self, payload):
            super().__init__(payload)
            type(self).live += 1
            type(self).max_live = max(
                type(self).max_live,
                type(self).live,
            )

        def __del__(self):
            type(self).live -= 1

    monkeypatch.setattr(
        statistics_module,
        "new_accumulator",
        lambda: TrackedAccumulator(original_new_accumulator()),
    )

    summary = run_statistics(_args(input_dir, tmp_path / "statistics"))

    assert summary["aggregate"]["scenarios"] == 3
    assert TrackedAccumulator.max_live <= 2


def test_rebuilds_split_summary_from_completed_shards(tmp_path, monkeypatch):
    input_dir = tmp_path / "training"
    input_dir.mkdir()
    for index in range(2):
        write_tfrecord(
            input_dir / f"training.tfrecord-{index:05d}-of-00002",
            [
                make_scenario(
                    f"scenario-{index}",
                    frame_count=11,
                ).SerializeToString()
            ],
        )
    output_dir = tmp_path / "statistics"
    first = run_statistics(_args(input_dir, output_dir))
    shard_summaries = sorted((output_dir / "shards").glob("*.summary.json"))
    before = [path.stat().st_mtime_ns for path in shard_summaries]
    (output_dir / "agent_action_counts.csv").write_text(
        "corrupt\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        statistics_module,
        "process_scenario",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("completed shard was recomputed")
        ),
    )
    resumed = run_statistics(_args(input_dir, output_dir))

    assert resumed["aggregate"] == first["aggregate"]
    assert [path.stat().st_mtime_ns for path in shard_summaries] == before
    assert (output_dir / "agent_action_counts.csv").read_text(
        encoding="utf-8"
    ).startswith("scope,")


def test_recomputes_later_shard_when_global_offset_changes(tmp_path):
    input_dir = tmp_path / "training"
    input_dir.mkdir()
    first_path = input_dir / "training.tfrecord-00000-of-00002"
    second_path = input_dir / "training.tfrecord-00001-of-00002"
    write_tfrecord(
        first_path,
        [make_scenario("scenario-0", frame_count=11).SerializeToString()],
    )
    write_tfrecord(
        second_path,
        [make_scenario("scenario-1", frame_count=11).SerializeToString()],
    )
    output_dir = tmp_path / "statistics"
    run_statistics(_args(input_dir, output_dir))
    second_detail = next(
        (output_dir / "shards").glob(
            "00001-*.current-frame-road-types.csv.gz"
        )
    )
    with gzip.open(
        second_detail,
        "rt",
        encoding="utf-8",
        newline="",
    ) as stream:
        assert next(csv.DictReader(stream))["global_index"] == "1"

    write_tfrecord(
        first_path,
        [
            make_scenario("scenario-0", frame_count=11).SerializeToString(),
            make_scenario("scenario-added", frame_count=11).SerializeToString(),
        ],
    )
    resumed = run_statistics(_args(input_dir, output_dir))

    assert resumed["aggregate"]["scenarios"] == 3
    with gzip.open(
        second_detail,
        "rt",
        encoding="utf-8",
        newline="",
    ) as stream:
        assert next(csv.DictReader(stream))["global_index"] == "2"


def test_parallel_statistics_preserves_scenario_order(tmp_path, monkeypatch):
    scenario_count = 24
    input_path = tmp_path / "training.tfrecord-00000-of-00001"
    write_tfrecord(
        input_path,
        [
            make_scenario(
                f"scenario-{index}",
                frame_count=11,
            ).SerializeToString()
            for index in range(scenario_count)
        ],
    )
    output_dir = tmp_path / "statistics"
    original_process_scenario = statistics_module.process_scenario

    class TrackedResult(dict):
        live = 0
        max_live = 0

        def __init__(self, payload):
            super().__init__(payload)
            type(self).live += 1
            type(self).max_live = max(
                type(self).max_live,
                type(self).live,
            )

        def __del__(self):
            type(self).live -= 1

    def delayed_process_scenario(*args):
        if args[2] == 0:
            time.sleep(0.15)
        return TrackedResult(original_process_scenario(*args))

    monkeypatch.setattr(
        statistics_module,
        "ProcessPoolExecutor",
        ThreadPoolExecutor,
    )
    monkeypatch.setattr(
        statistics_module,
        "process_scenario",
        delayed_process_scenario,
    )

    run_statistics(
        parse_args(
            [
                "--input-path",
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--workers",
                "2",
            ]
        )
    )

    detail_path = next(
        (output_dir / "shards").glob("*.current-frame-road-types.csv.gz")
    )
    with gzip.open(
        detail_path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as stream:
        assert [
            row["scenario_id"] for row in csv.DictReader(stream)
        ] == [
            f"scenario-{index}" for index in range(scenario_count)
        ]
    assert TrackedResult.max_live <= 6


def test_recovers_when_split_manifest_publish_is_interrupted(
    tmp_path,
    monkeypatch,
):
    input_path = tmp_path / "training.tfrecord-00000-of-00001"
    write_tfrecord(
        input_path,
        [make_scenario("scenario-a", frame_count=11).SerializeToString()],
    )
    output_dir = tmp_path / "statistics"
    run_statistics(_args(input_path, output_dir))
    original_replace = Path.replace
    injected = {"raised": False}

    def interrupted_replace(path, target):
        if (
            not injected["raised"]
            and path == output_dir / "summary.json.partial"
        ):
            injected["raised"] = True
            raise OSError("injected publish interruption")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupted_replace)
    with pytest.raises(OSError, match="injected"):
        run_statistics(_args(input_path, output_dir, "--overwrite"))
    monkeypatch.setattr(Path, "replace", original_replace)

    recovered = run_statistics(_args(input_path, output_dir))
    assert recovered["aggregate"]["scenarios"] == 1
    assert recovered["aggregate"]["errors"] == 0
    assert not list(output_dir.rglob("*.partial"))
