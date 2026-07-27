from __future__ import annotations

import csv
import gzip
import json

import pytest

from src.womd_labeling.annotate import parse_args as parse_annotate_args
from src.womd_labeling.annotate import annotate_paths
from src.womd_labeling.visualize import parse_args, visualize_paths

from .helpers import make_scenario, write_tfrecord


def test_renders_annotated_scenario_and_manifest(tmp_path):
    input_path = tmp_path / "training.tfrecord-00000-of-00001"
    write_tfrecord(
        input_path,
        [make_scenario("scenario-a", frame_count=11).SerializeToString()],
    )
    annotation_dir = tmp_path / "annotations"
    annotate_paths(
        parse_annotate_args(
            [
                "--input-path",
                str(input_path),
                "--output-dir",
                str(annotation_dir),
                "--workers",
                "1",
            ]
        )
    )
    output_dir = tmp_path / "visualizations"

    summary = visualize_paths(
        parse_args(
            [
                "--input-path",
                str(input_path),
                "--annotation-path",
                str(annotation_dir),
                "--output-dir",
                str(output_dir),
                "--workers",
                "1",
                "--max-scenarios",
                "1",
            ]
        )
    )

    assert summary["images_written"] == 1
    assert summary["errors"] == 0
    image_paths = list(output_dir.glob("*.png"))
    assert len(image_paths) == 1
    assert image_paths[0].stat().st_size > 0
    with (output_dir / "manifest.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["scenario_id"] == "scenario-a"
    assert rows[0]["status"] == "written"
    assert json.loads((output_dir / "summary.json").read_text())[
        "images_written"
    ] == 1
    assert not list(output_dir.glob("*.partial"))

    resumed = visualize_paths(
        parse_args(
            [
                "--input-path",
                str(input_path),
                "--annotation-path",
                str(annotation_dir),
                "--output-dir",
                str(output_dir),
                "--workers",
                "1",
                "--max-scenarios",
                "1",
            ]
        )
    )
    assert resumed["images_skipped"] == 1

    image_paths[0].write_bytes(b"")
    repaired = visualize_paths(
        parse_args(
            [
                "--input-path",
                str(input_path),
                "--annotation-path",
                str(annotation_dir),
                "--output-dir",
                str(output_dir),
                "--workers",
                "1",
                "--max-scenarios",
                "1",
            ]
        )
    )
    assert repaired["images_written"] == 1
    assert image_paths[0].stat().st_size > 0

    with pytest.raises(FileExistsError, match="--overwrite"):
        visualize_paths(
            parse_args(
                [
                    "--input-path",
                    str(input_path),
                    "--annotation-path",
                    str(annotation_dir),
                    "--output-dir",
                    str(output_dir),
                    "--workers",
                    "1",
                    "--max-scenarios",
                    "1",
                    "--no-resume",
                ]
            )
        )


def test_annotation_directory_resolver_accepts_gzip_jsonl(tmp_path):
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    path = annotation_dir / "shard.map-annotations.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({"scenario_id": "a", "ego_frames": []}) + "\n")

    args = parse_args(
        [
            "--input-path",
            str(tmp_path / "missing.tfrecord"),
            "--annotation-path",
            str(annotation_dir),
        ]
    )

    assert args.annotation_path == [str(annotation_dir)]


def test_visualization_loads_only_annotation_shards_it_reaches(tmp_path):
    input_dir = tmp_path / "training"
    input_dir.mkdir()
    first_input = input_dir / "training.tfrecord-00000-of-00002"
    second_input = input_dir / "training.tfrecord-00001-of-00002"
    write_tfrecord(
        first_input,
        [make_scenario("scenario-a", frame_count=11).SerializeToString()],
    )
    write_tfrecord(
        second_input,
        [make_scenario("scenario-b", frame_count=11).SerializeToString()],
    )
    annotation_dir = tmp_path / "annotations"
    annotate_paths(
        parse_annotate_args(
            [
                "--input-path",
                str(input_dir),
                "--output-dir",
                str(annotation_dir),
                "--workers",
                "1",
            ]
        )
    )
    second_annotation = annotation_dir / (
        "training.tfrecord-00001-of-00002.map-annotations.jsonl.gz"
    )
    second_annotation.write_bytes(b"must not be opened")

    output_dir = tmp_path / "visualizations"
    summary = visualize_paths(
        parse_args(
            [
                "--input-path",
                str(input_dir),
                "--annotation-path",
                str(annotation_dir),
                "--output-dir",
                str(output_dir),
                "--workers",
                "1",
                "--max-scenarios",
                "1",
            ]
        )
    )

    assert summary["images_written"] == 1
    assert summary["errors"] == 0
    with (output_dir / "manifest.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        assert [
            row["scenario_id"] for row in csv.DictReader(stream)
        ] == ["scenario-a"]
