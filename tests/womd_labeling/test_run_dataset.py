from __future__ import annotations

import json

import pytest

from src.womd_labeling import run_dataset as runner

from .helpers import make_scenario, write_tfrecord


def _touch_split(root, split):
    directory = root / split
    directory.mkdir(parents=True)
    (directory / f"{split}.tfrecord-00000-of-00001").write_bytes(b"")


def test_routes_each_split_to_isolated_stage_outputs(tmp_path, monkeypatch):
    input_root = tmp_path / "raw"
    for split in ("training", "validation", "testing"):
        _touch_split(input_root, split)
    output_root = tmp_path / "labels"
    calls = {
        "annotations": [],
        "statistics": [],
        "scenario_visualizations": [],
        "aggregate_visualization": [],
    }

    def fake_annotations(args):
        calls["annotations"].append(args)
        return {"errors": 0}

    def fake_statistics(args):
        calls["statistics"].append(args)
        return {
            "schema_version": "statistics-v1",
            "aggregate": {
                "scenarios": 2,
                "errors": 0,
                "road_counts": {"ROAD_SEGMENT": 2},
                "agent_size_counts": {"TYPE_VEHICLE\tSMALL": 3},
                "agent_action_counts": {"TYPE_VEHICLE\t6": 22},
                "action_diagnostics": {"valid_state_frames": 22},
            },
            "road_count_rows": [{"large": "payload"}] * 100,
            "output_files": {"summary": str(args.output_dir / "summary.json")},
        }

    def fake_scenarios(args):
        calls["scenario_visualizations"].append(args)
        return {"errors": 0}

    def fake_aggregate(args):
        calls["aggregate_visualization"].append(args)
        return {"annotation_errors": 0}

    monkeypatch.setattr(runner, "annotate_paths", fake_annotations)
    monkeypatch.setattr(runner, "run_statistics", fake_statistics)
    monkeypatch.setattr(runner, "visualize_paths", fake_scenarios)
    monkeypatch.setattr(runner, "plot_statistics", fake_aggregate)
    monkeypatch.setattr(
        runner,
        "_require_annotation_outputs",
        lambda *_: {},
    )
    monkeypatch.setattr(
        runner,
        "_require_statistics_outputs",
        lambda *_: {},
    )
    monkeypatch.setattr(
        runner,
        "resolve_annotation_paths",
        lambda *_: [],
    )

    summary = runner.run_dataset(
        runner.parse_args(
            [
                "--input-root",
                str(input_root),
                "--output-root",
                str(output_root),
                "--splits",
                "training",
                "validation",
                "testing",
                "--workers",
                "3",
                "--visualize-max-scenarios",
                "0",
            ]
        )
    )

    for index, split in enumerate(("training", "validation", "testing")):
        assert calls["annotations"][index].output_dir == (
            output_root / "annotations" / split
        ).resolve()
        assert calls["annotations"][index].resume is True
        assert calls["statistics"][index].output_dir == (
            output_root / "statistics" / split
        ).resolve()
        assert calls["statistics"][index].resume is True
        assert calls["scenario_visualizations"][index].output_dir == (
            output_root / "visualizations" / "scenarios" / split
        ).resolve()
        assert calls["scenario_visualizations"][index].max_scenarios is None
        assert calls["scenario_visualizations"][index].resume is True
        assert calls["aggregate_visualization"][index].output_prefix == (
            output_root / "visualizations" / "aggregate" / split
        ).resolve()
    assert summary["status"] == "complete"
    assert set(summary["splits"]) == {"training", "validation", "testing"}
    saved = json.loads((output_root / "run_summary.json").read_text())
    assert saved["status"] == "complete"
    for split in ("training", "validation", "testing"):
        stage = saved["splits"][split]["stages"]["statistics"]
        assert "road_count_rows" not in stage
        assert stage["scenarios"] == 2
        assert stage["agent_frame_count"] == 22
        assert stage["output_files"]["summary"].endswith("summary.json")
    assert (output_root / "run_summary.json").stat().st_size < 20_000


def test_missing_requested_split_is_rejected_before_stages_run(tmp_path):
    input_root = tmp_path / "raw"
    _touch_split(input_root, "training")

    with pytest.raises(FileNotFoundError, match="validation"):
        runner.run_dataset(
            runner.parse_args(
                [
                    "--input-root",
                    str(input_root),
                    "--output-root",
                    str(tmp_path / "labels"),
                    "--splits",
                    "training",
                    "validation",
                ]
            )
        )


def test_rejects_duplicate_splits_and_negative_visualization_limit(tmp_path):
    args = runner.parse_args(
        [
            "--input-root",
            str(tmp_path / "raw"),
            "--output-root",
            str(tmp_path / "labels"),
            "--splits",
            "training",
            "training",
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        runner.run_dataset(args)

    args = runner.parse_args(
        [
            "--input-root",
            str(tmp_path / "raw"),
            "--output-root",
            str(tmp_path / "labels"),
            "--visualize-max-scenarios",
            "-1",
        ]
    )
    with pytest.raises(ValueError, match="non-negative"):
        runner.run_dataset(args)


def test_visualization_stage_requires_compatible_annotations(tmp_path):
    input_root = tmp_path / "raw"
    _touch_split(input_root, "training")

    with pytest.raises(RuntimeError, match="annotations stage"):
        runner.run_dataset(
            runner.parse_args(
                [
                    "--input-root",
                    str(input_root),
                    "--output-root",
                    str(tmp_path / "labels"),
                    "--splits",
                    "training",
                    "--stages",
                    "scenario-visualizations",
                ]
            )
        )


def test_real_one_scenario_pipeline_resumes_all_stages(tmp_path):
    input_root = tmp_path / "raw"
    training_dir = input_root / "training"
    training_dir.mkdir(parents=True)
    write_tfrecord(
        training_dir / "training.tfrecord-00000-of-00001",
        [make_scenario("scenario-a", frame_count=11).SerializeToString()],
    )
    output_root = tmp_path / "labels"
    args = runner.parse_args(
        [
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--splits",
            "training",
            "--workers",
            "1",
            "--visualize-max-scenarios",
            "1",
            "--visualize-dpi",
            "72",
            "--aggregate-dpi",
            "72",
        ]
    )

    first = runner.run_dataset(args)
    resumed = runner.run_dataset(args)

    assert first["status"] == "complete"
    assert resumed["status"] == "complete"
    stages = resumed["splits"]["training"]["stages"]
    assert stages["annotations"]["shards_skipped"] == 1
    assert stages["statistics"]["resumed"] is True
    assert stages["scenario-visualizations"]["images_skipped"] == 1
    assert stages["aggregate-visualization"]["resumed"] is True
