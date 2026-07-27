from __future__ import annotations

import json

import pytest

from src.womd_labeling import run_dataset as runner


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
        return {"aggregate": {"errors": 0}}

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
        assert calls["statistics"][index].output_dir == (
            output_root / "statistics" / split
        ).resolve()
        assert calls["scenario_visualizations"][index].output_dir == (
            output_root / "visualizations" / "scenarios" / split
        ).resolve()
        assert calls["scenario_visualizations"][index].max_scenarios is None
        assert calls["aggregate_visualization"][index].output_prefix == (
            output_root / "visualizations" / "aggregate" / split
        ).resolve()
    assert summary["status"] == "complete"
    assert set(summary["splits"]) == {"training", "validation", "testing"}
    saved = json.loads((output_root / "run_summary.json").read_text())
    assert saved["status"] == "complete"


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
