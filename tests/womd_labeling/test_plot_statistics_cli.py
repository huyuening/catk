from __future__ import annotations

from pathlib import Path

from src.womd_labeling.annotate import annotate_paths
from src.womd_labeling.annotate import parse_args as parse_annotate_args
from src.womd_labeling.plot_statistics import parse_args, plot_statistics
from src.womd_labeling.statistics import parse_args as parse_statistics_args
from src.womd_labeling.statistics import run_statistics

from .helpers import make_scenario, write_tfrecord


def test_generates_all_aggregate_visualization_formats(tmp_path):
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
    statistics_dir = tmp_path / "statistics"
    run_statistics(
        parse_statistics_args(
            [
                "--input-path",
                str(input_path),
                "--output-dir",
                str(statistics_dir),
                "--workers",
                "1",
            ]
        )
    )
    output_prefix = tmp_path / "aggregate" / "womd_labels"

    summary = plot_statistics(
        parse_args(
            [
                "--statistics-dir",
                str(statistics_dir),
                "--map-annotation-path",
                str(annotation_dir),
                "--output-prefix",
                str(output_prefix),
                "--dpi",
                "72",
            ]
        )
    )

    assert summary["road_scenarios"] == 1
    assert summary["agent_count"] == 1
    assert summary["agent_frame_count"] == 11
    for suffix in (".png", ".pdf", ".svg"):
        path = output_prefix.with_suffix(suffix)
        assert path.is_file()
        assert path.stat().st_size > 0
    counts_path = output_prefix.with_name(output_prefix.name + "_counts.csv")
    assert counts_path.is_file()
    assert not list((tmp_path / "aggregate").glob("*.partial"))
