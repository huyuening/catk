"""Orchestrate WOMD labeling and visualization across dataset splits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from .annotate import annotate_paths
from .annotate import parse_args as parse_annotation_args
from .plot_statistics import plot_statistics
from .plot_statistics import parse_args as parse_plot_args
from .statistics import parse_args as parse_statistics_args
from .statistics import run_statistics
from .tfrecord_io import resolve_tfrecord_paths
from .visualize import parse_args as parse_visualization_args
from .visualize import visualize_paths


SCHEMA_VERSION = "catk-womd-labeling-run-v1"
SPLITS = ("training", "validation", "testing")
STAGES = (
    "annotations",
    "statistics",
    "scenario-visualizations",
    "aggregate-visualization",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Label raw WOMD training/validation/testing TFRecords and produce "
            "scenario-level plus aggregate visualizations."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=list(STAGES),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--visualize-max-scenarios",
        type=int,
        default=100,
        help="Scenario images per split; 0 renders every scenario.",
    )
    parser.add_argument(
        "--visualize-dpi",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--aggregate-dpi",
        type=int,
        default=600,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate and reuse complete stage outputs when possible.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _statistics_outputs_complete(
    output_dir: Path,
    input_paths: list[Path],
) -> dict | None:
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_inputs = [str(path.resolve()) for path in input_paths]
    if payload.get("input_files") != expected_inputs:
        return None
    output_files = payload.get("output_files")
    if not isinstance(output_files, dict):
        return None
    if not output_files or not all(Path(path).is_file() for path in output_files.values()):
        return None
    payload = dict(payload)
    payload["resumed"] = True
    return payload


def _aggregate_outputs_complete(output_prefix: Path) -> dict | None:
    paths = {
        "png": output_prefix.with_suffix(".png"),
        "pdf": output_prefix.with_suffix(".pdf"),
        "svg": output_prefix.with_suffix(".svg"),
        "counts": output_prefix.with_name(output_prefix.name + "_counts.csv"),
    }
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
        return None
    return {
        "resumed": True,
        "annotation_errors": 0,
        "output_files": {key: str(path) for key, path in paths.items()},
    }


def _stage_error_count(stage: str, result: dict) -> int:
    if stage == "statistics":
        return int(result.get("aggregate", {}).get("errors", 0))
    if stage == "aggregate-visualization":
        return int(result.get("annotation_errors", 0))
    return int(result.get("errors", 0))


def _validate_args(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.visualize_max_scenarios < 0:
        raise ValueError("--visualize-max-scenarios must be non-negative")
    if args.visualize_dpi < 72:
        raise ValueError("--visualize-dpi must be at least 72")
    if args.aggregate_dpi < 72:
        raise ValueError("--aggregate-dpi must be at least 72")
    if len(set(args.splits)) != len(args.splits):
        raise ValueError("--splits contains duplicate entries")
    if len(set(args.stages)) != len(args.stages):
        raise ValueError("--stages contains duplicate entries")


def run_dataset(args: argparse.Namespace) -> dict:
    """Run selected stages, preserving a split-level progress summary."""
    _validate_args(args)
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "run_summary.json"

    split_inputs: dict[str, tuple[Path, list[Path]]] = {}
    for split in args.splits:
        split_dir = input_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Requested WOMD split directory does not exist: {split_dir}"
            )
        try:
            input_paths = resolve_tfrecord_paths([split_dir])
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"No TFRecord shards found for requested split {split!r}: "
                f"{split_dir}"
            ) from exc
        split_inputs[split] = (split_dir, input_paths)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "requested_splits": list(args.splits),
        "requested_stages": list(args.stages),
        "workers": args.workers,
        "resume": args.resume,
        "overwrite": args.overwrite,
        "visualize_max_scenarios": args.visualize_max_scenarios,
        "errors": 0,
        "splits": {},
    }
    _write_summary(summary_path, summary)

    try:
        for split in args.splits:
            split_dir, input_paths = split_inputs[split]
            annotation_dir = (
                output_root / "annotations" / split
            ).resolve()
            statistics_dir = (
                output_root / "statistics" / split
            ).resolve()
            scenario_dir = (
                output_root / "visualizations" / "scenarios" / split
            ).resolve()
            aggregate_prefix = (
                output_root / "visualizations" / "aggregate" / split
            ).resolve()
            split_summary = {
                "status": "running",
                "input_dir": str(split_dir),
                "input_files": [str(path) for path in input_paths],
                "outputs": {
                    "annotations": str(annotation_dir),
                    "statistics": str(statistics_dir),
                    "scenario_visualizations": str(scenario_dir),
                    "aggregate_visualization": str(aggregate_prefix),
                },
                "stages": {},
                "errors": 0,
            }
            summary["splits"][split] = split_summary

            if "annotations" in args.stages:
                stage_argv = [
                    "--input-path",
                    str(split_dir),
                    "--output-dir",
                    str(annotation_dir),
                    "--workers",
                    str(args.workers),
                ]
                stage_argv.append("--resume" if args.resume else "--no-resume")
                if args.overwrite:
                    stage_argv.append("--overwrite")
                result = annotate_paths(parse_annotation_args(stage_argv))
                split_summary["stages"]["annotations"] = result
                split_summary["errors"] += _stage_error_count(
                    "annotations", result
                )

            if "statistics" in args.stages:
                result = None
                if args.resume and not args.overwrite:
                    result = _statistics_outputs_complete(
                        statistics_dir,
                        input_paths,
                    )
                if result is None:
                    stage_argv = [
                        "--input-path",
                        str(split_dir),
                        "--output-dir",
                        str(statistics_dir),
                        "--workers",
                        str(args.workers),
                    ]
                    if args.overwrite:
                        stage_argv.append("--overwrite")
                    result = run_statistics(
                        parse_statistics_args(stage_argv)
                    )
                split_summary["stages"]["statistics"] = result
                split_summary["errors"] += _stage_error_count(
                    "statistics", result
                )

            if "scenario-visualizations" in args.stages:
                stage_argv = [
                    "--input-path",
                    str(split_dir),
                    "--annotation-path",
                    str(annotation_dir),
                    "--output-dir",
                    str(scenario_dir),
                    "--workers",
                    str(args.workers),
                    "--dpi",
                    str(args.visualize_dpi),
                ]
                if args.visualize_max_scenarios:
                    stage_argv.extend(
                        [
                            "--max-scenarios",
                            str(args.visualize_max_scenarios),
                        ]
                    )
                if args.overwrite:
                    stage_argv.append("--overwrite")
                result = visualize_paths(
                    parse_visualization_args(stage_argv)
                )
                split_summary["stages"]["scenario-visualizations"] = result
                split_summary["errors"] += _stage_error_count(
                    "scenario-visualizations", result
                )

            if "aggregate-visualization" in args.stages:
                result = None
                if args.resume and not args.overwrite:
                    result = _aggregate_outputs_complete(aggregate_prefix)
                if result is None:
                    stage_argv = [
                        "--statistics-dir",
                        str(statistics_dir),
                        "--map-annotation-path",
                        str(annotation_dir),
                        "--output-prefix",
                        str(aggregate_prefix),
                        "--dpi",
                        str(args.aggregate_dpi),
                    ]
                    if args.overwrite:
                        stage_argv.append("--overwrite")
                    result = plot_statistics(parse_plot_args(stage_argv))
                split_summary["stages"]["aggregate-visualization"] = result
                split_summary["errors"] += _stage_error_count(
                    "aggregate-visualization", result
                )

            split_summary["status"] = "complete"
            summary["errors"] += split_summary["errors"]
            summary["updated_at"] = _utc_now()
            _write_summary(summary_path, summary)
    except Exception as exc:
        summary["status"] = "failed"
        summary["updated_at"] = _utc_now()
        summary["failure"] = f"{type(exc).__name__}: {exc}"
        _write_summary(summary_path, summary)
        raise

    summary["status"] = "complete"
    summary["completed_at"] = _utc_now()
    summary["updated_at"] = summary["completed_at"]
    _write_summary(summary_path, summary)
    return summary


def main() -> None:
    summary = run_dataset(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
