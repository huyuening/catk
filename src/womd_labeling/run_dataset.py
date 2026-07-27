"""Orchestrate WOMD labeling and visualization across dataset splits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from .annotate import (
    annotate_paths,
    annotation_config_fingerprint,
    source_file_fingerprint,
)
from .annotate import parse_args as parse_annotation_args
from .artifacts import (
    artifact_identity_matches,
    artifact_matches,
    stable_fingerprint,
)
from .map_annotation import (
    MAP_ANNOTATION_SCHEMA_VERSION,
    MapAnnotationConfig,
)
from .plot_statistics import (
    AGGREGATE_SCHEMA_VERSION,
    DEFAULT_FIGURE_WIDTH_CM,
    aggregate_dependency_record,
    plot_statistics,
)
from .plot_statistics import parse_args as parse_plot_args
from .statistics import _completed_statistics_run
from .statistics import parse_args as parse_statistics_args
from .statistics import run_statistics
from .statistics import statistics_run_fingerprint
from .tfrecord_io import resolve_tfrecord_paths
from .visualize import parse_args as parse_visualization_args
from .visualize import resolve_annotation_paths
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
    expected_args: argparse.Namespace,
) -> dict | None:
    return _completed_statistics_run(
        output_dir / "summary.json",
        run_fingerprint=statistics_run_fingerprint(
            expected_args,
            input_paths,
        ),
    )


def _annotation_outputs_complete(
    output_dir: Path,
    input_paths: list[Path],
) -> dict | None:
    summary_path = output_dir / "summary.json"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("input_files") != [str(path) for path in input_paths]:
        return None
    if payload.get("schema_version") != MAP_ANNOTATION_SCHEMA_VERSION:
        return None
    config = payload.get("config")
    if (
        not isinstance(config, dict)
        or stable_fingerprint(config) != payload.get("config_fingerprint")
        or payload.get("config_fingerprint")
        != annotation_config_fingerprint(MapAnnotationConfig())
        or payload.get("start_index") != 0
        or payload.get("max_scenarios") is not None
    ):
        return None
    expected_sources = {
        path.name: source_file_fingerprint(path) for path in input_paths
    }
    if payload.get("source_fingerprints") != expected_sources:
        return None
    output_files = payload.get("output_files")
    output_artifacts = payload.get("output_artifacts")
    if (
        not isinstance(output_files, list)
        or len(output_files) != len(input_paths)
        or not all(Path(path).is_file() for path in output_files)
        or not isinstance(output_artifacts, list)
        or len(output_artifacts) != len(output_files)
        or not all(
            artifact_identity_matches(record)
            for record in output_artifacts
        )
    ):
        return None
    return payload


def _require_annotation_outputs(
    output_dir: Path,
    input_paths: list[Path],
) -> dict:
    payload = _annotation_outputs_complete(output_dir, input_paths)
    if payload is None:
        raise RuntimeError(
            "Annotation outputs are missing, stale, or incomplete for "
            f"{output_dir}. Run the annotations stage first."
        )
    return payload


def _require_statistics_outputs(
    output_dir: Path,
    input_paths: list[Path],
    expected_args: argparse.Namespace,
) -> dict:
    payload = _statistics_outputs_complete(
        output_dir,
        input_paths,
        expected_args,
    )
    if payload is None:
        raise RuntimeError(
            "Statistics outputs are missing, stale, or incomplete for "
            f"{output_dir}. Run the statistics stage first."
        )
    return payload


def _aggregate_outputs_complete(
    output_prefix: Path,
    statistics_dir: Path,
    annotation_paths: list[Path],
    *,
    dpi: int,
) -> dict | None:
    summary_path = output_prefix.with_name(
        output_prefix.name + ".summary.json"
    )
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != AGGREGATE_SCHEMA_VERSION:
        return None
    artifacts = payload.get("output_artifacts")
    output_files = payload.get("output_files")
    if (
        not isinstance(artifacts, dict)
        or not artifacts
        or not isinstance(output_files, dict)
        or set(artifacts) != set(output_files) - {"summary"}
        or not all(artifact_matches(record) for record in artifacts.values())
    ):
        return None
    try:
        dependencies = aggregate_dependency_record(
            statistics_dir,
            annotation_paths,
        )
    except OSError:
        return None
    if payload.get("dependencies") != dependencies:
        return None
    expected_configuration = {
        "dpi": dpi,
        "width_cm": DEFAULT_FIGURE_WIDTH_CM,
        "html_fragment": None,
    }
    if payload.get("configuration") != expected_configuration:
        return None
    payload = dict(payload)
    payload["resumed"] = True
    return payload


def _stage_error_count(stage: str, result: dict) -> int:
    if stage == "statistics":
        return int(result.get("aggregate", {}).get("errors", 0))
    if stage == "aggregate-visualization":
        return int(result.get("annotation_errors", 0))
    return int(result.get("errors", 0))


def _select_keys(payload: dict, keys: tuple[str, ...]) -> dict:
    return {key: payload[key] for key in keys if key in payload}


def _compact_stage_result(stage: str, result: dict) -> dict:
    """Keep run_summary.json bounded independently of dataset size."""
    if stage == "annotations":
        return _select_keys(
            result,
            (
                "schema_version",
                "summary_path",
                "shards_selected",
                "shards_written",
                "shards_skipped",
                "scenarios_written",
                "scenarios_skipped",
                "scenarios_completed",
                "errors",
                "junctions",
                "signalized_junctions",
                "stop_controlled_junctions",
                "roundabout_junctions",
                "geometric_junctions",
                "ego_frames",
                "ego_valid_frames",
                "region_counts",
                "road_environment_counts",
                "road_environment_subtype_counts",
            ),
        )
    if stage == "statistics":
        aggregate = result.get("aggregate", {})
        action_diagnostics = aggregate.get("action_diagnostics", {})
        compact = _select_keys(
            result,
            (
                "schema_version",
                "frame_number",
                "frame_index",
                "table_row_counts",
                "output_files",
                "resumed",
            ),
        )
        compact.update(
            {
                "scenarios": int(aggregate.get("scenarios", 0)),
                "errors": int(aggregate.get("errors", 0)),
                "road_counts": aggregate.get("road_counts", {}),
                "agent_size_counts": aggregate.get(
                    "agent_size_counts", {}
                ),
                "agent_action_counts": aggregate.get(
                    "agent_action_counts", {}
                ),
                "agent_frame_count": int(
                    action_diagnostics.get("valid_state_frames", 0)
                ),
            }
        )
        return compact
    if stage == "scenario-visualizations":
        return _select_keys(
            result,
            (
                "output_dir",
                "preferred_frame_index",
                "workers",
                "mp_start_method",
                "scenarios_considered",
                "images_written",
                "images_skipped",
                "errors",
                "region_counts",
            ),
        )
    if stage == "aggregate-visualization":
        return _select_keys(
            result,
            (
                "statistics_dir",
                "road_scenarios",
                "road_unknown",
                "annotation_errors",
                "agent_count",
                "agent_frame_count",
                "output_files",
                "resumed",
            ),
        )
    raise ValueError(f"Unsupported stage: {stage}")


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
            statistics_argv = [
                "--input-path",
                str(split_dir),
                "--output-dir",
                str(statistics_dir),
                "--workers",
                str(args.workers),
                "--resume" if args.resume else "--no-resume",
            ]
            if args.overwrite:
                statistics_argv.append("--overwrite")
            statistics_args = parse_statistics_args(statistics_argv)

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
                split_summary["stages"]["annotations"] = (
                    _compact_stage_result("annotations", result)
                )
                split_summary["errors"] += _stage_error_count(
                    "annotations", result
                )

            if "statistics" in args.stages:
                result = run_statistics(statistics_args)
                split_summary["stages"]["statistics"] = (
                    _compact_stage_result("statistics", result)
                )
                split_summary["errors"] += _stage_error_count(
                    "statistics", result
                )

            if "scenario-visualizations" in args.stages:
                _require_annotation_outputs(annotation_dir, input_paths)
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
                stage_argv.append(
                    "--resume" if args.resume else "--no-resume"
                )
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
                split_summary["stages"]["scenario-visualizations"] = (
                    _compact_stage_result(
                        "scenario-visualizations",
                        result,
                    )
                )
                split_summary["errors"] += _stage_error_count(
                    "scenario-visualizations", result
                )

            if "aggregate-visualization" in args.stages:
                _require_annotation_outputs(annotation_dir, input_paths)
                _require_statistics_outputs(
                    statistics_dir,
                    input_paths,
                    statistics_args,
                )
                annotation_paths = resolve_annotation_paths(
                    [str(annotation_dir)]
                )
                result = None
                if args.resume and not args.overwrite:
                    result = _aggregate_outputs_complete(
                        aggregate_prefix,
                        statistics_dir,
                        annotation_paths,
                        dpi=args.aggregate_dpi,
                    )
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
                    if args.overwrite or (args.resume and result is None):
                        stage_argv.append("--overwrite")
                    result = plot_statistics(parse_plot_args(stage_argv))
                split_summary["stages"]["aggregate-visualization"] = (
                    _compact_stage_result(
                        "aggregate-visualization",
                        result,
                    )
                )
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
