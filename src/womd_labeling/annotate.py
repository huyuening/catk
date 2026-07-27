from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Iterable, Sequence, TextIO

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-womd")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, total=None, desc="Progress", unit="item", **_):
            self.total = total
            self.desc = desc
            self.unit = unit
            self.count = 0
            self.last_report = 0
            self.report_every = max(1, (total or 100) // 100)

        def __enter__(self):
            self._report()
            return self

        def __exit__(self, *_):
            if self.count != self.last_report:
                self._report()
            print(file=sys.stderr)

        def update(self, amount=1):
            self.count += amount
            if (
                self.count - self.last_report >= self.report_every
                or self.count == self.total
            ):
                self._report()

        def _report(self):
            total = "?" if self.total is None else str(self.total)
            print(
                f"\r{self.desc}: {self.count}/{total} {self.unit}",
                end="",
                file=sys.stderr,
                flush=True,
            )
            self.last_report = self.count


from .artifacts import artifact_identity_record
from .map_annotation import (
    MAP_ANNOTATION_SCHEMA_VERSION,
    MapAnnotationConfig,
    annotate_scenario,
)
from .proto import scenario_pb2
from .tfrecord_io import (
    count_tfrecord_records,
    iter_tfrecord,
    resolve_tfrecord_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "dataset" / "training.tfrecord-00000-of-01000"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "map_annotations"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build map-junction annotations and ego-centric per-frame "
            "road/intersection labels for WOMD TFRecord scenarios."
        )
    )
    parser.add_argument(
        "--input-path",
        nargs="+",
        default=[str(DEFAULT_INPUT_PATH)],
        help="TFRecord files, directories, or quoted glob patterns.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes. Increase after checking available CPU and RAM.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Global zero-based scenario index across all matched input shards.",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Optional global scenario cap for smoke tests.",
    )
    parser.add_argument(
        "--intersection-distance-m",
        "--near-distance-m",
        dest="near_distance_m",
        type=float,
        default=40.0,
        help="Classify matched lanes within this along-lane distance as intersection.",
    )
    parser.add_argument("--lane-half-width-m", type=float, default=2.0)
    parser.add_argument(
        "--lane-neighbor-extension-m",
        type=float,
        default=12.0,
        help="Extend lateral-neighbor index ranges by this longitudinal distance.",
    )
    parser.add_argument(
        "--max-lane-neighbor-distance-m",
        type=float,
        default=8.0,
        help="Maximum centerline distance for an extended lane neighbor.",
    )
    parser.add_argument(
        "--junction-merge-overlap-ratio",
        type=float,
        default=0.15,
        help=(
            "Merge same-control junction groups when their buffered core-lane "
            "overlap covers this fraction of the smaller group."
        ),
    )
    parser.add_argument("--arm-angle-threshold-deg", type=float, default=30.0)
    parser.add_argument("--max-map-match-distance-m", type=float, default=8.0)
    parser.add_argument(
        "--max-map-match-heading-error-deg", type=float, default=60.0
    )
    parser.add_argument(
        "--signalized-only",
        action="store_true",
        help="Exclude stop-controlled and geometry-only intersections.",
    )
    parser.add_argument("--min-junction-arms", type=int, default=3)
    parser.add_argument("--max-junction-arms", type=int, default=8)
    parser.add_argument(
        "--parking-max-speed-limit-mph",
        type=float,
        default=15.0,
        help="Maximum matched-lane speed limit for parking-lot inference.",
    )
    parser.add_argument(
        "--parking-context-radius-m",
        type=float,
        default=60.0,
        help="Radius used for parking vehicles and local low-speed lanes.",
    )
    parser.add_argument(
        "--parking-min-off-lane-stationary-vehicles",
        type=int,
        default=15,
        help="Minimum nearby stationary vehicles outside lane corridors.",
    )
    parser.add_argument(
        "--parking-dense-off-lane-stationary-vehicles",
        type=int,
        default=25,
        help="Dense parking threshold for off-lane stationary vehicles.",
    )
    parser.add_argument(
        "--parking-internal-lane-radius-m",
        type=float,
        default=30.0,
        help="Radius used to recognize a dense internal parking-lane network.",
    )
    parser.add_argument(
        "--parking-internal-min-lane-count",
        type=int,
        default=18,
        help="Minimum number of internal lanes around an enclosed parking area.",
    )
    parser.add_argument(
        "--parking-internal-min-branch-lane-count",
        type=int,
        default=4,
        help="Minimum number of branching lanes in an internal parking network.",
    )
    parser.add_argument(
        "--parking-internal-max-lane-length-m",
        type=float,
        default=90.0,
        help="Maximum matched-lane length for the internal-network proxy.",
    )
    parser.add_argument(
        "--parking-compact-edge-distance-m",
        type=float,
        default=40.0,
        help="Maximum forward/backward road-edge enclosure distance.",
    )
    parser.add_argument(
        "--freeway-ramp-max-lane-count",
        type=int,
        default=3,
        help="Maximum local lane count eligible for freeway-ramp inference.",
    )
    parser.add_argument(
        "--freeway-ramp-min-lane-count-gain",
        type=int,
        default=2,
        help="Required lane-count gain from a ramp to a wider freeway mainline.",
    )
    parser.add_argument(
        "--freeway-ramp-topology-hops",
        type=int,
        default=6,
        help="Maximum lane-topology hops used to find a connected mainline.",
    )
    parser.add_argument(
        "--compression",
        choices=("gzip", "none"),
        default="gzip",
        help="Output JSONL compression. Defaults to gzip.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing annotation shard outputs.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Validate and skip completed annotation shards. Enabled by default; "
            "--overwrite takes precedence."
        ),
    )
    parser.add_argument(
        "--no-count-total",
        action="store_true",
        help="Skip the initial TFRecord counting pass.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def selected_task_count(
    paths: Iterable[Path], start_index: int, max_scenarios: int | None
) -> int:
    available = sum(count_tfrecord_records(path) for path in paths)
    remaining = max(0, available - start_index)
    return remaining if max_scenarios is None else min(remaining, max_scenarios)


def output_path_for(path: Path, output_dir: Path, compression: str) -> Path:
    suffix = (
        ".map-annotations.jsonl.gz"
        if compression == "gzip"
        else ".map-annotations.jsonl"
    )
    return output_dir / f"{path.name}{suffix}"


def open_text_output(path: Path, compression: str) -> TextIO:
    if compression == "gzip":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def _stable_fingerprint(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def annotation_config_fingerprint(config: MapAnnotationConfig) -> str:
    return _stable_fingerprint(asdict(config))


def source_file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return _stable_fingerprint(
        {
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    )


def validate_completed_annotation(
    path: Path,
    *,
    source_file: str,
    expected_records: int,
    expected_indices: range | None = None,
    expected_config_fingerprint: str | None = None,
    expected_source_fingerprint: str | None = None,
) -> dict:
    """Validate a completed shard before resume skips its raw input."""
    opener = gzip.open if path.name.endswith(".gz") else open
    seen_indices: list[int] = []
    errors = 0
    aggregate = {
        "junctions": 0,
        "signalized_junctions": 0,
        "stop_controlled_junctions": 0,
        "roundabout_junctions": 0,
        "geometric_junctions": 0,
        "ego_frames": 0,
        "ego_valid_frames": 0,
        "region_counts": Counter(),
        "road_environment_counts": Counter(),
        "road_environment_subtype_counts": Counter(),
    }
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                row = json.loads(line)
                if row.get("schema_version") != MAP_ANNOTATION_SCHEMA_VERSION:
                    raise ValueError(
                        f"schema mismatch on line {line_number}"
                    )
                if row.get("source_file") != source_file:
                    raise ValueError(
                        f"source mismatch on line {line_number}"
                    )
                if (
                    expected_config_fingerprint is not None
                    and row.get("annotation_config_fingerprint")
                    != expected_config_fingerprint
                ):
                    raise ValueError(
                        f"annotation config mismatch on line {line_number}"
                    )
                if (
                    expected_source_fingerprint is not None
                    and row.get("source_fingerprint")
                    != expected_source_fingerprint
                ):
                    raise ValueError(
                        f"source fingerprint mismatch on line {line_number}"
                    )
                scenario_index = row.get("scenario_index")
                if not isinstance(scenario_index, int):
                    raise ValueError(
                        f"missing scenario_index on line {line_number}"
                    )
                seen_indices.append(scenario_index)
                if "error" in row:
                    errors += 1
                else:
                    merge_annotation_statistics(
                        aggregate,
                        row["statistics"],
                    )
    except Exception as exc:
        raise ValueError(
            f"Invalid completed annotation shard {path}: {exc}"
        ) from exc

    if len(seen_indices) != expected_records:
        raise ValueError(
            f"Invalid completed annotation shard {path}: expected "
            f"{expected_records} records, found {len(seen_indices)}"
        )
    if expected_indices is not None and seen_indices != list(expected_indices):
        raise ValueError(
            f"Invalid completed annotation shard {path}: scenario indices "
            "do not match the selected source range"
        )
    return {
        "path": str(path),
        "records": len(seen_indices),
        "errors": errors,
        "aggregate": aggregate,
    }


def process_record(
    payload: bytes,
    source_file: str,
    scenario_index: int,
    config: MapAnnotationConfig,
    config_fingerprint: str,
    source_fingerprint: str,
) -> dict:
    scenario = scenario_pb2.Scenario()
    try:
        scenario.ParseFromString(payload)
        annotation = annotate_scenario(
            scenario,
            config,
            scenario_index=scenario_index,
            source_file=source_file,
        )
        result = annotation.to_dict()
        result["annotation_config_fingerprint"] = config_fingerprint
        result["source_fingerprint"] = source_fingerprint
        statistics = result["statistics"]
        return {
            "json": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            "scenario_id": scenario.scenario_id,
            "statistics": statistics,
            "error": None,
        }
    except Exception as exc:
        error = {
            "schema_version": MAP_ANNOTATION_SCHEMA_VERSION,
            "source_file": source_file,
            "scenario_index": scenario_index,
            "scenario_id": scenario.scenario_id or None,
            "annotation_config_fingerprint": config_fingerprint,
            "source_fingerprint": source_fingerprint,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        return {
            "json": json.dumps(error, ensure_ascii=False, separators=(",", ":")),
            "scenario_id": scenario.scenario_id or None,
            "statistics": None,
            "error": error["error"],
        }


def merge_annotation_statistics(summary: dict, statistics: dict) -> None:
    summary["junctions"] += statistics["junction_count"]
    summary["signalized_junctions"] += statistics["signalized_junction_count"]
    summary["stop_controlled_junctions"] += statistics[
        "stop_controlled_junction_count"
    ]
    summary["roundabout_junctions"] += statistics[
        "roundabout_junction_count"
    ]
    summary["geometric_junctions"] += statistics["geometric_junction_count"]
    summary["ego_frames"] += statistics["ego_frame_count"]
    summary["ego_valid_frames"] += statistics["ego_valid_frame_count"]
    summary["region_counts"].update(statistics["region_counts"])
    summary["road_environment_counts"].update(
        statistics["road_environment_counts"]
    )
    summary["road_environment_subtype_counts"].update(
        statistics["road_environment_subtype_counts"]
    )


def update_summary(summary: dict, result: dict) -> None:
    summary["scenarios_written"] += 1
    if result["error"] is not None:
        summary["errors"] += 1
        return
    merge_annotation_statistics(summary, result["statistics"])


def merge_completed_aggregate(summary: dict, completed: dict) -> None:
    for key in (
        "junctions",
        "signalized_junctions",
        "stop_controlled_junctions",
        "roundabout_junctions",
        "geometric_junctions",
        "ego_frames",
        "ego_valid_frames",
    ):
        summary[key] += completed[key]
    for key in (
        "region_counts",
        "road_environment_counts",
        "road_environment_subtype_counts",
    ):
        summary[key].update(completed[key])


def selected_record_indices(
    *,
    shard_global_start: int,
    shard_record_count: int,
    selection_start: int,
    selection_count: int | None,
) -> range:
    shard_global_end = shard_global_start + shard_record_count
    selection_end = (
        None
        if selection_count is None
        else selection_start + selection_count
    )
    selected_global_start = max(shard_global_start, selection_start)
    selected_global_end = shard_global_end
    if selection_end is not None:
        selected_global_end = min(selected_global_end, selection_end)
    if selected_global_end <= selected_global_start:
        return range(0, 0)
    return range(
        selected_global_start - shard_global_start,
        selected_global_end - shard_global_start,
    )


def drain_one(
    pending,
    output,
    summary,
    progress,
    result_buffer,
    next_write_index,
):
    done, _ = wait(pending, return_when=FIRST_COMPLETED)
    for future in done:
        record_index = pending.pop(future)
        result_buffer[record_index] = future.result()
        progress.update(1)
    while next_write_index in result_buffer:
        result = result_buffer.pop(next_write_index)
        output.write(result["json"] + "\n")
        update_summary(summary, result)
        next_write_index += 1
    return next_write_index


def annotate_paths(args: argparse.Namespace) -> dict:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.max_scenarios is not None and args.max_scenarios < 1:
        raise ValueError("--max-scenarios must be positive")

    paths = resolve_tfrecord_paths(args.input_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = MapAnnotationConfig(
        near_distance_m=args.near_distance_m,
        lane_half_width_m=args.lane_half_width_m,
        lane_neighbor_extension_m=args.lane_neighbor_extension_m,
        max_lane_neighbor_distance_m=args.max_lane_neighbor_distance_m,
        junction_merge_overlap_ratio=args.junction_merge_overlap_ratio,
        arm_angle_threshold_deg=args.arm_angle_threshold_deg,
        max_map_match_distance_m=args.max_map_match_distance_m,
        max_map_match_heading_error_deg=args.max_map_match_heading_error_deg,
        include_stop_controlled=not args.signalized_only,
        min_junction_arms=args.min_junction_arms,
        max_junction_arms=args.max_junction_arms,
        parking_max_speed_limit_mph=args.parking_max_speed_limit_mph,
        parking_context_radius_m=args.parking_context_radius_m,
        parking_min_off_lane_stationary_vehicles=(
            args.parking_min_off_lane_stationary_vehicles
        ),
        parking_dense_off_lane_stationary_vehicles=(
            args.parking_dense_off_lane_stationary_vehicles
        ),
        parking_internal_lane_radius_m=args.parking_internal_lane_radius_m,
        parking_internal_min_lane_count=(
            args.parking_internal_min_lane_count
        ),
        parking_internal_min_branch_lane_count=(
            args.parking_internal_min_branch_lane_count
        ),
        parking_internal_max_lane_length_m=(
            args.parking_internal_max_lane_length_m
        ),
        parking_compact_edge_distance_m=(
            args.parking_compact_edge_distance_m
        ),
        freeway_ramp_max_lane_count=args.freeway_ramp_max_lane_count,
        freeway_ramp_min_lane_count_gain=(
            args.freeway_ramp_min_lane_count_gain
        ),
        freeway_ramp_topology_hops=args.freeway_ramp_topology_hops,
    )
    config_fingerprint = annotation_config_fingerprint(config)
    source_fingerprints = {
        path: source_file_fingerprint(path) for path in paths
    }

    shard_record_counts = {
        path: count_tfrecord_records(path) for path in paths
    }
    selection_by_path = {}
    next_global_index = 0
    for path in paths:
        count = shard_record_counts[path]
        selection_by_path[path] = selected_record_indices(
            shard_global_start=next_global_index,
            shard_record_count=count,
            selection_start=args.start_index,
            selection_count=args.max_scenarios,
        )
        next_global_index += count
    selected_total = sum(
        len(indices) for indices in selection_by_path.values()
    )
    total = None if args.no_count_total else selected_total
    summary = {
        "schema_version": MAP_ANNOTATION_SCHEMA_VERSION,
        "input_files": [str(path) for path in paths],
        "output_files": [],
        "config": asdict(config),
        "config_fingerprint": config_fingerprint,
        "source_fingerprints": {
            path.name: source_fingerprints[path] for path in paths
        },
        "workers": args.workers,
        "start_index": args.start_index,
        "max_scenarios": args.max_scenarios,
        "resume": args.resume,
        "overwrite": args.overwrite,
        "shards_selected": sum(
            bool(indices) for indices in selection_by_path.values()
        ),
        "shards_written": 0,
        "shards_skipped": 0,
        "scenarios_written": 0,
        "scenarios_skipped": 0,
        "errors": 0,
        "junctions": 0,
        "signalized_junctions": 0,
        "stop_controlled_junctions": 0,
        "roundabout_junctions": 0,
        "geometric_junctions": 0,
        "ego_frames": 0,
        "ego_valid_frames": 0,
        "region_counts": Counter(),
        "road_environment_counts": Counter(),
        "road_environment_subtype_counts": Counter(),
    }

    executor = (
        ProcessPoolExecutor(max_workers=args.workers)
        if args.workers > 1
        else None
    )
    try:
        with tqdm(
            total=total,
            desc="Annotating ego maps",
            unit="scenario",
            dynamic_ncols=True,
        ) as progress:
            for path in paths:
                selected_indices = selection_by_path[path]
                if not selected_indices:
                    continue
                output_path = output_path_for(
                    path, output_dir, args.compression
                )
                if output_path.exists() and not args.overwrite:
                    if not args.resume:
                        raise FileExistsError(
                            f"Output already exists: {output_path}. Use "
                            "--overwrite to replace it or --resume to validate "
                            "and skip it."
                        )
                    try:
                        completed = validate_completed_annotation(
                            output_path,
                            source_file=path.name,
                            expected_records=len(selected_indices),
                            expected_indices=selected_indices,
                            expected_config_fingerprint=config_fingerprint,
                            expected_source_fingerprint=(
                                source_fingerprints[path]
                            ),
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"Existing shard {output_path} cannot be resumed; "
                            "use --overwrite after inspecting it"
                        ) from exc
                    summary["output_files"].append(str(output_path))
                    summary["shards_skipped"] += 1
                    summary["scenarios_skipped"] += completed["records"]
                    summary["errors"] += completed["errors"]
                    merge_completed_aggregate(
                        summary,
                        completed["aggregate"],
                    )
                    progress.update(completed["records"])
                    continue
                partial_path = output_path.with_name(output_path.name + ".partial")
                wrote_this_shard = False
                with open_text_output(partial_path, args.compression) as output:
                    pending = {}
                    result_buffer = {}
                    next_write_index = None
                    for record_index, payload in iter_tfrecord(path):
                        if record_index not in selected_indices:
                            continue
                        wrote_this_shard = True
                        if next_write_index is None:
                            next_write_index = record_index
                        if executor is None:
                            result = process_record(
                                payload,
                                path.name,
                                record_index,
                                config,
                                config_fingerprint,
                                source_fingerprints[path],
                            )
                            output.write(result["json"] + "\n")
                            update_summary(summary, result)
                            progress.update(1)
                        else:
                            future = executor.submit(
                                process_record,
                                payload,
                                path.name,
                                record_index,
                                config,
                                config_fingerprint,
                                source_fingerprints[path],
                            )
                            pending[future] = record_index
                            while (
                                len(pending) + len(result_buffer)
                                >= args.workers * 3
                            ):
                                next_write_index = drain_one(
                                    pending,
                                    output,
                                    summary,
                                    progress,
                                    result_buffer,
                                    next_write_index,
                                )
                    while pending:
                        next_write_index = drain_one(
                            pending,
                            output,
                            summary,
                            progress,
                            result_buffer,
                            next_write_index,
                        )
                    if result_buffer:
                        raise RuntimeError(
                            f"Could not restore record order for {path}: "
                            f"remaining indices {sorted(result_buffer)}"
                        )

                if wrote_this_shard:
                    if (
                        source_file_fingerprint(path)
                        != source_fingerprints[path]
                    ):
                        partial_path.unlink(missing_ok=True)
                        raise RuntimeError(
                            f"Source TFRecord changed while annotating: {path}"
                        )
                    partial_path.replace(output_path)
                    summary["output_files"].append(str(output_path))
                    summary["shards_written"] += 1
                else:
                    partial_path.unlink(missing_ok=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    summary["region_counts"] = dict(sorted(summary["region_counts"].items()))
    summary["road_environment_counts"] = dict(
        sorted(summary["road_environment_counts"].items())
    )
    summary["road_environment_subtype_counts"] = dict(
        sorted(summary["road_environment_subtype_counts"].items())
    )
    summary["scenarios_completed"] = (
        summary["scenarios_written"] + summary["scenarios_skipped"]
    )
    summary["output_artifacts"] = [
        artifact_identity_record(Path(path))
        for path in summary["output_files"]
    ]
    summary_path = output_dir / "summary.json"
    summary["summary_path"] = str(summary_path)
    summary_partial = summary_path.with_name(summary_path.name + ".partial")
    summary_partial.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_partial.replace(summary_path)
    return summary


def main() -> None:
    args = parse_args()
    summary = annotate_paths(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
