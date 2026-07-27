from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
from dataclasses import asdict
import glob
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
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

        def __enter__(self):
            return self

        def __exit__(self, *_):
            print(file=sys.stderr)

        def update(self, amount=1):
            self.count += amount
            total = "?" if self.total is None else str(self.total)
            print(
                f"\r{self.desc}: {self.count}/{total} {self.unit}",
                end="",
                file=sys.stderr,
                flush=True,
            )


from .artifacts import (
    artifact_identity_matches,
    artifact_identity_record,
    stable_fingerprint,
)
from .map_annotation_visualization import (
    DEFAULT_MAP_FRAME_INDEX,
    MapVisualizationConfig,
    RenderedMapAnnotation,
    format_region_summary,
    render_initial_frame_map,
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
DEFAULT_ANNOTATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "map_annotations"
    / "training.tfrecord-00000-of-01000.map-annotations.jsonl.gz"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "map_annotation_visualizations"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render one ego-centric frame map image per WOMD scenario, "
            "using previously generated ego map annotations."
        )
    )
    parser.add_argument(
        "--input-path",
        nargs="+",
        default=[str(DEFAULT_INPUT_PATH)],
        help="TFRecord files, directories, or quoted glob patterns.",
    )
    parser.add_argument(
        "--annotation-path",
        nargs="+",
        default=[str(DEFAULT_ANNOTATION_PATH)],
        help="Map-annotation JSONL or JSONL.GZ files, directories, or globs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument(
        "--scenario-ids",
        nargs="+",
        default=None,
        help="Optional scenario IDs to render, independent of scenario index.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=DEFAULT_MAP_FRAME_INDEX,
        help=(
            "Frame index to render. Defaults to 10 and falls back to the "
            "first valid SDC frame."
        ),
    )
    parser.add_argument("--x-range", nargs=2, type=float, default=(-50.0, 100.0))
    parser.add_argument("--y-range", nargs=2, type=float, default=(-60.0, 60.0))
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Parallel rendering processes. Defaults to at most 4.",
    )
    available_start_methods = multiprocessing.get_all_start_methods()
    default_start_method = (
        "fork" if "fork" in available_start_methods else "spawn"
    )
    parser.add_argument(
        "--mp-start-method",
        choices=available_start_methods,
        default=default_start_method,
        help=(
            "Multiprocessing start method. Defaults to fork when available "
            "to avoid reloading Matplotlib in every worker."
        ),
    )
    parser.add_argument("--show-agent-ids", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace PNG files that already exist.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse validated existing PNG files. Enabled by default; "
            "--overwrite takes precedence."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_annotation_paths(entries: Iterable[str]) -> list[Path]:
    resolved = []
    for entry in entries:
        path = Path(entry).expanduser()
        if path.is_dir():
            matches = sorted(path.rglob("*.map-annotations.jsonl*"))
        else:
            matches = [Path(match) for match in sorted(glob.glob(str(path)))]
            if not matches and path.is_file():
                matches = [path]
        resolved.extend(candidate.resolve() for candidate in matches)

    unique = list(dict.fromkeys(resolved))
    if not unique:
        raise FileNotFoundError("No map-annotation JSONL files matched")
    return unique


def _open_annotation(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def annotation_source_file(path: Path) -> str:
    """Infer the TFRecord basename from an annotation shard filename."""
    for suffix in (
        ".map-annotations.jsonl.gz",
        ".map-annotations.jsonl",
    ):
        if path.name.endswith(suffix):
            source_file = path.name[: -len(suffix)]
            if source_file:
                return source_file
    raise ValueError(
        "Annotation shard name must end with "
        f"'.map-annotations.jsonl[.gz]': {path}"
    )


def index_annotation_paths(paths: Iterable[Path]) -> dict[str, Path]:
    """Map source TFRecord basenames to annotation shards without reading them."""
    indexed = {}
    for path in paths:
        source_file = annotation_source_file(path)
        if source_file in indexed:
            raise ValueError(
                f"Duplicate annotation shards for source {source_file!r}: "
                f"{indexed[source_file]} and {path}"
            )
        indexed[source_file] = path
    return indexed


def load_annotation_shard(
    path: Path,
    *,
    expected_source_file: str,
) -> tuple[dict[int, dict], dict[str, dict]]:
    """Load one annotation shard, bounded by one TFRecord shard."""
    by_index = {}
    by_scenario_id = {}
    with _open_annotation(path) as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            source_file = record.get("source_file")
            if source_file != expected_source_file:
                raise ValueError(
                    f"Expected source_file {expected_source_file!r} in {path} "
                    f"at line {line_number}, found {source_file!r}"
                )
            if record.get("error"):
                continue
            scenario_id = record.get("scenario_id")
            if not scenario_id:
                raise ValueError(
                    f"Missing scenario_id in {path} at line {line_number}"
                )
            scenario_index = record.get("scenario_index")
            if scenario_index is None:
                raise ValueError(
                    f"Missing scenario_index in {path} at line {line_number}"
                )
            scenario_index = int(scenario_index)
            if scenario_index in by_index:
                raise ValueError(
                    f"Duplicate scenario_index {scenario_index} in {path}"
                )
            if scenario_id in by_scenario_id:
                raise ValueError(
                    f"Duplicate scenario_id {scenario_id!r} in {path}"
                )
            by_index[scenario_index] = record
            by_scenario_id[scenario_id] = record
    return by_index, by_scenario_id


def _safe_id(scenario_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", scenario_id)


def _selected_total(
    input_paths: Iterable[Path],
    start_index: int,
    max_scenarios: int | None,
    scenario_ids: set[str] | None,
) -> int | None:
    if scenario_ids is not None:
        return len(scenario_ids)
    available = sum(count_tfrecord_records(path) for path in input_paths)
    remaining = max(0, available - start_index)
    return remaining if max_scenarios is None else min(remaining, max_scenarios)


def _existing_result(
    annotation: dict,
    output_path: Path,
    preferred_frame_index: int,
) -> RenderedMapAnnotation:
    frames = {
        int(frame["frame_index"]): frame
        for frame in annotation.get("ego_frames", [])
    }
    if not frames:
        raise ValueError("Annotation does not contain ego_frames")
    frame = frames.get(preferred_frame_index)
    if frame is None or not frame.get("valid", False):
        frame = next(
            (
                candidate
                for _, candidate in sorted(frames.items())
                if candidate.get("valid", False)
            ),
            None,
        )
    if frame is None:
        raise ValueError("Annotation does not contain a valid SDC frame")
    frame_index = int(frame["frame_index"])
    return RenderedMapAnnotation(
        frame_index=frame_index,
        region_type=frame.get("region_type", "UNKNOWN"),
        summary=format_region_summary(frame, "zh"),
        output_path=output_path,
    )


def _visualization_fingerprint(
    annotation: dict,
    preferred_frame_index: int,
    config: MapVisualizationConfig,
) -> str:
    return stable_fingerprint(
        {
            "schema_version": "catk-womd-scenario-visualization-v1",
            "scenario_id": annotation.get("scenario_id"),
            "scenario_index": annotation.get("scenario_index"),
            "annotation_config_fingerprint": annotation.get(
                "annotation_config_fingerprint"
            ),
            "source_fingerprint": annotation.get("source_fingerprint"),
            "preferred_frame_index": preferred_frame_index,
            "visualization_config": asdict(config),
        }
    )


def _sidecar_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".json")


def _write_visualization_sidecar(
    output_path: Path,
    *,
    fingerprint: str,
) -> None:
    sidecar_path = _sidecar_path(output_path)
    partial_path = sidecar_path.with_name(sidecar_path.name + ".partial")
    payload = {
        "schema_version": "catk-womd-scenario-visualization-v1",
        "fingerprint": fingerprint,
        "image_artifact": artifact_identity_record(output_path),
    }
    partial_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    partial_path.replace(sidecar_path)


def _visualization_is_reusable(
    output_path: Path,
    *,
    fingerprint: str,
) -> bool:
    try:
        payload = json.loads(
            _sidecar_path(output_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("schema_version")
        == "catk-womd-scenario-visualization-v1"
        and payload.get("fingerprint") == fingerprint
        and artifact_identity_matches(payload.get("image_artifact"))
    )


def _render_payload_task(
    payload: bytes,
    annotation: dict,
    output_path: str,
    preferred_frame_index: int,
    config: MapVisualizationConfig,
) -> dict:
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(payload)
    rendered = render_initial_frame_map(
        scenario,
        annotation,
        Path(output_path),
        preferred_frame_index=preferred_frame_index,
        config=config,
    )
    return {
        "frame_index": rendered.frame_index,
        "region_type": rendered.region_type,
        "summary": rendered.summary,
    }


def _rendered_result_dict(rendered: RenderedMapAnnotation) -> dict:
    return {
        "frame_index": rendered.frame_index,
        "region_type": rendered.region_type,
        "summary": rendered.summary,
    }


def visualize_paths(args: argparse.Namespace) -> dict:
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.max_scenarios is not None and args.max_scenarios < 1:
        raise ValueError("--max-scenarios must be positive")
    if args.frame_index < 0:
        raise ValueError("--frame-index must be non-negative")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    input_paths = resolve_tfrecord_paths(args.input_path)
    annotation_paths = resolve_annotation_paths(args.annotation_path)
    annotation_paths_by_source = index_annotation_paths(annotation_paths)
    missing_annotation_sources = [
        path.name
        for path in input_paths
        if path.name not in annotation_paths_by_source
    ]
    if missing_annotation_sources:
        raise FileNotFoundError(
            "Missing annotation shards for TFRecord sources: "
            + ", ".join(missing_annotation_sources)
        )
    requested_ids = None if args.scenario_ids is None else set(args.scenario_ids)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = MapVisualizationConfig(
        x_min_m=args.x_range[0],
        x_max_m=args.x_range[1],
        y_min_m=args.y_range[0],
        y_max_m=args.y_range[1],
        dpi=args.dpi,
        show_agent_ids=args.show_agent_ids,
    )
    total = _selected_total(
        input_paths,
        args.start_index,
        args.max_scenarios,
        requested_ids,
    )
    summary = {
        "input_files": [str(path) for path in input_paths],
        "annotation_files": [str(path) for path in annotation_paths],
        "output_dir": str(output_dir),
        "preferred_frame_index": args.frame_index,
        "workers": args.workers,
        "mp_start_method": args.mp_start_method,
        "scenarios_considered": 0,
        "images_written": 0,
        "images_skipped": 0,
        "errors": 0,
        "region_counts": Counter(),
        "visualization_config": asdict(config),
    }
    manifest_path = output_dir / "manifest.csv"
    manifest_partial = manifest_path.with_name(manifest_path.name + ".partial")
    manifest_events = manifest_path.with_name(
        manifest_path.name + ".events.partial"
    )
    manifest_events.unlink(missing_ok=True)
    found_requested_ids = set()
    global_index = 0
    selected = 0
    executor = (
        ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context(args.mp_start_method),
        )
        if args.workers > 1
        else None
    )
    pending = {}
    manifest_event_stream = manifest_events.open("w", encoding="utf-8")
    try:
        with tqdm(
            total=total,
            desc="Rendering ego map annotations",
            unit="scenario",
            dynamic_ncols=True,
        ) as progress:
            def record_result(
                metadata: dict,
                result: dict | None,
                *,
                status: str,
                error: str = "",
            ) -> None:
                if status == "written":
                    _write_visualization_sidecar(
                        metadata["output_path"],
                        fingerprint=metadata["fingerprint"],
                    )
                if result is None:
                    summary["errors"] += 1
                    frame_index = ""
                    region_type = ""
                    region_summary = ""
                else:
                    if status == "written":
                        summary["images_written"] += 1
                    elif status == "skipped_existing":
                        summary["images_skipped"] += 1
                    frame_index = result["frame_index"]
                    region_type = result["region_type"]
                    region_summary = result["summary"]
                    summary["region_counts"][region_type] += 1
                row = {
                    "scenario_index": metadata["scenario_index"],
                    "scenario_id": metadata["scenario_id"],
                    "frame_index": frame_index,
                    "region_type": region_type,
                    "annotation_summary": region_summary,
                    "status": status,
                    "image_path": metadata["output_path"].name,
                    "error": error,
                }
                manifest_event_stream.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )
                if requested_ids is not None:
                    found_requested_ids.add(metadata["scenario_id"])
                progress.update(1)

            def drain_pending() -> None:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    metadata = pending.pop(future)
                    try:
                        result = future.result()
                        record_result(metadata, result, status="written")
                    except Exception as exc:
                        record_result(
                            metadata,
                            None,
                            status="error",
                            error=f"{type(exc).__name__}: {exc}",
                        )

            stop_iteration = False
            for input_path in input_paths:
                if (
                    args.max_scenarios is not None
                    and selected >= args.max_scenarios
                ):
                    break
                annotations_by_index, annotations_by_scenario_id = (
                    load_annotation_shard(
                        annotation_paths_by_source[input_path.name],
                        expected_source_file=input_path.name,
                    )
                )
                for record_index, payload in iter_tfrecord(input_path):
                    scenario_index = global_index
                    global_index += 1
                    if scenario_index < args.start_index:
                        continue
                    annotation = annotations_by_index.get(record_index)
                    if annotation is None:
                        scenario = scenario_pb2.Scenario()
                        scenario.ParseFromString(payload)
                        scenario_id = scenario.scenario_id
                        annotation = annotations_by_scenario_id.get(scenario_id)
                    else:
                        scenario_id = annotation["scenario_id"]
                    if (
                        requested_ids is not None
                        and scenario_id not in requested_ids
                    ):
                        continue
                    if (
                        args.max_scenarios is not None
                        and selected >= args.max_scenarios
                    ):
                        stop_iteration = True
                        break
                    selected += 1
                    summary["scenarios_considered"] += 1
                    output_path = output_dir / (
                        f"{scenario_index:06d}-{_safe_id(scenario_id)}.png"
                    )
                    metadata = {
                        "scenario_index": scenario_index,
                        "scenario_id": scenario_id,
                        "output_path": output_path,
                        "fingerprint": (
                            None
                            if annotation is None
                            else _visualization_fingerprint(
                                annotation,
                                args.frame_index,
                                config,
                            )
                        ),
                    }
                    if annotation is None:
                        record_result(
                            metadata,
                            None,
                            status="error",
                            error=(
                                "KeyError: No map annotation for scenario "
                                f"{scenario_id}"
                            ),
                        )
                        continue
                    if output_path.exists() and not args.overwrite:
                        if not args.resume:
                            raise FileExistsError(
                                f"Output already exists: {output_path}. Use "
                                "--overwrite to replace it or --resume to "
                                "validate and skip it."
                            )
                        if _visualization_is_reusable(
                            output_path,
                            fingerprint=metadata["fingerprint"],
                        ):
                            try:
                                rendered = _existing_result(
                                    annotation,
                                    output_path,
                                    args.frame_index,
                                )
                                record_result(
                                    metadata,
                                    _rendered_result_dict(rendered),
                                    status="skipped_existing",
                                )
                            except Exception as exc:
                                record_result(
                                    metadata,
                                    None,
                                    status="error",
                                    error=f"{type(exc).__name__}: {exc}",
                                )
                            continue

                    if executor is None:
                        try:
                            result = _render_payload_task(
                                payload,
                                annotation,
                                str(output_path),
                                args.frame_index,
                                config,
                            )
                            record_result(metadata, result, status="written")
                        except Exception as exc:
                            record_result(
                                metadata,
                                None,
                                status="error",
                                error=f"{type(exc).__name__}: {exc}",
                            )
                    else:
                        future = executor.submit(
                            _render_payload_task,
                            payload,
                            annotation,
                            str(output_path),
                            args.frame_index,
                            config,
                        )
                        pending[future] = metadata
                        if len(pending) >= args.workers * 2:
                            drain_pending()
                if stop_iteration:
                    break

            while pending:
                drain_pending()
    finally:
        if executor is not None:
            executor.shutdown()
        manifest_event_stream.close()

    if requested_ids is not None:
        with manifest_events.open("a", encoding="utf-8") as event_stream:
            for missing_id in sorted(
                requested_ids - found_requested_ids
            ):
                summary["errors"] += 1
                row = {
                    "scenario_index": "",
                    "scenario_id": missing_id,
                    "frame_index": "",
                    "region_type": "",
                    "annotation_summary": "",
                    "status": "error",
                    "image_path": "",
                    "error": "Scenario ID was not found in TFRecord inputs",
                }
                event_stream.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )

    with manifest_partial.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "scenario_index",
                "scenario_id",
                "frame_index",
                "region_type",
                "annotation_summary",
                "status",
                "image_path",
                "error",
            ],
        )
        writer.writeheader()
        with manifest_events.open("r", encoding="utf-8") as events:
            for line in events:
                writer.writerow(json.loads(line))
    manifest_partial.replace(manifest_path)
    manifest_events.unlink()

    summary["region_counts"] = dict(sorted(summary["region_counts"].items()))
    summary_path = output_dir / "summary.json"
    summary_partial = summary_path.with_name(summary_path.name + ".partial")
    summary_partial.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_partial.replace(summary_path)
    return summary


def main() -> None:
    summary = visualize_paths(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
