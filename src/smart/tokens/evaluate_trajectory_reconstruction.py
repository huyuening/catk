# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is prohibited.

"""Evaluate CatK's batch reconstruction directly from WOMD TFRecords."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import importlib
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

os.environ.setdefault(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION",
    "python",
)

from .reconstruction_evaluation import (
    AGENT_METRICS,
    FRAME_METRICS,
    EvaluationAccumulator,
    MetricDefinition,
    RunningMoments,
    ScenarioMetricBatch,
    evaluate_scenario_pair,
)
from .exact_metric_store import (
    BufferKey,
    EvaluationCheckpoint,
    EvaluationIdentity,
    ExactMetricStore,
    ExactPercentiles,
    restore_checkpoint,
    write_checkpoint,
)
from .trajectory_filter_reconstructor import FILTER_STRENGTHS
from .womd_trajectory_reconstruction import (
    TrajectoryReconstructionConfig,
    reconstruct_scenario_for_vocabulary,
)


_TRAINING_SHARD_PATTERN = re.compile(
    r"^training\.tfrecord-(?P<index>\d{5})-of-(?P<total>\d{5})$"
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_METRIC_SCHEMA = "exact-reconstruction-v1"
_SCRATCH_MARKER = ".catk_exact_reconstruction_scratch"
_FINAL_OUTPUT_NAMES = (
    "agent_summary.csv",
    "frame_jerk_summary.csv",
    "summary.json",
    "reconstruction_summary.json",
    "run_config.json",
)
_SUMMARY_FIELDS = (
    "scope",
    "level",
    "metric",
    "variant",
    "unit",
    "count",
    "mean",
    "std",
    "min",
    "max",
    "range",
    "p01",
    "p99",
    "p99_minus_p01",
)


@dataclass(frozen=True)
class ReconstructionSettings:
    """Batch settings copied from a completed reconstruction run."""

    method: str
    filter_strength: str
    max_gap_frames: int
    batch_linear_jerk_weight: float
    batch_angular_jerk_weight: float

    def __post_init__(self) -> None:
        if self.method != "batch":
            raise ValueError(
                "reconstruction method must be 'batch' for this evaluator"
            )
        if self.filter_strength not in FILTER_STRENGTHS:
            valid = ", ".join(FILTER_STRENGTHS)
            raise ValueError(
                f"filter_strength must be one of: {valid}"
            )
        max_gap_frames = _integer_setting(
            self.max_gap_frames,
            "max_gap_frames",
        )
        if max_gap_frames < -1:
            raise ValueError("max_gap_frames must be at least -1")
        object.__setattr__(
            self,
            "batch_linear_jerk_weight",
            _weight_setting(
                self.batch_linear_jerk_weight,
                "batch_linear_jerk_weight",
            ),
        )
        object.__setattr__(
            self,
            "batch_angular_jerk_weight",
            _weight_setting(
                self.batch_angular_jerk_weight,
                "batch_angular_jerk_weight",
            ),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_reconstruction_config(
        self,
    ) -> TrajectoryReconstructionConfig:
        return TrajectoryReconstructionConfig(
            method="batch",
            filter_strength=self.filter_strength,
            max_gap_frames=self.max_gap_frames,
            batch_linear_jerk_weight=(
                self.batch_linear_jerk_weight
            ),
            batch_angular_jerk_weight=(
                self.batch_angular_jerk_weight
            ),
        )


@dataclass(frozen=True)
class ScenarioTask:
    """One serialized WOMD scenario and immutable reconstruction settings."""

    source_file: str
    record_index: int
    payload: bytes
    settings: ReconstructionSettings

    def __post_init__(self) -> None:
        if not isinstance(self.source_file, str) or not self.source_file:
            raise ValueError("scenario source_file must be non-empty")
        if (
            isinstance(self.record_index, bool)
            or not isinstance(self.record_index, int)
            or self.record_index < 0
        ):
            raise ValueError(
                "scenario record_index must be a non-negative integer"
            )
        if not isinstance(self.payload, bytes):
            raise ValueError("scenario payload must be bytes")
        if not isinstance(self.settings, ReconstructionSettings):
            raise ValueError(
                "scenario settings must be ReconstructionSettings"
            )


@dataclass(frozen=True)
class ScenarioEvaluationResult:
    """Compact in-memory metrics returned by one worker."""

    source_file: str
    record_index: int
    scenario_id: str
    metrics: ScenarioMetricBatch
    reconstruction_counts: dict[str, int]


def _required_value(
    value: dict,
    key: str,
) -> object:
    if key not in value:
        raise ValueError(
            f"reconstruction run config is missing {key!r}"
        )
    return value[key]


def _integer_setting(
    value: object,
    key: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _weight_setting(
    value: object,
    key: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{key} must be a finite non-negative number")
    return result


def load_reconstruction_settings(
    path: Path,
) -> ReconstructionSettings:
    """Read and strictly validate the batch reconstruction provenance."""

    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        try:
            value = json.load(stream)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid reconstruction run config JSON: {path}"
            ) from error
    if not isinstance(value, dict):
        raise ValueError(
            "reconstruction run config must be a JSON object"
        )

    method = _required_value(value, "method")
    if method != "batch":
        raise ValueError(
            "reconstruction method must be 'batch' for this evaluator"
        )
    filter_strength = _required_value(value, "filter_strength")
    if filter_strength not in FILTER_STRENGTHS:
        valid = ", ".join(FILTER_STRENGTHS)
        raise ValueError(
            f"filter_strength must be one of: {valid}"
        )
    max_gap_frames = _integer_setting(
        _required_value(value, "max_gap_frames"),
        "max_gap_frames",
    )
    if max_gap_frames < -1:
        raise ValueError("max_gap_frames must be at least -1")

    return ReconstructionSettings(
        method="batch",
        filter_strength=str(filter_strength),
        max_gap_frames=max_gap_frames,
        batch_linear_jerk_weight=_weight_setting(
            _required_value(
                value,
                "batch_linear_jerk_weight",
            ),
            "batch_linear_jerk_weight",
        ),
        batch_angular_jerk_weight=_weight_setting(
            _required_value(
                value,
                "batch_angular_jerk_weight",
            ),
            "batch_angular_jerk_weight",
        ),
    )


def _directory_training_shards(
    directory: Path,
) -> list[Path]:
    shards = sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file()
        and _TRAINING_SHARD_PATTERN.fullmatch(path.name) is not None
    )
    if not shards:
        raise FileNotFoundError(
            "No canonical training.tfrecord-#####-of-##### shards "
            f"found directly under {directory}"
        )

    parsed = [
        _TRAINING_SHARD_PATTERN.fullmatch(path.name)
        for path in shards
    ]
    totals = {
        int(match.group("total"))
        for match in parsed
        if match is not None
    }
    if len(totals) != 1:
        raise ValueError(
            f"inconsistent TFRecord shard totals under {directory}"
        )
    total = totals.pop()
    indices = [
        int(match.group("index"))
        for match in parsed
        if match is not None
    ]
    if indices != list(range(total)):
        raise ValueError(
            f"incomplete canonical training shard set under {directory}: "
            f"found {len(indices)} of {total}"
        )
    return shards


def resolve_input_paths(
    entries: Sequence[str],
) -> list[Path]:
    """Resolve explicit files or complete canonical training directories."""

    if not entries:
        raise ValueError("at least one input path is required")
    resolved: list[Path] = []
    for raw_entry in entries:
        expanded = os.path.expanduser(str(raw_entry))
        if glob.has_magic(expanded):
            matches = sorted(
                Path(match).resolve()
                for match in glob.glob(expanded)
                if Path(match).is_file()
                and _TRAINING_SHARD_PATTERN.fullmatch(
                    Path(match).name
                )
                is not None
            )
            if not matches:
                raise FileNotFoundError(
                    "No canonical training shards matched glob: "
                    f"{raw_entry}"
                )
            resolved.extend(matches)
            continue
        entry = Path(raw_entry).expanduser().resolve()
        if entry.is_file():
            resolved.append(entry)
        elif entry.is_dir():
            resolved.extend(_directory_training_shards(entry))
        else:
            raise FileNotFoundError(entry)
    ordered = sorted(resolved, key=lambda path: str(path))
    if len(ordered) != len(set(ordered)):
        raise ValueError("input TFRecord paths must be unique")
    return ordered


def iter_tfrecord(
    path: Path,
    *,
    expected_size: int | None = None,
    expected_mtime_ns: int | None = None,
) -> Iterator[tuple[int, bytes]]:
    """Read uncompressed TFRecord framing without importing TensorFlow."""

    path = Path(path)
    with path.open("rb") as stream:
        opened_stat = os.fstat(stream.fileno())
        _validate_tfrecord_metadata(
            path,
            opened_stat,
            expected_size=expected_size,
            expected_mtime_ns=expected_mtime_ns,
        )
        file_size = opened_stat.st_size
        try:
            record_index = 0
            while True:
                header = stream.read(12)
                if not header:
                    break
                if len(header) != 12:
                    raise ValueError(
                        f"{path}: truncated TFRecord header at record "
                        f"{record_index}"
                    )
                (length,) = struct.unpack("<Q", header[:8])
                remaining = file_size - stream.tell()
                if length + 4 > remaining:
                    raise ValueError(
                        f"{path}: truncated TFRecord payload at record "
                        f"{record_index}"
                    )
                payload = stream.read(length)
                footer_crc = stream.read(4)
                if len(payload) != length or len(footer_crc) != 4:
                    raise ValueError(
                        f"{path}: truncated TFRecord payload at record "
                        f"{record_index}"
                    )
                yield record_index, payload
                record_index += 1
        finally:
            _validate_tfrecord_metadata(
                path,
                os.fstat(stream.fileno()),
                expected_size=expected_size,
                expected_mtime_ns=expected_mtime_ns,
            )


def _validate_tfrecord_metadata(
    path: Path,
    stat_result,
    *,
    expected_size: int | None,
    expected_mtime_ns: int | None,
) -> None:
    mismatches = []
    if (
        expected_size is not None
        and int(stat_result.st_size) != int(expected_size)
    ):
        mismatches.append(
            f"size {stat_result.st_size} != {expected_size}"
        )
    if (
        expected_mtime_ns is not None
        and int(stat_result.st_mtime_ns) != int(expected_mtime_ns)
    ):
        mismatches.append(
            f"mtime_ns {stat_result.st_mtime_ns} != "
            f"{expected_mtime_ns}"
        )
    if mismatches:
        raise ValueError(
            f"{path}: TFRecord metadata changed: "
            + ", ".join(mismatches)
        )


def count_tfrecord_records(
    path: Path,
    limit: int | None = None,
) -> int:
    """Count records, optionally stopping after a positive smoke limit."""

    if limit is not None:
        limit = _integer_setting(limit, "limit")
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return 0
    count = 0
    for _record_index, _payload in iter_tfrecord(path):
        count += 1
        if limit is not None and count >= limit:
            break
    return count


def _load_scenario_class():
    try:
        from waymo_open_dataset.protos import scenario_pb2
    except ModuleNotFoundError:
        pb2_root = (
            _REPO_ROOT
            / "src"
            / "smart"
            / "tokens"
            / "womd_proto"
            / "pb2"
        )
        pb2_root_string = str(pb2_root)
        if pb2_root_string not in sys.path:
            sys.path.insert(0, pb2_root_string)
        scenario_pb2 = importlib.import_module("scenario_pb2")
    return scenario_pb2.Scenario


def _integer_reconstruction_counts(
    stats: object,
) -> dict[str, int]:
    if dataclasses.is_dataclass(stats):
        values = dataclasses.asdict(stats)
    elif isinstance(stats, Mapping):
        values = dict(stats)
    elif hasattr(stats, "__dict__"):
        values = vars(stats)
    else:
        raise TypeError(
            "batch reconstruction returned unsupported statistics"
        )
    return {
        str(key): int(value)
        for key, value in values.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }


def evaluate_scenario_task(
    task: ScenarioTask,
) -> ScenarioEvaluationResult:
    """Reconstruct one scenario in memory and return metric arrays only."""

    scenario_id = "<unparsed>"
    try:
        scenario_class = _load_scenario_class()
        scenario = scenario_class()
        scenario.ParseFromString(task.payload)
        scenario_id = str(scenario.scenario_id)
        if not scenario_id:
            raise ValueError("parsed WOMD scenario has no scenario_id")
        reconstructed, reconstruction_stats = (
            reconstruct_scenario_for_vocabulary(
                scenario,
                task.settings.to_reconstruction_config(),
            )
        )
        metrics = evaluate_scenario_pair(
            scenario,
            reconstructed,
        )
        reconstruction_counts = _integer_reconstruction_counts(
            reconstruction_stats
        )
    except Exception as error:
        raise RuntimeError(
            "failed reconstruction evaluation at "
            f"{task.source_file}, record {task.record_index}, "
            f"scenario {scenario_id}"
        ) from error
    return ScenarioEvaluationResult(
        source_file=task.source_file,
        record_index=task.record_index,
        scenario_id=scenario_id,
        metrics=metrics,
        reconstruction_counts=reconstruction_counts,
    )


def bounded_ordered_map(
    executor,
    function: Callable[[Any], Any],
    tasks: Iterable[Any],
    limit: int,
) -> Iterator[Any]:
    """Yield input-order results with at most ``limit`` futures retained."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("pending future limit must be a positive integer")
    task_iterator = iter(tasks)
    pending = deque()
    for _ in range(limit):
        try:
            item = next(task_iterator)
        except StopIteration:
            break
        pending.append(executor.submit(function, item))

    while pending:
        future = pending.popleft()
        yield future.result()
        try:
            item = next(task_iterator)
        except StopIteration:
            continue
        pending.append(executor.submit(function, item))


def _metric_definitions(
    level: str,
) -> tuple[MetricDefinition, ...]:
    if level == "agent":
        return AGENT_METRICS
    if level == "frame":
        return FRAME_METRICS
    raise ValueError("summary level must be 'agent' or 'frame'")


def _moment_groups(
    accumulator: EvaluationAccumulator,
    level: str,
) -> dict[str, dict[str, RunningMoments]]:
    if level == "agent":
        return accumulator.agent_moments
    if level == "frame":
        return accumulator.frame_moments
    raise ValueError("summary level must be 'agent' or 'frame'")


def _evaluation_scopes(
    accumulator: EvaluationAccumulator,
) -> list[str]:
    types = (
        set(accumulator.agent_moments)
        | set(accumulator.frame_moments)
    ) - {"all"}
    return ["all", *sorted(types)]


def collect_exact_percentiles(
    accumulator: EvaluationAccumulator,
    store: ExactMetricStore,
) -> dict[tuple[str, str, str], ExactPercentiles]:
    """Finalize exact type/all percentiles and verify every sample count."""

    store.flush_and_sync()
    counts = store.snapshot_counts()
    scopes = _evaluation_scopes(accumulator)
    type_scopes = scopes[1:]
    result: dict[
        tuple[str, str, str],
        ExactPercentiles,
    ] = {}
    for level in ("agent", "frame"):
        grouped = _moment_groups(accumulator, level)
        for definition in _metric_definitions(level):
            type_count = 0
            for scope in type_scopes:
                key = BufferKey(level, scope, definition.key)
                expected_count = counts.get(key.encoded, 0)
                actual_count = grouped.get(scope, {}).get(
                    definition.key,
                    RunningMoments(),
                ).count
                if actual_count != expected_count:
                    raise ValueError(
                        "metric moment/buffer count mismatch for "
                        f"{key.encoded}: {actual_count} != "
                        f"{expected_count}"
                    )
                type_count += expected_count
                result[(level, scope, definition.key)] = (
                    store.percentiles(key)
                )

            all_count = grouped.get("all", {}).get(
                definition.key,
                RunningMoments(),
            ).count
            if all_count != type_count:
                raise ValueError(
                    "global metric count does not equal type counts for "
                    f"{level}|all|{definition.key}: "
                    f"{all_count} != {type_count}"
                )
            result[(level, "all", definition.key)] = (
                store.combined_percentiles(
                    level,
                    definition.key,
                    type_scopes,
                )
            )
    return result


def summary_rows(
    accumulator: EvaluationAccumulator,
    percentiles: Mapping[
        tuple[str, str, str],
        ExactPercentiles,
    ],
    level: str,
) -> list[dict]:
    """Build stable all/type summary rows for one metric level."""

    definitions = _metric_definitions(level)
    grouped = _moment_groups(accumulator, level)
    rows = []
    for scope in _evaluation_scopes(accumulator):
        for definition in definitions:
            moments = grouped.get(scope, {}).get(
                definition.key,
                RunningMoments(),
            )
            exact = percentiles.get(
                (level, scope, definition.key),
                ExactPercentiles(None, None),
            )
            rows.append(
                {
                    "scope": scope,
                    "level": level,
                    "metric": definition.metric,
                    "variant": definition.variant,
                    "unit": definition.unit,
                    **moments.summary(
                        p01=exact.p01,
                        p99=exact.p99,
                    ),
                }
            )
    return rows


def _atomic_json(
    path: Path,
    value: object,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=_SUMMARY_FIELDS,
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reconstruction_summary(
    counts: Mapping[str, int],
) -> dict:
    normalized = {
        str(key): int(value)
        for key, value in sorted(counts.items())
    }
    rates = {}
    total_tracks = normalized.get("total_tracks", 0)
    processed_segments = normalized.get("processed_segments", 0)
    for key, value in normalized.items():
        if key != "total_tracks" and key.endswith("_tracks"):
            rates[f"{key}_per_total_track"] = (
                value / total_tracks
                if total_tracks
                else None
            )
        elif (
            key != "processed_segments"
            and key.endswith("_segments")
        ):
            rates[f"{key}_per_processed_segment"] = (
                value / processed_segments
                if processed_segments
                else None
            )
    return {
        "counts": normalized,
        "rates": rates,
    }


def _reject_unknown_output_entries(
    output_dir: Path,
) -> None:
    if not output_dir.exists():
        return
    unknown = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.name not in _FINAL_OUTPUT_NAMES
    )
    if unknown:
        raise ValueError(
            f"output directory contains unknown entries: {unknown}"
        )


def write_final_outputs(
    *,
    output_dir: Path,
    accumulator: EvaluationAccumulator,
    store: ExactMetricStore,
    reconstruction_counts: Mapping[str, int],
    run_config: Mapping[str, object],
) -> tuple[Path, ...]:
    """Finalize exact statistics and atomically write the five outputs."""

    output_dir = Path(output_dir)
    _reject_unknown_output_entries(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    percentiles = collect_exact_percentiles(
        accumulator,
        store,
    )
    agent_rows = summary_rows(
        accumulator,
        percentiles,
        "agent",
    )
    frame_rows = summary_rows(
        accumulator,
        percentiles,
        "frame",
    )
    reconstruction = _reconstruction_summary(
        reconstruction_counts
    )
    reconstruction_provenance = run_config.get(
        "reconstruction",
        {},
    )
    if not isinstance(reconstruction_provenance, Mapping):
        raise ValueError(
            "run_config reconstruction provenance must be a mapping"
        )
    summary = {
        "schema": "exact-reconstruction-v1",
        "scenario_count": accumulator.scenarios,
        "failure_count": 0,
        "agent_count": accumulator.agents,
        "reconstruction_provenance": dict(
            reconstruction_provenance
        ),
        "software_provenance": dict(
            run_config.get("software", {})
        ),
        "statistics": {
            "dtype": "float64",
            "standard_deviation": "population (ddof=0)",
            "percentiles": {
                "method": "NumPy linear",
                "values": [1, 99],
                "exact": True,
            },
        },
        "support": {
            "matched": (
                "raw jerk samples are reported only when every raw-valid "
                "jerk center is also valid after reconstruction"
            ),
            "reconstructed_full": (
                "all finite reconstructed-valid jerk centers"
            ),
            "xy_rmse": (
                "frames where both raw and reconstructed positions are valid"
            ),
        },
        "agent_metrics": agent_rows,
        "frame_metrics": frame_rows,
    }

    paths = tuple(
        output_dir / name
        for name in _FINAL_OUTPUT_NAMES
    )
    _atomic_csv(paths[0], agent_rows)
    _atomic_csv(paths[1], frame_rows)
    _atomic_json(paths[2], summary)
    _atomic_json(paths[3], reconstruction)
    _atomic_json(paths[4], dict(run_config))
    store.close()
    return paths


def build_evaluation_identity(
    paths: Sequence[Path],
    settings: ReconstructionSettings,
    max_scenarios: int | None,
) -> EvaluationIdentity:
    """Bind a checkpoint to exact input files and reconstruction settings."""

    shards = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        stat = path.stat()
        shards.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return EvaluationIdentity(
        input_shards=tuple(shards),
        reconstruction=settings.to_dict(),
        max_scenarios=max_scenarios,
        metric_schema=_METRIC_SCHEMA,
    )


def _scratch_marker_path(
    scratch_dir: Path,
) -> Path:
    return scratch_dir / _SCRATCH_MARKER


def _write_scratch_marker(
    scratch_dir: Path,
) -> None:
    marker = _scratch_marker_path(scratch_dir)
    temporary = marker.with_suffix(".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(f"{_METRIC_SCHEMA}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_scratch(
    scratch_dir: Path,
    *,
    resume: bool,
) -> None:
    allowed = {
        _SCRATCH_MARKER,
        "checkpoint.json",
        "metrics",
    }
    if not scratch_dir.exists():
        scratch_dir.mkdir(parents=True)
        _write_scratch_marker(scratch_dir)
        return
    if not scratch_dir.is_dir():
        raise ValueError(
            f"scratch path is not a directory: {scratch_dir}"
        )
    entries = list(scratch_dir.iterdir())
    if not entries:
        _write_scratch_marker(scratch_dir)
        return
    unknown = sorted(
        path.name
        for path in entries
        if path.name not in allowed
    )
    if unknown:
        raise ValueError(
            f"scratch directory contains unknown entries: {unknown}"
        )
    if not resume:
        raise ValueError(
            "scratch directory is non-empty; use --resume or choose "
            "a new scratch directory"
        )
    marker = _scratch_marker_path(scratch_dir)
    checkpoint = scratch_dir / "checkpoint.json"
    if not marker.is_file() or not checkpoint.is_file():
        raise ValueError(
            "scratch directory has no complete CatK checkpoint"
        )
    marker_value = marker.read_text(encoding="utf-8").strip()
    if marker_value != _METRIC_SCHEMA:
        raise ValueError("scratch marker identity is incompatible")


def _delete_owned_scratch(
    scratch_dir: Path,
) -> None:
    marker = _scratch_marker_path(scratch_dir)
    if (
        not marker.is_file()
        or marker.read_text(encoding="utf-8").strip()
        != _METRIC_SCHEMA
    ):
        raise ValueError(
            "refusing to delete scratch without the CatK ownership marker"
        )
    allowed = {
        _SCRATCH_MARKER,
        "checkpoint.json",
        "metrics",
    }
    unknown = [
        path.name
        for path in scratch_dir.iterdir()
        if path.name not in allowed
    ]
    if unknown:
        raise ValueError(
            "refusing to delete scratch containing unknown entries: "
            f"{sorted(unknown)}"
        )
    shutil.rmtree(scratch_dir)


def _validated_runtime_values(
    args,
) -> tuple[int, int | None, int]:
    workers = _integer_setting(args.workers, "workers")
    if workers < 1:
        raise ValueError("workers must be positive")
    max_scenarios = args.max_scenarios
    if max_scenarios is not None:
        max_scenarios = _integer_setting(
            max_scenarios,
            "max_scenarios",
        )
        if max_scenarios < 1:
            raise ValueError("max_scenarios must be positive")
    progress_every = _integer_setting(
        args.progress_every,
        "progress_every",
    )
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    return workers, max_scenarios, progress_every


def _scenario_tasks(
    path: Path,
    settings: ReconstructionSettings,
    limit: int | None,
    expected_metadata: Mapping[str, object],
) -> Iterator[ScenarioTask]:
    record_iterator = iter_tfrecord(
        path,
        expected_size=int(expected_metadata["size"]),
        expected_mtime_ns=int(expected_metadata["mtime_ns"]),
    )
    records: Iterable[tuple[int, bytes]] = record_iterator
    try:
        if limit is not None:
            records = islice(records, limit)
        for record_index, payload in records:
            yield ScenarioTask(
                source_file=str(path),
                record_index=record_index,
                payload=payload,
                settings=settings,
            )
    finally:
        record_iterator.close()


def _shard_fits_limit(
    path: Path,
    limit: int | None,
    expected_metadata: Mapping[str, object],
) -> bool:
    if limit is None:
        return True
    record_iterator = iter_tfrecord(
        path,
        expected_size=int(expected_metadata["size"]),
        expected_mtime_ns=int(expected_metadata["mtime_ns"]),
    )
    try:
        count = sum(
            1
            for _ in islice(
                record_iterator,
                limit + 1,
            )
        )
    finally:
        record_iterator.close()
    return count <= limit


def _result_stream(
    path: Path,
    settings: ReconstructionSettings,
    limit: int | None,
    expected_metadata: Mapping[str, object],
    workers: int,
    executor,
) -> Iterator[ScenarioEvaluationResult]:
    tasks = _scenario_tasks(
        path,
        settings,
        limit,
        expected_metadata,
    )
    if workers == 1:
        for task in tasks:
            yield evaluate_scenario_task(task)
        return
    yield from bounded_ordered_map(
        executor,
        evaluate_scenario_task,
        tasks,
        limit=workers * 2,
    )


def _consume_shard(
    *,
    path: Path,
    settings: ReconstructionSettings,
    remaining: int | None,
    expected_metadata: Mapping[str, object],
    workers: int,
    executor,
    accumulator: EvaluationAccumulator,
    reconstruction_counts: Counter,
    store: ExactMetricStore,
    progress,
) -> None:
    for result in _result_stream(
        path,
        settings,
        remaining,
        expected_metadata,
        workers,
        executor,
    ):
        if result.source_file != str(path):
            raise ValueError(
                "worker returned a result for the wrong TFRecord shard"
            )
        if result.scenario_id != result.metrics.scenario_id:
            raise ValueError(
                "worker scenario identity does not match its metric batch"
            )
        for key, value in result.reconstruction_counts.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"invalid reconstruction counter {key!r}: {value!r}"
                )
        accumulator.add_batch(result.metrics)
        reconstruction_counts.update(
            result.reconstruction_counts
        )
        store.append_batch(result.metrics)
        progress.update(1)
    _validate_tfrecord_metadata(
        path,
        path.stat(),
        expected_size=int(expected_metadata["size"]),
        expected_mtime_ns=int(expected_metadata["mtime_ns"]),
    )


def _run_config_value(
    *,
    identity: EvaluationIdentity,
    reconstruction_run_config: Path,
    output_dir: Path,
    scratch_dir: Path,
    workers: int,
    progress_every: int,
    resume: bool,
    keep_scratch: bool,
) -> dict:
    return {
        "schema": _METRIC_SCHEMA,
        "identity": identity.to_dict(),
        "reconstruction": dict(identity.reconstruction),
        "reconstruction_run_config": str(
            reconstruction_run_config
        ),
        "output_dir": str(output_dir),
        "scratch_dir": str(scratch_dir),
        "workers": workers,
        "progress_every": progress_every,
        "resume": bool(resume),
        "keep_scratch": bool(keep_scratch),
        "statistics": {
            "dtype": "float64",
            "standard_deviation_ddof": 0,
            "percentile_method": "NumPy linear",
            "percentiles": [1, 99],
            "exact": True,
        },
        "software": _software_provenance(),
    }


def _git_output(
    *arguments: str,
) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _software_provenance() -> dict:
    status = _git_output(
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "catk_git_commit": _git_output("rev-parse", "HEAD"),
        "catk_git_dirty": (
            bool(status)
            if status is not None
            else None
        ),
    }


def run_evaluation(
    args,
) -> tuple[Path, ...]:
    """Run a resumable, exact reconstruction evaluation."""

    from tqdm.auto import tqdm

    workers, max_scenarios, progress_every = (
        _validated_runtime_values(args)
    )
    raw_entries = args.input_path
    if isinstance(raw_entries, (str, Path)):
        raw_entries = [raw_entries]
    paths = resolve_input_paths(
        [str(entry) for entry in raw_entries]
    )
    settings = load_reconstruction_settings(
        Path(args.reconstruction_run_config)
    )
    identity = build_evaluation_identity(
        paths,
        settings,
        max_scenarios,
    )
    metadata_by_path = {
        shard["path"]: shard
        for shard in identity.input_shards
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    scratch_dir = Path(args.scratch_dir).expanduser().resolve()
    if (
        output_dir == scratch_dir
        or output_dir in scratch_dir.parents
        or scratch_dir in output_dir.parents
    ):
        raise ValueError(
            "output and scratch directories must be separate and non-nested"
        )
    _reject_unknown_output_entries(output_dir)
    _prepare_scratch(
        scratch_dir,
        resume=bool(args.resume),
    )

    checkpoint_path = scratch_dir / "checkpoint.json"
    store = ExactMetricStore(scratch_dir / "metrics")
    executor = (
        ProcessPoolExecutor(max_workers=workers)
        if workers > 1
        else None
    )
    progress = tqdm(
        total=max_scenarios,
        initial=0,
        unit="scenario",
        desc="Evaluate reconstruction",
        miniters=max(1, progress_every),
        disable=progress_every == 0,
    )
    try:
        if bool(args.resume) and checkpoint_path.is_file():
            checkpoint = restore_checkpoint(
                store,
                checkpoint_path,
                identity,
            )
            accumulator = EvaluationAccumulator.from_state(
                checkpoint.accumulator_state
            )
            reconstruction_counts = Counter(
                checkpoint.reconstruction_counts
            )
            completed = set(checkpoint.completed_shards)
            progress.update(accumulator.scenarios)
        else:
            accumulator = EvaluationAccumulator()
            reconstruction_counts = Counter()
            completed = set()
            write_checkpoint(
                checkpoint_path,
                EvaluationCheckpoint(
                    identity=identity,
                    completed_shards=[],
                    buffer_counts=store.snapshot_counts(),
                    accumulator_state=accumulator.to_state(),
                    reconstruction_counts={},
                ),
            )

        for path in paths:
            if (
                max_scenarios is not None
                and accumulator.scenarios >= max_scenarios
            ):
                break
            if str(path) in completed:
                continue
            remaining = (
                None
                if max_scenarios is None
                else max_scenarios - accumulator.scenarios
            )
            committed_counts = store.snapshot_counts()
            try:
                shard_is_complete = _shard_fits_limit(
                    path,
                    remaining,
                    metadata_by_path[str(path)],
                )
                _consume_shard(
                    path=path,
                    settings=settings,
                    remaining=remaining,
                    expected_metadata=metadata_by_path[str(path)],
                    workers=workers,
                    executor=executor,
                    accumulator=accumulator,
                    reconstruction_counts=reconstruction_counts,
                    store=store,
                    progress=progress,
                )
                if shard_is_complete:
                    store.flush_and_sync()
                    completed.add(str(path))
                    write_checkpoint(
                        checkpoint_path,
                        EvaluationCheckpoint(
                            identity=identity,
                            completed_shards=[
                                str(candidate)
                                for candidate in paths
                                if str(candidate) in completed
                            ],
                            buffer_counts=store.snapshot_counts(),
                            accumulator_state=accumulator.to_state(),
                            reconstruction_counts=dict(
                                reconstruction_counts
                            ),
                        ),
                    )
            except BaseException:
                store.truncate_to(committed_counts)
                raise

        run_config = _run_config_value(
            identity=identity,
            reconstruction_run_config=Path(
                args.reconstruction_run_config
            ).expanduser().resolve(),
            output_dir=output_dir,
            scratch_dir=scratch_dir,
            workers=workers,
            progress_every=progress_every,
            resume=bool(args.resume),
            keep_scratch=bool(args.keep_scratch),
        )
        outputs = write_final_outputs(
            output_dir=output_dir,
            accumulator=accumulator,
            store=store,
            reconstruction_counts=reconstruction_counts,
            run_config=run_config,
        )
        if not bool(args.keep_scratch):
            _delete_owned_scratch(scratch_dir)
        return outputs
    finally:
        progress.close()
        store.close()
        if executor is not None:
            executor.shutdown(
                wait=True,
                cancel_futures=True,
            )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun CatK's bundled batch reconstruction from original WOMD "
            "TFRecords and calculate exact raw/reconstructed metrics without "
            "saving reconstructed trajectories."
        )
    )
    parser.add_argument(
        "--input-path",
        nargs="+",
        required=True,
        metavar="ENTRY",
        help=(
            "One or more explicit TFRecord files, or a directory containing "
            "a complete canonical training shard set."
        ),
    )
    parser.add_argument(
        "--reconstruction-run-config",
        required=True,
        type=Path,
        metavar="PATH",
        help=(
            "run_config.json from the completed CatK batch reconstruction."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        metavar="PATH",
        help="Directory receiving the five final CSV/JSON files.",
    )
    parser.add_argument(
        "--scratch-dir",
        required=True,
        type=Path,
        metavar="PATH",
        help=(
            "Disk-backed exact float64 buffers and shard checkpoint directory."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Scenario reconstruction process count (default: 1).",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Optional positive global scenario limit for a smoke run.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help=(
            "Minimum progress refresh interval in scenarios; 0 disables "
            "the progress bar (default: 100)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last complete TFRecord-shard checkpoint.",
    )
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Retain exact scalar buffers after successful finalization.",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> None:
    outputs = run_evaluation(parse_args(argv))
    print("Exact reconstruction evaluation outputs:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
