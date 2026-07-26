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

import dataclasses
import importlib
import json
import math
import os
import re
import struct
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

os.environ.setdefault(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION",
    "python",
)

from .reconstruction_evaluation import (
    ScenarioMetricBatch,
    evaluate_scenario_pair,
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


@dataclass(frozen=True)
class ReconstructionSettings:
    """Batch settings copied from a completed reconstruction run."""

    method: str
    filter_strength: str
    max_gap_frames: int
    batch_linear_jerk_weight: float
    batch_angular_jerk_weight: float

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
) -> Iterator[tuple[int, bytes]]:
    """Read uncompressed TFRecord framing without importing TensorFlow."""

    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as stream:
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

    scenario_class = _load_scenario_class()
    scenario = scenario_class()
    scenario.ParseFromString(task.payload)
    scenario_id = str(scenario.scenario_id)
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
    return ScenarioEvaluationResult(
        source_file=task.source_file,
        record_index=task.record_index,
        scenario_id=scenario_id,
        metrics=metrics,
        reconstruction_counts=_integer_reconstruction_counts(
            reconstruction_stats
        ),
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
