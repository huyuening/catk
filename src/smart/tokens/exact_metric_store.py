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

"""Disk-backed exact percentile storage for reconstruction metrics."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping, Sequence

import numpy as np

from .reconstruction_evaluation import ScenarioMetricBatch


DTYPE = np.dtype("<f8")
CHECKPOINT_VERSION = 1
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True, order=True)
class BufferKey:
    """Identity of one type-specific scalar metric stream."""

    level: str
    scope: str
    metric_key: str

    def __post_init__(self) -> None:
        if self.level not in ("agent", "frame"):
            raise ValueError("buffer level must be 'agent' or 'frame'")
        for label, value in (
            ("scope", self.scope),
            ("metric_key", self.metric_key),
        ):
            if not value or _SAFE_COMPONENT.fullmatch(value) is None:
                raise ValueError(
                    f"buffer {label} contains unsafe characters: {value!r}"
                )

    @property
    def encoded(self) -> str:
        return f"{self.level}|{self.scope}|{self.metric_key}"

    @classmethod
    def decode(cls, value: str) -> "BufferKey":
        parts = value.split("|")
        if len(parts) != 3:
            raise ValueError(f"invalid encoded buffer key: {value!r}")
        return cls(*parts)


@dataclass(frozen=True)
class ExactPercentiles:
    """Exact first and ninety-ninth linear percentiles."""

    p01: float | None
    p99: float | None

    @property
    def p99_minus_p01(self) -> float | None:
        if self.p01 is None or self.p99 is None:
            return None
        return self.p99 - self.p01


def _require_integer(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _copy_json_mapping(
    value: object,
    label: str,
) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    try:
        encoded = json.dumps(
            dict(value),
            allow_nan=False,
            sort_keys=True,
        )
        copied = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{label} must contain finite JSON values"
        ) from error
    if not isinstance(copied, dict):
        raise ValueError(f"{label} must be a mapping")
    return copied


@dataclass(frozen=True)
class EvaluationIdentity:
    """Complete input and configuration identity for safe resume."""

    input_shards: tuple[dict, ...]
    reconstruction: dict
    max_scenarios: int | None
    metric_schema: str

    def __post_init__(self) -> None:
        shards: list[dict] = []
        for index, raw_shard in enumerate(self.input_shards):
            shard = _copy_json_mapping(
                raw_shard,
                f"input_shards[{index}]",
            )
            if set(shard) != {"path", "size", "mtime_ns"}:
                raise ValueError(
                    "each input shard must contain exactly path, size, "
                    "and mtime_ns"
                )
            if not isinstance(shard["path"], str) or not shard["path"]:
                raise ValueError("input shard path must be a non-empty string")
            shard["size"] = _require_integer(
                shard["size"],
                "input shard size",
                minimum=0,
            )
            shard["mtime_ns"] = _require_integer(
                shard["mtime_ns"],
                "input shard mtime_ns",
                minimum=0,
            )
            shards.append(shard)
        paths = [shard["path"] for shard in shards]
        if len(paths) != len(set(paths)):
            raise ValueError("input shard paths must be unique")
        reconstruction = _copy_json_mapping(
            self.reconstruction,
            "reconstruction identity",
        )
        max_scenarios = self.max_scenarios
        if max_scenarios is not None:
            max_scenarios = _require_integer(
                max_scenarios,
                "max_scenarios",
                minimum=1,
            )
        if (
            not isinstance(self.metric_schema, str)
            or not self.metric_schema
        ):
            raise ValueError("metric_schema must be a non-empty string")
        object.__setattr__(self, "input_shards", tuple(shards))
        object.__setattr__(self, "reconstruction", reconstruction)
        object.__setattr__(self, "max_scenarios", max_scenarios)

    def to_dict(self) -> dict:
        return {
            "input_shards": [dict(shard) for shard in self.input_shards],
            "reconstruction": dict(self.reconstruction),
            "max_scenarios": self.max_scenarios,
            "metric_schema": self.metric_schema,
        }

    @classmethod
    def from_dict(cls, value: object) -> "EvaluationIdentity":
        if not isinstance(value, Mapping):
            raise ValueError("checkpoint identity must be a mapping")
        required = {
            "input_shards",
            "reconstruction",
            "max_scenarios",
            "metric_schema",
        }
        if set(value) != required:
            raise ValueError(
                "checkpoint identity has missing or unexpected fields"
            )
        raw_shards = value["input_shards"]
        if (
            not isinstance(raw_shards, list)
            or any(not isinstance(item, Mapping) for item in raw_shards)
        ):
            raise ValueError(
                "checkpoint identity input_shards must be a list of mappings"
            )
        return cls(
            input_shards=tuple(dict(item) for item in raw_shards),
            reconstruction=_copy_json_mapping(
                value["reconstruction"],
                "checkpoint reconstruction identity",
            ),
            max_scenarios=value["max_scenarios"],
            metric_schema=value["metric_schema"],
        )


@dataclass(frozen=True)
class EvaluationCheckpoint:
    """Durable state committed after a complete TFRecord shard."""

    identity: EvaluationIdentity
    completed_shards: list[str]
    buffer_counts: dict[str, int]
    accumulator_state: dict
    reconstruction_counts: dict

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EvaluationIdentity):
            raise ValueError("checkpoint identity is invalid")
        completed = list(self.completed_shards)
        if any(not isinstance(path, str) or not path for path in completed):
            raise ValueError(
                "completed shard paths must be non-empty strings"
            )
        if len(completed) != len(set(completed)):
            raise ValueError("completed shard paths must be unique")
        input_paths = [
            shard["path"]
            for shard in self.identity.input_shards
        ]
        if completed != input_paths[: len(completed)]:
            raise ValueError(
                "completed shard paths must be an ordered input prefix"
            )

        if not isinstance(self.buffer_counts, Mapping):
            raise ValueError("buffer_counts must be a mapping")
        buffer_counts: dict[str, int] = {}
        for raw_key, raw_count in self.buffer_counts.items():
            if not isinstance(raw_key, str):
                raise ValueError("buffer count keys must be strings")
            BufferKey.decode(raw_key)
            buffer_counts[raw_key] = _require_integer(
                raw_count,
                f"buffer count for {raw_key!r}",
                minimum=0,
            )
        accumulator_state = _copy_json_mapping(
            self.accumulator_state,
            "accumulator_state",
        )
        reconstruction_counts = _copy_json_mapping(
            self.reconstruction_counts,
            "reconstruction_counts",
        )
        for key, value in reconstruction_counts.items():
            _require_integer(
                value,
                f"reconstruction count for {key!r}",
                minimum=0,
            )

        object.__setattr__(self, "completed_shards", completed)
        object.__setattr__(self, "buffer_counts", buffer_counts)
        object.__setattr__(
            self,
            "accumulator_state",
            accumulator_state,
        )
        object.__setattr__(
            self,
            "reconstruction_counts",
            reconstruction_counts,
        )

    def to_dict(self) -> dict:
        return {
            "version": CHECKPOINT_VERSION,
            "identity": self.identity.to_dict(),
            "completed_shards": list(self.completed_shards),
            "buffer_counts": dict(self.buffer_counts),
            "accumulator_state": dict(self.accumulator_state),
            "reconstruction_counts": dict(self.reconstruction_counts),
        }


def _checkpoint_from_dict(
    value: object,
) -> EvaluationCheckpoint:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint must be a JSON object")
    required = {
        "version",
        "identity",
        "completed_shards",
        "buffer_counts",
        "accumulator_state",
        "reconstruction_counts",
    }
    if set(value) != required:
        raise ValueError(
            "checkpoint has missing or unexpected fields"
        )
    version = _require_integer(
        value["version"],
        "checkpoint version",
    )
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {version}; "
            f"expected {CHECKPOINT_VERSION}"
        )
    completed = value["completed_shards"]
    if not isinstance(completed, list):
        raise ValueError("completed_shards must be a list")
    return EvaluationCheckpoint(
        identity=EvaluationIdentity.from_dict(value["identity"]),
        completed_shards=completed,
        buffer_counts=value["buffer_counts"],
        accumulator_state=value["accumulator_state"],
        reconstruction_counts=value["reconstruction_counts"],
    )


def write_checkpoint(
    path: Path,
    checkpoint: EvaluationCheckpoint,
) -> None:
    """Atomically replace a checkpoint after synchronizing its contents."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                checkpoint.to_dict(),
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    expected: EvaluationIdentity,
) -> EvaluationCheckpoint:
    """Load a checkpoint and reject any identity mismatch."""

    with Path(path).open("r", encoding="utf-8") as stream:
        try:
            value = json.load(stream)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid checkpoint JSON: {path}") from error
    checkpoint = _checkpoint_from_dict(value)
    if checkpoint.identity != expected:
        raise ValueError(
            "checkpoint identity does not match the current evaluation"
        )
    return checkpoint


def restore_checkpoint(
    store: "ExactMetricStore",
    path: Path,
    expected: EvaluationIdentity,
) -> EvaluationCheckpoint:
    """Load committed state and discard every uncommitted buffer suffix."""

    checkpoint = load_checkpoint(path, expected)
    store.truncate_to(checkpoint.buffer_counts)
    return checkpoint


def _linear_value(
    values: np.memmap,
    position: float,
) -> float:
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return float(
        values[lower]
        + fraction * (values[upper] - values[lower])
    )


def exact_percentiles(
    path: Path,
    count: int,
) -> ExactPercentiles:
    """Return exact NumPy-linear p01/p99 from a float64 binary file."""

    if count < 0:
        raise ValueError("metric count must be non-negative")
    if count == 0:
        return ExactPercentiles(None, None)
    expected_size = count * DTYPE.itemsize
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{path}: expected {expected_size} bytes for {count} samples, "
            f"found {actual_size}"
        )

    values = np.memmap(
        path,
        dtype=DTYPE,
        mode="r+",
        shape=(count,),
    )
    positions = (
        (count - 1) * 0.01,
        (count - 1) * 0.99,
    )
    kth = sorted(
        {
            int(math.floor(position))
            for position in positions
        }
        | {
            int(math.ceil(position))
            for position in positions
        }
    )
    values.partition(kth)
    result = ExactPercentiles(
        p01=_linear_value(values, positions[0]),
        p99=_linear_value(values, positions[1]),
    )
    values.flush()
    del values
    return result


class ExactMetricStore:
    """Append finite metric arrays and finalize their exact percentiles."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.buffer_root = self.root / "buffers"
        self.buffer_root.mkdir(parents=True, exist_ok=True)
        self._handles: dict[BufferKey, BinaryIO] = {}
        self._counts: dict[BufferKey, int] = {}
        self._discover_existing_buffers()

    def _discover_existing_buffers(self) -> None:
        for path in sorted(self.buffer_root.rglob("*.f64")):
            relative = path.relative_to(self.buffer_root)
            if len(relative.parts) != 3:
                raise ValueError(
                    f"unexpected metric buffer path: {path}"
                )
            level, scope, filename = relative.parts
            key = BufferKey(level, scope, Path(filename).stem)
            size = path.stat().st_size
            if size % DTYPE.itemsize:
                raise ValueError(
                    f"{path}: byte length is not float64-aligned"
                )
            self._counts[key] = size // DTYPE.itemsize

    def _path(self, key: BufferKey) -> Path:
        return (
            self.buffer_root
            / key.level
            / key.scope
            / f"{key.metric_key}.f64"
        )

    def _handle(self, key: BufferKey) -> BinaryIO:
        handle = self._handles.get(key)
        if handle is None:
            path = self._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("ab")
            self._handles[key] = handle
        return handle

    def _append(
        self,
        key: BufferKey,
        values: np.ndarray,
    ) -> None:
        finite = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = finite[np.isfinite(finite)]
        if len(finite) == 0:
            return
        encoded = np.ascontiguousarray(finite, dtype=DTYPE)
        self._handle(key).write(encoded.tobytes(order="C"))
        self._counts[key] = self._counts.get(key, 0) + len(encoded)

    def append_batch(self, batch: ScenarioMetricBatch) -> None:
        for level, grouped in (
            ("agent", batch.agent_values),
            ("frame", batch.frame_values),
        ):
            for scope, metrics in grouped.items():
                for metric_key, values in metrics.items():
                    self._append(
                        BufferKey(level, scope, metric_key),
                        values,
                    )

    def flush_and_sync(self) -> None:
        for handle in self._handles.values():
            handle.flush()
            os.fsync(handle.fileno())

    def _close_handle(self, key: BufferKey) -> None:
        handle = self._handles.pop(key, None)
        if handle is not None:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    def close(self) -> None:
        for key in list(self._handles):
            self._close_handle(key)

    def snapshot_counts(self) -> dict[str, int]:
        return {
            key.encoded: int(count)
            for key, count in sorted(self._counts.items())
        }

    def truncate_to(self, counts: Mapping[str, int]) -> None:
        target: dict[BufferKey, int] = {}
        for encoded, raw_count in counts.items():
            key = BufferKey.decode(str(encoded))
            count = int(raw_count)
            if count < 0:
                raise ValueError(
                    f"negative metric count for {encoded!r}"
                )
            target[key] = count

        self.close()
        for path in sorted(self.buffer_root.rglob("*.f64")):
            relative = path.relative_to(self.buffer_root)
            if len(relative.parts) != 3:
                raise ValueError(
                    f"unexpected metric buffer path: {path}"
                )
            key = BufferKey(
                relative.parts[0],
                relative.parts[1],
                Path(relative.parts[2]).stem,
            )
            if key not in target:
                path.unlink()

        for key, count in target.items():
            path = self._path(key)
            expected_size = count * DTYPE.itemsize
            if not path.is_file():
                if count == 0:
                    continue
                raise FileNotFoundError(path)
            actual_size = path.stat().st_size
            if actual_size < expected_size:
                raise ValueError(
                    f"{path}: committed size {expected_size} exceeds "
                    f"available size {actual_size}"
                )
            with path.open("r+b") as stream:
                stream.truncate(expected_size)
                stream.flush()
                os.fsync(stream.fileno())
        self._counts = target

    def percentiles(
        self,
        key: BufferKey,
    ) -> ExactPercentiles:
        self._close_handle(key)
        count = self._counts.get(key, 0)
        return exact_percentiles(self._path(key), count)

    def combined_percentiles(
        self,
        level: str,
        metric_key: str,
        scopes: Sequence[str],
    ) -> ExactPercentiles:
        keys = [
            BufferKey(level, scope, metric_key)
            for scope in sorted(set(scopes))
        ]
        for key in keys:
            self._close_handle(key)
        count = sum(self._counts.get(key, 0) for key in keys)
        if count == 0:
            return ExactPercentiles(None, None)

        combined_directory = self.root / "combined"
        combined_directory.mkdir(parents=True, exist_ok=True)
        combined_path = (
            combined_directory
            / f"{level}.{metric_key}.f64"
        )
        try:
            with combined_path.open("wb") as output:
                for key in keys:
                    source_path = self._path(key)
                    if self._counts.get(key, 0) == 0:
                        continue
                    with source_path.open("rb") as source:
                        shutil.copyfileobj(
                            source,
                            output,
                            length=1024 * 1024,
                        )
                output.flush()
                os.fsync(output.fileno())
            return exact_percentiles(combined_path, count)
        finally:
            combined_path.unlink(missing_ok=True)
            try:
                combined_directory.rmdir()
            except OSError:
                pass
