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
