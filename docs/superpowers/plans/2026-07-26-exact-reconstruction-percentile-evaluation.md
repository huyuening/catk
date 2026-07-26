# Exact Reconstruction Percentile Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable CatK CLI that reruns bundled batch reconstruction from original WOMD training TFRecords and emits exact p01/p99 agent-level and frame-level reconstruction metrics without saving reconstructed trajectories.

**Architecture:** Split the feature into a pure metric layer, an exact float64 scratch-buffer/checkpoint layer, and a CLI orchestration layer. Worker processes reconstruct one scenario in memory and return grouped scalar metric arrays; the parent appends them to type-specific binary buffers, checkpoints at shard boundaries, and uses memory-mapped in-place partition for exact percentiles.

**Tech Stack:** Python 3.11, NumPy, PyTorch-compatible WOMD protobufs, `concurrent.futures.ProcessPoolExecutor`, `tqdm`, CatK's bundled batch trajectory reconstruction, and `unittest`.

## Global Constraints

- Read the original WOMD TFRecords; never substitute `datasets/original/training` as the raw baseline.
- Require `method == "batch"` from the supplied reconstruction `run_config.json`.
- Reconstruct scenarios only in memory; never write reconstructed TFRecords, pickle caches, or trajectories.
- Calculate exact float64 `p01`, `p99`, and `p99_minus_p01`; do not use a quantile sketch.
- Preserve exact `count`, population `mean`, population `std`, `min`, `max`, and `range`.
- Report both agent-level and frame-level metrics for `all` and every encountered object type.
- Preserve the previous raw-matched-support and reconstructed-full-support definitions.
- Checkpoint only at complete TFRecord-shard boundaries and reject incompatible resume state.
- Successful output contains only `agent_summary.csv`, `frame_jerk_summary.csv`, `summary.json`, `reconstruction_summary.json`, and `run_config.json`.
- Add no third-party dependency.
- Do not modify or stage the user's existing `scripts/cache_womd.sh` change or unrelated untracked artifacts.

---

## File Structure

- Create `src/smart/tokens/reconstruction_evaluation.py`
  - Pure track/scenario metric calculation.
  - Metric schema, finite filtering, mergeable moments, and accumulator state.
- Create `src/smart/tokens/exact_metric_store.py`
  - Append-only float64 metric buffers.
  - Exact memory-mapped percentiles.
  - Atomic checkpoint state, validation, rollback, and resume.
- Create `src/smart/tokens/evaluate_trajectory_reconstruction.py`
  - CLI parsing, input/config resolution, TFRecord iteration, worker execution,
    shard orchestration, summary writing, and scratch cleanup.
- Create `tests/test_reconstruction_evaluation.py`
  - Metric stencil, support, RMSE, grouping, and moments.
- Create `tests/test_exact_metric_store.py`
  - Exact percentiles, global/type scopes, checkpointing, and truncation.
- Create `tests/test_evaluate_trajectory_reconstruction.py`
  - CLI configuration, TFRecord discovery, bounded execution, resume, and
    end-to-end output contract.
- Create `docs/reconstruction-evaluation.md`
  - Production and smoke commands, output interpretation, scratch sizing, and
    resume instructions.

---

### Task 1: Pure Reconstruction Metric Layer

**Files:**
- Create: `src/smart/tokens/reconstruction_evaluation.py`
- Create: `tests/test_reconstruction_evaluation.py`

**Interfaces:**
- Produces:
  - `MetricDefinition(metric: str, variant: str, unit: str, key: str)`
  - `AGENT_METRICS: tuple[MetricDefinition, ...]`
  - `FRAME_METRICS: tuple[MetricDefinition, ...]`
  - `RunningMoments.update_many(values: np.ndarray) -> None`
  - `RunningMoments.merge(other: RunningMoments) -> None`
  - `RunningMoments.to_state() -> dict`
  - `RunningMoments.from_state(state: Mapping) -> RunningMoments`
  - `ScenarioMetricBatch`
  - `evaluate_track(raw_track, reconstructed_track, timestamps) -> TrackMetricValues`
  - `evaluate_scenario_pair(raw_scenario, reconstructed_scenario) -> ScenarioMetricBatch`
  - `EvaluationAccumulator.add_batch(batch: ScenarioMetricBatch) -> None`
  - `EvaluationAccumulator.to_state() -> dict`
  - `EvaluationAccumulator.from_state(state: Mapping) -> EvaluationAccumulator`
- Consumes:
  - `compute_track_kinematics` from
    `src.smart.tokens.trajectory_filter_reconstructor`.

- [ ] **Step 1: Write failing moment and exact metric-schema tests**

Create `tests/test_reconstruction_evaluation.py` with finite filtering,
population standard deviation, serialization round-trip, and the exact schema:

```python
import copy
import unittest

import numpy as np

from src.smart.tokens.compare_trajectory_token_reconstruction import (
    _load_scenario_class,
)
from src.smart.tokens.reconstruction_evaluation import (
    AGENT_METRICS,
    FRAME_METRICS,
    RunningMoments,
    evaluate_scenario_pair,
    evaluate_track,
)


def build_track(count: int = 11):
    scenario_class = _load_scenario_class()
    scenario = scenario_class()
    scenario.scenario_id = "metric-test"
    scenario.current_time_index = min(10, count - 1)
    scenario.timestamps_seconds.extend(
        (np.arange(count, dtype=float) * 0.1).tolist()
    )
    track = scenario.tracks.add()
    track.id = 42
    track.object_type = 1
    for value in np.arange(count, dtype=float) * 0.1:
        state = track.states.add()
        state.center_x = float(value**3)
        state.center_y = float(0.5 * value**2)
        state.center_z = 0.0
        state.heading = float(0.2 * value**3)
        state.length = 4.5
        state.width = 1.8
        state.height = 1.5
        state.valid = True
    return scenario, track


class ReconstructionEvaluationTest(unittest.TestCase):
    def test_running_moments_filters_nonfinite_and_round_trips(self):
        moments = RunningMoments()
        moments.update_many(np.asarray([1.0, 2.0, 3.0, np.nan, np.inf]))
        restored = RunningMoments.from_state(moments.to_state())

        self.assertEqual(restored.count, 3)
        self.assertEqual(restored.mean, 2.0)
        self.assertAlmostEqual(restored.std, float(np.std([1.0, 2.0, 3.0])))
        self.assertEqual(restored.minimum, 1.0)
        self.assertEqual(restored.maximum, 3.0)

    def test_metric_schema_covers_agent_and_frame_variants(self):
        self.assertEqual(len(AGENT_METRICS), 9)
        self.assertEqual(len(FRAME_METRICS), 6)
        self.assertIn("xy_rmse_m", {definition.key for definition in AGENT_METRICS})
        self.assertIn(
            "reconstructed_full_angular_jerk_radps3",
            {definition.key for definition in FRAME_METRICS},
        )
```

- [ ] **Step 2: Run the focused test and verify the missing module failure**

Run:

```bash
python -m unittest discover -s tests \
  -p 'test_reconstruction_evaluation.py' -v
```

Expected: FAIL with `ModuleNotFoundError` for
`src.smart.tokens.reconstruction_evaluation`.

- [ ] **Step 3: Implement metric definitions and mergeable moments**

Create `src/smart/tokens/reconstruction_evaluation.py` with:

```python
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from .trajectory_filter_reconstructor import compute_track_kinematics


OBJECT_TYPE_NAMES = {
    0: "unset",
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
    4: "other",
}


@dataclass(frozen=True)
class MetricDefinition:
    metric: str
    variant: str
    unit: str
    key: str


AGENT_METRICS = (
    MetricDefinition("linear_jerk_rms", "raw_matched_support", "m/s^3",
                     "raw_linear_jerk_rms_mps3"),
    MetricDefinition("linear_jerk_rms", "reconstructed_matched_support", "m/s^3",
                     "reconstructed_linear_jerk_rms_mps3"),
    MetricDefinition("linear_jerk_rms", "reconstructed_full_support", "m/s^3",
                     "reconstructed_full_linear_jerk_rms_mps3"),
    MetricDefinition("linear_jerk_support_coverage", "reconstructed_vs_raw",
                     "fraction", "linear_jerk_matched_coverage"),
    MetricDefinition("angular_jerk_rms", "raw_matched_support", "rad/s^3",
                     "raw_angular_jerk_rms_radps3"),
    MetricDefinition("angular_jerk_rms", "reconstructed_matched_support",
                     "rad/s^3", "reconstructed_angular_jerk_rms_radps3"),
    MetricDefinition("angular_jerk_rms", "reconstructed_full_support",
                     "rad/s^3", "reconstructed_full_angular_jerk_rms_radps3"),
    MetricDefinition("angular_jerk_support_coverage", "reconstructed_vs_raw",
                     "fraction", "angular_jerk_matched_coverage"),
    MetricDefinition("xy_rmse", "reconstructed_vs_raw", "m", "xy_rmse_m"),
)

FRAME_METRICS = (
    MetricDefinition("linear_jerk", "raw_matched_support", "m/s^3",
                     "raw_linear_jerk_mps3"),
    MetricDefinition("linear_jerk", "reconstructed_matched_support", "m/s^3",
                     "reconstructed_linear_jerk_mps3"),
    MetricDefinition("linear_jerk", "reconstructed_full_support", "m/s^3",
                     "reconstructed_full_linear_jerk_mps3"),
    MetricDefinition("angular_jerk", "raw_matched_support", "rad/s^3",
                     "raw_angular_jerk_radps3"),
    MetricDefinition("angular_jerk", "reconstructed_matched_support", "rad/s^3",
                     "reconstructed_angular_jerk_radps3"),
    MetricDefinition("angular_jerk", "reconstructed_full_support", "rad/s^3",
                     "reconstructed_full_angular_jerk_radps3"),
)


@dataclass
class RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update_many(self, values: np.ndarray) -> None:
        finite = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            return
        batch_mean = float(np.mean(finite))
        self.merge(
            RunningMoments(
                count=len(finite),
                mean=batch_mean,
                m2=float(np.sum((finite - batch_mean) ** 2)),
                minimum=float(np.min(finite)),
                maximum=float(np.max(finite)),
            )
        )

    def merge(self, other: "RunningMoments") -> None:
        if not other.count:
            return
        if not self.count:
            self.count = other.count
            self.mean = other.mean
            self.m2 = other.m2
            self.minimum = other.minimum
            self.maximum = other.maximum
            return
        combined = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += (
            other.m2
            + delta * delta * self.count * other.count / combined
        )
        self.mean += delta * other.count / combined
        self.count = combined
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.m2 / self.count)) if self.count else math.nan

    def to_state(self) -> dict:
        return {
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "minimum": self.minimum if self.count else None,
            "maximum": self.maximum if self.count else None,
        }

    @classmethod
    def from_state(cls, state: Mapping) -> "RunningMoments":
        count = int(state["count"])
        return cls(
            count=count,
            mean=float(state["mean"]),
            m2=float(state["m2"]),
            minimum=float(state["minimum"]) if count else math.inf,
            maximum=float(state["maximum"]) if count else -math.inf,
        )
```

- [ ] **Step 4: Run the focused test and verify the foundational tests pass**

Run the same unittest discovery command.

Expected: the two foundational tests PASS.

- [ ] **Step 5: Add failing track, support, grouping, and RMSE tests**

Extend the test file:

```python
    def test_track_metrics_use_raw_support_and_xy_rmse(self):
        scenario, raw = build_track(count=25)
        reconstructed = copy.deepcopy(raw)
        for state in reconstructed.states:
            state.center_x += 3.0
            state.center_y += 4.0
        for index in range(11, 14):
            raw.states[index].valid = False

        evaluation = evaluate_track(
            raw, reconstructed, scenario.timestamps_seconds
        )

        self.assertAlmostEqual(evaluation.agent_values["xy_rmse_m"], 5.0)
        self.assertEqual(
            len(evaluation.frame_values["raw_linear_jerk_mps3"]),
            len(evaluation.frame_values["reconstructed_linear_jerk_mps3"]),
        )
        self.assertGreater(
            len(evaluation.frame_values[
                "reconstructed_full_linear_jerk_mps3"
            ]),
            len(evaluation.frame_values["raw_linear_jerk_mps3"]),
        )

    def test_scenario_pair_groups_values_by_object_type(self):
        raw_scenario, _ = build_track(count=25)
        reconstructed = copy.deepcopy(raw_scenario)

        batch = evaluate_scenario_pair(raw_scenario, reconstructed)

        self.assertEqual(batch.scenario_id, "metric-test")
        self.assertEqual(batch.agent_count, 1)
        self.assertEqual(set(batch.agent_values), {"vehicle"})
        self.assertEqual(batch.agent_values["vehicle"]["xy_rmse_m"].shape, (1,))
```

- [ ] **Step 6: Run the new tests and verify missing evaluation types fail**

Expected: FAIL because `TrackMetricValues`, `ScenarioMetricBatch`,
`evaluate_track`, and `evaluate_scenario_pair` are not yet implemented.

- [ ] **Step 7: Implement track/scenario evaluation and accumulator state**

Add:

```python
@dataclass(frozen=True)
class TrackMetricValues:
    object_type_name: str
    agent_values: dict[str, float]
    frame_values: dict[str, np.ndarray]


@dataclass(frozen=True)
class ScenarioMetricBatch:
    scenario_id: str
    agent_count: int
    agent_values: dict[str, dict[str, np.ndarray]]
    frame_values: dict[str, dict[str, np.ndarray]]


def _finite_rms(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite**2))) if len(finite) else math.nan


def _paired_samples(raw_values, reconstructed_values, raw_valid, reconstructed_valid):
    raw_support = np.asarray(raw_valid, dtype=bool) & np.isfinite(raw_values)
    reconstructed_support = (
        np.asarray(reconstructed_valid, dtype=bool)
        & np.isfinite(reconstructed_values)
    )
    matched = raw_support & reconstructed_support
    coverage = (
        float(np.sum(matched)) / float(np.sum(raw_support))
        if np.any(raw_support)
        else math.nan
    )
    if int(np.sum(matched)) != int(np.sum(raw_support)):
        paired_raw = np.empty(0, dtype=np.float64)
        paired_reconstructed = np.empty(0, dtype=np.float64)
    else:
        paired_raw = np.asarray(raw_values, dtype=np.float64)[raw_support]
        paired_reconstructed = np.asarray(
            reconstructed_values, dtype=np.float64
        )[raw_support]
    return (
        paired_raw,
        paired_reconstructed,
        np.asarray(reconstructed_values, dtype=np.float64)[reconstructed_support],
        coverage,
    )
```

Implement `evaluate_track` by calling `compute_track_kinematics` for both
tracks, applying `_paired_samples` separately to linear and angular jerk,
calculating paired/full RMS values, and calculating XY RMSE on frames valid and
finite in both tracks. Implement `evaluate_scenario_pair` by joining tracks on
integer track ID, concatenating frame arrays per stable object-type name, and
forming one-dimensional float64 agent arrays.

Add `EvaluationAccumulator` with nested `RunningMoments` for both the `all`
scope and the type scope, plus lossless `to_state`/`from_state` methods:

```python
@dataclass
class EvaluationAccumulator:
    scenarios: int = 0
    agents: int = 0
    agent_moments: dict[str, dict[str, RunningMoments]] = field(default_factory=dict)
    frame_moments: dict[str, dict[str, RunningMoments]] = field(default_factory=dict)

    def add_batch(self, batch: ScenarioMetricBatch) -> None:
        self.scenarios += 1
        self.agents += batch.agent_count
        for level, grouped in (
            ("agent", batch.agent_values),
            ("frame", batch.frame_values),
        ):
            target = self.agent_moments if level == "agent" else self.frame_moments
            for scope, metrics in grouped.items():
                for output_scope in ("all", scope):
                    scope_moments = target.setdefault(output_scope, {})
                    for key, values in metrics.items():
                        scope_moments.setdefault(key, RunningMoments()).update_many(values)
```

- [ ] **Step 8: Run the focused tests**

Expected: all reconstruction-evaluation tests PASS.

- [ ] **Step 9: Commit the pure metric layer**

```bash
git add \
  src/smart/tokens/reconstruction_evaluation.py \
  tests/test_reconstruction_evaluation.py
git commit -m "feat: add exact reconstruction metric layer"
```

---

### Task 2: Exact Float64 Metric Buffer and Percentiles

**Files:**
- Create: `src/smart/tokens/exact_metric_store.py`
- Create: `tests/test_exact_metric_store.py`

**Interfaces:**
- Consumes:
  - `ScenarioMetricBatch` from Task 1.
- Produces:
  - `BufferKey(level: str, scope: str, metric_key: str)`
  - `ExactPercentiles(p01: float | None, p99: float | None)`
  - `ExactMetricStore.append_batch(batch: ScenarioMetricBatch) -> None`
  - `ExactMetricStore.flush_and_sync() -> None`
  - `ExactMetricStore.snapshot_counts() -> dict[str, int]`
  - `ExactMetricStore.truncate_to(counts: Mapping[str, int]) -> None`
  - `ExactMetricStore.percentiles(key: BufferKey) -> ExactPercentiles`
  - `ExactMetricStore.combined_percentiles(level, metric_key, scopes) -> ExactPercentiles`
  - `ExactMetricStore.close() -> None`

- [ ] **Step 1: Write failing exact percentile tests**

Create `tests/test_exact_metric_store.py`:

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.smart.tokens.exact_metric_store import (
    BufferKey,
    ExactMetricStore,
)
from src.smart.tokens.reconstruction_evaluation import ScenarioMetricBatch


def metric_batch(scope: str, values: np.ndarray) -> ScenarioMetricBatch:
    return ScenarioMetricBatch(
        scenario_id="scenario",
        agent_count=len(values),
        agent_values={scope: {"xy_rmse_m": values}},
        frame_values={scope: {"raw_linear_jerk_mps3": values - 50.0}},
    )


class ExactMetricStoreTest(unittest.TestCase):
    def test_percentiles_match_numpy_linear_method(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExactMetricStore(Path(directory))
            values = np.arange(100, dtype=np.float64)
            store.append_batch(metric_batch("vehicle", values))
            store.flush_and_sync()
            actual = store.percentiles(
                BufferKey("agent", "vehicle", "xy_rmse_m")
            )
            expected = np.percentile(values, [1, 99], method="linear")
            store.close()

        self.assertEqual(actual.p01, expected[0])
        self.assertEqual(actual.p99, expected[1])

    def test_store_filters_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExactMetricStore(Path(directory))
            store.append_batch(
                metric_batch(
                    "vehicle",
                    np.asarray([1.0, np.nan, np.inf, 4.0]),
                )
            )
            counts = store.snapshot_counts()
            store.close()

        key = BufferKey("agent", "vehicle", "xy_rmse_m").encoded
        self.assertEqual(counts[key], 2)

    def test_combined_percentiles_use_every_type_without_duplicate_all_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExactMetricStore(Path(directory))
            store.append_batch(metric_batch("vehicle", np.asarray([0.0, 1.0])))
            store.append_batch(metric_batch("pedestrian", np.asarray([8.0, 9.0])))
            store.flush_and_sync()
            actual = store.combined_percentiles(
                "agent", "xy_rmse_m", ["pedestrian", "vehicle"]
            )
            store.close()

        expected = np.percentile([0.0, 1.0, 8.0, 9.0], [1, 99])
        np.testing.assert_allclose([actual.p01, actual.p99], expected)
```

- [ ] **Step 2: Run the focused test and verify the missing module failure**

Run:

```bash
python -m unittest discover -s tests -p 'test_exact_metric_store.py' -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement append-only buffer keys and writes**

Create `src/smart/tokens/exact_metric_store.py`:

```python
from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .reconstruction_evaluation import ScenarioMetricBatch


DTYPE = np.dtype("<f8")


@dataclass(frozen=True, order=True)
class BufferKey:
    level: str
    scope: str
    metric_key: str

    @property
    def encoded(self) -> str:
        return f"{self.level}|{self.scope}|{self.metric_key}"

    @classmethod
    def decode(cls, value: str) -> "BufferKey":
        level, scope, metric_key = value.split("|", 2)
        return cls(level, scope, metric_key)


@dataclass(frozen=True)
class ExactPercentiles:
    p01: float | None
    p99: float | None

    @property
    def p99_minus_p01(self) -> float | None:
        if self.p01 is None or self.p99 is None:
            return None
        return self.p99 - self.p01
```

Use paths of the form
`buffers/<level>/<scope>/<metric_key>.f64`. Keep one append handle per key,
write only finite contiguous little-endian float64 values, and count samples.
`append_batch` must append every type-specific agent and frame array without
creating an `all` buffer.

- [ ] **Step 4: Implement exact in-place linear percentiles**

Implement:

```python
def _linear_percentile(memmap: np.memmap, q: float) -> float:
    position = (len(memmap) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return float(memmap[lower] + fraction * (memmap[upper] - memmap[lower]))


def exact_percentiles(path: Path, count: int) -> ExactPercentiles:
    if count == 0:
        return ExactPercentiles(None, None)
    values = np.memmap(path, dtype=DTYPE, mode="r+", shape=(count,))
    positions = [(count - 1) * 0.01, (count - 1) * 0.99]
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
    p01 = _linear_percentile(values, 0.01)
    p99 = _linear_percentile(values, 0.99)
    values.flush()
    del values
    return ExactPercentiles(p01, p99)
```

Before reading, validate that file size equals `count * DTYPE.itemsize`.
Implement combined percentiles by copying the selected type buffers into one
temporary `combined/<level>/<metric_key>.f64`, finalizing it, and deleting it
in `finally`.

- [ ] **Step 5: Run the focused tests**

Expected: all three tests PASS.

- [ ] **Step 6: Add failing truncation and non-destructive count tests**

Add tests that snapshot counts, append more values, truncate to the snapshot,
reopen the store, and verify both file length and p01/p99 match the committed
prefix:

```python
    def test_truncate_discards_uncommitted_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ExactMetricStore(root)
            store.append_batch(metric_batch("vehicle", np.asarray([1.0, 2.0])))
            store.flush_and_sync()
            committed = store.snapshot_counts()
            store.append_batch(metric_batch("vehicle", np.asarray([100.0, 200.0])))
            store.flush_and_sync()
            store.truncate_to(committed)
            actual = store.percentiles(
                BufferKey("agent", "vehicle", "xy_rmse_m")
            )
            counts = store.snapshot_counts()
            store.close()

        key = BufferKey("agent", "vehicle", "xy_rmse_m").encoded
        self.assertEqual(counts[key], 2)
        np.testing.assert_allclose([actual.p01, actual.p99], [1.01, 1.99])
```

- [ ] **Step 7: Implement flush, fsync, reopen, and truncation**

`flush_and_sync` must flush Python handles and call `os.fsync`. `truncate_to`
must close active handles, remove files absent from the committed mapping,
truncate each retained file to its exact committed byte length, validate
alignment, restore counts, and lazily reopen handles on the next append.

- [ ] **Step 8: Run the focused tests**

Expected: all exact-store tests PASS.

- [ ] **Step 9: Commit exact metric storage**

```bash
git add \
  src/smart/tokens/exact_metric_store.py \
  tests/test_exact_metric_store.py
git commit -m "feat: store exact reconstruction percentiles"
```

---

### Task 3: Atomic Shard Checkpoint State

**Files:**
- Modify: `src/smart/tokens/exact_metric_store.py`
- Modify: `tests/test_exact_metric_store.py`

**Interfaces:**
- Consumes:
  - `EvaluationAccumulator.to_state()` and `.from_state()` from Task 1.
  - `ExactMetricStore.snapshot_counts()` and `.truncate_to()` from Task 2.
- Produces:
  - `CHECKPOINT_VERSION: int`
  - `EvaluationIdentity`
  - `EvaluationCheckpoint`
  - `write_checkpoint(path: Path, checkpoint: EvaluationCheckpoint) -> None`
  - `load_checkpoint(path: Path, expected: EvaluationIdentity) -> EvaluationCheckpoint`
  - `restore_checkpoint(store, path, expected) -> EvaluationCheckpoint`

- [ ] **Step 1: Write failing checkpoint identity and resume tests**

Append:

```python
from src.smart.tokens.exact_metric_store import (
    EvaluationCheckpoint,
    EvaluationIdentity,
    load_checkpoint,
    restore_checkpoint,
    write_checkpoint,
)


def identity(size: int = 100) -> EvaluationIdentity:
    return EvaluationIdentity(
        input_shards=(
            {
                "path": "/data/training.tfrecord-00000-of-01000",
                "size": size,
                "mtime_ns": 123,
            },
        ),
        reconstruction={
            "method": "batch",
            "filter_strength": "strong",
            "max_gap_frames": -1,
            "batch_linear_jerk_weight": 1.0,
            "batch_angular_jerk_weight": 1.0,
        },
        max_scenarios=None,
        metric_schema="exact-reconstruction-v1",
    )


class CheckpointTest(unittest.TestCase):
    def test_checkpoint_round_trip_and_identity_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            checkpoint = EvaluationCheckpoint(
                identity=identity(),
                completed_shards=["/data/training.tfrecord-00000-of-01000"],
                buffer_counts={},
                accumulator_state={"scenarios": 496, "agents": 10,
                                   "agent_moments": {}, "frame_moments": {}},
                reconstruction_counts={"total_tracks": 10},
            )
            write_checkpoint(path, checkpoint)
            restored = load_checkpoint(path, identity())

            self.assertEqual(restored.completed_shards, checkpoint.completed_shards)
            with self.assertRaisesRegex(ValueError, "identity"):
                load_checkpoint(path, identity(size=101))

    def test_restore_truncates_bytes_after_last_committed_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ExactMetricStore(root / "metrics")
            store.append_batch(metric_batch("vehicle", np.asarray([1.0, 2.0])))
            store.flush_and_sync()
            committed = store.snapshot_counts()
            checkpoint = EvaluationCheckpoint(
                identity=identity(),
                completed_shards=[
                    "/data/training.tfrecord-00000-of-01000"
                ],
                buffer_counts=committed,
                accumulator_state={"scenarios": 1, "agents": 2,
                                   "agent_moments": {}, "frame_moments": {}},
                reconstruction_counts={},
            )
            path = root / "checkpoint.json"
            write_checkpoint(path, checkpoint)
            store.append_batch(metric_batch("vehicle", np.asarray([99.0])))
            store.flush_and_sync()

            restored = restore_checkpoint(store, path, identity())
            counts = store.snapshot_counts()
            store.close()

        self.assertEqual(
            restored.completed_shards,
            ["/data/training.tfrecord-00000-of-01000"],
        )
        self.assertEqual(counts, committed)
```

- [ ] **Step 2: Run the tests and verify missing checkpoint types fail**

Expected: FAIL on missing imports.

- [ ] **Step 3: Implement canonical identity and checkpoint serialization**

Use frozen dataclasses whose `to_dict` methods emit JSON-safe lists and
dictionaries. `load_checkpoint` must:

- require the exact integer checkpoint version;
- parse and compare the complete `EvaluationIdentity`;
- require unique, ordered completed shard strings;
- require non-negative buffer counts;
- require accumulator and reconstruction mappings.

Write atomically:

```python
def write_checkpoint(path: Path, checkpoint: EvaluationCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(checkpoint.to_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
```

`restore_checkpoint` calls `load_checkpoint` and then
`store.truncate_to(checkpoint.buffer_counts)`.

- [ ] **Step 4: Run the focused tests**

Expected: all exact-store and checkpoint tests PASS.

- [ ] **Step 5: Commit atomic checkpoint support**

```bash
git add \
  src/smart/tokens/exact_metric_store.py \
  tests/test_exact_metric_store.py
git commit -m "feat: checkpoint reconstruction metric shards"
```

---

### Task 4: TFRecord, Configuration, and Scenario Worker CLI Layer

**Files:**
- Create: `src/smart/tokens/evaluate_trajectory_reconstruction.py`
- Create: `tests/test_evaluate_trajectory_reconstruction.py`

**Interfaces:**
- Consumes:
  - `evaluate_scenario_pair` from Task 1.
  - CatK `TrajectoryReconstructionConfig` and
    `reconstruct_scenario_for_vocabulary`.
- Produces:
  - `ReconstructionSettings`
  - `load_reconstruction_settings(path: Path) -> ReconstructionSettings`
  - `resolve_input_paths(entries: Sequence[str]) -> list[Path]`
  - `iter_tfrecord(path: Path) -> Iterator[tuple[int, bytes]]`
  - `count_tfrecord_records(path: Path, limit: int | None = None) -> int`
  - `ScenarioTask`
  - `ScenarioEvaluationResult`
  - `evaluate_scenario_task(task: ScenarioTask) -> ScenarioEvaluationResult`
  - `bounded_ordered_map(executor, function, tasks, limit) -> Iterator`

- [ ] **Step 1: Write failing input and configuration tests**

Create `tests/test_evaluate_trajectory_reconstruction.py`:

```python
import json
import struct
import tempfile
import unittest
from pathlib import Path

from src.smart.tokens.evaluate_trajectory_reconstruction import (
    count_tfrecord_records,
    load_reconstruction_settings,
    resolve_input_paths,
)


def write_tfrecord(path: Path, payloads: list[bytes]) -> None:
    with path.open("wb") as stream:
        for payload in payloads:
            stream.write(struct.pack("<Q", len(payload)))
            stream.write(b"\0" * 4)
            stream.write(payload)
            stream.write(b"\0" * 4)


class EvaluationCliInputTest(unittest.TestCase):
    def test_loads_only_batch_reconstruction_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            path.write_text(json.dumps({
                "method": "batch",
                "filter_strength": "strong",
                "max_gap_frames": -1,
                "batch_linear_jerk_weight": 1.0,
                "batch_angular_jerk_weight": 2.0,
                "num_workers": 99,
            }))
            settings = load_reconstruction_settings(path)

        self.assertEqual(settings.method, "batch")
        self.assertEqual(settings.batch_angular_jerk_weight, 2.0)

    def test_rejects_non_batch_run_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            path.write_text(json.dumps({
                "method": "filter",
                "filter_strength": "strong",
                "max_gap_frames": -1,
                "batch_linear_jerk_weight": 1.0,
                "batch_angular_jerk_weight": 1.0,
            }))
            with self.assertRaisesRegex(ValueError, "method.*batch"):
                load_reconstruction_settings(path)

    def test_resolves_canonical_training_shards_and_counts_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "training.tfrecord-00000-of-00002"
            second = root / "training.tfrecord-00001-of-00002"
            write_tfrecord(first, [b"a", b"bc"])
            write_tfrecord(second, [b"def"])
            (root / "training.tfrecord-00000-of-00002-new").touch()

            resolved = resolve_input_paths([str(root)])

            self.assertEqual(resolved, [first.resolve(), second.resolve()])
            self.assertEqual(count_tfrecord_records(first), 2)
```

- [ ] **Step 2: Run the focused tests and verify the missing module failure**

Run:

```bash
python -m unittest discover -s tests \
  -p 'test_evaluate_trajectory_reconstruction.py' -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement CLI parsing, configuration, and TFRecord helpers**

Create `src/smart/tokens/evaluate_trajectory_reconstruction.py` with:

```python
@dataclass(frozen=True)
class ReconstructionSettings:
    method: str
    filter_strength: str
    max_gap_frames: int
    batch_linear_jerk_weight: float
    batch_angular_jerk_weight: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_reconstruction_config(self) -> TrajectoryReconstructionConfig:
        return TrajectoryReconstructionConfig(
            method="batch",
            filter_strength=self.filter_strength,
            max_gap_frames=self.max_gap_frames,
            batch_linear_jerk_weight=self.batch_linear_jerk_weight,
            batch_angular_jerk_weight=self.batch_angular_jerk_weight,
        )
```

Validate required keys, finite non-negative jerk weights, supported filter
strength, and `max_gap_frames >= -1`. Resolve only canonical
`*.tfrecord-#####-of-#####` shard names or an explicitly named file; reject
audit copies such as `-new`. Port strict TFRecord length/CRC-field boundary
reading and counting without adding TensorFlow as a dependency.

- [ ] **Step 4: Run input/configuration tests**

Expected: all three tests PASS.

- [ ] **Step 5: Add failing scenario-worker and bounded-order tests**

Build one 91-frame synthetic scenario with a cubic trajectory, serialize it,
and assert the worker returns non-empty agent/frame metrics and reconstruction
counters. Assert that no file appears in the temporary directory. Also test a
fake executor whose futures complete in reverse order while
`bounded_ordered_map` yields input order and never exceeds the requested
pending limit.

The worker assertion must include:

```python
self.assertEqual(result.metrics.scenario_id, "worker-test")
self.assertEqual(result.metrics.agent_count, 1)
self.assertIn("vehicle", result.metrics.frame_values)
self.assertGreater(result.reconstruction_counts["processed_tracks"], 0)
self.assertEqual(list(root.iterdir()), [])
```

- [ ] **Step 6: Run the tests and verify missing worker behavior fails**

Expected: FAIL because task/result types, reconstruction worker, and bounded
ordered mapping are absent.

- [ ] **Step 7: Implement the in-memory scenario worker**

Define:

```python
@dataclass(frozen=True)
class ScenarioTask:
    source_file: str
    record_index: int
    payload: bytes
    settings: ReconstructionSettings


@dataclass(frozen=True)
class ScenarioEvaluationResult:
    source_file: str
    record_index: int
    scenario_id: str
    metrics: ScenarioMetricBatch
    reconstruction_counts: dict[str, int]
```

The worker must parse a fresh Scenario, call
`reconstruct_scenario_for_vocabulary`, evaluate the original/reconstructed
pair, convert only integer reconstruction counters, and return the compact
metric batch. It must not expose the reconstructed Scenario in its return
value or accept an output path.

Implement `bounded_ordered_map` with a `collections.deque`: submit up to
`limit`, wait on and yield the leftmost future, then submit exactly one
replacement. This preserves deterministic order and limits serialized metric
arrays in flight.

- [ ] **Step 8: Run the focused CLI-layer tests**

Expected: all tests PASS.

- [ ] **Step 9: Commit the CLI worker layer**

```bash
git add \
  src/smart/tokens/evaluate_trajectory_reconstruction.py \
  tests/test_evaluate_trajectory_reconstruction.py
git commit -m "feat: evaluate reconstructed WOMD scenarios in memory"
```

---

### Task 5: Shard Orchestration, Resume, and Final Outputs

**Files:**
- Modify: `src/smart/tokens/evaluate_trajectory_reconstruction.py`
- Modify: `src/smart/tokens/reconstruction_evaluation.py`
- Modify: `tests/test_evaluate_trajectory_reconstruction.py`

**Interfaces:**
- Consumes:
  - Tasks 1--4 public interfaces.
- Produces:
  - `build_evaluation_identity(...) -> EvaluationIdentity`
  - `run_evaluation(args: argparse.Namespace) -> tuple[Path, ...]`
  - `summary_rows(accumulator, percentiles, level) -> list[dict]`
  - `write_final_outputs(...) -> tuple[Path, ...]`
  - module `main()`.

- [ ] **Step 1: Write failing summary schema and atomic output tests**

Add a test that creates an accumulator and exact store from known vehicle and
pedestrian batches, calls the final-output helper, and asserts:

```python
expected_fields = {
    "scope", "level", "metric", "variant", "unit", "count",
    "mean", "std", "min", "max", "range",
    "p01", "p99", "p99_minus_p01",
}
self.assertEqual(set(agent_rows[0]), expected_fields)
self.assertAlmostEqual(
    float(overall_xy["p99_minus_p01"]),
    float(overall_xy["p99"]) - float(overall_xy["p01"]),
)
self.assertEqual(
    sorted(path.name for path in output_dir.iterdir()),
    [
        "agent_summary.csv",
        "frame_jerk_summary.csv",
        "reconstruction_summary.json",
        "run_config.json",
        "summary.json",
    ],
)
```

- [ ] **Step 2: Run and verify missing output functions fail**

Expected: FAIL on missing imports.

- [ ] **Step 3: Implement summary rows and atomic final files**

Extend `RunningMoments` without importing the storage module back into the
pure metric layer:

```python
def summary(
    self,
    *,
    p01: float | None,
    p99: float | None,
) -> dict:
    if not self.count:
        return {
            "count": 0, "mean": None, "std": None,
            "min": None, "max": None, "range": None,
            "p01": None, "p99": None, "p99_minus_p01": None,
        }
    return {
        "count": self.count,
        "mean": self.mean,
        "std": self.std,
        "min": self.minimum,
        "max": self.maximum,
        "range": self.maximum - self.minimum,
        "p01": p01,
        "p99": p99,
        "p99_minus_p01": (
            p99 - p01 if p01 is not None and p99 is not None else None
        ),
    }
```

Build stable scope order (`all`, then sorted type names), definition order from
`AGENT_METRICS`/`FRAME_METRICS`, and write every CSV/JSON through a sibling
`.tmp` file followed by `os.replace`. Include support definitions, percentile
method, `ddof=0`, counts, reconstruction provenance, and derived reconstruction
rates.

- [ ] **Step 4: Run summary/output tests**

Expected: PASS.

- [ ] **Step 5: Write failing shard commit and resume integration test**

Create two synthetic TFRecord shards. Patch `evaluate_scenario_task` so the
first run completes shard zero and raises during shard one. Verify the
checkpoint records only shard zero. Rerun with `--resume`, count worker calls,
and assert shard zero is skipped while shard one completes. Also assert that
changing the reconstruction weight or input file size causes identity
rejection.

- [ ] **Step 6: Run and verify orchestration test fails**

Expected: FAIL because `run_evaluation` and shard checkpoint orchestration are
not implemented.

- [ ] **Step 7: Implement production orchestration**

Implement this sequence:

```python
def run_evaluation(args):
    paths = resolve_input_paths(args.input_path)
    settings = load_reconstruction_settings(args.reconstruction_run_config)
    identity = build_evaluation_identity(paths, settings, args.max_scenarios)
    store = ExactMetricStore(args.scratch_dir / "metrics")
    checkpoint_path = args.scratch_dir / "checkpoint.json"

    if args.resume and checkpoint_path.is_file():
        checkpoint = restore_checkpoint(store, checkpoint_path, identity)
        accumulator = EvaluationAccumulator.from_state(
            checkpoint.accumulator_state
        )
        reconstruction_counts = Counter(checkpoint.reconstruction_counts)
        completed = set(checkpoint.completed_shards)
    else:
        reject_nonempty_scratch_without_resume(args.scratch_dir)
        accumulator = EvaluationAccumulator()
        reconstruction_counts = Counter()
        completed = set()

    for path in paths:
        if str(path) in completed:
            continue
        committed_counts = store.snapshot_counts()
        try:
            consume_complete_shard(path, settings, args, accumulator,
                                   reconstruction_counts, store)
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
                    reconstruction_counts=dict(reconstruction_counts),
                ),
            )
        except BaseException:
            store.truncate_to(committed_counts)
            raise
```

Respect the global `--max-scenarios` count, including a partial final shard.
Use workers directly when `workers == 1`; otherwise use one process pool for
the run and `bounded_ordered_map` with `limit=workers * 2`. Update `tqdm` by
completed scenarios.

After all work:

1. close append handles;
2. compute type and combined exact percentiles;
3. validate moment counts equal exact-store counts;
4. atomically write all five final outputs;
5. delete scratch unless `--keep-scratch`;
6. print resolved output paths.

- [ ] **Step 8: Add CLI arguments and `main()`**

Expose:

```text
--input-path ENTRY [ENTRY ...]        required
--reconstruction-run-config PATH      required
--output-dir PATH                     required
--scratch-dir PATH                    required
--workers N                           default 1
--max-scenarios N                     default unrestricted
--progress-every N                    default 100
--resume                              flag
--keep-scratch                        flag
```

Reject:

- non-positive workers;
- non-positive `max_scenarios`;
- negative `progress_every`;
- output and scratch resolving to the same directory;
- an output directory containing files outside the five known final names;
- a full run whose identity differs from a retained checkpoint.

- [ ] **Step 9: Run all three feature test files**

Run:

```bash
python -m unittest discover -s tests \
  -p 'test_*reconstruction_evaluation*.py' -v
python -m unittest discover -s tests \
  -p 'test_exact_metric_store.py' -v
python -m unittest discover -s tests \
  -p 'test_evaluate_trajectory_reconstruction.py' -v
```

Expected: all PASS.

- [ ] **Step 10: Commit orchestration and outputs**

```bash
git add \
  src/smart/tokens/reconstruction_evaluation.py \
  src/smart/tokens/evaluate_trajectory_reconstruction.py \
  tests/test_evaluate_trajectory_reconstruction.py
git commit -m "feat: run resumable exact reconstruction evaluation"
```

---

### Task 6: User Documentation and Full Verification

**Files:**
- Create: `docs/reconstruction-evaluation.md`
- Modify: `tests/test_evaluate_trajectory_reconstruction.py`

**Interfaces:**
- Consumes the final CLI from Task 5.
- Produces the remote smoke/full commands and output interpretation.

- [ ] **Step 1: Add a failing CLI help/output-contract test**

Use `subprocess.run` to invoke:

```bash
python -m src.smart.tokens.evaluate_trajectory_reconstruction --help
```

Assert exit code zero and that help includes
`--reconstruction-run-config`, `--scratch-dir`, `--resume`,
`--keep-scratch`, and `--max-scenarios`.

- [ ] **Step 2: Run the help test and fix any public CLI mismatch**

Expected: PASS after the public names exactly match the design.

- [ ] **Step 3: Write production documentation**

Create `docs/reconstruction-evaluation.md` with this smoke command:

```bash
cd /root/workspace/catk

export WOMD_TRAIN=/mnt/pfs/waymo_motion_1_3_0/uncompressed/scenario/training
export RECON_OUTPUT=/mnt/pfs/waymo_motion_1_3_0/catk_batch_vocab_v1
export METRIC_OUTPUT="$RECON_OUTPUT/exact_reconstruction_metrics_smoke"
export METRIC_SCRATCH="$RECON_OUTPUT/.exact_metric_scratch_smoke"

python -m src.smart.tokens.evaluate_trajectory_reconstruction \
  --input-path "$WOMD_TRAIN" \
  --reconstruction-run-config "$RECON_OUTPUT/run_config.json" \
  --output-dir "$METRIC_OUTPUT" \
  --scratch-dir "$METRIC_SCRATCH" \
  --workers 8 \
  --max-scenarios 16
```

Document the unrestricted run:

```bash
export METRIC_OUTPUT="$RECON_OUTPUT/exact_reconstruction_metrics"
export METRIC_SCRATCH="$RECON_OUTPUT/.exact_metric_scratch"

python -m src.smart.tokens.evaluate_trajectory_reconstruction \
  --input-path "$WOMD_TRAIN" \
  --reconstruction-run-config "$RECON_OUTPUT/run_config.json" \
  --output-dir "$METRIC_OUTPUT" \
  --scratch-dir "$METRIC_SCRATCH" \
  --workers 24 \
  --resume
```

Explain:

- scratch requires approximately 70--90 GB;
- rerunning the same full command resumes committed shards;
- `p01`/`p99` use exact float64 linear interpolation;
- matched and full support differences;
- signed frame jerk versus non-negative per-agent RMS;
- scratch auto-deletion on success and retention on failure;
- no reconstructed data is written.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m unittest discover -s tests \
  -p 'test_reconstruction_evaluation.py' -v
python -m unittest discover -s tests \
  -p 'test_exact_metric_store.py' -v
python -m unittest discover -s tests \
  -p 'test_evaluate_trajectory_reconstruction.py' -v
```

Expected: all PASS.

- [ ] **Step 5: Run the complete CatK test suite**

Run:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Expected: all tests PASS with only the repository's established optional
dependency skips.

- [ ] **Step 6: Run static and repository checks**

Run:

```bash
python -m src.smart.tokens.evaluate_trajectory_reconstruction --help
python -m compileall \
  src/smart/tokens/reconstruction_evaluation.py \
  src/smart/tokens/exact_metric_store.py \
  src/smart/tokens/evaluate_trajectory_reconstruction.py
git diff --check
git status --short
```

Confirm that no reconstructed TFRecord, `.pkl`, metric scratch file, slide
artifact, `.DS_Store`, or `scripts/cache_womd.sh` is staged.

- [ ] **Step 7: Commit documentation**

```bash
git add \
  docs/reconstruction-evaluation.md \
  tests/test_evaluate_trajectory_reconstruction.py
git commit -m "docs: explain exact reconstruction evaluation"
```

- [ ] **Step 8: Review the final commit range**

Run:

```bash
git log --oneline --decorate -8
PLAN_COMMIT="$(git log -1 --format=%H -- \
  docs/superpowers/plans/2026-07-26-exact-reconstruction-percentile-evaluation.md)"
git diff --stat "${PLAN_COMMIT}..HEAD"
git status --short --branch
```

Expected: the feature commits contain only the three implementation modules,
three test files, and the reconstruction-evaluation documentation; unrelated
user changes remain unstaged.
