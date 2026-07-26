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

"""Pure metrics for raw-versus-reconstructed WOMD trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

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
    """Stable output metadata for one scalar metric stream."""

    metric: str
    variant: str
    unit: str
    key: str


AGENT_METRICS = (
    MetricDefinition(
        "linear_jerk_rms",
        "raw_matched_support",
        "m/s^3",
        "raw_linear_jerk_rms_mps3",
    ),
    MetricDefinition(
        "linear_jerk_rms",
        "reconstructed_matched_support",
        "m/s^3",
        "reconstructed_linear_jerk_rms_mps3",
    ),
    MetricDefinition(
        "linear_jerk_rms",
        "reconstructed_full_support",
        "m/s^3",
        "reconstructed_full_linear_jerk_rms_mps3",
    ),
    MetricDefinition(
        "linear_jerk_support_coverage",
        "reconstructed_vs_raw",
        "fraction",
        "linear_jerk_matched_coverage",
    ),
    MetricDefinition(
        "angular_jerk_rms",
        "raw_matched_support",
        "rad/s^3",
        "raw_angular_jerk_rms_radps3",
    ),
    MetricDefinition(
        "angular_jerk_rms",
        "reconstructed_matched_support",
        "rad/s^3",
        "reconstructed_angular_jerk_rms_radps3",
    ),
    MetricDefinition(
        "angular_jerk_rms",
        "reconstructed_full_support",
        "rad/s^3",
        "reconstructed_full_angular_jerk_rms_radps3",
    ),
    MetricDefinition(
        "angular_jerk_support_coverage",
        "reconstructed_vs_raw",
        "fraction",
        "angular_jerk_matched_coverage",
    ),
    MetricDefinition(
        "xy_rmse",
        "reconstructed_vs_raw",
        "m",
        "xy_rmse_m",
    ),
)

FRAME_METRICS = (
    MetricDefinition(
        "linear_jerk",
        "raw_matched_support",
        "m/s^3",
        "raw_linear_jerk_mps3",
    ),
    MetricDefinition(
        "linear_jerk",
        "reconstructed_matched_support",
        "m/s^3",
        "reconstructed_linear_jerk_mps3",
    ),
    MetricDefinition(
        "linear_jerk",
        "reconstructed_full_support",
        "m/s^3",
        "reconstructed_full_linear_jerk_mps3",
    ),
    MetricDefinition(
        "angular_jerk",
        "raw_matched_support",
        "rad/s^3",
        "raw_angular_jerk_radps3",
    ),
    MetricDefinition(
        "angular_jerk",
        "reconstructed_matched_support",
        "rad/s^3",
        "reconstructed_angular_jerk_radps3",
    ),
    MetricDefinition(
        "angular_jerk",
        "reconstructed_full_support",
        "rad/s^3",
        "reconstructed_full_angular_jerk_radps3",
    ),
)


@dataclass
class RunningMoments:
    """Mergeable population moments with finite-value filtering."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update_many(self, values: np.ndarray) -> None:
        finite = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = finite[np.isfinite(finite)]
        if len(finite) == 0:
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
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.m2 = other.m2
            self.minimum = other.minimum
            self.maximum = other.maximum
            return
        combined_count = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += (
            other.m2
            + delta * delta * self.count * other.count / combined_count
        )
        self.mean += delta * other.count / combined_count
        self.count = combined_count
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)

    @property
    def std(self) -> float:
        if self.count == 0:
            return math.nan
        return math.sqrt(max(0.0, self.m2 / self.count))

    def to_state(self) -> dict:
        return {
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "minimum": self.minimum if self.count else None,
            "maximum": self.maximum if self.count else None,
        }

    def summary(
        self,
        *,
        p01: float | None,
        p99: float | None,
    ) -> dict:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "range": None,
                "p01": None,
                "p99": None,
                "p99_minus_p01": None,
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
                p99 - p01
                if p01 is not None and p99 is not None
                else None
            ),
        }

    @classmethod
    def from_state(cls, state: Mapping) -> "RunningMoments":
        count = int(state["count"])
        return cls(
            count=count,
            mean=float(state["mean"]),
            m2=float(state["m2"]),
            minimum=(
                float(state["minimum"])
                if count
                else math.inf
            ),
            maximum=(
                float(state["maximum"])
                if count
                else -math.inf
            ),
        )


@dataclass(frozen=True)
class TrackMetricValues:
    """Agent scalar metrics and frame samples for one matched track."""

    object_type_name: str
    agent_values: dict[str, float]
    frame_values: dict[str, np.ndarray]


@dataclass(frozen=True)
class ScenarioMetricBatch:
    """Type-grouped metric arrays produced by one scenario."""

    scenario_id: str
    agent_count: int
    agent_values: dict[str, dict[str, np.ndarray]]
    frame_values: dict[str, dict[str, np.ndarray]]


@dataclass(frozen=True)
class _PairedMetricSamples:
    raw: np.ndarray
    reconstructed: np.ndarray
    reconstructed_full: np.ndarray
    raw_support_count: int
    matched_count: int
    reconstructed_full_count: int
    matched_coverage: float


def _finite_rms(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return math.nan
    return float(np.sqrt(np.mean(finite**2)))


def _paired_metric_samples(
    raw_values: np.ndarray,
    reconstructed_values: np.ndarray,
    raw_validity: np.ndarray,
    reconstructed_validity: np.ndarray,
) -> _PairedMetricSamples:
    raw_values = np.asarray(raw_values, dtype=np.float64)
    reconstructed_values = np.asarray(
        reconstructed_values,
        dtype=np.float64,
    )
    raw_support = np.asarray(raw_validity, dtype=bool) & np.isfinite(
        raw_values
    )
    reconstructed_support = np.asarray(
        reconstructed_validity,
        dtype=bool,
    ) & np.isfinite(reconstructed_values)
    matched_support = raw_support & reconstructed_support
    raw_support_count = int(np.sum(raw_support))
    matched_count = int(np.sum(matched_support))
    reconstructed_full_count = int(np.sum(reconstructed_support))
    matched_coverage = (
        matched_count / raw_support_count
        if raw_support_count
        else math.nan
    )
    if matched_count == raw_support_count:
        raw_paired = raw_values[raw_support]
        reconstructed_paired = reconstructed_values[raw_support]
    else:
        raw_paired = np.empty(0, dtype=np.float64)
        reconstructed_paired = np.empty(0, dtype=np.float64)
    return _PairedMetricSamples(
        raw=raw_paired,
        reconstructed=reconstructed_paired,
        reconstructed_full=reconstructed_values[reconstructed_support],
        raw_support_count=raw_support_count,
        matched_count=matched_count,
        reconstructed_full_count=reconstructed_full_count,
        matched_coverage=matched_coverage,
    )


def _track_arrays(track, count: int) -> tuple[np.ndarray, np.ndarray]:
    states = track.states[:count]
    position = np.asarray(
        [
            [state.center_x, state.center_y]
            for state in states
        ],
        dtype=np.float64,
    )
    valid = np.asarray(
        [state.valid for state in states],
        dtype=bool,
    )
    return position, valid


def evaluate_track(
    raw_track,
    reconstructed_track,
    timestamps,
) -> TrackMetricValues:
    """Evaluate one track pair on raw-matched and reconstructed-full support."""

    time = list(timestamps)
    count = min(
        len(raw_track.states),
        len(reconstructed_track.states),
        len(time),
    )
    raw_position, raw_valid = _track_arrays(raw_track, count)
    reconstructed_position, reconstructed_valid = _track_arrays(
        reconstructed_track,
        count,
    )
    raw_kinematics = compute_track_kinematics(
        raw_track,
        time[:count],
    )
    reconstructed_kinematics = compute_track_kinematics(
        reconstructed_track,
        time[:count],
    )
    linear_jerk = _paired_metric_samples(
        raw_kinematics.linear_jerk,
        reconstructed_kinematics.linear_jerk,
        raw_kinematics.jerk_validity,
        reconstructed_kinematics.jerk_validity,
    )
    angular_jerk = _paired_metric_samples(
        raw_kinematics.angular_jerk,
        reconstructed_kinematics.angular_jerk,
        raw_kinematics.jerk_validity,
        reconstructed_kinematics.jerk_validity,
    )

    comparable = (
        raw_valid
        & reconstructed_valid
        & np.all(np.isfinite(raw_position), axis=1)
        & np.all(np.isfinite(reconstructed_position), axis=1)
    )
    if np.any(comparable):
        displacement_squared = np.sum(
            (
                reconstructed_position[comparable]
                - raw_position[comparable]
            )
            ** 2,
            axis=1,
        )
        xy_rmse = float(np.sqrt(np.mean(displacement_squared)))
    else:
        xy_rmse = math.nan

    object_type = int(raw_track.object_type)
    return TrackMetricValues(
        object_type_name=OBJECT_TYPE_NAMES.get(
            object_type,
            str(object_type),
        ),
        agent_values={
            "raw_linear_jerk_rms_mps3": _finite_rms(
                linear_jerk.raw
            ),
            "reconstructed_linear_jerk_rms_mps3": _finite_rms(
                linear_jerk.reconstructed
            ),
            "reconstructed_full_linear_jerk_rms_mps3": _finite_rms(
                linear_jerk.reconstructed_full
            ),
            "linear_jerk_matched_coverage": (
                linear_jerk.matched_coverage
            ),
            "raw_angular_jerk_rms_radps3": _finite_rms(
                angular_jerk.raw
            ),
            "reconstructed_angular_jerk_rms_radps3": _finite_rms(
                angular_jerk.reconstructed
            ),
            "reconstructed_full_angular_jerk_rms_radps3": _finite_rms(
                angular_jerk.reconstructed_full
            ),
            "angular_jerk_matched_coverage": (
                angular_jerk.matched_coverage
            ),
            "xy_rmse_m": xy_rmse,
        },
        frame_values={
            "raw_linear_jerk_mps3": linear_jerk.raw,
            "reconstructed_linear_jerk_mps3": (
                linear_jerk.reconstructed
            ),
            "reconstructed_full_linear_jerk_mps3": (
                linear_jerk.reconstructed_full
            ),
            "raw_angular_jerk_radps3": angular_jerk.raw,
            "reconstructed_angular_jerk_radps3": (
                angular_jerk.reconstructed
            ),
            "reconstructed_full_angular_jerk_radps3": (
                angular_jerk.reconstructed_full
            ),
        },
    )


def _group_track_metrics(
    evaluations: list[TrackMetricValues],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
]:
    agent_lists: dict[str, dict[str, list[float]]] = {}
    frame_lists: dict[str, dict[str, list[np.ndarray]]] = {}
    for evaluation in evaluations:
        scope_agent = agent_lists.setdefault(
            evaluation.object_type_name,
            {},
        )
        for key, value in evaluation.agent_values.items():
            scope_agent.setdefault(key, []).append(value)
        scope_frame = frame_lists.setdefault(
            evaluation.object_type_name,
            {},
        )
        for key, values in evaluation.frame_values.items():
            scope_frame.setdefault(key, []).append(
                np.asarray(values, dtype=np.float64)
            )

    agent_values = {
        scope: {
            key: np.asarray(values, dtype=np.float64)
            for key, values in metrics.items()
        }
        for scope, metrics in agent_lists.items()
    }
    frame_values = {
        scope: {
            key: (
                np.concatenate(values).astype(
                    np.float64,
                    copy=False,
                )
                if values
                else np.empty(0, dtype=np.float64)
            )
            for key, values in metrics.items()
        }
        for scope, metrics in frame_lists.items()
    }
    return agent_values, frame_values


def evaluate_scenario_pair(
    raw_scenario,
    reconstructed_scenario,
) -> ScenarioMetricBatch:
    """Evaluate every track in a matched raw/reconstructed scenario pair."""

    if raw_scenario.scenario_id != reconstructed_scenario.scenario_id:
        raise ValueError("raw and reconstructed scenario IDs differ")
    reconstructed_by_id = {
        int(track.id): track
        for track in reconstructed_scenario.tracks
    }
    raw_ids = [int(track.id) for track in raw_scenario.tracks]
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("raw scenario track IDs must be unique")
    if set(raw_ids) != set(reconstructed_by_id):
        raise ValueError("raw and reconstructed track ID sets differ")

    evaluations = []
    for raw_track in raw_scenario.tracks:
        reconstructed_track = reconstructed_by_id[int(raw_track.id)]
        if int(raw_track.object_type) != int(
            reconstructed_track.object_type
        ):
            raise ValueError("raw and reconstructed object types differ")
        evaluations.append(
            evaluate_track(
                raw_track,
                reconstructed_track,
                raw_scenario.timestamps_seconds,
            )
        )
    agent_values, frame_values = _group_track_metrics(evaluations)
    return ScenarioMetricBatch(
        scenario_id=str(raw_scenario.scenario_id),
        agent_count=len(evaluations),
        agent_values=agent_values,
        frame_values=frame_values,
    )


@dataclass
class EvaluationAccumulator:
    """Exact moments for all/type scopes across scenario batches."""

    scenarios: int = 0
    agents: int = 0
    agent_moments: dict[
        str,
        dict[str, RunningMoments],
    ] = field(default_factory=dict)
    frame_moments: dict[
        str,
        dict[str, RunningMoments],
    ] = field(default_factory=dict)

    def add_batch(self, batch: ScenarioMetricBatch) -> None:
        self.scenarios += 1
        self.agents += batch.agent_count
        for target, grouped in (
            (self.agent_moments, batch.agent_values),
            (self.frame_moments, batch.frame_values),
        ):
            for scope, metrics in grouped.items():
                for output_scope in ("all", scope):
                    scope_moments = target.setdefault(
                        output_scope,
                        {},
                    )
                    for key, values in metrics.items():
                        scope_moments.setdefault(
                            key,
                            RunningMoments(),
                        ).update_many(values)

    @staticmethod
    def _moments_to_state(
        grouped: dict[str, dict[str, RunningMoments]],
    ) -> dict:
        return {
            scope: {
                key: moments.to_state()
                for key, moments in metrics.items()
            }
            for scope, metrics in grouped.items()
        }

    @staticmethod
    def _moments_from_state(state: Mapping) -> dict:
        return {
            str(scope): {
                str(key): RunningMoments.from_state(moment_state)
                for key, moment_state in metrics.items()
            }
            for scope, metrics in state.items()
        }

    def to_state(self) -> dict:
        return {
            "scenarios": self.scenarios,
            "agents": self.agents,
            "agent_moments": self._moments_to_state(
                self.agent_moments
            ),
            "frame_moments": self._moments_to_state(
                self.frame_moments
            ),
        }

    @classmethod
    def from_state(cls, state: Mapping) -> "EvaluationAccumulator":
        return cls(
            scenarios=int(state["scenarios"]),
            agents=int(state["agents"]),
            agent_moments=cls._moments_from_state(
                state["agent_moments"]
            ),
            frame_moments=cls._moments_from_state(
                state["frame_moments"]
            ),
        )
