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
# its affiliates is strictly prohibited.

"""Body-frame dynamics extracted continuously from complete trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FullTrajectoryDynamics:
    """Per-frame dynamics and the frames at which they are available."""

    values: np.ndarray
    valid: np.ndarray


class TransitionDynamicsAccumulator:
    """Accumulate class-specific previous/current-token dynamics."""

    _CLASS_NAMES = ("veh", "ped", "cyc")

    def __init__(self, n_agent_types: int, n_token: int) -> None:
        if n_agent_types != len(self._CLASS_NAMES):
            raise ValueError("n_agent_types must be 3")
        if n_token < 1:
            raise ValueError("n_token must be positive")
        self.n_agent_types = n_agent_types
        self.n_token = n_token
        self.pair_sum = np.zeros(
            (n_agent_types, n_token, n_token, 3),
            dtype=np.float64,
        )
        self.pair_count = np.zeros(
            (n_agent_types, n_token, n_token),
            dtype=np.int64,
        )
        self.marginal_sum = np.zeros(
            (n_agent_types, n_token, 3),
            dtype=np.float64,
        )
        self.marginal_count = np.zeros(
            (n_agent_types, n_token),
            dtype=np.int64,
        )

    def add(
        self,
        agent_type: np.ndarray,
        previous_token: np.ndarray,
        current_token: np.ndarray,
        values: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        """Add flattened token-transition occurrences."""

        agent_type = np.asarray(agent_type)
        previous_token = np.asarray(previous_token)
        current_token = np.asarray(current_token)
        values = np.asarray(values, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)
        expected_shape = agent_type.shape
        if (
            previous_token.shape != expected_shape
            or current_token.shape != expected_shape
            or valid.shape != expected_shape
            or values.shape != (*expected_shape, 3)
        ):
            raise ValueError(
                "agent_type, token indices, values, and valid must share "
                "their occurrence dimensions"
            )

        accepted = valid & np.isfinite(values).all(axis=-1)
        if not accepted.any():
            return
        classes = agent_type[accepted].astype(np.int64, copy=False)
        previous = previous_token[accepted].astype(np.int64, copy=False)
        current = current_token[accepted].astype(np.int64, copy=False)
        accepted_values = values[accepted]
        if (
            np.any(classes < 0)
            or np.any(classes >= self.n_agent_types)
            or np.any(previous < 0)
            or np.any(previous >= self.n_token)
            or np.any(current < 0)
            or np.any(current >= self.n_token)
        ):
            raise ValueError("agent type or token index is outside the accumulator")

        np.add.at(
            self.pair_sum,
            (classes, previous, current),
            accepted_values,
        )
        np.add.at(self.pair_count, (classes, previous, current), 1)
        np.add.at(
            self.marginal_sum,
            (classes, current),
            accepted_values,
        )
        np.add.at(self.marginal_count, (classes, current), 1)

    def finalize(
        self,
        isolated_fallback: np.ndarray,
        *,
        shrinkage_count: float = 8.0,
    ) -> tuple[np.ndarray, dict]:
        """Return a dense shrunk table and JSON-serializable coverage stats."""

        isolated_fallback = np.asarray(isolated_fallback, dtype=np.float64)
        expected_shape = (self.n_agent_types, self.n_token, 3)
        if isolated_fallback.shape != expected_shape:
            raise ValueError(
                f"isolated_fallback must have shape {expected_shape}"
            )
        if not np.isfinite(isolated_fallback).all():
            raise ValueError("isolated_fallback must be finite")
        if not np.isfinite(shrinkage_count) or shrinkage_count <= 0.0:
            raise ValueError("shrinkage_count must be finite and positive")

        marginal = isolated_fallback.copy()
        observed_tokens = self.marginal_count > 0
        marginal[observed_tokens] = (
            self.marginal_sum[observed_tokens]
            / self.marginal_count[observed_tokens, None]
        )
        values = (
            self.pair_sum
            + shrinkage_count * marginal[:, None, :, :]
        ) / (self.pair_count[..., None] + shrinkage_count)
        if not np.isfinite(values).all():
            raise ValueError("final transition dynamics contain non-finite values")

        statistics = {
            "occurrences": int(self.pair_count.sum()),
            "occurrences_by_class": {
                name: int(self.pair_count[index].sum())
                for index, name in enumerate(self._CLASS_NAMES)
            },
            "observed_pairs": {
                name: int(np.count_nonzero(self.pair_count[index]))
                for index, name in enumerate(self._CLASS_NAMES)
            },
            "observed_tokens": {
                name: int(np.count_nonzero(self.marginal_count[index]))
                for index, name in enumerate(self._CLASS_NAMES)
            },
        }
        return values.astype(np.float16), statistics


def _true_runs(mask: np.ndarray):
    start = None
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(mask)


def extract_full_trajectory_dynamics(
    position: np.ndarray,
    heading: np.ndarray,
    valid_mask: np.ndarray,
    *,
    dt: float = 0.1,
    clipping_limits: Sequence[float] = (15.0, 3.0, 15.0),
) -> FullTrajectoryDynamics:
    """Calculate `[a_lon, angular_speed, a_lat]` on continuous valid runs."""

    position = np.asarray(position, dtype=np.float64)
    heading = np.asarray(heading, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    limits = np.asarray(clipping_limits, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] < 2:
        raise ValueError("position must have shape [steps, >=2]")
    if heading.shape != position.shape[:1] or valid_mask.shape != position.shape[:1]:
        raise ValueError(
            "position, heading, and valid_mask must share the step dimension"
        )
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if (
        limits.shape != (3,)
        or not np.isfinite(limits).all()
        or np.any(limits <= 0.0)
    ):
        raise ValueError(
            "clipping_limits must contain three finite positive values"
        )

    values = np.zeros((len(position), 3), dtype=np.float32)
    output_valid = np.zeros(len(position), dtype=bool)
    usable = (
        valid_mask
        & np.isfinite(position[:, :2]).all(axis=1)
        & np.isfinite(heading)
    )
    for start, end in _true_runs(usable):
        if end - start < 3:
            continue

        xy = position[start:end, :2]
        theta = np.unwrap(heading[start:end])
        velocity = np.column_stack(
            [
                np.gradient(xy[:, axis], dt, edge_order=2)
                for axis in range(2)
            ]
        )
        acceleration = np.column_stack(
            [
                np.gradient(velocity[:, axis], dt, edge_order=2)
                for axis in range(2)
            ]
        )
        angular_speed = np.gradient(theta, dt, edge_order=2)
        cosine = np.cos(theta)
        sine = np.sin(theta)
        run_values = np.column_stack(
            (
                acceleration[:, 0] * cosine + acceleration[:, 1] * sine,
                angular_speed,
                -acceleration[:, 0] * sine + acceleration[:, 1] * cosine,
            )
        )
        finite = np.isfinite(run_values).all(axis=1)
        run_output = values[start:end]
        run_output[finite] = np.clip(run_values[finite], -limits, limits)
        output_valid[start:end][finite] = True

    return FullTrajectoryDynamics(values=values, valid=output_valid)
