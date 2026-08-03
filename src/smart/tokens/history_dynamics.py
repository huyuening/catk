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

"""History-only body-frame dynamics extracted during CatK preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from .trajectory_filter_reconstructor import (
    TrajectoryFilterConfig,
    _gradient,
    _timestamps_for_count,
    reconstruct_trajectory_arrays,
)


@dataclass(frozen=True)
class HistoryDynamicsResult:
    """Token endpoint dynamics together with their reconstructed history."""

    values: np.ndarray
    valid: np.ndarray
    reconstructed_position: np.ndarray
    reconstructed_heading: np.ndarray
    reconstructed_valid: np.ndarray


@torch.no_grad()
def estimate_raw_history_dynamics(
    position: Tensor,
    heading: Tensor,
    valid_mask: Tensor,
    *,
    num_historical_steps: int = 11,
    token_shift_steps: int = 5,
    dt: float = 0.1,
    max_abs_longitudinal_accel_mps2: float = 15.0,
    max_abs_angular_speed_radps: float = 3.0,
    max_abs_lateral_accel_mps2: float = 15.0,
) -> tuple[Tensor, Tensor]:
    """Calculate causal endpoint dynamics without reconstructing trajectories.

    The input tensors are used as stored by ordinary CatK preprocessing. At
    each history-token endpoint, backward finite differences over the last
    three cached positions produce world-frame acceleration. The acceleration
    is projected through the cached endpoint heading, while wrapped heading
    difference produces angular speed. No smoothing, fitting, gap filling, or
    heading correction is applied here.
    """

    if position.ndim != 3 or position.size(-1) < 2:
        raise ValueError("position must have shape [n_agent, n_step, >=2]")
    if (
        heading.shape != position.shape[:2]
        or valid_mask.shape != position.shape[:2]
    ):
        raise ValueError("heading and valid_mask must match position[:2]")
    if num_historical_steps > position.size(1) or num_historical_steps < 3:
        raise ValueError(
            "num_historical_steps must fit the available trajectory"
        )
    if (
        token_shift_steps < 2
        or (num_historical_steps - 1) % token_shift_steps
    ):
        raise ValueError(
            "token_shift_steps must produce endpoints with 3-frame support"
        )
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")

    limits_tuple = (
        max_abs_longitudinal_accel_mps2,
        max_abs_angular_speed_radps,
        max_abs_lateral_accel_mps2,
    )
    if any(
        not np.isfinite(value) or value <= 0 for value in limits_tuple
    ):
        raise ValueError(
            "dynamics clipping limits must be finite and positive"
        )

    output_dtype = position.dtype
    compute_dtype = (
        torch.float32
        if output_dtype in (torch.float16, torch.bfloat16)
        else output_dtype
    )
    endpoints = torch.arange(
        token_shift_steps,
        num_historical_steps,
        token_shift_steps,
        device=position.device,
    )
    xy = position[..., :2].to(dtype=compute_dtype)
    theta = heading.to(device=position.device, dtype=compute_dtype)
    valid = valid_mask.to(device=position.device, dtype=torch.bool)

    p0 = xy[:, endpoints - 2]
    p1 = xy[:, endpoints - 1]
    p2 = xy[:, endpoints]
    theta_previous = theta[:, endpoints - 1]
    theta_current = theta[:, endpoints]
    feature_valid = (
        valid[:, endpoints - 2]
        & valid[:, endpoints - 1]
        & valid[:, endpoints]
        & torch.isfinite(p0).all(dim=-1)
        & torch.isfinite(p1).all(dim=-1)
        & torch.isfinite(p2).all(dim=-1)
        & torch.isfinite(theta_previous)
        & torch.isfinite(theta_current)
    )

    velocity_previous = (p1 - p0) / dt
    velocity_current = (p2 - p1) / dt
    acceleration = (velocity_current - velocity_previous) / dt
    delta_heading = torch.atan2(
        torch.sin(theta_current - theta_previous),
        torch.cos(theta_current - theta_previous),
    )
    angular_speed = delta_heading / dt
    cosine = torch.cos(theta_current)
    sine = torch.sin(theta_current)
    longitudinal = (
        acceleration[..., 0] * cosine + acceleration[..., 1] * sine
    )
    lateral = (
        -acceleration[..., 0] * sine + acceleration[..., 1] * cosine
    )
    values = torch.stack((longitudinal, angular_speed, lateral), dim=-1)
    limits = values.new_tensor(limits_tuple)
    values = torch.maximum(torch.minimum(values, limits), -limits)
    values = torch.where(
        feature_valid.unsqueeze(-1), values, torch.zeros_like(values)
    )
    return values.to(output_dtype), feature_valid


def extract_history_dynamics(
    position: np.ndarray,
    heading: np.ndarray,
    valid_mask: np.ndarray,
    agent_type: int,
    *,
    timestamps: Iterable[float] | None = None,
    num_historical_steps: int = 11,
    token_shift_steps: int = 5,
    min_token_observed_frames: int = 3,
    filter_config: TrajectoryFilterConfig | None = None,
    max_abs_longitudinal_accel_mps2: float = 15.0,
    max_abs_angular_speed_radps: float = 3.0,
    max_abs_lateral_accel_mps2: float = 15.0,
) -> HistoryDynamicsResult:
    """Reconstruct one 10 Hz history and sample body dynamics at token ends.

    CatK types use the zero-based convention (vehicle=0, pedestrian=1,
    cyclist=2). The complete observable frames ``0..10`` are passed through
    the same reverse-aware xy/heading filter used for trajectory-vocabulary
    reconstruction. Internal gaps are filled as part of that single history
    reconstruction. No future state after ``num_historical_steps`` is read.

    The output order is signed longitudinal acceleration, angular speed, and
    signed lateral acceleration. Linear acceleration is obtained by twice
    differentiating reconstructed xy and projecting it into reconstructed body
    heading theta. Angular speed is the derivative of reconstructed theta.
    Standard CatK history therefore produces values at frames 5 and 10.
    """

    position = np.asarray(position, dtype=float)
    heading = np.asarray(heading, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if position.ndim != 2 or position.shape[1] < 2:
        raise ValueError("position must have shape [steps, >=2]")
    if (
        heading.shape != position.shape[:1]
        or valid_mask.shape != position.shape[:1]
    ):
        raise ValueError(
            "position, heading, and valid_mask must share the step dimension"
        )
    if agent_type not in (0, 1, 2):
        raise ValueError("agent_type must be 0, 1, or 2")
    if num_historical_steps < 1 or num_historical_steps > len(position):
        raise ValueError("num_historical_steps must be within the available trajectory")
    if token_shift_steps < 1 or (num_historical_steps - 1) % token_shift_steps:
        raise ValueError(
            "num_historical_steps - 1 must be divisible by token_shift_steps"
        )
    if min_token_observed_frames < 2:
        raise ValueError("min_token_observed_frames must be at least two")

    limits = np.asarray(
        [
            max_abs_longitudinal_accel_mps2,
            max_abs_angular_speed_radps,
            max_abs_lateral_accel_mps2,
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(limits)) or np.any(limits <= 0):
        raise ValueError("dynamics clipping limits must be finite and positive")

    position_history = position[:num_historical_steps]
    heading_history = heading[:num_historical_steps]
    valid_history = valid_mask[:num_historical_steps]
    if timestamps is None:
        raw_time = np.arange(num_historical_steps, dtype=float) * 0.1
    else:
        raw_time = np.asarray(
            list(timestamps)[:num_historical_steps], dtype=float
        )
        if len(raw_time) != num_historical_steps:
            raise ValueError("timestamps must cover num_historical_steps")
    time = _timestamps_for_count(raw_time, num_historical_steps)

    config = filter_config or TrajectoryFilterConfig(
        position_window=11,
        z_window=11,
        heading_window=11,
        max_gap_frames=None,
    )
    reconstruction = reconstruct_trajectory_arrays(
        position_history,
        heading_history,
        valid_history,
        time,
        object_type=agent_type + 1,
        config=config,
    )

    endpoint_steps = np.arange(
        token_shift_steps, num_historical_steps, token_shift_steps, dtype=int
    )
    values = np.zeros((len(endpoint_steps), 3), dtype=np.float32)
    feature_valid = np.zeros(len(endpoint_steps), dtype=bool)

    for run_start, run_end in _true_runs(reconstruction.valid):
        if run_end - run_start < 3:
            continue
        run_time = time[run_start:run_end]
        run_xy = reconstruction.positions[run_start:run_end, :2]
        run_heading = reconstruction.heading[run_start:run_end]

        velocity_x = _gradient(run_xy[:, 0], run_time)
        velocity_y = _gradient(run_xy[:, 1], run_time)
        acceleration_x = _gradient(velocity_x, run_time)
        acceleration_y = _gradient(velocity_y, run_time)
        angular_speed = _gradient(np.unwrap(run_heading), run_time)

        cosine = np.cos(run_heading)
        sine = np.sin(run_heading)
        longitudinal_acceleration = acceleration_x * cosine + acceleration_y * sine
        lateral_acceleration = -acceleration_x * sine + acceleration_y * cosine
        run_values = np.column_stack(
            (longitudinal_acceleration, angular_speed, lateral_acceleration)
        )

        for output_index, endpoint in enumerate(endpoint_steps):
            if not (run_start <= endpoint < run_end):
                continue
            interval_start = endpoint - token_shift_steps
            observed_support = int(
                np.sum(valid_history[interval_start : endpoint + 1])
            )
            local_index = endpoint - run_start
            candidate = run_values[local_index]
            if (
                observed_support < min_token_observed_frames
                or not np.all(np.isfinite(candidate))
            ):
                continue
            values[output_index] = np.clip(candidate, -limits, limits)
            feature_valid[output_index] = True

    return HistoryDynamicsResult(
        values=values,
        valid=feature_valid,
        reconstructed_position=reconstruction.positions.astype(np.float32),
        reconstructed_heading=reconstruction.heading.astype(np.float32),
        reconstructed_valid=reconstruction.valid,
    )


def _true_runs(mask: np.ndarray):
    """Local iterator kept private to avoid exposing filter implementation."""

    start = None
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(mask)
