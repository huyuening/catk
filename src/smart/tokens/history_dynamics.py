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

"""Causal history-only dynamics for optional CatK agent conditioning."""

from typing import Sequence

import torch
from torch import Tensor


@torch.no_grad()
def estimate_history_dynamics(
    position: Tensor,
    valid_mask: Tensor,
    agent_type: Tensor,
    *,
    num_historical_steps: int = 11,
    token_shift_steps: int = 5,
    dt: float = 0.1,
    fit_window_steps: int = 6,
    min_speed_mps: Sequence[float] = (0.5, 0.2, 0.3),
    ridge: float = 1.0e-6,
    max_abs_longitudinal_accel_mps2: float = 15.0,
    max_abs_angular_speed_radps: float = 3.0,
    max_abs_lateral_accel_mps2: float = 15.0,
) -> Tensor:
    """Estimate token-aligned dynamics from observable history.

    Each CatK history token receives ``[a_parallel, omega, a_perp]`` from its
    own reconstructed 10 Hz interval.  For the standard 11-frame history and
    five-step token shift this returns ``[n_agent, 2, 3]`` for frames 0--5 and
    5--10.  A quadratic curve is fitted independently to x/y in each interval;
    its endpoint derivatives define velocity and acceleration.  The local
    tangent follows motion rather than object heading, so the same quantities
    remain meaningful for vehicles, pedestrians, and cyclists.

    Only ``position[:, :num_historical_steps]`` is inspected.  At low speed,
    angular and lateral terms are zeroed because the course direction is not
    identifiable; longitudinal acceleration may still use the net historical
    direction to represent starting or stopping.
    """

    if position.ndim != 3 or position.size(-1) < 2:
        raise ValueError("position must have shape [n_agent, n_step, >=2]")
    if valid_mask.shape != position.shape[:2]:
        raise ValueError(
            "valid_mask must match the first two position dimensions: "
            f"{valid_mask.shape} != {position.shape[:2]}"
        )
    if agent_type.ndim != 1 or agent_type.size(0) != position.size(0):
        raise ValueError("agent_type must have shape [n_agent]")
    if num_historical_steps < 1 or num_historical_steps > position.size(1):
        raise ValueError(
            "num_historical_steps must be within the available trajectory: "
            f"{num_historical_steps} not in [1, {position.size(1)}]"
        )
    if fit_window_steps < 3:
        raise ValueError("fit_window_steps must be at least 3")
    if token_shift_steps < 1:
        raise ValueError("token_shift_steps must be positive")
    if (num_historical_steps - 1) % token_shift_steps != 0:
        raise ValueError(
            "num_historical_steps - 1 must be divisible by token_shift_steps"
        )
    if dt <= 0:
        raise ValueError("dt must be positive")
    if ridge <= 0:
        raise ValueError("ridge must be positive")

    type_thresholds = tuple(float(value) for value in min_speed_mps)
    if len(type_thresholds) != 3:
        raise ValueError("min_speed_mps must contain [vehicle, pedestrian, cyclist]")
    if any(value < 0 for value in type_thresholds):
        raise ValueError("min_speed_mps values must be non-negative")
    if torch.any((agent_type < 0) | (agent_type > 2)):
        raise ValueError("agent_type values must be 0, 1, or 2")

    output_dtype = position.dtype
    fit_dtype = (
        torch.float32
        if position.dtype in (torch.float16, torch.bfloat16)
        else position.dtype
    )
    thresholds = torch.as_tensor(
        type_thresholds, dtype=fit_dtype, device=position.device
    )[agent_type.long()]
    limits = torch.tensor(
        [
            max_abs_longitudinal_accel_mps2,
            max_abs_angular_speed_radps,
            max_abs_lateral_accel_mps2,
        ],
        dtype=fit_dtype,
        device=position.device,
    )
    if torch.any(limits <= 0):
        raise ValueError("dynamics clipping limits must be positive")

    token_dynamics = []
    for endpoint in range(
        token_shift_steps, num_historical_steps, token_shift_steps
    ):
        window_steps = min(fit_window_steps, endpoint + 1)
        start = endpoint + 1 - window_steps
        token_dynamics.append(
            _estimate_interval_dynamics(
                position=position[:, start : endpoint + 1, :2],
                valid_mask=valid_mask[:, start : endpoint + 1],
                speed_thresholds=thresholds,
                dt=dt,
                ridge=ridge,
                limits=limits,
                fit_dtype=fit_dtype,
            )
        )
    return torch.stack(token_dynamics, dim=1).to(output_dtype)


def _estimate_interval_dynamics(
    position: Tensor,
    valid_mask: Tensor,
    speed_thresholds: Tensor,
    *,
    dt: float,
    ridge: float,
    limits: Tensor,
    fit_dtype: torch.dtype,
) -> Tensor:
    """Fit one reconstructed token interval and return endpoint dynamics."""

    n_agent, window_steps = position.shape[:2]
    pos = position.to(fit_dtype)
    valid = valid_mask.bool()
    finite = torch.isfinite(pos).all(dim=-1)
    valid = valid & finite
    # Global WOMD coordinates can be large compared with sub-metre motion.
    # Translation does not affect derivatives and materially improves the
    # conditioning of the float32 polynomial fit.
    anchor = torch.where(
        valid[:, -1:].unsqueeze(-1), pos[:, -1:], torch.zeros_like(pos[:, -1:])
    )
    pos = pos - anchor
    pos = torch.where(valid.unsqueeze(-1), pos, torch.zeros_like(pos))

    times = (
        torch.arange(window_steps, device=position.device, dtype=fit_dtype)
        - (window_steps - 1)
    ) * dt
    design = torch.stack(
        [torch.ones_like(times), times, 0.5 * times.square()], dim=-1
    )  # [window, position/velocity/acceleration coefficients]

    weights = valid.to(fit_dtype)
    design_batch = design.unsqueeze(0).expand(n_agent, -1, -1)
    weighted_design_t = design_batch.transpose(1, 2) * weights.unsqueeze(1)
    normal = weighted_design_t @ design_batch
    rhs = weighted_design_t @ pos
    regularizer = torch.eye(3, dtype=fit_dtype, device=position.device) * ridge
    coefficients = torch.linalg.solve(normal + regularizer.unsqueeze(0), rhs)
    velocity = coefficients[:, 1]
    acceleration = coefficients[:, 2]

    valid_count = valid.sum(dim=-1)
    endpoint_valid = valid[:, -1]
    enough_history = (valid_count >= 3) & endpoint_valid

    first_valid_index = valid.to(torch.int64).argmax(dim=-1)
    batch_index = torch.arange(n_agent, device=position.device)
    first_position = pos[batch_index, first_valid_index]
    net_displacement = pos[:, -1] - first_position

    speed = torch.linalg.vector_norm(velocity, dim=-1)
    displacement_norm = torch.linalg.vector_norm(net_displacement, dim=-1)
    turning_valid = enough_history & (speed >= speed_thresholds)

    # Course tangent for omega/a_perp; a trailing displacement fallback keeps
    # signed a_parallel useful around starts and stops.
    velocity_tangent = velocity / speed.clamp_min(1.0e-6).unsqueeze(-1)
    fallback_tangent = net_displacement / displacement_norm.clamp_min(
        1.0e-6
    ).unsqueeze(-1)
    longitudinal_tangent = torch.where(
        turning_valid.unsqueeze(-1), velocity_tangent, fallback_tangent
    )
    longitudinal_valid = enough_history & (
        turning_valid | (displacement_norm > 1.0e-4)
    )

    acceleration_parallel = (acceleration * longitudinal_tangent).sum(dim=-1)
    acceleration_parallel = torch.where(
        longitudinal_valid,
        acceleration_parallel,
        torch.zeros_like(acceleration_parallel),
    )
    acceleration_perpendicular = (
        velocity_tangent[:, 0] * acceleration[:, 1]
        - velocity_tangent[:, 1] * acceleration[:, 0]
    )
    acceleration_perpendicular = torch.where(
        turning_valid,
        acceleration_perpendicular,
        torch.zeros_like(acceleration_perpendicular),
    )
    angular_speed = acceleration_perpendicular / speed.clamp_min(1.0e-6)
    angular_speed = torch.where(
        turning_valid, angular_speed, torch.zeros_like(angular_speed)
    )

    dynamics = torch.stack(
        [acceleration_parallel, angular_speed, acceleration_perpendicular], dim=-1
    )
    dynamics = torch.maximum(torch.minimum(dynamics, limits), -limits)
    dynamics = torch.nan_to_num(dynamics)
    return dynamics
